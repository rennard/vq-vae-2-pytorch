import argparse
import sys
import os
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import AudioDataset
from torchaudio import save


from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist

output_sample_path = "sample/"


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2**15
        + 2**14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--nocuda", type=str)
    parser.add_argument("path", type=str)

    return parser.parse_args()


def run(args):
    device = None if args.nocuda else "cuda"

    if not Path(output_sample_path).exists():
        Path(output_sample_path).mkdir()

    args.distributed = dist.get_world_size() > 1

    # transform = transforms.Compose(
    #     [
    #         transforms.Resize(args.size),
    #         transforms.CenterCrop(args.size),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #     ]
    # )

    dataset = AudioDataset(args.path)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=args.batch // args.n_gpu, sampler=sampler, num_workers=2
    )

    model = VQVAE(1, embed_dim=1024).to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)

        if dist.is_primary():
            torch.save(model.state_dict(), f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")


def output_samples(index, epoch, sample, out):
    original_file_name = (
        f"sample/original_{str(epoch + 1).zfill(5)}_{str(index).zfill(5)}.mp3"
    )
    sample_file_name = (
        f"sample/sample_{str(epoch + 1).zfill(5)}_{str(index).zfill(5)}.mp3"
    )

    sample = sample[0]
    out = out[0]

    # TODO figure out how to get this
    sample_rate = 44100

    save(
        original_file_name,
        sample,
        sample_rate,
    )
    save(
        sample_file_name,
        sample,
        sample_rate,
    )


def train(epoch, loader, model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (song, sample_rate) in enumerate(loader):
        model.zero_grad()

        audio = song.to(device)

        out, latent_loss = model(audio)
        recon_loss = criterion(out, audio)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * audio.shape[0]
        part_mse_n = audio.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        # short circuit if not primary, we only want to sample for primary
        if not dist.is_primary():
            pass

        lr = optimizer.param_groups[0]["lr"]

        loader.set_description(
            (
                f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                f"lr: {lr:.5f}"
            )
        )

        if i % 100 == 0:
            model.eval()

            # TODO not sure if I need to do this
            sample = audio[:sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            output_samples(i, epoch, sample, out)
            model.train()


if __name__ == "__main__":
    args = get_arguments()
    dist.launch(run, args.n_gpu, 1, 0, args.dist_url, args=(args,))
