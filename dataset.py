import pandas as pd
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.info = pd.read_csv(f"{self.path}/dataset_info.csv")

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        audio_path = f"{self.path}/{self.info.iloc[index][1]}"
        signal, sr = torchaudio.load(audio_path)
        print(signal.shape)
        print(sr)
        print("///////")
        cut_length = 1000000

        # trim audio arbitrarily
        # TODO reshape audio properly
        signal = signal.narrow(1, 0, cut_length)
        # signal.shape -> [1, cut_length]

        signal = signal[0]
        signal = signal.unsqueeze(0).unsqueeze(0)
        # signal.shape -> [1, 1, cut_length]

        return signal
