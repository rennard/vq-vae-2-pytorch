import pandas as pd
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.info = pd.read_csv(f'{self.path}/dataset_info.csv')

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        audio_path = f"{self.path}/{self.info.iloc[index][1]}"
        print(self.info.iloc(index)[1])
        signal, sr = torchaudio.load(audio_path)
        return signal
