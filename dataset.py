import torch
from torch.utils.data import Dataset
from typing import Tuple

class EmbeddedDataset(Dataset):
    def __init__(
        self, 
        emd_path: str,
        label_path: str
    ):
        self.emd = torch.load(emd_path)
        self.label = torch.load(label_path)
        
    def __len__(self) -> int:
        return len(self.emd)
    
    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.emd[index], self.label[index]

