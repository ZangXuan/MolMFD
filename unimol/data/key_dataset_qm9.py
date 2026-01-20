from functools import lru_cache
from unicore.data import BaseWrapperDataset

class KeyDataset_qm9(BaseWrapperDataset):
    def __init__(self, dataset, key, tgt_idx):
        self.dataset = dataset
        self.key = key
        self.tgt_idx = tgt_idx

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return self.dataset[idx][self.key][self.tgt_idx]