import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, sp):
        self.src = src_texts
        self.tgt = tgt_texts
        self.sp = sp

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_ids = [1] + self.sp.encode(self.src[idx]) + [2]
        tgt_ids = [1] + self.sp.encode(self.tgt[idx]) + [2]
        return src_ids, tgt_ids


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    max_src = max(len(x) for x in src_batch)
    max_tgt = max(len(x) for x in tgt_batch)

    src_pad = [x + [0]*(max_src-len(x)) for x in src_batch]
    tgt_pad = [x + [0]*(max_tgt-len(x)) for x in tgt_batch]

    return torch.tensor(src_pad), torch.tensor(tgt_pad)