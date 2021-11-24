from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, url, targets, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.url = url
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.url)

    def __getitem__(self, index):
        url = str(self.url[index])
        url = " ".join(url.split())

        inputs = self.tokenizer.encode_plus(
            url,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
