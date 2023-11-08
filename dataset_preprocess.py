import torch
from torch.utils.data import Dataset


class AlfredDataset(Dataset):
    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text, task_type
    ):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

        if 'both' in target_text:
            self.prefix = self.data['prefix_both']
        else:
            self.prefix = self.data['prefix_{}'.format(task_type)]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        src = '{}: {}'.format(self.prefix[index], self.source_text[index])
        trg = str(self.target_text[index])

        source = self.tokenizer.batch_encode_plus(
            [src],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target = self.tokenizer.batch_encode_plus(
            [trg],
            max_length=self.target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }
