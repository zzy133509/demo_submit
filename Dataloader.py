import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
import os
import random
import numpy as np
import torch



class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, label2id=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = [
            f"{row['title']} {row['content']}" for _, row in self.df.iterrows()
        ]

        if label2id is not None:
            self.labels = torch.tensor(
                self.df['label'].map(label2id).values, dtype=torch.long
            )
        else:
            self.labels = torch.tensor(
                self.df['label_id'].values, dtype=torch.long
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'label': self.labels[idx].item()
        }


def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('_!_')
            if len(parts) >= 5:
                id_, label, category, title, content = parts[:5]
                data.append({
                    'id': id_,
                    'label': int(label),
                    'category': category,
                    'title': title,
                    'content': content
                })
    return pd.DataFrame(data)


def load_all_data(train_file, dev_file, test_file):
    print("Loading data...")
    train_df = load_data(train_file)
    dev_df = load_data(dev_file)
    test_df = load_data(test_file)
    print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

    all_labels = sorted(train_df['label'].unique())
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_classes = len(all_labels)
    print(f"Number of classes: {num_classes}")

    train_df['label_id'] = train_df['label'].map(label2id)
    dev_df['label_id'] = dev_df['label'].map(label2id)
    test_df['label_id'] = test_df['label'].map(label2id)
    return train_df, dev_df, test_df, label2id, id2label, num_classes


def build_datasets(train_df, dev_df, test_df, tokenizer, max_len):
    train_dataset = NewsDataset(train_df, tokenizer, max_len)
    dev_dataset = NewsDataset(dev_df, tokenizer, max_len)
    test_dataset = NewsDataset(test_df, tokenizer, max_len)
    return train_dataset, dev_dataset, test_dataset


def build_loaders(batch_size, train_dataset, dev_dataset, test_dataset, tokenizer):
    def collate_fn(batch):
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        features = [
            {'input_ids': item['input_ids'], 'attention_mask': item['attention_mask']}
            for item in batch
        ]

        batch_enc = tokenizer.pad(features, padding=True, return_tensors='pt')
        batch_enc['label'] = labels
        return batch_enc

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, dev_loader, test_loader

def set_seed(seed=42):
    """设置所有随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 时使用

    # 保证 cuDNN 确定性（会略微降低性能，但结果可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[set_seed] 随机种子已设置为: {seed}")