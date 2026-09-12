import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import swanlab
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import json
from Dataloader import load_all_data, build_datasets, build_loaders, NewsDataset,set_seed
from train import train_epoch, evaluate
import sys
from dotenv import load_dotenv


env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("SWANLAB_API_KEY")
os.environ["SWANLAB_API_KEY"] = api_key

Conifg_path = 'config.json'
with open(Conifg_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

SEED = config.get('seed')
set_seed(SEED)


DATA_DIR = config['data_dir']
MODEL_PATH = config['model_path']
TRAIN_FILE = os.path.join(DATA_DIR, config['train_file'])
DEV_FILE = os.path.join(DATA_DIR, config['dev_file'])
TEST_FILE = os.path.join(DATA_DIR, config['test_file'])
MAX_LEN = config['max_len']
EPOCHS = config['epochs']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = config['learning_rate']
BATCH_SIZE = config['batch_size']
DROPOUT = config['dropout']
BEST_DEV = config['best_dev_acc_init']
print(f"Using device: {DEVICE}")
print(f"Model will be loaded from: {MODEL_PATH}")
print(f"Random seed: {SEED}")


train_df, dev_df, test_df, label2id, id2label, num_classes = load_all_data(
    TRAIN_FILE, DEV_FILE, TEST_FILE
)

print("Loading tokenizer ")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

print("Building datasets...")
train_dataset, dev_dataset, test_dataset = build_datasets(
    train_df, dev_df, test_df, tokenizer, MAX_LEN
)

class BertClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super().__init__()
        print("Loading BERT model from local...")
        self.bert = BertModel.from_pretrained(MODEL_PATH, local_files_only=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        dropped = self.dropout(pooled)
        return self.classifier(dropped)


def run_experiment(lr, batch_size, dropout, best_dev, run_name=None, save_path=None, seed=SEED):
    set_seed(seed)

    train_loader, dev_loader, test_loader = build_loaders(
        batch_size, train_dataset, dev_dataset, test_dataset, tokenizer
    )
    model = BertClassifier(num_classes, dropout=dropout).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    swanlab.init(
        project="Bert-News-Classification",
        logdir="D:/swanlog",
        experiment_name=run_name or f"lr{lr}_bs{batch_size}_do{dropout}",
        config={"learning_rate": lr, "batch_size": batch_size, "dropout": dropout,
                "epochs": EPOCHS, "seed": seed},
        mode="offline"
    )

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, DEVICE)
        swanlab.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "dev_loss": dev_loss,
            "dev_acc": dev_acc
        })
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, dev_acc={dev_acc:.4f}")

    if dev_acc > best_dev:
        best_dev = dev_acc
        if save_path is not None:
            torch.save(model.state_dict(), save_path)
            print(f"Model weights saved to: {save_path}")

    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    swanlab.log({"test_acc": test_acc})
    print(f"Test Acc: {test_acc:.4f}")
    swanlab.finish()
    return test_acc, model

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    SAVE_PATH = os.path.join(DATA_DIR, "bert_classifier_weights.pt")
    print(f"\n=== Using lr={LR}, batch_size={BATCH_SIZE}, dropout={DROPOUT} ===")
    test_acc, model = run_experiment(
        lr=LR, batch_size=BATCH_SIZE, dropout=DROPOUT,
        best_dev=BEST_DEV, run_name="best_train", save_path=SAVE_PATH
    )
    print(f"Test Accuracy: {test_acc:.4f}")