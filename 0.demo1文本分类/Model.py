import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, num_classes, model_path, dropout=0.1):
        super().__init__()
        print("Loading BERT model from local...")
        self.bert = BertModel.from_pretrained(model_path, local_files_only=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        dropped = self.dropout(pooled)
        return self.classifier(dropped)