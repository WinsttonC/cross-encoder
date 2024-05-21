import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertConfig, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm 
import numpy as np

DATASET_PATH = ''
NUM_ATTENTION_HEADS = 8
NUM_ENCODER_BLOCKS = 4
LR = 1e-5
BATCH_SIZE = 10
num_epochs = 5

dataset = pd.read_csv(DATASET_PATH)

embeddings = dataset['emb'].tolist()
labels = dataset['label'].tolist()

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.classes = np.unique(labels)
        self.class_indices = {cls: np.where(labels == cls)[0] for cls in self.classes}
        
        # Проверка на четность
        assert batch_size % len(self.classes) == 0, "Batch size должен быть кратен количеству классов"
        self.samples_per_class = batch_size // len(self.classes)

    def __iter__(self):
        indices = []
        num_batches = len(self.labels) // self.batch_size
        
        for _ in range(num_batches):
            batch_indices = []
            for cls in self.classes:
                class_indices = np.random.choice(self.class_indices[cls], self.samples_per_class, replace=False)
                batch_indices.extend(class_indices)
            np.random.shuffle(batch_indices)
            indices.extend(batch_indices)
        
        return iter(indices)

    def __len__(self):
        return len(self.labels) // self.batch_size * self.batch_size

# Создание кастомного Dataset
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'embedding': torch.tensor(self.embeddings[idx], dtype=torch.float),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    
# Разделение данных на обучающую и тестовую выборки
train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42
)
balanced_sampler = BalancedBatchSampler(train_labels, BATCH_SIZE)

train_dataset = EmbeddingDataset(train_embeddings, train_labels)
test_dataset = EmbeddingDataset(test_embeddings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=balanced_sampler)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# # Определение BERT модели с n энкодер блоками
class BertClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(BertClassifier, self).__init__()
        config = BertConfig(
            hidden_size=1024,
            num_hidden_layers=NUM_ENCODER_BLOCKS,  
            num_attention_heads=NUM_ATTENTION_HEADS,
            intermediate_size=4096,
            max_position_embeddings=512,
        )
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
    
    def forward(self, x):
        outputs = self.bert(inputs_embeds=x)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        return logits

model = BertClassifier()
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
all_losses = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    losses = []
    model.train()
    for batch in tqdm(train_loader):
        embeddings = batch['embedding'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(embeddings.unsqueeze(1)) # tensor[0,0,1,0,1,1,0]
        loss = criterion(outputs, labels) # tensor[0,0,1,0,1,1,0] -> 0.31
        losses.append(loss.item())
        loss.backward() # 0.31 -> []
        optimizer.step() # 0.31 - 

    all_losses.extend(losses)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {np.mean(losses)}')
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }
    torch.save(checkpoint, 'model_checkpoint.pt')
    
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        embeddings = batch['embedding'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(embeddings.unsqueeze(1))
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')
print(f'Accuracy: {accuracy}\nF1 Score: {f1}')
