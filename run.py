import os
import numpy as np
from dataset_utils import HandlersDataset
from focal_loss import FocalLoss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import importlib
from model import HAN_Classification
from train import ClassificationTrainer
from data_processing.scripts.helpers import convert_to_binary_classification

#loading data
load_dir = "./data_processing"

if os.path.exists(os.path.join(load_dir, "config.pt")):
    config = torch.load(os.path.join(load_dir, "config.pt"), weights_only=False)
    E = config['E']
    D = config['D']
    L = config['L']
    print(f"   ...config: E={E}, D={D}, L={L}")
else:
    print("can't find config.pt")

day_dict = torch.load(os.path.join(load_dir, "day_dict.pt"), weights_only=False)

train_s = torch.load(os.path.join(load_dir, "train_samples.pt"), weights_only=False)
val_s   = torch.load(os.path.join(load_dir, "val_samples.pt"), weights_only=False)
test_s  = torch.load(os.path.join(load_dir, "test_samples.pt"), weights_only=False)

print(f"   ...sample loaded: train ({len(train_s)}), validation ({len(val_s)}), test ({len(test_s)})")

# pipeline 产出的样本标签已是 0/1 二分类，阈值 0.5 保持原样
THRESHOLD_RISK = 0.5
train_s_cls = convert_to_binary_classification(train_s, THRESHOLD_RISK)
val_s_cls   = convert_to_binary_classification(val_s, THRESHOLD_RISK)
test_s_cls  = convert_to_binary_classification(test_s, THRESHOLD_RISK)

# 第一阶段仅使用文本 embedding，不使用 day features
D = 0
L = 50
train_ds = HandlersDataset(train_s_cls, day_dict, L=L, E=E, D=0)
val_ds   = HandlersDataset(val_s_cls,   day_dict, L=L, E=E, D=0)
test_ds  = HandlersDataset(test_s_cls,  day_dict, L=L, E=E, D=0)

train_labels = []
for idx in range(len(train_ds)):
    sample = train_ds[idx]
    train_labels.append(sample['y'].item())  #
train_labels = np.array(train_labels)

class_sample_count = np.array([
    len(np.where(train_labels == 0)[0]),  
    len(np.where(train_labels == 1)[0])   
])

class_sample_count = np.maximum(class_sample_count, 1)
weight = 1. / class_sample_count  
samples_weight = np.array([weight[t] for t in train_labels])
samples_weight = torch.from_numpy(samples_weight).float() 

sampler = WeightedRandomSampler(
    weights=samples_weight,
    num_samples=len(samples_weight),
    replacement=True
)

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    sampler=sampler,
    num_workers=0,    # Mac 上避免多进程 dataloader 问题
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


#training
Num_epoches = 10 #change
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
cls_weights = torch.tensor([1.0, 1.4]).to(device)  #change the weights to your preference
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.4]).to(device))
# criterion = FocalLoss(alpha=0.25, gamma=1.5, weight=cls_weights)

model = HAN_Classification(
    embedding_dim=E, 
    gru_hidden_dim=64,        
    gru_num_layers=1,
    prediction_hidden_dim=32,
    num_classes=2,            
    dropout=0.4
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

save_dir = "./checkpoints"

trainer = ClassificationTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    device=device,
    save_path=save_dir
)

trainer.train(num_epochs=Num_epoches, patience=5)

#testing performance on test set
model_path = "./checkpoints/best_model.pt"  
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Successfully loaded best model from {model_path}")
except Exception as e:
    print(f"❌ Load model failed: {e}")
    exit()

model.eval() 
all_preds = []
all_labels = []
total_loss = 0.0

criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.4]).to(device))

with torch.no_grad():  
    for batch in test_loader:
        x_text = batch['x_text'].to(device)
        x_mask = batch['x_mask'].to(device)
        y = batch['y'].to(device).long()
        
        logits, _, _ = model(x_text, x_mask)  
        
        loss = criterion(logits, y)
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1) 
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

cm = confusion_matrix(all_labels, all_preds)
report = classification_report(
    all_labels, 
    all_preds, 
    target_names=['Safe (0)', 'Risk (1)'],
    digits=4 
)

tn, fp, fn, tp = cm.ravel()  
risk_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0 
risk_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 
risk_f1 = 2 * (risk_prec * risk_recall) / (risk_prec + risk_recall) if (risk_prec + risk_recall) > 0 else 0.0
acc = (tp + tn) / (tp + tn + fp + fn) 
avg_loss = total_loss / len(test_loader)

print("\n" + "="*50)
print("📊 Test Set Results")
print("="*50)
print(f"Average Loss: {avg_loss:.4f}")
print(f"Overall Accuracy: {acc:.4%}")
print(f"Risk Precision: {risk_prec:.4f}")
print(f"Risk Recall: {risk_recall:.4f}")
print(f"Risk F1-Score: {risk_f1:.4f}")

print("\n🔍 Confusion Matrix:")
print(f"           Pred_Safe  Pred_Risk")
print(f"Actual_0:     {tn:<8} {fp:<8}")
print(f"Actual_1:     {fn:<8} {tp:<8}")

print("\n📋 Detailed Classification Report:")
print(report)
