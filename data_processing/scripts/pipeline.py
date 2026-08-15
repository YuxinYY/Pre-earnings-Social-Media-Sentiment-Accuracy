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
import config as project_config

#loading data
# 数据统一放在 config.py 指定的 data_dir（默认外部磁盘 T9）
load_dir = project_config.args.data_dir
# 用法: HAN_SAMPLE_DIR=<data_dir> python run.py（默认取 config.data_dir）
sample_dir = os.environ.get("HAN_SAMPLE_DIR", load_dir)
run_tag = os.environ.get("HAN_RUN_TAG", "default")
print(f"   ...samples from {sample_dir} (run tag: {run_tag})")

if os.path.exists(os.path.join(sample_dir, "config.pt")):
    config = torch.load(os.path.join(sample_dir, "config.pt"), weights_only=False)
    E = config['E']
    D = config['D']
    L = config['L']
    print(f"   ...config: E={E}, D={D}, L={L}")
else:
    print("can't find config.pt")

day_dict_path = os.environ.get("HAN_DAY_DICT", os.path.join(load_dir, "day_dict.pt"))
print(f"   ...day_dict: {day_dict_path}")
day_dict = torch.load(day_dict_path, weights_only=False)

train_s = torch.load(os.path.join(sample_dir, "train_samples.pt"), weights_only=False)
val_s   = torch.load(os.path.join(sample_dir, "val_samples.pt"), weights_only=False)
test_s  = torch.load(os.path.join(sample_dir, "test_samples.pt"), weights_only=False)

print(f"   ...sample loaded: train ({len(train_s)}), validation ({len(val_s)}), test ({len(test_s)})")

# pipeline 产出的样本标签已是 0/1 二分类，阈值 0.5 保持原样
THRESHOLD_RISK = 0.5
train_s_cls = convert_to_binary_classification(train_s, THRESHOLD_RISK)
val_s_cls   = convert_to_binary_classification(val_s, THRESHOLD_RISK)
test_s_cls  = convert_to_binary_classification(test_s, THRESHOLD_RISK)

# day features（day_dict 里存的是 log1p(当日帖子数)）默认关闭，只用文本 embedding。
# 设 HAN_USE_DAY_FEAT=1 打开，用 config.pt 里记录的维度 D。
use_day_feat = os.environ.get("HAN_USE_DAY_FEAT", "0") == "1"
# D 直接从 day_dict 实际内容推断，避免 config.pt 与打过补丁的 day_dict 不一致
D = int(next(iter(day_dict.values()))["day_features"].numel()) if use_day_feat else 0
L = 50
print(f"   ...day features: {'ON' if use_day_feat else 'OFF'} (D={D})")
train_ds = HandlersDataset(train_s_cls, day_dict, L=L, E=E, D=D)
val_ds   = HandlersDataset(val_s_cls,   day_dict, L=L, E=E, D=D)
test_ds  = HandlersDataset(test_s_cls,  day_dict, L=L, E=E, D=D)

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
Num_epoches = 20 #change
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
# 标签语义：1 = 未来 5 日 CAPM 异质波动率高于全期中位数（HighIVOL），0 = 偏低（LowIVOL）
# pipeline.py 已按中位数把 ivol_5 二分类，样本标签为 0/1，阈值 0.5 保持原样
# 类别平衡已由上面的 WeightedRandomSampler 处理，loss 不再叠加类权重
# （之前 [1.0, 1.4] 在 60% 正类的训练集上进一步推高正类，导致模型全猜一类）
criterion = torch.nn.CrossEntropyLoss()

model = HAN_Classification(
    embedding_dim=E, 
    gru_hidden_dim=64,        
    gru_num_layers=1,
    prediction_hidden_dim=32,
    num_classes=2,
    dropout=0.4,
    day_feature_dim=D
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

save_dir = f"./checkpoints/{run_tag}"

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
model_path = os.path.join(save_dir, "best_model.pt")
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

criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():  
    for batch in test_loader:
        x_text = batch['x_text'].to(device)
        x_mask = batch['x_mask'].to(device)
        x_day = batch['x_day_feat'].to(device) if 'x_day_feat' in batch else None
        y = batch['y'].to(device).long()

        logits, _, _ = model(x_text, x_mask, x_day)
        
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
    target_names=['LowIVOL (0)', 'HighIVOL (1)'],
    digits=4
)

tn, fp, fn, tp = cm.ravel()
out_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
out_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
out_f1 = 2 * (out_prec * out_recall) / (out_prec + out_recall) if (out_prec + out_recall) > 0 else 0.0
acc = (tp + tn) / (tp + tn + fp + fn)
avg_loss = total_loss / len(test_loader)

print("\n" + "="*50)
print("📊 Test Set Results")
print("="*50)
print(f"Average Loss: {avg_loss:.4f}")
print(f"Overall Accuracy: {acc:.4%}")
print(f"HighIVOL Precision: {out_prec:.4f}")
print(f"HighIVOL Recall: {out_recall:.4f}")
print(f"HighIVOL F1-Score: {out_f1:.4f}")

print("\n🔍 Confusion Matrix:")
print(f"           Pred_Low    Pred_High")
print(f"Actual_0:     {tn:<10} {fp:<8}")
print(f"Actual_1:     {fn:<10} {tp:<8}")

print("\n📋 Detailed Classification Report:")
print(report)
