import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from newmodel import HAN

import config
args = config.args

class Trainer:
    def __init__(self, model, train_loader, val_loader, 
                 optimizer, scheduler, criterion, device, save_path):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_X, batch_y in self.train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            logits, _ = self.model(batch_X)
            loss = self.criterion(logits, batch_y)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print("WARNING: NaN loss detected, skipping batch")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            # Update weights
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                logits, _ = self.model(batch_X)
                predictions = torch.sigmoid(logits)
                loss = self.criterion(logits, batch_y)
                total_loss += loss.item()
                
                all_preds.append(predictions.cpu())
                all_targets.append(batch_y.cpu())
        
        # Calculate metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        predicted_classes = (all_preds > 0.5).float()
        accuracy = (predicted_classes == all_targets).float().mean().item()
        
        # Additional metrics
        tp = ((predicted_classes == 1) & (all_targets == 1)).sum().item()
        fp = ((predicted_classes == 1) & (all_targets == 0)).sum().item()
        tn = ((predicted_classes == 0) & (all_targets == 0)).sum().item()
        fn = ((predicted_classes == 0) & (all_targets == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return total_loss / len(self.val_loader), accuracy, precision, recall, f1
    
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        torch.save(checkpoint, self.save_path / 'best_model.pt')
        print(f"✓ Model saved at epoch {epoch}")
    
    def train(self, num_epochs):
        print("Starting training...")
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Acc:    {val_acc:.4f}")
            print(f"  Val Prec:   {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")
            print(f"  LR:         {current_lr:.6f}")
            
            # Save best model (but don't stop)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint(epoch)
        
        print(f"\nTraining complete!")
        print(f"Best model from epoch {self.best_epoch+1} with val loss {self.best_val_loss:.4f}")

def prepare_dataloaders(X_train, y_train, X_val, y_val, batch_size=64):
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_path = Path(args.dataset_save_dir)
    
    X_train = np.load(data_path / 'X_train.npy')
    y_train = np.load(data_path / 'y_train.npy')
    X_val = np.load(data_path / 'X_val.npy')
    y_val = np.load(data_path / 'y_val.npy')
    
    print(f"\n=== Data Statistics ===")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Train - Positive: {int(y_train.sum())}/{len(y_train)} ({y_train.mean()*100:.1f}%)")
    print(f"Val   - Positive: {int(y_val.sum())}/{len(y_val)} ({y_val.mean()*100:.1f}%)")
    
    # Class weight
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = neg_count / pos_count
    print(f"Using pos_weight: {pos_weight:.3f}\n")
    
    # Create dataloaders
    train_loader, val_loader = prepare_dataloaders(
        X_train, y_train, X_val, y_val, 
        batch_size=64
    )
    
    # Initialize model
    model = HAN(
        embedding_dim=768,
        gru_hidden_dim=128,
        gru_num_layers=2,
        prediction_hidden_dim=128,
        dropout=0.4
    )
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    model.apply(init_weights)
    print("✓ Model weights initialized")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device)
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.001,
        weight_decay=1e-5
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        verbose=True,
        min_lr=1e-6
    )
    
    # Create trainer
    trainer = Trainer(
        model, train_loader, val_loader, 
        optimizer, scheduler, criterion, device, 
        save_path='./checkpoints'
    )
    
    # Train for full 100 epochs (no early stopping)
    trainer.train(num_epochs=100)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()