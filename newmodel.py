import torch
import torch.nn as nn


class HAN(nn.Module):
    def __init__(self, 
                 embedding_dim=768,
                 gru_hidden_dim=128, # 64, 128, 256
                 gru_num_layers=2, # 1, 2, 3, 2-3 is standard choice
                 prediction_hidden_dim=128, # 64, 128, 256
                 dropout=0.3
                 ):
        super(HAN, self).__init__()

        #layer one: bi GRU
        self.bi_gru = nn.GRU(
            input_size=embedding_dim,      
            hidden_size=gru_hidden_dim,    # set as 128
            num_layers=gru_num_layers,    
            batch_first=True,
            bidirectional=True, #bi GRU
            dropout=dropout if gru_num_layers > 1 else 0
        )

        #layer two: sequantial attention mechanism, assigns different weights to different days
        self.attn = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, 1),  
            # nn.Sigmoid(),  #or Tanh or other
            nn.Tanh(),
            nn.Softmax(dim=1)
        )

        #layer three: 
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, prediction_hidden_dim), # dim: 256 -> 128
            nn.ReLU(), #alternative:nn.ELU()
            nn.Dropout(dropout),
            nn.Linear(prediction_hidden_dim, 64), # dim: 128 -> 64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # dim: 64 -> 1
            # nn.Sigmoid()
        )

    def forward(self, x):
            # x shape: (batch, 60, 768)
            
            # Step 1: GRU
            gru_out, _ = self.bi_gru(x) 
            
            # Step 2: Temporal attention
            attn_weights = self.attn(gru_out)
            attended = torch.sum(attn_weights * gru_out, dim=1) 
            
            # Step 3: Prediction
            output = self.fc(attended)
            
            return output.squeeze(), attn_weights.squeeze()
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss