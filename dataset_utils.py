# dataset_utils.py
import torch
from torch.utils.data import Dataset, DataLoader

class HandlersDataset(Dataset):
    def __init__(self, samples, day_dict, L, E, D=0):
        self.samples = samples
        self.day_dict = day_dict
        self.L = L
        self.E = E
        self.D = D
        
        # 预定义全 0 特征，用于没新闻的日子
        if D > 0:
            self.empty_day_feat = torch.zeros((D,), dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 注意：s[3] 现在应该是我们转换后的分类标签 (0, 1, 2)
        tic, anchor_date, lookback_dates, y = self.samples[idx]
        
        x_text_list = []
        x_mask_list = []
        x_day_feat_list = []
        
        for date in lookback_dates:
            item = self.day_dict.get((tic, date))
            
            if item is None:
                padded_text = torch.zeros((self.L, self.E), dtype=torch.float32)
                mask = torch.zeros((self.L,), dtype=torch.bool)
                day_feat = self.empty_day_feat if self.D > 0 else None
            else:
                # 兼容不同格式：如果 item 直接是 tensor，或者是一个 dict
                if isinstance(item, dict):
                    real_text = item["text"]
                    day_feat = item.get("day_features", self.empty_day_feat if self.D > 0 else None)
                else:
                    real_text = item
                    day_feat = self.empty_day_feat if self.D > 0 else None

                N = min(real_text.shape[0], self.L) # 防止超过 L
                
                padded_text = torch.zeros((self.L, self.E), dtype=torch.float32)
                padded_text[:N] = real_text[:N]
                
                mask = torch.zeros((self.L,), dtype=torch.bool)
                mask[:N] = True
            
            x_text_list.append(padded_text)
            x_mask_list.append(mask)
            if self.D > 0:
                x_day_feat_list.append(day_feat)
        
        out = {
            "x_text": torch.stack(x_text_list),      # (W, L, E)
            "x_mask": torch.stack(x_mask_list),      # (W, L)
            # 【关键修改】改为 long 类型，适配分类任务
            "y": torch.tensor(y, dtype=torch.long)   
        }
        
        if self.D > 0:
            out["x_day_feat"] = torch.stack(x_day_feat_list) # (W, D)
            
        return out