import torch
import torch.nn as nn
import torch.nn.functional as F

class DayStatGRU(nn.Module):
    """
    去掉 post-level attention 的简化版：输入直接是每天的统计量向量 (B, W, F)。

    与 HAN_Classification 的关系:
        HAN 的 Layer 1 (news-level attention) 负责把当天 50 条帖子的 768 维 embedding
        压成一个日向量；这里改成用 build_day_features.py 算好的可解释统计量直接作为日向量，
        Layer 2/3/4（Bi-GRU → temporal attention → 分类头）结构保持不变。

    为什么这么改:
        Bi-GRU 吃 768 维输入时独占了 HAN 44 万参数里的 73%，而训练样本只有 4,640 个
        （参数/样本 95:1），模型要么塌缩要么过拟合。换成 F≈21 维输入后参数量降到约
        1/8，参数/样本降到约 11:1，才能干净地检验"信号到底存不存在"。
        temporal attention 保留，所以"20 天窗口里哪一天最重要"这个可解释性仍然在。
    """

    def __init__(self, feature_dim, gru_hidden_dim=32, gru_num_layers=1,
                 attn_dim=64, prediction_hidden_dim=32, num_classes=2, dropout=0.3):
        super(DayStatGRU, self).__init__()
        # 各统计量量纲差异大（log 帖子量 ~10，情绪 ~0.1），先做归一化
        self.input_norm = nn.LayerNorm(feature_dim)

        self.bi_gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_num_layers > 1 else 0,
        )
        self.temporal_attn_linear = nn.Linear(gru_hidden_dim * 2, attn_dim)
        self.temporal_context_vector = nn.Parameter(torch.randn(attn_dim, 1))

        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, prediction_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prediction_hidden_dim, num_classes),
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.temporal_context_vector)

    def forward(self, x_day, x_mask=None, unused=None):
        # x_day: (B, W, F)。签名与 HAN 对齐，方便复用 ClassificationTrainer
        h = self.input_norm(x_day)
        gru_out, _ = self.bi_gru(h)                                   # (B, W, H*2)
        gru_out = self.dropout(gru_out)

        v = torch.tanh(self.temporal_attn_linear(gru_out))            # (B, W, attn_dim)
        score = torch.matmul(v, self.temporal_context_vector).squeeze(-1)  # (B, W)
        weights = F.softmax(score, dim=1)

        doc = torch.sum(gru_out * weights.unsqueeze(-1), dim=1)       # (B, H*2)
        logits = self.fc(doc)
        return logits, weights, None


class HAN_Classification(nn.Module):
    def __init__(self, 
                 embedding_dim=768,
                 gru_hidden_dim=128,
                 gru_num_layers=1,
                 prediction_hidden_dim=64,
                 num_classes=3,  # 修改点1：增加分类数参数
                 dropout=0.3,
                 day_feature_dim=0,      # 每天除文本外的数值特征维度（如帖子量），0 表示不用
                 day_feature_hidden=16):
        super(HAN_Classification, self).__init__()

        # ========================================
        # Layer 0 (可选): Day-level 数值特征
        # 帖子量这类特征与 embedding 量纲差异很大，先过一层 Linear 让模型自己学缩放，
        # 再拼到日向量上一起进 GRU。（D=1 时不能用 LayerNorm——单维归一化恒等于 0）
        # ========================================
        self.day_feature_dim = day_feature_dim
        if day_feature_dim > 0:
            self.day_feat_proj = nn.Sequential(
                nn.Linear(day_feature_dim, day_feature_hidden),
                nn.ReLU(),
            )
            gru_input_dim = embedding_dim + day_feature_hidden
        else:
            gru_input_dim = embedding_dim

        # ========================================
        # Layer 1: News-level Attention (文本级注意力)
        # 学习同一天内不同新闻的权重
        # ========================================
        self.news_attn_linear = nn.Linear(embedding_dim, 128)
        # 这个 context_vector (u_w) 是 HAN 的灵魂，它是模型“学习”出来的代表性新闻特征
        self.news_context_vector = nn.Parameter(torch.randn(128, 1)) 
        
        # ========================================
        # Layer 2: Temporal (Day-level) Bi-GRU
        # ========================================
        self.bi_gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden_dim,    
            num_layers=gru_num_layers,    
            batch_first=True,
            bidirectional=True, 
            dropout=dropout if gru_num_layers > 1 else 0
        )

        # ========================================
        # Layer 3: Temporal Attention (日期级注意力)
        # ========================================
        self.temporal_attn_linear = nn.Linear(gru_hidden_dim * 2, 128)
        self.temporal_context_vector = nn.Parameter(torch.randn(128, 1))

        # ========================================
        # Layer 4: Prediction (Classification Head)
        # ========================================
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, prediction_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prediction_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 修改点2：最后一层改为 num_classes
            nn.Linear(32, num_classes) 
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        # 初始化 Context Vector 对收敛很重要
        nn.init.xavier_uniform_(self.news_context_vector)
        nn.init.xavier_uniform_(self.temporal_context_vector)

    def news_level_attention(self, x, mask):
        """
        x: (BW, L, E) -> (Batch*Window, News_Limit, Embedding)
        mask: (BW, L)
        """
        # 1. 线性变换得到隐藏表示 u_it
        u = torch.tanh(self.news_attn_linear(x)) # (BW, L, 128)
        
        # 2. 计算隐藏表示与“新闻级上下文向量”的相似度 (即权重得分)
        # 这一步是在学习：相对于整体而言，这篇新闻有多重要
        att_score = torch.matmul(u, self.news_context_vector).squeeze(-1) # (BW, L)
        
        if mask is not None:
            # Mask 掉没有新闻的填充位
            att_score = att_score.masked_fill(~mask, -1e9)
            is_all_masked = (~mask).all(dim=1, keepdim=True)
            att_score = att_score.masked_fill(is_all_masked, 0.0)
        
        # 3. Softmax 归一化
        att_weights = F.softmax(att_score, dim=1) # (BW, L)
        
        if mask is not None:
            att_weights = att_weights.masked_fill(is_all_masked, 0.0)
        
        # 4. 加权求和得到“日向量”
        day_vector = torch.sum(x * att_weights.unsqueeze(-1), dim=1) # (BW, E)
        return day_vector, att_weights

    def forward(self, x_text, x_mask, x_day_feat=None):
        # x_text: (Batch, Window, L, E); x_day_feat: (Batch, Window, D) 或 None
        B, W, L, E = x_text.size()
        
        # 1. 并行处理所有日期
        x_flat = x_text.view(B * W, L, E)
        mask_flat = x_mask.view(B * W, L)
        
        # 2. 文本级注意力聚合成日向量
        day_vectors_flat, news_weights = self.news_level_attention(x_flat, mask_flat)
        day_vectors_flat = self.dropout(day_vectors_flat)
        
        # 3. 恢复序列结构
        day_vectors = day_vectors_flat.view(B, W, E) # (B, W, E)

        # 3.5 可选：拼接 day-level 数值特征（帖子量等）
        if self.day_feature_dim > 0:
            if x_day_feat is None:
                raise ValueError(
                    f"模型以 day_feature_dim={self.day_feature_dim} 构建，forward 必须传入 x_day_feat"
                )
            day_vectors = torch.cat([day_vectors, self.day_feat_proj(x_day_feat)], dim=-1)

        # 4. 时间序列建模 (Bi-GRU)
        gru_out, _ = self.bi_gru(day_vectors) # (B, W, Hidden*2)
        
        # 5. 日期级注意力 (Temporal Attention)
        v_t = torch.tanh(self.temporal_attn_linear(gru_out)) # (B, W, 128)
        att_score_t = torch.matmul(v_t, self.temporal_context_vector).squeeze(-1) # (B, W)
        att_weights_t = F.softmax(att_score_t, dim=1) # (B, W)
        
        # 6. 加权求和得到最终特征向量
        doc_vector = torch.sum(gru_out * att_weights_t.unsqueeze(-1), dim=1) # (B, Hidden*2)
        
        # 7. 分类输出
        logits = self.fc(doc_vector) # (B, num_classes)
       
        # 注意：分类任务通常不在这里加 Softmax，因为 nn.CrossEntropyLoss 会自动处理
        return logits, att_weights_t, news_weights.view(B, W, L)
