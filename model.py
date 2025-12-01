import torch
import torch.nn as nn
import math
from tqdm import tqdm
import torch.nn.functional as F


class GatedFusionModule(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super(GatedFusionModule, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加LayerNorm提升稳定性
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )
        
        self.feature_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, domain_feat, esm_feat, go_feat):
        """
        门控融合
        
        参数:
            domain_feat: [batch_size, hidden_dim]
            esm_feat: [batch_size, hidden_dim]
            go_feat: [batch_size, hidden_dim]
        
        返回:
            fused_feat: [batch_size, hidden_dim]
            gate_weights: [batch_size, 3] (用于分析)
        """
        # 1. 拼接所有特征用于计算门控权重
        concat_feat = torch.cat([domain_feat, esm_feat, go_feat], dim=-1)  # [B, hidden_dim*3]
        
        gate_logits = self.gate_network(concat_feat)  # [B, 3]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, 3]
        
        stacked_feats = torch.stack([domain_feat, esm_feat, go_feat], dim=1)  # [B, 3, hidden_dim]
        batch_size, num_feats, hidden_dim = stacked_feats.shape
        
        stacked_feats_flat = stacked_feats.view(-1, hidden_dim)  # [B*3, hidden_dim]
        transformed_feats_flat = self.feature_transform(stacked_feats_flat)  # [B*3, hidden_dim]
        transformed_feats = transformed_feats_flat.view(batch_size, num_feats, hidden_dim)  # [B, 3, hidden_dim]
        
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # [B, 3, 1]
        fused_feat = (transformed_feats * gate_weights_expanded).sum(dim=1)  # [B, hidden_dim]
        
        return fused_feat, gate_weights


class FeatureDropout(nn.Module):
    def __init__(self, dropout_prob=0.15):
        super(FeatureDropout, self).__init__()
        self.dropout_prob = dropout_prob
        
    def forward(self, domain_feat, esm_feat, go_feat):
        """
        训练时随机将Domain或ESM特征置零，但GO特征永远保留
        
        参数:
            domain_feat: [batch_size, hidden_dim] - 蛋白质结构域特征
            esm_feat: [batch_size, hidden_dim] - 蛋白质序列特征
            go_feat: [batch_size, hidden_dim] - GO功能特征（不会被dropout）
        
        返回:
            处理后的三个特征
        """
        if not self.training:
            return domain_feat, esm_feat, go_feat
        
        batch_size = domain_feat.size(0)
        device = domain_feat.device
        
        rand_vals = torch.rand(batch_size, 2, device=device)  # 只有2个特征可能被dropout
        
        protein_masks = (rand_vals > self.dropout_prob).float()  # [B, 2]
        
        row_sums = protein_masks.sum(dim=1)  # [B]
        zero_rows = (row_sums == 0)  # [B] bool tensor
        
        if zero_rows.any():
            num_zero_rows = zero_rows.sum().item()
            random_positions = torch.randint(0, 2, (num_zero_rows,), device=device)
            
            zero_row_indices = torch.where(zero_rows)[0]
            protein_masks[zero_row_indices, random_positions] = 1.0
        
        domain_mask = protein_masks[:, 0:1]  # [B, 1]
        esm_mask = protein_masks[:, 1:2]     # [B, 1]
        
        domain_feat = domain_feat * domain_mask
        esm_feat = esm_feat * esm_mask
        
        return domain_feat, esm_feat, go_feat


class CustomModel(nn.Module):
    def __init__(self, esm_dim, nlp_dim, inter_size, hidden_dim=512, dropout=0.3,
                 feature_dropout_prob=0.15):
        """
        二分类模型：判断蛋白质序列是否具有特定GO功能
        使用门控融合 (Gated Fusion) 模式
        
        任务：给定蛋白质（Domain + ESM特征）和GO功能描述，预测该蛋白质是否具有该功能
        
        参数:
            esm_dim: ESM embedding维度（蛋白质序列特征）
            nlp_dim: NLP(GO) embedding维度（功能描述特征）
            inter_size: domain embedding维度（蛋白质结构域特征）
            hidden_dim: 隐藏层维度
            dropout: dropout概率
            feature_dropout_prob: 蛋白质特征级dropout概率（仅作用于Domain和ESM，不影响GO）
        """
        super(CustomModel, self).__init__()
        
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.nlp_proj = nn.Sequential(
            nn.Linear(nlp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.domain_proj = nn.Sequential(
            nn.Linear(inter_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.feature_dropout = FeatureDropout(dropout_prob=feature_dropout_prob)
        
        self.gated_fusion = GatedFusionModule(
            hidden_dim=hidden_dim,
            dropout=dropout * 0.7
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, esm_embedding, domain_embedding, nlp_embedding, batch_size, 
                return_attention_weights=False):
        """
        前向传播
        
        参数:
            esm_embedding: [batch_size * num_labels, esm_dim] - 蛋白质序列特征
            domain_embedding: [batch_size * num_labels, inter_size] - 蛋白质结构域特征
            nlp_embedding: [num_labels, nlp_dim] - GO功能描述特征
            batch_size: 批次大小
            return_attention_weights: 是否返回门控权重
        
        返回:
            logits: [batch_size, num_labels] 二分类logits
            gate_weights: 门控权重 [batch_size * num_labels, 3] (可选)
        """
        num_labels = nlp_embedding.size(0)
        
        esm_feat = self.esm_proj(esm_embedding)  # [batch_size * num_labels, hidden_dim]
        domain_feat = self.domain_proj(domain_embedding)  # [batch_size * num_labels, hidden_dim]
        nlp_feat = self.nlp_proj(nlp_embedding)  # [num_labels, hidden_dim]
        
        nlp_feat_expanded = nlp_feat.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_labels, hidden_dim]
        nlp_feat_batched = nlp_feat_expanded.reshape(-1, nlp_feat.size(-1))  # [batch_size * num_labels, hidden_dim]
        
        domain_feat, esm_feat, nlp_feat_batched = self.feature_dropout(
            domain_feat, esm_feat, nlp_feat_batched
        )
        
        fused_feat, gate_weights = self.gated_fusion(
            domain_feat, esm_feat, nlp_feat_batched
        )  # fused_feat: [batch_size * num_labels, hidden_dim]
           # gate_weights: [batch_size * num_labels, 3]
        
        fused = self.fusion(fused_feat)  # [batch_size * num_labels, hidden_dim//2]
        
        logits = self.classifier(fused)  # [batch_size * num_labels, 1]
        
        logits = logits.view(batch_size, num_labels)
        
        if return_attention_weights:
            return logits, gate_weights
        
        return logits


# ==================== 测试代码 ====================
if __name__ == "__main__":
    import time
    
    # 模拟数据
    batch_size = 16
    num_labels = 100
    esm_dim = 1280
    nlp_dim = 768
    inter_size = 512
    
    print("=" * 80)
    print("门控融合模式（Gated Fusion）测试")
    print("⚠️  修正版：GO功能嵌入永不被dropout")
    print("=" * 80)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        
        # 准备测试数据
        esm_embedding = torch.randn(batch_size * num_labels, esm_dim).to(device)
        domain_embedding = torch.randn(batch_size * num_labels, inter_size).to(device)
        nlp_embedding = torch.randn(num_labels, nlp_dim).to(device)
        
        print(f"\n创建模型...")
        model = CustomModel(
            esm_dim=esm_dim,
            nlp_dim=nlp_dim,
            inter_size=inter_size,
            hidden_dim=512,
            feature_dropout_prob=0.15
        ).to(device)
        
        # 测试Feature Dropout逻辑
        print("\n" + "="*80)
        print("测试Feature Dropout逻辑")
        print("="*80)
        
        model.train()
        
        # 投影特征
        esm_feat = model.esm_proj(esm_embedding)
        domain_feat = model.domain_proj(domain_embedding)
        nlp_feat = model.nlp_proj(nlp_embedding)
        nlp_feat_expanded = nlp_feat.unsqueeze(0).expand(batch_size, -1, -1)
        nlp_feat_batched = nlp_feat_expanded.reshape(-1, nlp_feat.size(-1))
        
        # 应用Feature Dropout
        domain_dropped, esm_dropped, go_dropped = model.feature_dropout(
            domain_feat, esm_feat, nlp_feat_batched
        )
        
        # 检查GO是否被保留
        print(f"\n原始GO特征范数: {nlp_feat_batched.norm(dim=1).mean().item():.4f}")
        print(f"Dropout后GO特征范数: {go_dropped.norm(dim=1).mean().item():.4f}")
        print(f"GO特征是否完全相同: {torch.allclose(nlp_feat_batched, go_dropped)}")
        
        # 检查Domain和ESM的dropout情况
        domain_dropout_ratio = (domain_dropped.norm(dim=1) == 0).float().mean().item()
        esm_dropout_ratio = (esm_dropped.norm(dim=1) == 0).float().mean().item()
        both_kept_ratio = ((domain_dropped.norm(dim=1) > 0) & (esm_dropped.norm(dim=1) > 0)).float().mean().item()
        
        print(f"\nDomain被dropout的比例: {domain_dropout_ratio*100:.2f}%")
        print(f"ESM被dropout的比例: {esm_dropout_ratio*100:.2f}%")
        print(f"Domain和ESM都保留的比例: {both_kept_ratio*100:.2f}%")
        print(f"✅ 验证：至少一个蛋白质特征被保留")
        
        # 性能测试
        print("\n" + "="*80)
        print("性能测试 (50次迭代)")
        print("="*80)
        
        # 预热
        print("预热中...")
        for _ in range(3):
            _ = model(esm_embedding, domain_embedding, nlp_embedding, batch_size)
        
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        num_iterations = 50
        
        start_time = time.time()
        for _ in tqdm(range(num_iterations), desc="Gated Fusion"):
            logits, gate_weights = model(
                esm_embedding, domain_embedding, nlp_embedding, 
                batch_size, return_attention_weights=True
            )
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        throughput = batch_size / avg_time
        mem = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"\n结果:")
        print(f"  - 平均推理时间: {avg_time*1000:.2f} ms")
        print(f"  - 吞吐量: {throughput:.2f} samples/sec")
        print(f"  - 显存占用: {mem:.2f} MB")
        print(f"  - 输出shape: {logits.shape}")
        print(f"  - 门控权重shape: {gate_weights.shape}")
        
        # 门控权重详细分析
        print("\n" + "="*80)
        print("门控权重详细分析")
        print("="*80)
        
        model.eval()
        with torch.no_grad():
            logits, gate_weights = model(
                esm_embedding, domain_embedding, nlp_embedding,
                batch_size, return_attention_weights=True
            )
            
            # 统计门控权重分布
            gate_weights_cpu = gate_weights.cpu().numpy()
            
            print(f"\n门控权重统计 (基于 {batch_size * num_labels} 个样本):")
            print(f"  Domain权重: 均值={gate_weights_cpu[:, 0].mean():.4f}, "
                  f"标准差={gate_weights_cpu[:, 0].std():.4f}")
            print(f"  ESM权重:    均值={gate_weights_cpu[:, 1].mean():.4f}, "
                  f"标准差={gate_weights_cpu[:, 1].std():.4f}")
            print(f"  GO权重:     均值={gate_weights_cpu[:, 2].mean():.4f}, "
                  f"标准差={gate_weights_cpu[:, 2].std():.4f}")
            
            # 显示几个样本
            print(f"\n前5个样本的门控权重:")
            print(f"{'样本':<6} {'Domain':<10} {'ESM':<10} {'GO':<10} {'总和':<6}")
            print("-" * 50)
            for i in range(min(5, batch_size * num_labels)):
                w = gate_weights_cpu[i]
                print(f"{i:<6} {w[0]:<10.4f} {w[1]:<10.4f} {w[2]:<10.4f} {w.sum():<6.4f}")
        
        torch.cuda.empty_cache()
    
    else:
        print("CUDA不可用，跳过GPU测试")
        
        # CPU测试
        print("\nCPU测试...")
        esm_embedding = torch.randn(batch_size * num_labels, esm_dim)
        domain_embedding = torch.randn(batch_size * num_labels, inter_size)
        nlp_embedding = torch.randn(num_labels, nlp_dim)
        
        model = CustomModel(
            esm_dim=esm_dim,
            nlp_dim=nlp_dim,
            inter_size=inter_size,
            hidden_dim=512
        )
        
        model.eval()
        with torch.no_grad():
            logits, gate_weights = model(
                esm_embedding, domain_embedding, nlp_embedding,
                batch_size, return_attention_weights=True
            )
        
        print(f"输出shape: {logits.shape}")
        print(f"门控权重shape: {gate_weights.shape}")
        print(f"第一个样本的门控权重: {gate_weights[0].numpy()}")
        print("✅ CPU测试通过")