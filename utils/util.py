import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn.functional as F
import math
import os 
from datetime import datetime

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_f1):
        score = val_f1
        
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()
            
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def evaluate_annotations(gold, hypo, ontology_name='test'):
    temp_gold = []
    temp_hypo = []
    
    for i in range(len(gold)):
        g = gold[i].cpu().numpy() if hasattr(gold[i], 'cpu') else gold[i]
        h = hypo[i].cpu().numpy() if hasattr(hypo[i], 'cpu') else hypo[i]
        
        if np.sum(g) > 0: # 过滤掉没有真实标签的蛋白
            temp_gold.append(g)
            temp_hypo.append(h)
            
    print(f"{ontology_name}: {len(temp_gold)} proteins with annotations")
    
    if len(temp_gold) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    y_true = np.array(temp_gold)
    y_pred = np.array(temp_hypo)
    
    thresholds = np.linspace(0, 1, 101)
    avg_prec_list = []
    avg_rec_list = []
    f_list = []
    
    # 总样本数 N
    N = y_true.shape[0]
    
    for t in thresholds:
        y_pred_binary = (y_pred >= t).astype(int)
        
        # 向量化计算 TP, FP, FN
        tp = np.sum((y_true == 1) & (y_pred_binary == 1), axis=1)
        fp = np.sum((y_true == 0) & (y_pred_binary == 1), axis=1)
        fn = np.sum((y_true == 1) & (y_pred_binary == 0), axis=1)
        
        # 预测出的标签数量
        pred_count = tp + fp
        
        # 找出那些“至少预测了一个标签”的蛋白
        has_pred_mask = pred_count > 0
        n_with_pred = np.sum(has_pred_mask)
        
        if n_with_pred > 0:
            # 只计算有预测的蛋白的 Precision
            p_vals = tp[has_pred_mask] / pred_count[has_pred_mask]
            avg_prec = np.sum(p_vals) / n_with_pred
        else:
            avg_prec = 0.0
            
        # 真实标签数量 (之前过滤过，保证 > 0)
        real_count = tp + fn
        r_vals = tp / real_count
        avg_rec = np.sum(r_vals) / N  # 分母始终是总样本数 N
        
        avg_prec_list.append(avg_prec)
        avg_rec_list.append(avg_rec)
        
        if (avg_prec + avg_rec) > 0:
            f1 = 2 * avg_prec * avg_rec / (avg_prec + avg_rec)
        else:
            f1 = 0.0
        f_list.append(f1)
        
    fmax = max(f_list)
    max_idx = np.argmax(f_list)
    best_p = avg_prec_list[max_idx]
    best_r = avg_rec_list[max_idx]
    best_t = thresholds[max_idx]
    
    prec_array = np.array(avg_prec_list)
    rec_array = np.array(avg_rec_list)
    
    # 排序以计算面积
    sorted_indices = np.argsort(rec_array)
    rec_sorted = rec_array[sorted_indices]
    prec_sorted = prec_array[sorted_indices]
    
    aupr = np.trapz(prec_sorted, rec_sorted)
    
    return fmax, best_p, best_r, aupr, best_t

def propagate_predictions(preds, adj_matrix):
    """
    传播预测分数，确保子节点的分数不高于父节点
    
    Args:
        preds: torch.Tensor, shape (n_samples, n_labels), 预测分数
        adj_matrix: torch.Tensor, shape (n_labels, n_labels), 邻接矩阵
                   adj_matrix[i,j]=1 表示 j 是 i 的父节点
    
    Returns:
        propagated_preds: torch.Tensor, 传播后的预测分数
    """
    device = preds.device
    n_samples, n_labels = preds.shape
    
    # 转换邻接矩阵
    if isinstance(adj_matrix, torch.Tensor):
        if adj_matrix.is_sparse:
            adj_matrix = adj_matrix.to_dense()
        adj_matrix = adj_matrix.to(device)
    else:
        adj_matrix = torch.tensor(adj_matrix, device=device)
    
    # 复制预测结果
    # print('adj_matrix.shape', adj_matrix.shape)
    propagated_preds = preds.clone()
    
    # 对每个标签，确保其所有祖先的分数不低于它
    for label_idx in range(n_labels):
        # 找到所有父节点
        parents = torch.where(adj_matrix[label_idx, :] > 0)[0]
        
        if len(parents) > 0:
            # 当前标签的分数
            child_scores = propagated_preds[:, label_idx:label_idx+1]  # (n_samples, 1)
            # 父节点的分数
            parent_scores = propagated_preds[:, parents]  # (n_samples, n_parents)
            # 更新父节点分数为 max(父节点当前分数, 子节点分数)
            propagated_preds[:, parents] = torch.max(parent_scores, child_scores)
    
    return propagated_preds


def compute_propagated_metrics(labels, preds, adj_matrix):
    """
    计算传播后的评估指标
    
    Args:
        labels: torch.Tensor, shape (n_samples, n_labels), 真实标签
        preds: torch.Tensor, shape (n_samples, n_labels), 预测分数
        adj_matrix: torch.Tensor, 邻接矩阵
    
    Returns:
        prop_fmax: float, 传播后的最大Fmax分数
        prop_aupr: float, 传播后的AUPR
        prop_preds: torch.Tensor, 传播后的预测分数
    """
    # 传播预测分数
    prop_preds = propagate_predictions(preds, adj_matrix)
    
    prop_fmax, prop_precision, prop_recall, prop_aupr,prop_th = evaluate_annotations(labels, prop_preds,"prop")
    
    return prop_fmax, prop_precision,prop_recall,prop_aupr,prop_th, prop_preds

def save_results(config, metrics_output_test, seed, ctime):
    """保存训练结果（包含未见标签分析）"""
    os.makedirs(config['output_path'], exist_ok=True)
    output_file = os.path.join(config['output_path'], f"{config['model']}_{config.get('text_mode', 'default')}_{ctime}.txt")
    
    with open(output_file, 'w') as file_prec:
        file_prec.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_prec.write(f"Seed: {seed}\n")
        file_prec.write(f"Model: ProtNote (CNN-based with NLP embeddings)\n")
        file_prec.write(f"Training mode: Fixed epochs, evaluate after training\n")
        file_prec.write(f"Total epochs: {config['epoch_num']}\n")
        file_prec.write(f"Hidden dim: {config.get('hidden_dim', 512)}, Dropout: {config.get('dropout', 0.3)}\n")
        file_prec.write(f"Learning rate scheduler: Warmup + Cosine Annealing\n")
        file_prec.write(f"Warmup ratio: {config.get('warmup_ratio', 0.1)}\n")
        file_prec.write(f"Base LR: {config['learning_rate']}\n")
        
        file_prec.write(f"\n{'='*80}\n")
        file_prec.write(f"FINAL EVALUATION RESULTS (with Unseen Label Analysis)\n")
        file_prec.write(f"{'='*80}\n")

        for key in metrics_output_test.keys():
            metrics = metrics_output_test[key]
            
            file_prec.write(f"\n{'='*30} {key} {'='*30}\n")
            
            # 整体指标
            file_prec.write(f"\nOverall Metrics:\n")
            file_prec.write(f"  Avg Fmax:          {metrics['Fmax']:.4f}\n")
            file_prec.write(f"  Avg Precision:     {metrics['p']:.4f}\n")
            file_prec.write(f"  Avg Recall:        {metrics['r']:.4f}\n")
            file_prec.write(f"  AUPR:              {metrics['aupr']:.4f}\n")
            file_prec.write(f"  Threshold:         {metrics['threshold']:.4f}\n")
            file_prec.write(f"  Prop-Fmax:         {metrics['prop_Fmax']:.4f} ★\n")
            file_prec.write(f"  Prop-Precision:    {metrics['prop_precision']:.4f}\n")
            file_prec.write(f"  Prop-Recall:       {metrics['prop_recall']:.4f}\n")
            file_prec.write(f"  Prop-AUPR:         {metrics['prop_aupr']:.4f}\n")
            file_prec.write(f"  Prop-Threshold:    {metrics['prop_threshold']:.4f}\n")
            
            # 未见标签指标
            if metrics['unseen'] is not None:
                unseen = metrics['unseen']
                file_prec.write(f"\nUnseen Labels ({metrics['unseen_count']} labels, {unseen['sample_count']} samples):\n")
                file_prec.write(f"  Avg Fmax:          {unseen['Fmax']:.4f}\n")
                file_prec.write(f"  Avg Precision:     {unseen['precision']:.4f}\n")
                file_prec.write(f"  Avg Recall:        {unseen['recall']:.4f}\n")
                file_prec.write(f"  AUPR:              {unseen['aupr']:.4f}\n")
                file_prec.write(f"  Threshold:         {unseen['threshold']:.4f}\n")
                if 'prop_Fmax' in unseen:
                    file_prec.write(f"  Prop-Fmax:         {unseen['prop_Fmax']:.4f} ★\n")
                    file_prec.write(f"  Prop-Precision:    {unseen['prop_precision']:.4f}\n")
                    file_prec.write(f"  Prop-Recall:       {unseen['prop_recall']:.4f}\n")
                    file_prec.write(f"  Prop-AUPR:         {unseen['prop_aupr']:.4f}\n")
                    file_prec.write(f"  Prop-Threshold:    {unseen['prop_threshold']:.4f}\n")
            
            # 已见标签指标
            if metrics['seen'] is not None:
                seen = metrics['seen']
                file_prec.write(f"\nSeen Labels ({metrics['seen_count']} labels, {seen['sample_count']} samples):\n")
                file_prec.write(f"  Avg Fmax:          {seen['Fmax']:.4f}\n")
                file_prec.write(f"  Avg Precision:     {seen['precision']:.4f}\n")
                file_prec.write(f"  Avg Recall:        {seen['recall']:.4f}\n")
                file_prec.write(f"  AUPR:              {seen['aupr']:.4f}\n")
                file_prec.write(f"  Threshold:         {seen['threshold']:.4f}\n")
                if 'prop_Fmax' in seen:
                    file_prec.write(f"  Prop-Fmax:         {seen['prop_Fmax']:.4f} ★\n")
                    file_prec.write(f"  Prop-Precision:    {seen['prop_precision']:.4f}\n")
                    file_prec.write(f"  Prop-Recall:       {seen['prop_recall']:.4f}\n")
                    file_prec.write(f"  Prop-AUPR:         {seen['prop_aupr']:.4f}\n")
                    file_prec.write(f"  Prop-Threshold:    {seen['prop_threshold']:.4f}\n")
            
            # 调和平均
            if metrics['harmonic_mean'] is not None:
                file_prec.write(f"\nHarmonic Mean:     {metrics['harmonic_mean']:.4f} ★★\n")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # 在控制台打印汇总
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    for key in metrics_output_test.keys():
        metrics = metrics_output_test[key]
        print(f"\n{key}:")
        print(f"  Overall:")
        print(f"    Avg Fmax:      {metrics['Fmax']:.4f}")
        print(f"    Prop-Fmax:     {metrics['prop_Fmax']:.4f} ★")
        print(f"    AUPR:          {metrics['aupr']:.4f}")
        print(f"    Prop-AUPR:     {metrics['prop_aupr']:.4f}")
        
        if metrics['unseen'] is not None:
            print(f"  Unseen ({metrics['unseen_count']} labels):")
            print(f"    Avg Fmax:      {metrics['unseen']['Fmax']:.4f}")
            if 'prop_Fmax' in metrics['unseen']:
                print(f"    Prop-Fmax:     {metrics['unseen']['prop_Fmax']:.4f} ★")
            print(f"    Prop-AUPR:     {metrics['unseen']['prop_aupr']:.4f}")
        
        if metrics['seen'] is not None:
            print(f"  Seen ({metrics['seen_count']} labels):")
            print(f"    Avg Fmax:      {metrics['seen']['Fmax']:.4f}")
            if 'prop_Fmax' in metrics['seen']:
                print(f"    Prop-Fmax:     {metrics['seen']['prop_Fmax']:.4f} ★")
            print(f"    Prop-AUPR:     {metrics['seen']['prop_aupr']:.4f}")
        
        if metrics['harmonic_mean'] is not None:
            print(f"  Harmonic Mean:   {metrics['harmonic_mean']:.4f} ★★")
    
    print(f"{'='*80}\n")