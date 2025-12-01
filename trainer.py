import torch
import torch.nn as nn   
from tqdm import tqdm
from utils.util import (evaluate_annotations, compute_propagated_metrics, 
                       get_cosine_schedule_with_warmup,FocalLoss)
import os
from datetime import datetime
from models.Binary_domain_gated import CustomModel
import math
from test_zero import (identify_unseen_labels, print_unseen_label_analysis,
                      evaluate_unseen_labels, compute_harmonic_mean)
from torch.nn import DataParallel

def create_model_and_optimizer(config, train_domain_features, pos_weight=None, total_steps=None, adj=None):
    model = CustomModel(
        esm_dim=config['embed_dim'],
        nlp_dim=config['nlp_dim'],
        inter_size=train_domain_features.shape[1],
        hidden_dim=config.get('hidden_dim', 512),
        dropout=config.get('dropout', 0.3)
    ).cuda()
    
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion = FocalLoss(gamma=2,alpha=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    if total_steps is None:
        total_steps = 1000 
    
    warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=config.get('num_cycles', 0.5)
    )

    print("\n" + "="*50)
    print("Model Configuration:")
    print("="*50)
    print(f"ESM Embedding Dim: {config['embed_dim']}")
    print(f"NLP Embedding Dim: {config['nlp_dim']}")
    print(f"Domain Feature Size: {train_domain_features.shape[1]}")
    print(f"Hidden Dim: {config.get('hidden_dim', 512)}")
    print(f"Dropout: {config.get('dropout', 0.3)}")
    print(f"\nTrainable parameters:")
    
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")
            trainable_params += param.numel()
    
    print(f"\nTotal trainable parameters: {trainable_params:,}")
    print(f"Estimated total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Learning rate schedule: Warmup + Cosine Annealing")
    print(f"Training mode: Early stopping based on test loss")
    print("="*50 + "\n")

    return model, criterion, optimizer, scheduler


def train_one_epoch_efficient(model, train_dataloader, list_embedding, criterion, optimizer, 
                              scheduler, epoch, key):
    model.train()
    loss_mean = 0
    
    list_embedding = list_embedding.cuda()  # [num_go_terms, nlp_dim]
    num_go_terms = list_embedding.shape[0]
    
    for batch_idx, batch_data in tqdm(enumerate(train_dataloader), 
                                    desc=f"Epoch {epoch+1} Training",
                                    total=len(train_dataloader)):
        optimizer.zero_grad()
        
        batch_embeddings = batch_data['embedding'].cuda()
        batch_domain_features = batch_data['domain_feature'].cuda()
        batch_labels = batch_data['labels'].cuda()
        batch_size = batch_embeddings.shape[0]
        
        # 批量处理所有样本-GO配对
        esm_expanded = batch_embeddings.unsqueeze(1).expand(-1, num_go_terms, -1)
        domain_expanded = batch_domain_features.unsqueeze(1).expand(-1, num_go_terms, -1)
        
        esm_flat = esm_expanded.reshape(-1, esm_expanded.size(-1))
        domain_flat = domain_expanded.reshape(-1, domain_expanded.size(-1))
        
        outputs = model(esm_flat, domain_flat, list_embedding, batch_size)
        
        loss = criterion(outputs, batch_labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        loss_mean += loss.item()
        
        if (batch_idx + 1) % 50 == 0:
            print('{}  Epoch [{}], Step [{}/{}], LR: {:.6f}, Loss: {:.4f}'.format(
                key, epoch + 1, batch_idx + 1,
                len(train_dataloader), current_lr, loss_mean / (batch_idx + 1)))
    
    avg_loss = loss_mean / len(train_dataloader)
    print(f"\nEpoch {epoch+1} Training Summary:")
    print(f"  Avg Training Loss: {avg_loss:.4f}")
    
    return avg_loss


def evaluate_test_loss(model, test_dataloader, list_embedding, criterion, key):
    """
    计算测试集的loss（用于早停）
    
    Args:
        model: 模型
        test_dataloader: 测试数据加载器
        list_embedding: GO term文本嵌入
        criterion: 损失函数
        key: 本体类型
    
    Returns:
        avg_test_loss: 平均测试loss
    """
    model.eval()
    test_loss = 0
    
    list_embedding = list_embedding.cuda()
    num_go_terms = list_embedding.shape[0]
    
    with torch.no_grad():
        for batch_data in tqdm(test_dataloader, desc=f"Calculating test loss for {key}"):
            batch_embeddings = batch_data['embedding'].cuda()
            batch_domain_features = batch_data['domain_feature'].cuda()
            batch_labels = batch_data['labels'].cuda()
            batch_size = batch_embeddings.shape[0]
            
            # 批量处理
            esm_expanded = batch_embeddings.unsqueeze(1).expand(-1, num_go_terms, -1)
            domain_expanded = batch_domain_features.unsqueeze(1).expand(-1, num_go_terms, -1)
            
            esm_flat = esm_expanded.reshape(-1, esm_expanded.size(-1))
            domain_flat = domain_expanded.reshape(-1, domain_expanded.size(-1))
            
            output = model(esm_flat, domain_flat, list_embedding, batch_size)
            
            loss = criterion(output, batch_labels)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_dataloader)
    return avg_test_loss


def evaluate_model_with_unseen(model, test_dataloader, list_embedding, ia_list, key, 
                               adj_matrix, unseen_indices, seen_indices):
    """
    评估模型（包含未见标签分析）- 修改为与第二个代码相同的评估指标
    
    Args:
        model: 训练好的模型
        test_dataloader: 测试数据加载器
        list_embedding: GO term文本嵌入
        ia_list: Information Accretion列表
        key: 本体类型
        adj_matrix: 邻接矩阵，用于标签传播
        unseen_indices: 未见标签索引
        seen_indices: 已见标签索引
    
    Returns:
        metrics: 包含所有评估指标的字典
    """
    model.eval()
    _labels = []
    _preds = []
    sigmoid = torch.nn.Sigmoid()
    
    list_embedding = list_embedding.cuda()
    num_go_terms = list_embedding.shape[0]
    
    print(f"\n{'='*80}")
    print(f"Evaluating {key} on test set (with unseen label analysis)...")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        for batch_data in tqdm(test_dataloader, desc=f"Evaluating {key}"):
            batch_embeddings = batch_data['embedding'].cuda()
            batch_domain_features = batch_data['domain_feature'].cuda()
            batch_labels = batch_data['labels']
            batch_size = batch_embeddings.shape[0]
            
            # 批量处理
            esm_expanded = batch_embeddings.unsqueeze(1).expand(-1, num_go_terms, -1)
            domain_expanded = batch_domain_features.unsqueeze(1).expand(-1, num_go_terms, -1)
            
            esm_flat = esm_expanded.reshape(-1, esm_expanded.size(-1))
            domain_flat = domain_expanded.reshape(-1, domain_expanded.size(-1))
            
            output = model(esm_flat, domain_flat, list_embedding, batch_size)
            
            output = sigmoid(output).cpu()
            
            _labels.append(batch_labels)
            _preds.append(output)
    
    all_labels = torch.cat(_labels, dim=0)
    all_preds = torch.cat(_preds, dim=0)
    
    # ========== 整体评估（使用Fmax等指标，与第二个代码一致）==========
    f, p, r, aupr, th = evaluate_annotations(all_labels, all_preds)
    prop_fmax, prop_precision, prop_recall, prop_aupr, prop_th, prop_preds = compute_propagated_metrics(
        all_labels, all_preds, adj_matrix
    )
    
    print(f"\n{'='*80}")
    print(f"Overall Results for {key}:")
    print(f"{'='*80}")
    print(f"  Avg Fmax:          {100 * f:.2f}%")
    print(f"  Avg Precision:     {100 * p:.2f}%")
    print(f"  Avg Recall:        {100 * r:.2f}%")
    print(f"  AUPR:              {100 * aupr:.2f}%")
    print(f"  Threshold:         {th:.4f}")
    print(f"  Prop-Fmax:         {100 * prop_fmax:.2f}% ★")
    print(f"  Prop-Precision:    {100 * prop_precision:.2f}%")
    print(f"  Prop-Recall:       {100 * prop_recall:.2f}%")
    print(f"  Prop-AUPR:         {100 * prop_aupr:.2f}%")
    print(f"  Prop-Threshold:    {prop_th:.4f}")
    
    # ========== 未见/已见标签分析 ==========
    label_specific_metrics = evaluate_unseen_labels(
        all_labels, all_preds, unseen_indices, seen_indices, adj_matrix
    )
    
    # 打印未见标签指标
    if label_specific_metrics['unseen'] is not None:
        unseen = label_specific_metrics['unseen']
        print(f"\n{'='*80}")
        print(f"Unseen Labels Performance ({unseen['count']} labels, {unseen['sample_count']} samples):")
        print(f"{'='*80}")
        print(f"  Avg Fmax:          {100 * unseen['Fmax']:.2f}%")
        print(f"  Avg Precision:     {100 * unseen['precision']:.2f}%")
        print(f"  Avg Recall:        {100 * unseen['recall']:.2f}%")
        print(f"  AUPR:              {100 * unseen['aupr']:.2f}%")
        print(f"  Threshold:         {unseen['threshold']:.4f}")
        if 'prop_Fmax' in unseen:
            print(f"  Prop-Fmax:         {100 * unseen['prop_Fmax']:.2f}% ★")
            print(f"  Prop-Precision:    {100 * unseen['prop_precision']:.2f}%")
            print(f"  Prop-Recall:       {100 * unseen['prop_recall']:.2f}%")
            print(f"  Prop-AUPR:         {100 * unseen['prop_aupr']:.2f}%")
            print(f"  Prop-Threshold:    {unseen['prop_threshold']:.4f}")
    else:
        print(f"\nNo unseen labels with positive samples")
    
    # 打印已见标签指标
    if label_specific_metrics['seen'] is not None:
        seen = label_specific_metrics['seen']
        print(f"\n{'='*80}")
        print(f"Seen Labels Performance ({seen['count']} labels, {seen['sample_count']} samples):")
        print(f"{'='*80}")
        print(f"  Avg Fmax:          {100 * seen['Fmax']:.2f}%")
        print(f"  Avg Precision:     {100 * seen['precision']:.2f}%")
        print(f"  Avg Recall:        {100 * seen['recall']:.2f}%")
        print(f"  AUPR:              {100 * seen['aupr']:.2f}%")
        print(f"  Threshold:         {seen['threshold']:.4f}")
        if 'prop_Fmax' in seen:
            print(f"  Prop-Fmax:         {100 * seen['prop_Fmax']:.2f}% ★")
            print(f"  Prop-Precision:    {100 * seen['prop_precision']:.2f}%")
            print(f"  Prop-Recall:       {100 * seen['prop_recall']:.2f}%")
            print(f"  Prop-AUPR:         {100 * seen['prop_aupr']:.2f}%")
            print(f"  Prop-Threshold:    {seen['prop_threshold']:.4f}")
    else:
        print(f"\nNo seen labels with positive samples")
    
    # 计算调和平均
    harmonic_mean = None
    if (label_specific_metrics['unseen'] is not None and 
        label_specific_metrics['seen'] is not None and
        'prop_aupr' in label_specific_metrics['unseen'] and
        'prop_aupr' in label_specific_metrics['seen']):
        
        harmonic_mean = compute_harmonic_mean(
            label_specific_metrics['unseen']['prop_aupr'],
            label_specific_metrics['seen']['prop_aupr']
        )
        
        print(f"\n{'='*80}")
        print(f"Harmonic Mean (H):")
        print(f"{'='*80}")
        print(f"  H = 2 × ({label_specific_metrics['unseen']['prop_aupr']:.4f} × {label_specific_metrics['seen']['prop_aupr']:.4f}) / ({label_specific_metrics['unseen']['prop_aupr']:.4f} + {label_specific_metrics['seen']['prop_aupr']:.4f})")
        print(f"  H = {100 * harmonic_mean:.2f}% ★★")
    
    print(f"{'='*80}\n")
    
    metrics = {
        'p': p,
        'r': r,
        'Fmax': f,
        'aupr': aupr,
        'threshold': th,
        'prop_Fmax': prop_fmax,
        'prop_precision': prop_precision,
        'prop_recall': prop_recall,
        'prop_aupr': prop_aupr,
        'prop_threshold': prop_th,
        'unseen': label_specific_metrics['unseen'],
        'seen': label_specific_metrics['seen'],
        'harmonic_mean': harmonic_mean
    }
    
    return metrics


def train_model_for_ontology(config, key, train_dataloader, test_dataloader, 
                            list_embedding, ia_list, ctime, 
                            metrics_output_test, train_domain_features, 
                            adj_matrix=None, pos_weight=None,
                            training_labels_binary=None, test_labels_binary=None,
                            label_list=None):
    """
    为特定本体训练模型（使用早停策略，基于测试集loss）
    
    Args:
        config: 配置字典（需包含以下键）
            - patience: 早停耐心值（默认10）
            - min_delta: 最小改善阈值（默认0.0001）
            - max_epochs: 最大训练轮数（默认100）
        key: 本体类型 (e.g., 'mf', 'bp', 'cc')
        train_dataloader: 训练数据加载器
        test_dataloader: 测试数据加载器
        list_embedding: GO term文本嵌入
        ia_list: Information Accretion列表
        ctime: 时间戳
        metrics_output_test: 保存指标的字典
        train_domain_features: 领域特征
        adj_matrix: 邻接矩阵，用于标签传播
        pos_weight: 正样本权重
        training_labels_binary: 训练集标签
        test_labels_binary: 测试集标签
        label_list: 标签列表
    
    Returns:
        model: 训练好的模型
    """
    # ========== 早停参数 ==========
    patience = config.get('patience', 10)  # 耐心值：连续多少个epoch没有改善就停止
    min_delta = config.get('min_delta', 0.0001)  # 最小改善阈值
    max_epochs = config.get('epoch_num', 100)  # 最大训练轮数
    
    print(f"\n{'='*80}")
    print(f"Early Stopping Configuration:")
    print(f"{'='*80}")
    print(f"  Patience: {patience} epochs")
    print(f"  Min Delta: {min_delta}")
    print(f"  Max Epochs: {max_epochs}")
    print(f"  Criterion: Test set loss")
    print(f"{'='*80}\n")
    
    # ========== 识别未见标签 ==========
    unseen_indices, seen_indices, train_counts, test_counts = identify_unseen_labels(
        training_labels_binary, test_labels_binary
    )
    
    # 打印未见标签分析
    print_unseen_label_analysis(key, unseen_indices, seen_indices, 
                                train_counts, test_counts, label_list)
    
    # 估计总训练步数（用于scheduler，基于max_epochs）
    estimated_total_steps = len(train_dataloader) * max_epochs
    
    # 创建模型
    model, criterion, optimizer, scheduler = create_model_and_optimizer(
        config, train_domain_features, pos_weight, estimated_total_steps, adj_matrix
    )
    
    # ========== 早停变量 ==========
    best_test_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    
    print(f"\n{'='*80}")
    print(f"Training {key} with early stopping")
    print(f"Monitoring test set loss for early stopping...")
    print(f"{'='*80}\n")
    
    # ========== 训练循环（带早停）==========
    for epoch in range(max_epochs):
        # 训练一个epoch
        train_loss = train_one_epoch_efficient(
            model, train_dataloader, list_embedding, criterion, 
            optimizer, scheduler, epoch, key
        )
        
        # 计算测试集loss
        test_loss = evaluate_test_loss(
            model, test_dataloader, list_embedding, criterion, key
        )
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  Test Loss:     {test_loss:.4f}")
        
        # 检查是否有改善
        if test_loss < best_test_loss - min_delta:
            print(f"  ✓ Test loss improved from {best_test_loss:.4f} to {test_loss:.4f}")
            best_test_loss = test_loss
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ✗ No improvement in test loss (patience: {patience_counter}/{patience})")
        
        print(f"  Best test loss so far: {best_test_loss:.4f} (epoch {best_epoch+1})")
        
        # 早停检查
        if patience_counter >= patience:
            print(f"\n{'='*80}")
            print(f"Early stopping triggered!")
            print(f"  No improvement for {patience} consecutive epochs")
            print(f"  Best test loss: {best_test_loss:.4f} at epoch {best_epoch+1}")
            print(f"  Total epochs trained: {epoch+1}")
            print(f"{'='*80}\n")
            break
    else:
        print(f"\n{'='*80}")
        print(f"Reached maximum epochs ({max_epochs})")
        print(f"  Best test loss: {best_test_loss:.4f} at epoch {best_epoch+1}")
        print(f"{'='*80}\n")
    
    # ========== 恢复最佳模型 ==========
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model from epoch {best_epoch+1}")
    
    print(f"\n{'='*80}")
    print(f"Training completed for {key}!")
    print(f"Starting final evaluation with unseen label analysis...")
    print(f"{'='*80}\n")
    
    # ========== 训练结束后进行一次评估（包含未见标签分析）==========
    metrics = evaluate_model_with_unseen(
        model, test_dataloader, list_embedding, ia_list, key, 
        adj_matrix, unseen_indices, seen_indices
    )
    
    # 保存指标
    if key not in metrics_output_test:
        metrics_output_test[key] = {}
    
    for metric_name, metric_value in metrics.items():
        metrics_output_test[key][metric_name] = metric_value
    
    # 保存未见标签信息和训练信息
    metrics_output_test[key]['unseen_count'] = len(unseen_indices)
    metrics_output_test[key]['seen_count'] = len(seen_indices)
    metrics_output_test[key]['best_epoch'] = best_epoch + 1
    metrics_output_test[key]['total_epochs'] = epoch + 1
    metrics_output_test[key]['best_test_loss'] = best_test_loss
    
    # ========== 保存最终模型 ==========
    ckpt_dir = './ckpt/cafa5/domain_text_gated/'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{ctime}domain_text_gated_{key}_best.pt")
    
    torch.save({
        'epoch': best_epoch + 1,
        'total_epochs': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_test_loss': best_test_loss,
        'metrics': metrics_output_test[key],
        'config': config,
        'unseen_indices': unseen_indices,
        'seen_indices': seen_indices
    }, ckpt_path)
    
    print(f"\n{'='*80}")
    print(f"Best model saved:")
    print(f"  Path: {ckpt_path}")
    print(f"  Best epoch: {best_epoch+1}")
    print(f"  Total epochs: {epoch+1}")
    print(f"  Best test loss: {best_test_loss:.4f}")
    print(f"  Overall Prop-Fmax: {metrics['prop_Fmax']:.4f}")
    print(f"  Overall Avg Fmax: {metrics['Fmax']:.4f}")
    if metrics['harmonic_mean'] is not None:
        print(f"  Harmonic Mean: {metrics['harmonic_mean']:.4f} ★★")
    if metrics['unseen'] is not None and 'prop_Fmax' in metrics['unseen']:
        print(f"  Unseen Prop-Fmax: {metrics['unseen']['prop_Fmax']:.4f}")
    if metrics['seen'] is not None and 'prop_Fmax' in metrics['seen']:
        print(f"  Seen Prop-Fmax: {metrics['seen']['prop_Fmax']:.4f}")
    print(f"{'='*80}\n")
    
    return model