from datetime import datetime
import sys
import torch
import os
import numpy as np

sys.path.append(r"../utils")
sys.path.append(r"../models")
sys.path.append(r"../trainer")

from dataset import (obo_graph, load_datasets, process_labels_for_ontology,
                     create_dataloaders, create_ontology_adjacency_matrix)
from config import setup_environment, get_config
from nlp_embed import load_nlp_model, compute_nlp_embeddings_list
from embed import (compute_esm_embeddings, load_domain_features,
                   load_domain_features_with_pretrained_encoder)
from sklearn import preprocessing
import torch.nn as nn
from tqdm import tqdm
from util import evaluate_annotations, compute_propagated_metrics

def identify_unseen_labels(train_labels, test_labels):
    """
    识别训练集中未出现但在测试集中出现的标签
    
    Args:
        train_labels: 训练集标签 (可以是list、numpy array或tensor)
        test_labels: 测试集标签 (可以是list、numpy array或tensor)
    
    Returns:
        unseen_label_indices: 未见标签的索引列表
        seen_label_indices: 已见标签的索引列表
        train_label_counts: 训练集中每个标签的出现次数
        test_label_counts: 测试集中每个标签的出现次数
    """
    # 统一转换为numpy array
    if isinstance(train_labels, list):
        train_labels = np.array(train_labels, dtype=np.float32)
    elif isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.cpu().numpy()
    
    if isinstance(test_labels, list):
        test_labels = np.array(test_labels, dtype=np.float32)
    elif isinstance(test_labels, torch.Tensor):
        test_labels = test_labels.cpu().numpy()
    
    # 计算每个标签在训练集中出现的次数
    train_label_counts = train_labels.sum(axis=0)
    
    # 计算每个标签在测试集中出现的次数
    test_label_counts = test_labels.sum(axis=0)
    
    # 未见标签:训练集中从未出现(count=0)但测试集中出现(count>0)
    unseen_mask = (train_label_counts == 0) & (test_label_counts > 0)
    unseen_label_indices = np.where(unseen_mask)[0].tolist()
    
    # 已见标签:训练集中出现过且测试集中也出现
    seen_mask = (train_label_counts > 0) & (test_label_counts > 0)
    seen_label_indices = np.where(seen_mask)[0].tolist()
    
    # 转换回torch tensor以便后续使用
    train_label_counts = torch.from_numpy(train_label_counts)
    test_label_counts = torch.from_numpy(test_label_counts)
    
    return unseen_label_indices, seen_label_indices, train_label_counts, test_label_counts


def evaluate_unseen_labels(all_labels, all_preds, unseen_indices, seen_indices, adj_matrix=None):
    """
    针对未见标签和已见标签分别计算评估指标
    
    Args:
        all_labels: 真实标签
        all_preds: 预测结果
        unseen_indices: 未见标签的索引
        seen_indices: 已见标签的索引
        adj_matrix: 本体邻接矩阵(用于传播)
    
    Returns:
        metrics: 包含未见和已见标签评估指标的字典
    """
    metrics = {}
    
    # 评估未见标签
    if len(unseen_indices) > 0:
        unseen_labels = all_labels[:, unseen_indices]
        unseen_preds = all_preds[:, unseen_indices]
        
        # 确保有正样本
        if unseen_labels.sum().item() > 0:
            f, p, r, aupr, th = evaluate_annotations(unseen_labels, unseen_preds,"unseen")
            
            metrics['unseen'] = {
                'count': len(unseen_indices),
                'Fmax': f,
                'precision': p,
                'recall': r,
                'aupr': aupr,
                'threshold': th,
                'sample_count': int(unseen_labels.sum().item())
            }
            
            # 如果提供了邻接矩阵,计算传播后的指标
            if adj_matrix is not None:
                try:
                    # 将稀疏矩阵转换为密集矩阵以便索引
                    if adj_matrix.is_sparse:
                        adj_matrix_dense = adj_matrix.to_dense()
                    else:
                        adj_matrix_dense = adj_matrix
                    
                    # 为未见标签创建子邻接矩阵
                    unseen_adj = adj_matrix_dense[unseen_indices][:, unseen_indices]
                    
                    # 转换回稀疏矩阵(如果需要)
                    if adj_matrix.is_sparse:
                        unseen_adj = unseen_adj.to_sparse()
                    
                    prop_Fmax, prop_precision, prop_recall, prop_aupr, prop_th, _ = compute_propagated_metrics(
                        unseen_labels, unseen_preds, unseen_adj
                    )
                    metrics['unseen']['prop_Fmax'] = prop_Fmax
                    metrics['unseen']['prop_precision'] = prop_precision
                    metrics['unseen']['prop_recall'] = prop_recall
                    metrics['unseen']['prop_aupr'] = prop_aupr
                    metrics['unseen']['prop_threshold'] = prop_th
                except Exception as e:
                    print(f"Warning: Could not compute propagated metrics for unseen labels: {e}")
        else:
            print("Warning: No positive samples for unseen labels in test set")
            metrics['unseen'] = None
    else:
        metrics['unseen'] = None
    
    # 评估已见标签
    if len(seen_indices) > 0:
        seen_labels = all_labels[:, seen_indices]
        seen_preds = all_preds[:, seen_indices]
        
        # 确保有正样本
        if seen_labels.sum().item() > 0:
            f, p, r, aupr, th = evaluate_annotations(seen_labels, seen_preds,"seen")
            
            metrics['seen'] = {
                'count': len(seen_indices),
                'Fmax': f,
                'precision': p,
                'recall': r,
                'aupr': aupr,
                'threshold': th,
                'sample_count': int(seen_labels.sum().item())
            }
            
            # 如果提供了邻接矩阵,计算传播后的指标
            if adj_matrix is not None:
                try:
                    # 将稀疏矩阵转换为密集矩阵以便索引
                    if adj_matrix.is_sparse:
                        adj_matrix_dense = adj_matrix.to_dense()
                    else:
                        adj_matrix_dense = adj_matrix
                    
                    # 为已见标签创建子邻接矩阵
                    seen_adj = adj_matrix_dense[seen_indices][:, seen_indices]
                    
                    # 转换回稀疏矩阵(如果需要)
                    if adj_matrix.is_sparse:
                        seen_adj = seen_adj.to_sparse()
                    
                    prop_Fmax, prop_precision, prop_recall, prop_aupr, prop_th, _ = compute_propagated_metrics(
                        seen_labels, seen_preds, seen_adj
                    )
                    metrics['seen']['prop_Fmax'] = prop_Fmax
                    metrics['seen']['prop_precision'] = prop_precision
                    metrics['seen']['prop_recall'] = prop_recall
                    metrics['seen']['prop_aupr'] = prop_aupr
                    metrics['seen']['prop_threshold'] = prop_th
                except Exception as e:
                    print(f"Warning: Could not compute propagated metrics for seen labels: {e}")
        else:
            print("Warning: No positive samples for seen labels in test set")
            metrics['seen'] = None
    else:
        metrics['seen'] = None
    
    return metrics

def compute_harmonic_mean(unseen_fmax, seen_fmax):
    """
    计算调和平均数
    H = 2 / (1/Prop-Fmax_seen + 1/Prop-Fmax_unseen) 
      = (2 × Prop-Fmax_seen × Prop-Fmax_unseen) / (Prop-Fmax_seen + Prop-Fmax_unseen)
    
    Args:
        unseen_fmax: 未见标签的Prop-Fmax
        seen_fmax: 已见标签的Prop-Fmax
    
    Returns:
        harmonic_mean: 调和平均数
    """
    if unseen_fmax is None or seen_fmax is None:
        return None
    
    if unseen_fmax == 0 and seen_fmax == 0:
        return 0.0
    
    if unseen_fmax == 0 or seen_fmax == 0:
        return 0.0
    
    harmonic_mean = (2 * unseen_fmax * seen_fmax) / (unseen_fmax + seen_fmax)
    return harmonic_mean

def print_unseen_label_analysis(key, unseen_indices, seen_indices, train_counts, test_counts, label_list):
    """打印未见标签的详细分析"""
    print(f"\n{'='*80}")
    print(f"Label Analysis for {key}")
    print(f"{'='*80}")
    print(f"  Total labels:        {len(label_list)}")
    print(f"  Unseen labels:       {len(unseen_indices)} (appear only in test set)")
    print(f"  Seen labels:         {len(seen_indices)} (appear in both train and test)")
    
    # 训练集中有但测试集中没有的标签
    train_only = ((train_counts > 0) & (test_counts == 0)).sum().item()
    print(f"  Train-only labels:   {train_only}")
    
    if len(unseen_indices) > 0:
        print(f"\n  Unseen Label Details (top 10):")
        print(f"  {'Index':<8} {'GO Term':<15} {'Test Count':<12}")
        print(f"  {'-'*40}")
        
        # 按测试集出现次数排序
        unseen_test_counts = [(idx, test_counts[idx].item()) for idx in unseen_indices]
        unseen_test_counts.sort(key=lambda x: x[1], reverse=True)
        
        for idx, count in unseen_test_counts[:10]:  # 显示前10个
            go_term = label_list[idx] if idx < len(label_list) else "N/A"
            print(f"  {idx:<8} {go_term:<15} {int(count):<12}")
        
        if len(unseen_indices) > 10:
            print(f"  ... and {len(unseen_indices) - 10} more")
        
        # 统计信息
        total_unseen_samples = sum(count for _, count in unseen_test_counts)
        print(f"\n  Total unseen label annotations in test set: {int(total_unseen_samples)}")
    
    print(f"{'='*80}\n")

def save_test_results(config, all_results, output_dir='./test_results'):
    """保存测试结果 - 增强版,包含未见标签信息和调和平均"""
    os.makedirs(output_dir, exist_ok=True)
    ctime = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = os.path.join(output_dir, f"{config['model']}_{ctime}.txt")
    
    with open(output_file, 'w') as f:
        f.write(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: CustomModel (Binary Classification with Pre-trained Inter Embedding)\n")
        f.write(f"Analysis Type: Zero-shot GO term prediction (unseen labels)\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"OVERALL TEST RESULTS\n")
        f.write(f"{'='*80}\n")
        
        for key, result in all_results.items():
            metrics = result['metrics']
            checkpoint_path = result['checkpoint_path']
            
            f.write(f"\n{key}:\n")
            f.write(f"  Model path:        {checkpoint_path}\n")
            f.write(f"  Avg Precision:     {metrics['p']:.4f}\n")
            f.write(f"  Avg Recall:        {metrics['r']:.4f}\n")
            f.write(f"  Avg Fmax:          {metrics['Fmax']:.4f}\n")
            f.write(f"  AUPR:              {metrics['aupr']:.4f}\n")
            f.write(f"  Threshold:         {metrics['threshold']:.4f}\n")
            f.write(f"  Prop-Fmax:         {metrics['prop_Fmax']:.4f} ★\n")
            f.write(f"  Prop-Precision:    {metrics['prop_precision']:.4f}\n")
            f.write(f"  Prop-Recall:       {metrics['prop_recall']:.4f}\n")
            f.write(f"  Prop-AUPR:         {metrics['prop_aupr']:.4f}\n")
            f.write(f"  Prop-Threshold:    {metrics['prop_threshold']:.4f}\n")
            if metrics['harmonic_mean'] is not None:
                f.write(f"  Harmonic Mean (H): {metrics['harmonic_mean']:.4f} ★★\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"UNSEEN LABELS PERFORMANCE (Zero-shot)\n")
        f.write(f"{'='*80}\n")
        
        for key, result in all_results.items():
            metrics = result['metrics']
            unseen = metrics.get('unseen')
            
            f.write(f"\n{key}:\n")
            if unseen is not None:
                f.write(f"  Label count:       {unseen['count']}\n")
                f.write(f"  Sample count:      {unseen['sample_count']}\n")
                f.write(f"  Avg Precision:     {unseen['precision']:.4f}\n")
                f.write(f"  Avg Recall:        {unseen['recall']:.4f}\n")
                f.write(f"  Avg Fmax:          {unseen['Fmax']:.4f}\n")
                f.write(f"  AUPR:              {unseen['aupr']:.4f}\n")
                f.write(f"  Threshold:         {unseen['threshold']:.4f}\n")
                if 'prop_Fmax' in unseen:
                    f.write(f"  Prop-Fmax:         {unseen['prop_Fmax']:.4f} ★\n")
                    f.write(f"  Prop-Precision:    {unseen['prop_precision']:.4f}\n")
                    f.write(f"  Prop-Recall:       {unseen['prop_recall']:.4f}\n")
                    f.write(f"  Prop-AUPR:         {unseen['prop_aupr']:.4f}\n")
                    f.write(f"  Prop-Threshold:    {unseen['prop_threshold']:.4f}\n")
            else:
                f.write(f"  No unseen labels with positive samples\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"SEEN LABELS PERFORMANCE\n")
        f.write(f"{'='*80}\n")
        
        for key, result in all_results.items():
            metrics = result['metrics']
            seen = metrics.get('seen')
            
            f.write(f"\n{key}:\n")
            if seen is not None:
                f.write(f"  Label count:       {seen['count']}\n")
                f.write(f"  Sample count:      {seen['sample_count']}\n")
                f.write(f"  Avg Precision:     {seen['precision']:.4f}\n")
                f.write(f"  Avg Recall:        {seen['recall']:.4f}\n")
                f.write(f"  Avg Fmax:          {seen['Fmax']:.4f}\n")
                f.write(f"  AUPR:              {seen['aupr']:.4f}\n")
                f.write(f"  Threshold:         {seen['threshold']:.4f}\n")
                if 'prop_Fmax' in seen:
                    f.write(f"  Prop-Fmax:         {seen['prop_Fmax']:.4f} ★\n")
                    f.write(f"  Prop-Precision:    {seen['prop_precision']:.4f}\n")
                    f.write(f"  Prop-Recall:       {seen['prop_recall']:.4f}\n")
                    f.write(f"  Prop-AUPR:         {seen['prop_aupr']:.4f}\n")
                    f.write(f"  Prop-Threshold:    {seen['prop_threshold']:.4f}\n")
            else:
                f.write(f"  No seen labels with positive samples\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"HARMONIC MEAN (H)\n")
        f.write(f"{'='*80}\n")
        
        for key, result in all_results.items():
            metrics = result['metrics']
            f.write(f"\n{key}:\n")
            if metrics['harmonic_mean'] is not None:
                f.write(f"  H = 2 × (Prop-Fmax_unseen × Prop-Fmax_seen) / (Prop-Fmax_unseen + Prop-Fmax_seen)\n")
                f.write(f"  H = {metrics['harmonic_mean']:.4f} ★★\n")
            else:
                f.write(f"  Cannot compute harmonic mean (missing unseen or seen labels)\n")
    
    print(f"\n{'='*80}")
    print(f"Test results saved to: {output_file}")
    print(f"{'='*80}\n")
