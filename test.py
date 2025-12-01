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
                   load_domain_features_with_pretrained_encoder,load_text_pretrained_domain_features)
from Binary_domain_gated import CustomModel
from sklearn import preprocessing
import torch.nn as nn
from tqdm import tqdm
from util import evaluate_annotations, compute_propagated_metrics
from test_zero import identify_unseen_labels,evaluate_unseen_labels,compute_harmonic_mean,print_unseen_label_analysis,save_test_results

def load_trained_model(checkpoint_path, config, train_domain_features):
    print(f"\nLoading model from: {checkpoint_path}")
    
    model = CustomModel(
        esm_dim=config['embed_dim'],
        nlp_dim=config['nlp_dim'],
        inter_size=train_domain_features.shape[1],
        hidden_dim=config.get('hidden_dim', 512),
        dropout=config.get('dropout', 0.3)
    ).cuda()
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Trained epochs: {checkpoint.get('epoch', 'N/A')}")
    
    return model, checkpoint


def evaluate_model_test(model, test_dataloader, list_embedding, ia_list, key, adj_matrix, 
                       unseen_indices, seen_indices):
    model.eval()
    _labels = []
    _preds = []
    sigmoid = torch.nn.Sigmoid()
    
    list_embedding = list_embedding.cuda()
    num_go_terms = list_embedding.shape[0]
    
    print(f"\n{'='*80}")
    print(f"Evaluating {key} on test set...")
    
    with torch.no_grad():
        for batch_data in tqdm(test_dataloader, desc=f"Testing {key}"):
            batch_embeddings = batch_data['embedding'].cuda()
            batch_domain_features = batch_data['domain_feature'].cuda()
                        
            batch_labels = batch_data['labels']
            batch_size = batch_embeddings.shape[0]
            
            esm_expanded = batch_embeddings.unsqueeze(1).expand(-1, num_go_terms, -1)
            domain_expanded = batch_domain_features.unsqueeze(1).expand(-1, num_go_terms, -1)
            
            esm_flat = esm_expanded.reshape(-1, esm_expanded.size(-1))
            domain_flat = domain_expanded.reshape(-1, domain_expanded.size(-1))
            
            logits_flat = model(esm_flat, domain_flat, list_embedding,batch_size)
            output = logits_flat.reshape(batch_size, num_go_terms)
            
            output = sigmoid(output).cpu()
            
            _labels.append(batch_labels)
            _preds.append(output)
    
    all_labels = torch.cat(_labels, dim=0)
    all_preds = torch.cat(_preds, dim=0)
    
    f, p, r, aupr, th = evaluate_annotations(all_labels, all_preds)
    
    prop_Fmax, prop_precision, prop_recall, prop_aupr, prop_th, _ = compute_propagated_metrics(
        all_labels, all_preds, adj_matrix
    )
    
    # 计算未见/已见标签的指标
    label_specific_metrics = evaluate_unseen_labels(
        all_labels, all_preds, unseen_indices, seen_indices, adj_matrix
    )
    
    print(f"\n{'='*80}")
    print(f"Overall Test Results for {key}:")
    print(f"{'='*80}")
    print(f"  Avg Fmax:          {100 * f:.2f}%")
    print(f"  Avg Precision:     {100 * p:.2f}%")
    print(f"  Avg Recall:        {100 * r:.2f}%")
    print(f"  AUPR:              {100 * aupr:.2f}%")
    print(f"  Threshold:         {th:.4f}")
    print(f"  Prop-Fmax:         {100 * prop_Fmax:.2f}% ★")
    print(f"  Prop-Precision:    {100 * prop_precision:.2f}%")
    print(f"  Prop-Recall:       {100 * prop_recall:.2f}%")
    print(f"  Prop-AUPR:         {100 * prop_aupr:.2f}%")
    print(f"  Prop-Threshold:    {prop_th:.4f}")
    
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
        print(f"\n{'='*80}")
        print(f"No unseen labels with positive samples")
        print(f"{'='*80}")
    
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
        print(f"\n{'='*80}")
        print(f"No seen labels with positive samples")
        print(f"{'='*80}")
    
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
        print(f"  H = {100 * harmonic_mean:.2f}% ★★")
    
    print(f"{'='*80}\n")
    
    metrics = {
        'p': p,
        'r': r,
        'Fmax': f,
        'aupr': aupr,
        'threshold': th,
        'prop_Fmax': prop_Fmax,
        'prop_precision': prop_precision,
        'prop_recall': prop_recall,
        'prop_aupr': prop_aupr,
        'prop_threshold': prop_th,
        'unseen': label_specific_metrics['unseen'],
        'seen': label_specific_metrics['seen'],
        'harmonic_mean': harmonic_mean
    }
    
    return metrics, all_preds

def main_test():
    seed = setup_environment()
    config = get_config(run_mode="full", text_mode="all", occ_num=0,
                       batch_size_train=32, batch_size_test=32,
                       nlp_model_type="qwen_4b", epoch_num=50)
    
    print('='*80)
    print('Start testing at: {}'.format(datetime.now().strftime("%Y%m%d%H%M%S")))
    print('='*80)
    
    model_paths = {
        'biological_process': '/d/cuiby/paper/cut_0/ckpt/cafa5/domain_text_gated/20251122214650domain_text_gated_biological_process_best.pt',
        'cellular_component': '/d/cuiby/paper/cut_0/ckpt/cafa5/domain_text_gated/20251122214650domain_text_gated_cellular_component_best.pt',
        # 'molecular_function': '/d/cuiby/paper/cut_0/ckpt/cafa5/domain_text_gated/20251122214650domain_text_gated_molecular_function_best.pt'
        'molecular_function': '/d/cuiby/paper/cut_0/ckpt/cafa5/domain_text_ddp/20251127160823domain_text_ddp_molecular_function_best.pt'
        
    }
    
    nlp_tokenizer, nlp_model = load_nlp_model(config['nlp_path'])
    # nlp_tokenizer, nlp_model = None, None
    
    # label_space = {
    #     'biological_process': [],
    #     'molecular_function': [],
    #     'cellular_component': []
    # }
    label_space = {
        'molecular_function': [],
        'biological_process': [],
        'cellular_component': []
    }
    enc = preprocessing.LabelEncoder()
    
    onto, ia_dict = obo_graph(config['obo_path'], config['ia_path'])
    
    train_id, training_sequences, training_labels, test_id, test_sequences, test_labels = load_datasets(
        config, onto, label_space)
    
    # 计算ESM嵌入
    _, test_esm_embeddings = compute_esm_embeddings(
        config, training_sequences, test_sequences)
    
    # 加载结构域特征
    train_domain_features,test_domain_features=load_text_pretrained_domain_features(train_id,test_id,config['domain_text_path'])
    
    all_results = {}
    
    # 对每个本体进行测试
    for key in label_space.keys():
        print(f"Testing for ontology: {key}")
        if key !="molecular_function":
            continue
        
        # 处理标签
        label_list, training_labels_binary, test_labels_binary, enc, ia_list, onto_parent, label_num = process_labels_for_ontology(
            config, key, label_space, training_labels, test_labels, onto, enc, ia_dict)
        
        # 识别未见标签
        unseen_indices, seen_indices, train_counts, test_counts = identify_unseen_labels(
            training_labels_binary, test_labels_binary)
        
        # 打印未见标签分析
        print_unseen_label_analysis(key, unseen_indices, seen_indices, 
                                    train_counts, test_counts, label_list)
        
        # 创建邻接矩阵
        adj_matrix = create_ontology_adjacency_matrix(onto_parent, label_num, key, config)
        
        # 计算list embeddings
        list_nlp = compute_nlp_embeddings_list(
            config, nlp_model, nlp_tokenizer, key, label_list, onto).cuda()
        
        train_nlp = None
        test_nlp = None
        
        # 创建数据加载器(只需要测试集)
        _, test_dataloader = create_dataloaders(
            config, training_sequences, training_labels_binary, _, train_nlp,
            test_sequences, test_labels_binary, test_esm_embeddings, test_nlp,
            train_domain_features, test_domain_features)
        
        # 加载训练好的模型
        checkpoint_path = model_paths[key]
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Model file not found: {checkpoint_path}")
            continue
        
        model, checkpoint = load_trained_model(checkpoint_path, config, train_domain_features)
        
        # 评估模型(包含未见标签分析)
        metrics, predictions = evaluate_model_test(
            model, test_dataloader, list_nlp, ia_list, key, adj_matrix,
            unseen_indices, seen_indices)
        
        # 保存结果
        all_results[key] = {
            'metrics': metrics,
            'checkpoint_path': checkpoint_path,
            'predictions': predictions,
            'unseen_count': len(unseen_indices),
            'seen_count': len(seen_indices)
        }
    
    # 保存所有测试结果
    save_test_results(config, all_results)
    
    # 打印汇总
    print(f"\n{'='*80}")
    print(f"TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    for key, result in all_results.items():
        metrics = result['metrics']
        print(f"\n{key}:")
        print(f"  Overall Metrics:")
        print(f"    Avg Fmax:      {metrics['Fmax']:.4f}")
        print(f"    Prop-Fmax:     {metrics['prop_Fmax']:.4f} ★")
        print(f"    AUPR:          {metrics['aupr']:.4f}")
        print(f"    Prop-AUPR:     {metrics['prop_aupr']:.4f}")
        
        if metrics['unseen'] is not None:
            print(f"  Unseen Labels ({result['unseen_count']} labels):")
            print(f"    Avg Fmax:      {metrics['unseen']['Fmax']:.4f}")
            if 'prop_Fmax' in metrics['unseen']:
                print(f"    Prop-Fmax:     {metrics['unseen']['prop_Fmax']:.4f} ★")
            print(f"    AUPR:          {metrics['unseen']['aupr']:.4f}")
            if 'prop_aupr' in metrics['unseen']:
                print(f"    Prop-AUPR:     {metrics['unseen']['prop_aupr']:.4f}")
        
        if metrics['seen'] is not None:
            print(f"  Seen Labels ({result['seen_count']} labels):")
            print(f"    Avg Fmax:      {metrics['seen']['Fmax']:.4f}")
            if 'prop_Fmax' in metrics['seen']:
                print(f"    Prop-Fmax:     {metrics['seen']['prop_Fmax']:.4f} ★")
            print(f"    AUPR:          {metrics['seen']['aupr']:.4f}")
            if 'prop_aupr' in metrics['seen']:
                print(f"    Prop-AUPR:     {metrics['seen']['prop_aupr']:.4f}")
        
        if metrics['harmonic_mean'] is not None:
            print(f"  Harmonic Mean:   {metrics['harmonic_mean']:.4f} ★★")
    
    print(f"{'='*80}\n")
    
    print('End testing at: {}'.format(datetime.now().strftime("%Y%m%d%H%M%S")))


if __name__ == "__main__":
    main_test()