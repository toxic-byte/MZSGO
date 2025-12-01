import os
import pickle
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel
import re

MAXLEN = 2048

def load_nlp_model(nlp_path):
    """加载NLP模型和tokenizer"""
    nlp_tokenizer = AutoTokenizer.from_pretrained(nlp_path)
    nlp_model = AutoModel.from_pretrained(nlp_path)
    nlp_model.cuda()
    nlp_model.eval()
    return nlp_tokenizer, nlp_model


def nlp_embedding(nlp_model, nlp_tokenizer, nlp_dim, sample_ids, label_list, key, top_list, cache_path=None, onto=None, pooling='mean', name_flag="name"):
    """
    生成或加载NLP文本嵌入
    
    参数:
        nlp_model: BiomedBERT模型
        nlp_tokenizer: 对应的tokenizer
        nlp_dim: 嵌入维度
        sample_ids: 样本ID列表,用于对齐
        label_list: 每个样本的GO标签列表
        key: GO命名空间 (biological_process, molecular_function, cellular_component)
        top_list: 高频标签列表
        cache_path: 缓存文件路径,如果提供则尝试加载/保存
        onto: GO本体对象
        pooling: 池化方式 ('mean', 'max', 'cls', None)
        name_flag: 使用GO的名称("name")、定义("def")或两者("all")
    
    返回:
        match_embedding: 文本嵌入列表,与sample_ids对齐
    """
    # 如果提供了缓存路径且文件存在,直接加载
    if cache_path and os.path.exists(cache_path):
        print(f"Loading NLP embeddings from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        # 验证缓存数据的对齐性和池化方式
        if (len(cached_data['embeddings']) == len(sample_ids) and 
            cached_data.get('pooling') == pooling):
            if cached_data['sample_ids'] == sample_ids:
                print(f"Cache loaded successfully. {len(cached_data['embeddings'])} embeddings loaded.")
                return cached_data['embeddings']
            else:
                print("Warning: Sample IDs mismatch. Regenerating embeddings...")
        else:
            print(f"Warning: Cache mismatch (pooling: {cached_data.get('pooling')} vs {pooling}). Regenerating embeddings...")
    
    # 生成新的嵌入
    print(f"Generating NLP embeddings")
    print(f"Processing {len(label_list)} samples for ontology: {key}")
    print(f"Pooling method: {pooling}")
    
    match_embedding = []
    
    for index, multi_tag in enumerate(tqdm(label_list, desc="Generating NLP embeddings")):
        if multi_tag == []:
            # 对于没有标签的样本,创建零向量
            if pooling is None:
                match_embedding.append(torch.zeros(1, nlp_dim).cuda())
            else:
                match_embedding.append(torch.zeros(nlp_dim).cuda())
            continue
        
        context = ''
        for _tag in multi_tag:
            if _tag not in top_list:
                continue
            for ont in onto:
                if ont.namespace != key:
                    continue
                _tag = 'GO:' + _tag
                if _tag in ont.terms_dict.keys():
                    if name_flag == "name":
                        # print("Using GO names as context.")
                        tag_context = ont.terms_dict[_tag]['name']  # 只要名字,并且将一个蛋白质的所有GO名字拼接
                        context = context + tag_context + ' '
                    elif name_flag == "def":
                        # print("Using GO definitions as context.")
                        tag_context = ont.terms_dict[_tag]['def']
                        tag_contents = re.findall(r'"(.*?)"', tag_context)
                        if context == '':
                            context = context + tag_contents[0]
                        else:
                            context = context + ' ' + tag_contents[0]
        
        # 如果没有找到任何上下文,使用零向量
        if context == '':
            if pooling is None:
                match_embedding.append(torch.zeros(1, nlp_dim).cuda())
            else:
                match_embedding.append(torch.zeros(nlp_dim).cuda())
            continue
        
        # 分段处理长文本
        seq_len = 512
        max_len = MAXLEN // 2
        if len(context) > max_len:
            context = context[:max_len]
        
        num_seqs = len(context) // seq_len + (1 if len(context) % seq_len != 0 else 0)
        last_embed = []
        
        with torch.no_grad():
            for i in range(num_seqs):
                start_index = i * seq_len
                end_index = min((i + 1) * seq_len, len(context))
                context_sample = context[start_index:end_index]
                inputs = nlp_tokenizer(context_sample, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = nlp_model(**inputs)
                last_hidden_states = outputs.last_hidden_state.squeeze(0).detach()
                last_embed.append(last_hidden_states)
        
        # 合并所有段落的embeddings
        embed = torch.cat(last_embed, dim=0)  # [total_seq_len, nlp_dim]
        
        # 应用池化
        if pooling == 'mean':
            # 平均池化
            pooled_embed = embed.mean(dim=0)  # [nlp_dim]
        elif pooling == 'max':
            # 最大池化
            pooled_embed = embed.max(dim=0)[0]  # [nlp_dim]
        elif pooling == 'cls':
            # 使用CLS token (第一个token)
            pooled_embed = embed[0]  # [nlp_dim]
        elif pooling is None:
            # 不池化,保持原样
            pooled_embed = embed.unsqueeze(0)  # [1, seq_len, nlp_dim]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        match_embedding.append(pooled_embed)
    
    # 如果使用了池化,转换为tensor
    if pooling is not None:
        match_embedding = torch.stack(match_embedding)  # [num_samples, nlp_dim]
    
    # 保存到缓存
    if cache_path:
        print(f"Saving NLP embeddings to cache: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_data = {
            'sample_ids': sample_ids,
            'embeddings': match_embedding,
            'key': key,
            'num_samples': len(sample_ids),
            'pooling': pooling,
            'embedding_dim': nlp_dim
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Cache saved successfully.")
    
    print(f"NLP embedding generation completed. Total samples: {len(match_embedding)}")
    if pooling is not None:
        print(f"Embedding shape: {match_embedding.shape}")
    
    return match_embedding


# =====================
# NLP Embedding处理
# =====================
def compute_nlp_embeddings(config, nlp_model, nlp_tokenizer, key, train_id, test_id, training_labels, label_list, onto):
    """计算NLP embeddings"""
    if config['run_mode'] == "sample":
        train_nlp_cache = os.path.join(config['cache_dir'], f"nlp/train_nlp_embeddings_{key}_{config['text_mode']}_sample.pkl")
    elif config['run_mode'] == "full":
        train_nlp_cache = os.path.join(config['cache_dir'], f"nlp/train_nlp_embeddings_{key}_{config['text_mode']}.pkl")
    
    print(f"\n--- Processing Train NLP Embeddings for {key} ---")
    train_nlp = nlp_embedding(
        nlp_model,
        nlp_tokenizer,
        config['nlp_dim'],
        train_id,
        training_labels[key], 
        key, 
        label_list,
        cache_path=train_nlp_cache,
        onto=onto,
        pooling='mean',
        name_flag=config['text_mode']
    )
    
    # 测试集使用零向量作为占位符
    print(f"\n--- Creating placeholder NLP Embeddings for Test Set ---")
    test_nlp = torch.zeros(len(test_id), config['nlp_dim'])
    print(f"Test NLP embeddings shape: {test_nlp.shape}")
    
    return train_nlp, test_nlp


# 测试集也加上文本向量
def compute_nlp_embeddings_with_test(config, nlp_model, nlp_tokenizer, key, train_id, test_id, training_labels, test_labels, label_list, onto):
    """计算NLP embeddings"""
    if config['run_mode'] == "sample":
        train_nlp_cache = os.path.join(config['cache_dir'], f"nlp/train_nlp_embeddings_{key}_{config['text_mode']}_sample.pkl")
        test_nlp_cache = os.path.join(config['cache_dir'], f"nlp/test_nlp_embeddings_{key}_{config['text_mode']}_sample.pkl")
    elif config['run_mode'] == "full":
        train_nlp_cache = os.path.join(config['cache_dir'], f"nlp/train_nlp_embeddings_{key}_{config['text_mode']}.pkl")
        test_nlp_cache = os.path.join(config['cache_dir'], f"nlp/test_nlp_embeddings_{key}_{config['text_mode']}.pkl")
    
    print(f"\n--- Processing Train NLP Embeddings for {key} ---")
    train_nlp = nlp_embedding(
        nlp_model,
        nlp_tokenizer,
        config['nlp_dim'],
        train_id,
        training_labels[key], 
        key, 
        label_list,
        cache_path=train_nlp_cache,
        onto=onto,
        pooling='mean',
        name_flag=config['text_mode']
    )

    test_nlp = nlp_embedding(
        nlp_model,
        nlp_tokenizer,
        config['nlp_dim'],
        test_id,
        test_labels[key], 
        key, 
        label_list,
        cache_path=test_nlp_cache,
        onto=onto,
        pooling='mean',
        name_flag=config['text_mode']
    )
    
    return train_nlp, test_nlp


def list_embedding(nlp_model, nlp_tokenizer, nlp_dim, key, top_list, cache_path=None, onto=None, pooling='mean', name_flag="all"):
    """
    生成或加载NLP文本嵌入(为top_list中的每个标签生成向量)
    
    参数:
        nlp_model: BiomedBERT模型
        nlp_tokenizer: 对应的tokenizer
        nlp_dim: 嵌入维度
        key: GO命名空间 (biological_process, molecular_function, cellular_component)
        top_list: 高频标签列表
        cache_path: 缓存文件路径,如果提供则尝试加载/保存
        onto: GO本体对象
        pooling: 池化方式 ('mean', 'max', 'cls')
        name_flag: 使用GO的名称("name")、定义("def")或两者("all")
    
    返回:
        embeddings: 所有top_list标签的向量 [len(top_list), nlp_dim]
    """
    # 如果提供了缓存路径且文件存在,直接加载
    if cache_path and os.path.exists(cache_path):
        print(f"Loading NLP embeddings from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        # 验证缓存数据
        if (cached_data.get('pooling') == pooling and 
            cached_data.get('key') == key and
            cached_data.get('top_list') == top_list and
            cached_data.get('name_flag') == name_flag):
            print(f"Cache loaded successfully. Shape: {cached_data['embeddings'].shape}")
            return cached_data['embeddings']
        else:
            print("Warning: Cache mismatch. Regenerating embeddings...")
    
    # 生成新的嵌入
    print(f"Generating NLP embeddings for {len(top_list)} GO terms")
    print(f"Ontology namespace: {key}")
    print(f"Pooling method: {pooling}")
    print(f"Using GO {name_flag} as context")
    
    all_embeddings = []
    
    for _tag in tqdm(top_list, desc="Processing GO terms"):
        context = ''
        
        # 查找该标签的文本描述
        for ont in onto:
            if ont.namespace != key:
                continue
            
            _tag_with_prefix = 'GO:' + _tag if not _tag.startswith('GO:') else _tag
            
            if _tag_with_prefix in ont.terms_dict.keys():
                term_info = ont.terms_dict[_tag_with_prefix]
                
                if name_flag == "name":
                    context = term_info['name']
                elif name_flag == "def":
                    tag_context = term_info['def']
                    tag_contents = re.findall(r'"(.*?)"', tag_context)
                    if tag_contents:
                        context = tag_contents[0]
                elif name_flag == "all":
                    # 拼接name和def
                    name_part = term_info['name']
                    def_part = ''
                    tag_context = term_info['def']
                    tag_contents = re.findall(r'"(.*?)"', tag_context)
                    if tag_contents:
                        def_part = tag_contents[0]
                    
                    # 使用冒号分隔name和def
                    if name_part and def_part:
                        context = f"{name_part}: {def_part}"
                    elif name_part:
                        context = name_part
                    elif def_part:
                        context = def_part
                else:
                    raise ValueError(f"Unknown name_flag: {name_flag}. Must be 'name', 'def', or 'all'")
                break
        
        # 如果没有找到上下文,使用零向量
        if context == '':
            print(f"Warning: No context found for {_tag}, using zero vector...")
            all_embeddings.append(torch.zeros(nlp_dim).cuda())
            continue
        # print(f"Context for {_tag}: {context}")
        # 分段处理长文本
        seq_len = 512
        max_len = MAXLEN // 2
        if len(context) > max_len:
            context = context[:max_len]
        
        num_seqs = len(context) // seq_len + (1 if len(context) % seq_len != 0 else 0)
        last_embed = []
        
        with torch.no_grad():
            for i in range(num_seqs):
                start_index = i * seq_len
                end_index = min((i + 1) * seq_len, len(context))
                context_sample = context[start_index:end_index]
                inputs = nlp_tokenizer(context_sample, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = nlp_model(**inputs)
                last_hidden_states = outputs.last_hidden_state.squeeze(0).detach()
                last_embed.append(last_hidden_states)
        
        # 合并所有段落的embeddings
        embed = torch.cat(last_embed, dim=0)  # [total_seq_len, nlp_dim]
        
        # 应用池化
        if pooling == 'mean':
            pooled_embed = embed.mean(dim=0)  # [nlp_dim]
        elif pooling == 'max':
            pooled_embed = embed.max(dim=0)[0]  # [nlp_dim]
        elif pooling == 'cls':
            pooled_embed = embed[0]  # [nlp_dim]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        all_embeddings.append(pooled_embed)
    
    # 堆叠所有向量
    embeddings = torch.stack(all_embeddings)  # [len(top_list), nlp_dim]
    
    print(f"NLP embedding generation completed.")
    print(f"Embedding shape: {embeddings.shape}")
    
    # 保存到缓存
    if cache_path:
        print(f"Saving NLP embeddings to cache: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_data = {
            'top_list': top_list,
            'embeddings': embeddings,
            'key': key,
            'num_terms': len(top_list),
            'pooling': pooling,
            'embedding_dim': nlp_dim,
            'name_flag': name_flag
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Cache saved successfully.")
    
    return embeddings


def compute_nlp_embeddings_list(config, nlp_model, nlp_tokenizer, key, label_list, onto):
    """计算NLP embeddings"""
    if config['run_mode'] == "sample":
        list_nlp_cache = os.path.join(config['cache_dir'], f"go_text/{config['occ_num']}/train_nlp_embeddings_{key}_{config['text_mode']}_{config['nlp_model_type']}_sample.pkl")
    elif config['run_mode'] == "full":
        list_nlp_cache = os.path.join(config['cache_dir'], f"go_text/{config['occ_num']}/train_nlp_embeddings_{key}_{config['text_mode']}_{config['nlp_model_type']}.pkl")
    elif config['run_mode'] == "zero":
        list_nlp_cache = os.path.join(config['cache_dir'], f"go_text/{config['occ_num']}/train_nlp_embeddings_{key}_{config['text_mode']}_{config['nlp_model_type']}_zero.pkl")
    
    print(f"\n--- Processing Train NLP Embeddings for {key} ---")
    embeddings = list_embedding(
        nlp_model,
        nlp_tokenizer,
        config['nlp_dim'],
        key, 
        label_list,
        cache_path=list_nlp_cache,
        onto=onto,
        pooling='mean',
        name_flag=config['text_mode']
    )

    return embeddings

def load_pretrained_go_embeddings(embedding_path, label_list, device='cuda'):
    """
    加载预训练的GO embeddings
    
    Args:
        embedding_path: 预训练embeddings文件路径
        label_list: GO term列表 (可能有或没有 'GO:' 前缀)
        device: 设备类型
    
    Returns:
        torch.Tensor: GO embeddings张量
    """
    # 加载预训练的embeddings
    with open(embedding_path, 'rb') as f:
        pretrained_embeddings = pickle.load(f)
    
    # 获取embedding维度
    embedding_dim = next(iter(pretrained_embeddings.values())).shape[0]
    
    # 检查预训练embeddings的键格式
    sample_key = next(iter(pretrained_embeddings.keys()))
    has_go_prefix = sample_key.startswith('GO:')
    
    print(f"Pretrained embeddings sample key: {sample_key}")
    print(f"Label list sample: {label_list[:5]}")
    
    # 初始化embeddings矩阵
    go_embeddings = []
    missing_terms = []
    found_count = 0
    
    for go_term in label_list:
        # 标准化GO term格式
        if has_go_prefix and not go_term.startswith('GO:'):
            # 预训练有GO:前缀，但label_list没有，需要添加
            lookup_term = f"GO:{go_term}"
        elif not has_go_prefix and go_term.startswith('GO:'):
            # 预训练没有GO:前缀，但label_list有，需要去除
            lookup_term = go_term.replace('GO:', '')
        else:
            # 格式一致，直接使用
            lookup_term = go_term
        
        if lookup_term in pretrained_embeddings:
            go_embeddings.append(pretrained_embeddings[lookup_term])
            found_count += 1
        else:
            # 如果GO term不存在，使用零向量
            missing_terms.append(go_term)
            go_embeddings.append(np.zeros(embedding_dim))
    
    print(f"\nGO Embeddings Loading Summary:")
    print(f"  Total GO terms: {len(label_list)}")
    print(f"  Found in pretrained: {found_count}")
    print(f"  Missing: {len(missing_terms)}")
    
    if missing_terms:
        print(f"  Missing terms (first 10): {missing_terms[:10]}")
    
    # 转换为tensor
    go_embeddings = torch.FloatTensor(np.array(go_embeddings)).to(device)
    
    return go_embeddings


def combine_nlp_and_go_embeddings(list_nlp, go_embeddings, method='concat', weights=None):
    """
    结合NLP embeddings和预训练的GO embeddings
    
    Args:
        list_nlp: NLP模型生成的embeddings (tensor: [num_labels, nlp_dim])
        go_embeddings: 预训练的GO embeddings (tensor: [num_labels, go_dim])
        method: 结合方法
            - 'concat': 直接拼接 (默认)
            - 'weighted_sum': 加权求和 (需要相同维度)
            - 'attention': 注意力融合
            - 'gated': 门控融合
        weights: 当method='weighted_sum'时使用的权重 [w_nlp, w_go]
    
    Returns:
        torch.Tensor: 结合后的embeddings
    """
    
    print(f"\nCombining embeddings:")
    print(f"  NLP embeddings shape: {list_nlp.shape}")
    print(f"  GO embeddings shape: {go_embeddings.shape}")
    
    if method == 'concat':
        # 方法1: 直接拼接
        combined_embeddings = torch.cat([list_nlp, go_embeddings], dim=1)
        print(f"  Combined embeddings shape (concat): {combined_embeddings.shape}")
        
    elif method == 'weighted_sum':
        # 方法2: 加权求和 (需要相同维度)
        assert list_nlp.shape[1] == go_embeddings.shape[1], \
            f"For weighted_sum, embeddings must have same dimension. Got {list_nlp.shape[1]} and {go_embeddings.shape[1]}"
        
        if weights is None:
            weights = [0.5, 0.5]  # 默认等权重
        
        w_nlp, w_go = weights
        combined_embeddings = w_nlp * list_nlp + w_go * go_embeddings
        print(f"  Combined embeddings shape (weighted_sum): {combined_embeddings.shape}")
        print(f"  Weights: NLP={w_nlp}, GO={w_go}")
        
    elif method == 'attention':
        # 方法3: 简单的注意力融合
        device = list_nlp.device
        hidden_dim = max(list_nlp.shape[1], go_embeddings.shape[1])
        
        # 线性投影层
        proj_nlp = torch.nn.Linear(list_nlp.shape[1], hidden_dim).to(device)
        proj_go = torch.nn.Linear(go_embeddings.shape[1], hidden_dim).to(device)
        
        # 投影
        nlp_projected = proj_nlp(list_nlp)
        go_projected = proj_go(go_embeddings)
        
        # 计算注意力权重
        attn_scores = torch.nn.functional.softmax(
            torch.stack([nlp_projected, go_projected], dim=1).mean(dim=2), 
            dim=1
        )
        
        # 加权融合
        combined_embeddings = (attn_scores[:, 0:1] * nlp_projected + 
                              attn_scores[:, 1:2] * go_projected)
        print(f"  Combined embeddings shape (attention): {combined_embeddings.shape}")
        
    elif method == 'gated':
        # 方法4: 门控融合
        device = list_nlp.device
        
        # 投影到相同维度
        hidden_dim = list_nlp.shape[1]
        proj_go = torch.nn.Linear(go_embeddings.shape[1], hidden_dim).to(device)
        go_projected = proj_go(go_embeddings)
        
        # 门控机制
        gate = torch.nn.Linear(hidden_dim * 2, hidden_dim).to(device)
        gate_input = torch.cat([list_nlp, go_projected], dim=1)
        gate_values = torch.sigmoid(gate(gate_input))
        
        # 融合
        combined_embeddings = gate_values * list_nlp + (1 - gate_values) * go_projected
        print(f"  Combined embeddings shape (gated): {combined_embeddings.shape}")
        
    elif method == 'mlp_fusion':
        # 方法5: MLP融合 (新增)
        device = list_nlp.device
        input_dim = list_nlp.shape[1] + go_embeddings.shape[1]
        output_dim = list_nlp.shape[1]  # 或者自定义
        
        mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(output_dim * 2, output_dim)
        ).to(device)
        
        concatenated = torch.cat([list_nlp, go_embeddings], dim=1)
        combined_embeddings = mlp(concatenated)
        print(f"  Combined embeddings shape (mlp_fusion): {combined_embeddings.shape}")
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: concat, weighted_sum, attention, gated, mlp_fusion")
    
    return combined_embeddings


def create_combined_embeddings(config, nlp_model, nlp_tokenizer, ontology_key, 
                               label_list, onto, pretrained_embedding_path,
                               combine_method='concat', weights=None):
    """
    创建结合了NLP和预训练GO embeddings的联合嵌入
    
    Args:
        config: 配置字典
        nlp_model: NLP模型
        nlp_tokenizer: NLP分词器
        ontology_key: 本体类型 (如 'biological_process')
        label_list: GO term列表
        onto: 本体对象
        pretrained_embedding_path: 预训练embeddings路径
        combine_method: 结合方法 ('concat', 'weighted_sum', 'attention', 'gated', 'mlp_fusion')
        weights: 加权求和时的权重 [w_nlp, w_go]
    
    Returns:
        torch.Tensor: 联合嵌入
    """
    print(f"\n{'='*60}")
    print(f"Creating combined embeddings for {ontology_key}")
    print(f"{'='*60}")
    
    # 1. 计算NLP embeddings
    print("\n1. Computing NLP embeddings...")
    list_nlp = compute_nlp_embeddings_list(
        config, nlp_model, nlp_tokenizer, ontology_key, label_list, onto
    ).cuda()
    print(f"   NLP embeddings computed: {list_nlp.shape}")
    
    # 2. 加载预训练的GO embeddings
    print("\n2. Loading pretrained GO embeddings...")
    go_embeddings = load_pretrained_go_embeddings(
        pretrained_embedding_path, 
        label_list,
        device='cuda'
    )
    print(f"   GO embeddings loaded: {go_embeddings.shape}")
    
    # 3. 结合两种embeddings
    print(f"\n3. Combining embeddings using method: {combine_method}")
    combined_embeddings = combine_nlp_and_go_embeddings(
        list_nlp, 
        go_embeddings, 
        method=combine_method,
        weights=weights
    )
    
    print(f"\n{'='*60}")
    print(f"Combined embeddings created successfully!")
    print(f"Final shape: {combined_embeddings.shape}")
    print(f"{'='*60}\n")
    
    return combined_embeddings