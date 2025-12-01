from math import e
import os
import pickle
import torch
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import esm
import sys
from scipy.sparse import csr_matrix, save_npz, load_npz
from collections import defaultdict
from collections import Counter

# model, tokenizer = esm.pretrained.esm2_t33_650M_UR50D()
# num_layers = 33

# model, tokenizer = esm.pretrained.esm2_t30_150M_UR50D()
# num_layers = 30

# model, tokenizer = esm.pretrained.esm1b_t33_650M_UR50S()
# num_layers = 31

# model.cuda()
# model.eval()

def precompute_esm_embeddings(sequences, cache_file, pooling='mean'):
    cache_dir = os.path.dirname(cache_file)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"创建缓存目录: {cache_dir}")

    if os.path.exists(cache_file):
        print(f"Loading cached ESM embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"Computing ESM embeddings for {len(sequences)} sequences...")
    batch_converter = tokenizer.get_batch_converter()
    embeddings = []
    
    for i, seq in enumerate(tqdm(sequences, desc="Computing ESM embeddings")):
        batch_labels, batch_strs, batch_tokens = batch_converter([("x", seq)])
        with torch.no_grad():
            batch_tokens = batch_tokens.cuda()
            results = model(batch_tokens, repr_layers=[num_layers])
            token_representations = results["representations"][num_layers]
            
            # 提取去掉 <cls>/<eos> 的部分
            plm_embed = token_representations[0, 1:1 + len(seq), :].cpu()
            
            # 池化处理
            if pooling == 'mean':
                pooled_embed = plm_embed.mean(dim=0)  # [embed_dim]
            elif pooling == 'max':
                pooled_embed, _ = plm_embed.max(dim=0)  # [embed_dim]
            elif pooling == 'cls':
                pooled_embed = token_representations[0, 0, :].cpu()  # [embed_dim]
            else:
                raise ValueError(f"Unsupported pooling method: {pooling}")
            
            embeddings.append(pooled_embed)
    
    # 保存缓存
    print(f"Saving ESM embeddings to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings

def compute_esm_embeddings(config, training_sequences, test_sequences):
    """预计算ESM embeddings"""
    print("Computing ESM embeddings...")
    
    if config['run_mode'] == "sample":
        train_esm_cache = os.path.join(config['cache_dir'], f"esm/{config['esm_type']}/train_esm_embeddings_sample.pkl")
        test_esm_cache = os.path.join(config['cache_dir'], f"esm/{config['esm_type']}/test_esm_embeddings_sample.pkl")
    elif config['run_mode'] == "full":
        train_esm_cache = os.path.join(config['cache_dir'], f"esm/{config['esm_type']}/train_esm_embeddings_mean.pkl")
        test_esm_cache = os.path.join(config['cache_dir'], f"esm/{config['esm_type']}/test_esm_embeddings_mean.pkl")
    elif config['run_mode'] == "zero":
        train_esm_cache = os.path.join(config['cache_dir'], f"esm/{config['esm_type']}/train_esm_embeddings_sample.pkl")
        test_esm_cache = os.path.join(config['cache_dir'], f"esm/{config['esm_type']}/test_esm_embeddings_zero.pkl")
    train_esm_embeddings = precompute_esm_embeddings(training_sequences, train_esm_cache, pooling='mean')
    test_esm_embeddings = precompute_esm_embeddings(test_sequences, test_esm_cache, pooling='mean')
    
    return train_esm_embeddings, test_esm_embeddings

##########################提取蛋白质的one-hot编码############################################
def precompute_onehot_embeddings(sequences, cache_file):
    """预计算one-hot编码的embeddings"""
    cache_dir = os.path.dirname(cache_file)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"创建缓存目录: {cache_dir}")

    if os.path.exists(cache_file):
        print(f"Loading cached One-hot embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"Computing One-hot embeddings for {len(sequences)} sequences...")
    
    # 定义氨基酸字母表（20种标准氨基酸）
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
    
    embeddings = []
    
    for seq in tqdm(sequences, desc="Computing One-hot embeddings"):
        # 为每个序列创建one-hot编码
        seq_length = len(seq)
        onehot = torch.zeros(seq_length, len(amino_acids))
        
        for i, aa in enumerate(seq):
            if aa in aa_to_idx:
                onehot[i, aa_to_idx[aa]] = 1
            # 如果遇到未知氨基酸，保持全0
        
        embeddings.append(onehot)  # 直接添加完整的 [seq_length, 20] 张量
    
    # 保存缓存
    print(f"Saving One-hot embeddings to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings


def compute_onehot_embeddings(config, training_sequences, test_sequences):
    """预计算One-hot embeddings"""
    print("Computing One-hot embeddings...")
    
    if config['run_mode'] == "sample":
        train_onehot_cache = os.path.join(config['cache_dir'], "onehot/train_onehot_embeddings_sample.pkl")
        test_onehot_cache = os.path.join(config['cache_dir'], "onehot/test_onehot_embeddings_sample.pkl")
    elif config['run_mode'] == "full":
        train_onehot_cache = os.path.join(config['cache_dir'], "onehot/train_onehot_embeddings_full.pkl")
        test_onehot_cache = os.path.join(config['cache_dir'], "onehot/test_onehot_embeddings_full.pkl")
    elif config['run_mode'] == "zero":
        train_onehot_cache = os.path.join(config['cache_dir'], "onehot/train_onehot_embeddings_sample.pkl")
        test_onehot_cache = os.path.join(config['cache_dir'], "onehot/test_onehot_embeddings_zero.pkl")
    train_onehot_embeddings = precompute_onehot_embeddings(training_sequences, train_onehot_cache)
    test_onehot_embeddings = precompute_onehot_embeddings(test_sequences, test_onehot_cache)
    
    return train_onehot_embeddings, test_onehot_embeddings

def conv_onehot_embeddings(config, training_sequences, test_sequences):
    """
    使用ProteInfer模型预计算one-hot的卷积向量
    
    Args:
        config: 配置字典
        training_sequences: 训练序列列表
        test_sequences: 测试序列列表
    
    Returns:
        train_conv_embeddings: 训练集的卷积embeddings列表
        test_conv_embeddings: 测试集的卷积embeddings列表
    """
    from models.protein_encoders import ProteInfer
    import pickle
    import os
    from tqdm import tqdm
    
    print("Computing convolution One-hot embeddings with ProteInfer...")
    
    # 设置缓存文件路径
    if config['run_mode'] == "sample":
        train_conv_cache = os.path.join(config['cache_dir'], "onehot/train_conv_onehot_embeddings_sample.pkl")
        test_conv_cache = os.path.join(config['cache_dir'], "onehot/test_conv_onehot_embeddings_sample.pkl")
    elif config['run_mode'] == "full":
        train_conv_cache = os.path.join(config['cache_dir'], "onehot/train_conv_onehot_embeddings_full.pkl")
        test_conv_cache = os.path.join(config['cache_dir'], "onehot/test_conv_onehot_embeddings_full.pkl")
    
    # 检查缓存
    cache_dir = os.path.dirname(train_conv_cache)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"创建缓存目录: {cache_dir}")
    
    # 如果缓存存在，直接加载
    if os.path.exists(train_conv_cache) and os.path.exists(test_conv_cache):
        print(f"Loading cached convolution embeddings from cache...")
        with open(train_conv_cache, 'rb') as f:
            train_conv_embeddings = pickle.load(f)
        with open(test_conv_cache, 'rb') as f:
            test_conv_embeddings = pickle.load(f)
        return train_conv_embeddings, test_conv_embeddings
    
    # 加载ProteInfer模型
    proteinfer_model = ProteInfer.from_pretrained(
        weights_path="/d/cuiby/paper_data/models/GO_model_weights13703706.pkl"
    ).cuda()
    proteinfer_model.eval()
    
    # 定义氨基酸字母表
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
    
    def process_sequences(sequences, desc):
        """处理序列批次"""
        conv_embeddings = []
        
        with torch.no_grad():
            for seq in tqdm(sequences, desc=desc):
                # 创建one-hot编码
                seq_length = len(seq)
                onehot = torch.zeros(seq_length, len(amino_acids))
                
                for i, aa in enumerate(seq):
                    if aa in aa_to_idx:
                        onehot[i, aa_to_idx[aa]] = 1
                
                # 转换为模型输入格式: [1, 20, seq_length]
                onehot_transposed = onehot.transpose(0, 1).unsqueeze(0).cuda()
                sequence_length = torch.tensor([seq_length]).cuda()
                
                # 通过ProteInfer获取嵌入
                embedding = proteinfer_model.get_embeddings(
                    onehot_transposed, 
                    sequence_length
                )
                
                # 将结果移回CPU并保存
                conv_embeddings.append(embedding.cpu())
                
                # 清理GPU内存
                del onehot_transposed, sequence_length, embedding
                if len(conv_embeddings) % 100 == 0:
                    torch.cuda.empty_cache()
        
        return conv_embeddings
    
    # 处理训练集和测试集
    train_conv_embeddings = process_sequences(training_sequences, "Processing training sequences")
    test_conv_embeddings = process_sequences(test_sequences, "Processing test sequences")
    
    # 保存缓存
    print(f"Saving convolution embeddings to cache...")
    with open(train_conv_cache, 'wb') as f:
        pickle.dump(train_conv_embeddings, f)
    with open(test_conv_cache, 'wb') as f:
        pickle.dump(test_conv_embeddings, f)
    
    print(f"Convolution embeddings saved successfully!")
    
    return train_conv_embeddings, test_conv_embeddings
####################################################蛋白质结构域部分####################################################
def load_domain_features(domain_file_path, protein_ids, cache_file=None):
    """
    加载蛋白质结构域信息并转换为one-hot向量
    
    Args:
        domain_file_path: 结构域文件路径
        protein_ids: 蛋白质ID列表
        cache_file: 缓存文件路径（可选）
    
    Returns:
        domain_features: one-hot编码的结构域特征矩阵 [num_proteins, num_domains]
        domain_encoder: MultiLabelBinarizer对象，用于后续处理
        domain_names: 所有结构域的名称列表
    """
    # 如果提供了缓存文件且存在，直接加载
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached domain features from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print(f"Domain feature matrix shape: {cached_data['domain_features'].shape}")
        print(f"Total unique domains: {len(cached_data['domain_names'])}")
        return cached_data['domain_features'], cached_data['domain_encoder'], cached_data['domain_names']
    
    print(f"Loading domain features from {domain_file_path}")
    
    # 读取结构域文件
    domain_dict = {}
    with open(domain_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                protein_id = parts[0]
                domains = parts[1].split(';')
                domain_dict[protein_id] = domains
            else:
                # 如果没有结构域信息，设为空列表
                protein_id = parts[0]
                domain_dict[protein_id] = []
    
    print(f"Total proteins with domain annotations: {len(domain_dict)}")
    
    # 为每个蛋白质ID准备结构域列表
    domain_lists = []
    for protein_id in protein_ids:
        if protein_id in domain_dict:
            domain_lists.append(domain_dict[protein_id])
        else:
            # 如果蛋白质没有结构域注释，使用空列表
            domain_lists.append([])
    
    # 使用MultiLabelBinarizer进行one-hot编码
    domain_encoder = MultiLabelBinarizer()
    domain_features = domain_encoder.fit_transform(domain_lists)
    
    domain_names = domain_encoder.classes_
    print(f"Total unique domains: {len(domain_names)}")
    print(f"Domain feature matrix shape: {domain_features.shape}")
    
    # 统计信息
    proteins_with_domains = np.sum(domain_features.sum(axis=1) > 0)
    avg_domains_per_protein = domain_features.sum() / len(protein_ids)
    print(f"Proteins with at least one domain: {proteins_with_domains}/{len(protein_ids)}")
    print(f"Average domains per protein: {avg_domains_per_protein:.2f}")
    
    # 如果提供了缓存文件路径，保存处理结果
    if cache_file:
        cache_dir = os.path.dirname(cache_file)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            print(f"创建缓存目录: {cache_dir}")
        
        print(f"Saving domain features to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'domain_features': domain_features,
                'domain_encoder': domain_encoder,
                'domain_names': domain_names,
                'protein_ids': protein_ids
            }, f)
    
    return domain_features, domain_encoder, domain_names


def load_domain_features_with_pretrained_encoder(domain_file_path, protein_ids, 
                                                  domain_encoder, cache_file=None):
    """
    使用预训练的encoder加载结构域特征（用于测试集）
    
    Args:
        domain_file_path: 结构域文件路径
        protein_ids: 蛋白质ID列表
        domain_encoder: 已经fit过的MultiLabelBinarizer对象
        cache_file: 缓存文件路径（可选）
    
    Returns:
        domain_features: one-hot编码的结构域特征矩阵
    """
    # 如果提供了缓存文件且存在，直接加载
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached domain features from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print(f"Domain feature matrix shape: {cached_data['domain_features'].shape}")
        return cached_data['domain_features']
    
    print(f"Loading domain features from {domain_file_path} with pretrained encoder")
    
    # 读取结构域文件
    domain_dict = {}
    with open(domain_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                protein_id = parts[0]
                domains = parts[1].split(';')
                domain_dict[protein_id] = domains
            else:
                protein_id = parts[0]
                domain_dict[protein_id] = []
    
    # 为每个蛋白质ID准备结构域列表
    domain_lists = []
    for protein_id in protein_ids:
        if protein_id in domain_dict:
            domain_lists.append(domain_dict[protein_id])
        else:
            domain_lists.append([])
    
    # 使用已有的encoder进行transform
    domain_features = domain_encoder.transform(domain_lists)
    
    print(f"Domain feature matrix shape: {domain_features.shape}")
    
    # 统计信息
    proteins_with_domains = np.sum(domain_features.sum(axis=1) > 0)
    avg_domains_per_protein = domain_features.sum() / len(protein_ids)
    print(f"Proteins with at least one domain: {proteins_with_domains}/{len(protein_ids)}")
    print(f"Average domains per protein: {avg_domains_per_protein:.2f}")
    
    # 保存缓存
    if cache_file:
        cache_dir = os.path.dirname(cache_file)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            print(f"创建缓存目录: {cache_dir}")
        
        print(f"Saving domain features to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'domain_features': domain_features,
                'protein_ids': protein_ids
            }, f)
    
    return domain_features

def load_domain_bag_features(domain_file_path, cache_dir,protein_ids, domain_to_idx=None,config=None):
    """
    加载蛋白质结构域特征并转换为稀疏矩阵，支持缓存
    
    参数:
        domain_file_path: 结构域文件路径
        protein_ids: 蛋白质ID列表 (train_id 或 test_id)
        domain_to_idx: 结构域到索引的映射字典 (如果为None则基于当前数据集构建)
        cache_dir: 缓存文件存储目录
    
    返回:
        稀疏矩阵形式的结构域特征和结构域映射字典
    """
    
    # 创建缓存目录
    os.makedirs(cache_dir, exist_ok=True)
    
    # 生成缓存文件名（基于蛋白质ID和是否训练集）
    is_training = domain_to_idx is None
    mode = "train" if is_training else "test"
    
    cache_features_path = os.path.join(cache_dir, f"domain_features_{mode}_{config['run_mode']}.npz")
    cache_mapping_path = os.path.join(cache_dir, f'domain_mapping_{mode}_{config["run_mode"]}.pkl')
    
    # 检查缓存是否存在
    if os.path.exists(cache_features_path):
        if is_training and os.path.exists(cache_mapping_path):
            print(f"Loading cached domain features from {cache_features_path}")
            print(f"Loading cached domain mapping from {cache_mapping_path}")
            
            sparse_features = load_npz(cache_features_path)
            with open(cache_mapping_path, 'rb') as f:
                domain_to_idx = pickle.load(f)
            
            print(f"Loaded cached data: {sparse_features.shape[0]} proteins, {sparse_features.shape[1]} domains")
            print(f"Non-zero elements: {sparse_features.nnz}")
            if sparse_features.shape[0] * sparse_features.shape[1] > 0:
                sparsity = 100 * (1 - sparse_features.nnz / (sparse_features.shape[0] * sparse_features.shape[1]))
                print(f"Sparsity: {sparsity:.2f}%")
            
            return sparse_features, domain_to_idx
        elif not is_training and os.path.exists(cache_features_path):
            print(f"Loading cached domain features from {cache_features_path}")
            
            sparse_features = load_npz(cache_features_path)
            
            print(f"Loaded cached data: {sparse_features.shape[0]} proteins, {sparse_features.shape[1]} domains")
            print(f"Non-zero elements: {sparse_features.nnz}")
            if sparse_features.shape[0] * sparse_features.shape[1] > 0:
                sparsity = 100 * (1 - sparse_features.nnz / (sparse_features.shape[0] * sparse_features.shape[1]))
                print(f"Sparsity: {sparsity:.2f}%")
            
            return sparse_features, domain_to_idx
    
    # 如果缓存不存在，处理原始数据
    print(f"Cache not found. Processing domain features from {domain_file_path}")
    
    # 读取结构域文件
    protein_domains = defaultdict(list)
    all_domains = set()
    
    with open(domain_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 格式: protein_id\tdomain1;domain2;domain3
            parts = line.split('\t')
            if len(parts) != 2:
                continue
                
            protein_id = parts[0].strip()
            
            # 提取结构域信息（分号分隔）
            domains = [d.strip() for d in parts[1].split(';') if d.strip()]
            
            if domains:
                protein_domains[protein_id] = domains
                # 只统计当前数据集中蛋白质的domain
                if protein_id in protein_ids:
                    all_domains.update(domains)
    
    # 如果没有提供domain_to_idx，则基于当前数据集构建（训练集）
    if is_training:
        domain_to_idx = {domain: idx for idx, domain in enumerate(sorted(all_domains))}
        num_domains = len(domain_to_idx)
        print(f"Building domain vocabulary from training set")
        print(f"Total unique domains in training set: {num_domains}")
    else:
        num_domains = len(domain_to_idx)
        print(f"Using existing domain vocabulary with {num_domains} domains")
    
    print(f"Proteins with domain info in file: {len(protein_domains)}")
    
    # 构建稀疏矩阵
    rows = []
    cols = []
    data = []
    
    proteins_with_domains = 0
    unknown_domains = set()  # 记录测试集中出现但训练集没有的domain
    
    for i, protein_id in enumerate(protein_ids):
        if protein_id in protein_domains:
            proteins_with_domains += 1
            domains = protein_domains[protein_id]
            for domain in domains:
                if domain in domain_to_idx:
                    rows.append(i)
                    cols.append(domain_to_idx[domain])
                    data.append(1)  # 二值特征，表示该结构域存在
                elif not is_training:
                    # 测试集中遇到训练集没见过的domain
                    unknown_domains.add(domain)
    
    if not is_training and unknown_domains:
        print(f"Warning: {len(unknown_domains)} domains in test set not seen in training set (will be ignored)")
    
    # 创建稀疏矩阵
    sparse_features = csr_matrix(
        (data, (rows, cols)), 
        shape=(len(protein_ids), num_domains),
        dtype=np.float32
    )
    
    print(f"Proteins in current dataset with domain info: {proteins_with_domains}/{len(protein_ids)}")
    print(f"Sparse matrix shape: {sparse_features.shape}")
    print(f"Non-zero elements: {sparse_features.nnz}")
    if sparse_features.shape[0] * sparse_features.shape[1] > 0:
        sparsity = 100 * (1 - sparse_features.nnz / (sparse_features.shape[0] * sparse_features.shape[1]))
        print(f"Sparsity: {sparsity:.2f}%")
    
    # 保存到缓存
    print(f"Saving domain features to {cache_features_path}")
    save_npz(cache_features_path, sparse_features)
    
    if is_training:
        print(f"Saving domain mapping to {cache_mapping_path}")
        with open(cache_mapping_path, 'wb') as f:
            pickle.dump(domain_to_idx, f)
    
    return sparse_features, domain_to_idx


def create_random_embeddings(real_embeddings):
    """
    用随机向量替换原始ESM嵌入
    """
    print("生成随机向量替换原始嵌入...")
    
    # 获取原始嵌入的维度信息
    if real_embeddings and isinstance(real_embeddings[0], torch.Tensor):
        original_dim = real_embeddings[0].shape[0]
        print(f"原始嵌入维度: {original_dim}, 样本数量: {len(real_embeddings)}")
    else:
        # 默认维度
        original_dim = 640
        print(f"使用默认维度: {original_dim}, 样本数量: {len(real_embeddings)}")
    
    # 生成随机向量（保持PyTorch Tensor格式）
    random_embeddings = []
    for i in range(len(real_embeddings)):
        # 生成随机向量，可以使用不同的分布
        random_vec = torch.randn(original_dim)  # 标准正态分布
        # 或者使用均匀分布: torch.rand(original_dim)
        # 或者使用特定范围: torch.FloatTensor(original_dim).uniform_(-1, 1)
        
        random_embeddings.append(random_vec)
    
    print(f"生成完成: {len(random_embeddings)}个随机向量，每个维度{original_dim}")
    return random_embeddings

def debug_list_structure(real_embeddings, sample_size=10):
    """
    详细分析列表的结构信息
    """
    print("=" * 60)
    print("LIST 结构分析报告")
    print("=" * 60)
    
    # 基本信息
    print(f"1. 列表基本信息:")
    print(f"   - 总长度: {len(real_embeddings)}")
    print(f"   - 类型: {type(real_embeddings)}")
    
    # 检查是否有None值
    none_count = sum(1 for item in real_embeddings if item is None)
    print(f"   - None值数量: {none_count}")
    
    # 检查前几个元素的类型和形状
    print(f"\n2. 前{sample_size}个元素分析:")
    for i, item in enumerate(real_embeddings[:sample_size]):
        print(f"   索引 {i}:")
        print(f"     - 类型: {type(item)}")
        
        if hasattr(item, 'shape'):
            print(f"     - 形状: {item.shape}")
            print(f"     - 数据类型: {item.dtype}")
        elif hasattr(item, '__len__'):
            print(f"     - 长度: {len(item)}")
            # 如果是嵌套结构，进一步分析
            if len(item) > 0:
                first_subitem = item[0] if hasattr(item, '__getitem__') else None
                print(f"     - 第一个子元素类型: {type(first_subitem)}")
        else:
            print(f"     - 值: {item}")
    
    # 分析所有元素的形状/长度分布
    print(f"\n3. 所有元素的形状/长度分布:")
    shapes = []
    for item in real_embeddings:
        if item is None:
            shapes.append('None')
        elif hasattr(item, 'shape'):
            shapes.append(f"array{item.shape}")
        elif hasattr(item, '__len__'):
            shapes.append(f"len{len(item)}")
        else:
            shapes.append(f"scalar({type(item).__name__})")
    
    shape_counts = Counter(shapes)
    for shape, count in shape_counts.most_common(10):  # 显示前10种最常见的形状
        print(f"   {shape}: {count}个 ({count/len(real_embeddings)*100:.1f}%)")
    
    # 检查是否可以转换为NumPy数组
    print(f"\n4. NumPy数组转换测试:")
    try:
        test_array = np.array(real_embeddings[:100])  # 先用前100个测试
        print(f"   - 前100个可以转换，结果形状: {test_array.shape}")
        print(f"   - 结果数据类型: {test_array.dtype}")
    except Exception as e:
        print(f"   - 转换失败: {e}")
    
    # 检查具体的不一致之处
    print(f"\n5. 形状不一致的详细分析:")
    if len(real_embeddings) > 1:
        first_item = real_embeddings[0]
        if hasattr(first_item, 'shape'):
            first_shape = first_item.shape
        elif hasattr(first_item, '__len__'):
            first_shape = len(first_item)
        else:
            first_shape = 'scalar'
        
        print(f"   - 第一个元素作为参考: {first_shape}")
        
        # 找出与第一个元素形状不同的索引
        different_indices = []
        for i, item in enumerate(real_embeddings[1:], 1):
            if hasattr(item, 'shape'):
                current_shape = item.shape
            elif hasattr(item, '__len__'):
                current_shape = len(item)
            else:
                current_shape = 'scalar'
            
            if current_shape != first_shape:
                different_indices.append((i, current_shape))
        
        if different_indices:
            print(f"   - 发现 {len(different_indices)} 个形状不同的元素:")
            for idx, shape in different_indices[:5]:  # 只显示前5个
                print(f"     索引 {idx}: {shape}")
            if len(different_indices) > 5:
                print(f"     ... 还有 {len(different_indices)-5} 个")
        else:
            print("   - 所有元素形状一致")

def load_pretrained_domain_features(train_ids, test_ids, pretrained_domain_path):
    """
    从预训练的结构域特征文件中加载训练集和测试集的结构域嵌入
    
    Args:
        train_ids: 训练集蛋白质ID列表
        test_ids: 测试集蛋白质ID列表  
        pretrained_domain_path: 预训练结构域特征文件路径
    
    Returns:
        train_domain_features: 训练集结构域特征张量 [num_train, 768]
        test_domain_features: 测试集结构域特征张量 [num_test, 768]
    """
    
    # 加载预训练的结构域特征字典
    print(f"Loading pretrained domain features from: {pretrained_domain_path}")
    
    with open(pretrained_domain_path, 'rb') as f:
        domain_features_dict = pickle.load(f)
    
    print(f"=== 预训练结构域特征字典信息 ===")
    print(f"字典类型: {type(domain_features_dict)}")
    print(f"包含的蛋白质数量: {len(domain_features_dict)}")
    
    # 检查第一个蛋白质ID和特征形状
    first_key = list(domain_features_dict.keys())[0]
    first_embedding = domain_features_dict[first_key]
    print(f"第一个蛋白质ID: {first_key}")
    print(f"对应的embedding类型: {type(first_embedding)}")
    print(f"embedding形状: {first_embedding.shape}")
    print(f"embedding数据类型: {first_embedding.dtype}")
    
    # 初始化训练集和测试集特征列表
    train_features = []
    test_features = []
    
    # 用于跟踪未找到的蛋白质ID
    train_missing = []
    test_missing = []
    
    # 处理训练集
    print(f"\nProcessing training set ({len(train_ids)} proteins)...")
    for protein_id in train_ids:
        if protein_id in domain_features_dict:
            train_features.append(domain_features_dict[protein_id])
        else:
            train_missing.append(protein_id)
            # 如果找不到，使用零向量作为占位符
            train_features.append(np.zeros(768, dtype=np.float32))
    
    # 处理测试集
    print(f"Processing test set ({len(test_ids)} proteins)...")
    for protein_id in test_ids:
        if protein_id in domain_features_dict:
            test_features.append(domain_features_dict[protein_id])
        else:
            test_missing.append(protein_id)
            # 如果找不到，使用零向量作为占位符
            test_features.append(np.zeros(768, dtype=np.float32))
    
    # 打印统计信息
    print(f"\n=== 加载结果统计 ===")
    print(f"训练集: {len(train_features)} 个特征, 其中 {len(train_missing)} 个未找到")
    print(f"测试集: {len(test_features)} 个特征, 其中 {len(test_missing)} 个未找到")
    
    if train_missing:
        print(f"训练集中未找到的蛋白质ID (前10个): {train_missing[:10]}")
    if test_missing:
        print(f"测试集中未找到的蛋白质ID (前10个): {test_missing[:10]}")
    
    # 转换为numpy数组
    train_features_array = np.array(train_features, dtype=np.float32)
    test_features_array = np.array(test_features, dtype=np.float32)
    
    print(f"\n=== 最终特征形状 ===")
    print(f"训练集特征形状: {train_features_array.shape}")
    print(f"测试集特征形状: {test_features_array.shape}")
    
    # 转换为PyTorch张量
    train_domain_features = torch.FloatTensor(train_features_array)
    test_domain_features = torch.FloatTensor(test_features_array)
    
    print(f"转换为PyTorch张量后的形状:")
    print(f"训练集: {train_domain_features.shape}")
    print(f"测试集: {test_domain_features.shape}")
    
    return train_domain_features, test_domain_features

def load_graph_pretrained_domain_features(
    train_id, 
    test_id,
    mapping_path='/e/cuiby/paper/struct_model/data/domain/domain_embeddings_mapping.pkl',
    embeddings_path='/e/cuiby/paper/struct_model/data/domain/domain_embeddings.npy',
    protein_domain_path='/e/cuiby/paper/pretrain/data/swissprot_domains.txt',
    aggregation='mean'  # 'mean', 'max', 'sum'
):
    """
    加载图预训练的结构域特征
    
    Args:
        train_id: 训练集蛋白质ID列表
        test_id: 测试集蛋白质ID列表
        mapping_path: domain映射文件路径
        embeddings_path: domain embeddings文件路径
        protein_domain_path: 蛋白质到结构域映射文件路径
        aggregation: 聚合多个结构域的方式 ('mean', 'max', 'sum')
    
    Returns:
        train_domain_features: 训练集结构域特征张量
        test_domain_features: 测试集结构域特征张量
    """
    
    print("Loading graph pretrained domain features...")
    
    # 1. 加载domain映射字典
    with open(mapping_path, 'rb') as f:
        mapping_dict = pickle.load(f)
    
    domain_to_idx = mapping_dict['domain_to_idx']
    idx_to_domain = mapping_dict['idx_to_domain']
    
    print(f"Number of domains: {len(domain_to_idx)}")
    
    # 2. 加载domain embeddings
    domain_embeddings = np.load(embeddings_path)
    print(f"Domain embeddings shape: {domain_embeddings.shape}")
    embedding_dim = domain_embeddings.shape[1]
    
    # 3. 加载蛋白质到结构域的映射
    protein_to_domains = {}
    with open(protein_domain_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                protein_id = parts[0]
                domains = parts[1].split(';')
                protein_to_domains[protein_id] = domains
    
    print(f"Number of proteins with domain annotations: {len(protein_to_domains)}")
    
    # 4. 定义聚合函数
    def aggregate_domain_embeddings(domains, method='mean'):
        """将多个结构域的embeddings聚合为一个向量"""
        valid_embeddings = []
        
        for domain in domains:
            if domain in domain_to_idx:
                idx = domain_to_idx[domain]
                valid_embeddings.append(domain_embeddings[idx])
        
        if len(valid_embeddings) == 0:
            # 没有有效的结构域，返回零向量
            return np.zeros(embedding_dim, dtype=np.float32)
        
        valid_embeddings = np.array(valid_embeddings)
        
        if method == 'mean':
            return np.mean(valid_embeddings, axis=0)
        elif method == 'max':
            return np.max(valid_embeddings, axis=0)
        elif method == 'sum':
            return np.sum(valid_embeddings, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    # 5. 为训练集构建特征
    train_domain_features = []
    train_missing = 0
    train_no_valid_domains = 0
    
    for protein_id in train_id:
        if protein_id in protein_to_domains:
            domains = protein_to_domains[protein_id]
            embedding = aggregate_domain_embeddings(domains, method=aggregation)
            train_domain_features.append(embedding)
            
            # 检查是否所有结构域都无效
            valid_domains = [d for d in domains if d in domain_to_idx]
            if len(valid_domains) == 0:
                train_no_valid_domains += 1
        else:
            # 蛋白质没有结构域注释
            train_domain_features.append(np.zeros(embedding_dim, dtype=np.float32))
            train_missing += 1
    
    # 6. 为测试集构建特征
    test_domain_features = []
    test_missing = 0
    test_no_valid_domains = 0
    
    for protein_id in test_id:
        if protein_id in protein_to_domains:
            domains = protein_to_domains[protein_id]
            embedding = aggregate_domain_embeddings(domains, method=aggregation)
            test_domain_features.append(embedding)
            
            # 检查是否所有结构域都无效
            valid_domains = [d for d in domains if d in domain_to_idx]
            if len(valid_domains) == 0:
                test_no_valid_domains += 1
        else:
            # 蛋白质没有结构域注释
            test_domain_features.append(np.zeros(embedding_dim, dtype=np.float32))
            test_missing += 1
    
    # 7. 转换为numpy数组和PyTorch张量
    train_domain_features = np.array(train_domain_features, dtype=np.float32)
    test_domain_features = np.array(test_domain_features, dtype=np.float32)
    
    train_domain_features = torch.FloatTensor(train_domain_features)
    test_domain_features = torch.FloatTensor(test_domain_features)
    
    # 8. 打印统计信息
    print(f"\n{'='*60}")
    print(f"Train domain features shape: {train_domain_features.shape}")
    print(f"Test domain features shape: {test_domain_features.shape}")
    print(f"Aggregation method: {aggregation}")
    print(f"\nTrain set statistics:")
    print(f"  - Proteins without domain annotations: {train_missing}/{len(train_id)} ({100*train_missing/len(train_id):.2f}%)")
    print(f"  - Proteins with no valid domains: {train_no_valid_domains}/{len(train_id)} ({100*train_no_valid_domains/len(train_id):.2f}%)")
    print(f"\nTest set statistics:")
    print(f"  - Proteins without domain annotations: {test_missing}/{len(test_id)} ({100*test_missing/len(test_id):.2f}%)")
    print(f"  - Proteins with no valid domains: {test_no_valid_domains}/{len(test_id)} ({100*test_no_valid_domains/len(test_id):.2f}%)")
    print(f"{'='*60}\n")
    
    return train_domain_features, test_domain_features

def load_text_pretrained_domain_features(
    train_id, 
    test_id,
    embeddings_path=None,
    protein_domain_path='/d/cuiby/paper/pretrain/data/swissprot_domains.txt',
    aggregation='mean'  # 'mean', 'max', 'sum'
):
    """
    加载文本预训练的结构域特征
    
    Args:
        train_id: 训练集蛋白质ID列表
        test_id: 测试集蛋白质ID列表
        embeddings_path: domain embeddings的pkl文件路径
        protein_domain_path: 蛋白质到结构域映射文件路径
        aggregation: 聚合多个结构域的方式 ('mean', 'max', 'sum')
    
    Returns:
        train_domain_features: 训练集结构域特征张量
        test_domain_features: 测试集结构域特征张量
    """
    
    print("Loading text pretrained domain features...")
    
    # 1. 加载domain embeddings (pkl格式)
    with open(embeddings_path, 'rb') as f:
        domain_embeddings_dict = pickle.load(f)
    
    print(f"Number of domains with embeddings: {len(domain_embeddings_dict)}")
    
    # 获取embedding维度
    sample_domain = list(domain_embeddings_dict.keys())[0]
    embedding_dim = domain_embeddings_dict[sample_domain]['embedding'].shape[0]
    print(f"Domain embedding dimension: {embedding_dim}")
    
    # 2. 加载蛋白质到结构域的映射
    protein_to_domains = {}
    with open(protein_domain_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                protein_id = parts[0]
                domains = parts[1].split(';')
                protein_to_domains[protein_id] = domains
    
    print(f"Number of proteins with domain annotations: {len(protein_to_domains)}")
    
    # 3. 定义聚合函数
    def aggregate_domain_embeddings(domains, method='mean'):
        """将多个结构域的embeddings聚合为一个向量"""
        valid_embeddings = []
        
        for domain in domains:
            if domain in domain_embeddings_dict:
                embedding = domain_embeddings_dict[domain]['embedding']
                valid_embeddings.append(embedding)
        
        if len(valid_embeddings) == 0:
            # 没有有效的结构域，返回零向量
            return np.zeros(embedding_dim, dtype=np.float32)
        
        valid_embeddings = np.array(valid_embeddings)
        
        if method == 'mean':
            return np.mean(valid_embeddings, axis=0)
        elif method == 'max':
            return np.max(valid_embeddings, axis=0)
        elif method == 'sum':
            return np.sum(valid_embeddings, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    # 4. 为训练集构建特征
    train_domain_features = []
    train_missing = 0
    train_no_valid_domains = 0
    
    for protein_id in train_id:
        if protein_id in protein_to_domains:
            domains = protein_to_domains[protein_id]
            embedding = aggregate_domain_embeddings(domains, method=aggregation)
            train_domain_features.append(embedding)
            
            # 检查是否所有结构域都无效
            valid_domains = [d for d in domains if d in domain_embeddings_dict]
            if len(valid_domains) == 0:
                train_no_valid_domains += 1
        else:
            # 蛋白质没有结构域注释
            train_domain_features.append(np.zeros(embedding_dim, dtype=np.float32))
            train_missing += 1
    
    # 5. 为测试集构建特征
    test_domain_features = []
    test_missing = 0
    test_no_valid_domains = 0
    
    for protein_id in test_id:
        if protein_id in protein_to_domains:
            domains = protein_to_domains[protein_id]
            embedding = aggregate_domain_embeddings(domains, method=aggregation)
            test_domain_features.append(embedding)
            
            # 检查是否所有结构域都无效
            valid_domains = [d for d in domains if d in domain_embeddings_dict]
            if len(valid_domains) == 0:
                test_no_valid_domains += 1
        else:
            # 蛋白质没有结构域注释
            test_domain_features.append(np.zeros(embedding_dim, dtype=np.float32))
            test_missing += 1
    
    # 6. 转换为numpy数组和PyTorch张量
    train_domain_features = np.array(train_domain_features, dtype=np.float32)
    test_domain_features = np.array(test_domain_features, dtype=np.float32)
    
    train_domain_features = torch.FloatTensor(train_domain_features)
    test_domain_features = torch.FloatTensor(test_domain_features)
    
    # 7. 打印统计信息
    print(f"\n{'='*60}")
    print(f"Train domain features shape: {train_domain_features.shape}")
    print(f"Test domain features shape: {test_domain_features.shape}")
    print(f"Aggregation method: {aggregation}")
    print(f"\nTrain set statistics:")
    print(f"  - Proteins without domain annotations: {train_missing}/{len(train_id)} ({100*train_missing/len(train_id):.2f}%)")
    print(f"  - Proteins with no valid domains: {train_no_valid_domains}/{len(train_id)} ({100*train_no_valid_domains/len(train_id):.2f}%)")
    print(f"\nTest set statistics:")
    print(f"  - Proteins without domain annotations: {test_missing}/{len(test_id)} ({100*test_missing/len(test_id):.2f}%)")
    print(f"  - Proteins with no valid domains: {test_no_valid_domains}/{len(test_id):.2f}%)")
    print(f"{'='*60}\n")
    
    return train_domain_features, test_domain_features