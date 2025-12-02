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

model, tokenizer = esm.pretrained.esm2_t33_650M_UR50D()
num_layers = 33

model.cuda()
model.eval()

def precompute_esm_embeddings(sequences, cache_file, pooling='mean'):
    cache_dir = os.path.dirname(cache_file)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

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
            
            plm_embed = token_representations[0, 1:1 + len(seq), :].cpu()
            
            if pooling == 'mean':
                pooled_embed = plm_embed.mean(dim=0)  # [embed_dim]
            elif pooling == 'max':
                pooled_embed, _ = plm_embed.max(dim=0)  # [embed_dim]
            elif pooling == 'cls':
                pooled_embed = token_representations[0, 0, :].cpu()  # [embed_dim]
            else:
                raise ValueError(f"Unsupported pooling method: {pooling}")
            
            embeddings.append(pooled_embed)
    
    print(f"Saving ESM embeddings to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings

def compute_esm_embeddings(config, training_sequences, test_sequences):
    print("Computing ESM embeddings...")
    
    if config['run_mode'] == "sample":
        train_esm_cache = os.path.join(config['cache_dir'], f"esm/{config['esm_type']}/train_esm_embeddings_sample.pkl")
        test_esm_cache = os.path.join(config['cache_dir'], f"esm/{config['esm_type']}/test_esm_embeddings_sample.pkl")
    elif config['run_mode'] == "full":
        train_esm_cache = os.path.join(config['cache_dir'], f"esm/{config['esm_type']}/train_esm_embeddings_mean.pkl")
        test_esm_cache = os.path.join(config['cache_dir'], f"esm/{config['esm_type']}/test_esm_embeddings_mean.pkl")
    elif config['run_mode'] == "zero":
        train_esm_cache = None
        test_esm_cache = os.path.join(config['cache_dir'], f"esm/{config['esm_type']}/test_esm_embeddings_zero.pkl")
    train_esm_embeddings = precompute_esm_embeddings(training_sequences, train_esm_cache, pooling='mean')
    test_esm_embeddings = precompute_esm_embeddings(test_sequences, test_esm_cache, pooling='mean')
    
    return train_esm_embeddings, test_esm_embeddings

def load_text_pretrained_domain_features(
    train_id, 
    test_id,
    embeddings_path=None,
    protein_domain_path=None,
    aggregation='mean'  # 'mean', 'max', 'sum'
):
   
    print("Loading text pretrained domain features...")
    with open(embeddings_path, 'rb') as f:
        domain_embeddings_dict = pickle.load(f)
    
    print(f"Number of domains with embeddings: {len(domain_embeddings_dict)}")
    
    sample_domain = list(domain_embeddings_dict.keys())[0]
    embedding_dim = domain_embeddings_dict[sample_domain]['embedding'].shape[0]
    print(f"Domain embedding dimension: {embedding_dim}")
    
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
    
    def aggregate_domain_embeddings(domains, method='mean'):
        valid_embeddings = []
        
        for domain in domains:
            if domain in domain_embeddings_dict:
                embedding = domain_embeddings_dict[domain]['embedding']
                valid_embeddings.append(embedding)
        
        if len(valid_embeddings) == 0:
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
    
    train_domain_features = []
    train_missing = 0
    train_no_valid_domains = 0
    
    for protein_id in train_id:
        if protein_id in protein_to_domains:
            domains = protein_to_domains[protein_id]
            embedding = aggregate_domain_embeddings(domains, method=aggregation)
            train_domain_features.append(embedding)
            
            valid_domains = [d for d in domains if d in domain_embeddings_dict]
            if len(valid_domains) == 0:
                train_no_valid_domains += 1
        else:
            train_domain_features.append(np.zeros(embedding_dim, dtype=np.float32))
            train_missing += 1
    
    test_domain_features = []
    test_missing = 0
    test_no_valid_domains = 0
    
    for protein_id in test_id:
        if protein_id in protein_to_domains:
            domains = protein_to_domains[protein_id]
            embedding = aggregate_domain_embeddings(domains, method=aggregation)
            test_domain_features.append(embedding)
            
            valid_domains = [d for d in domains if d in domain_embeddings_dict]
            if len(valid_domains) == 0:
                test_no_valid_domains += 1
        else:
            test_domain_features.append(np.zeros(embedding_dim, dtype=np.float32))
            test_missing += 1
    
    train_domain_features = np.array(train_domain_features, dtype=np.float32)
    test_domain_features = np.array(test_domain_features, dtype=np.float32)
    
    train_domain_features = torch.FloatTensor(train_domain_features)
    test_domain_features = torch.FloatTensor(test_domain_features)
    
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