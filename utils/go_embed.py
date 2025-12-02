import os
import pickle
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel
import re

MAXLEN = 2048

def load_nlp_model_path(nlp_path):
    nlp_tokenizer = AutoTokenizer.from_pretrained(nlp_path)
    nlp_model = AutoModel.from_pretrained(nlp_path)
    nlp_model.cuda()
    nlp_model.eval()
    return nlp_tokenizer, nlp_model

def load_nlp_model_name(model_name):
    nlp_tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp_model = AutoModel.from_pretrained(model_name)
    nlp_model.cuda()
    nlp_model.eval()
    return nlp_tokenizer, nlp_model

def list_embedding(nlp_model, nlp_tokenizer, nlp_dim, key, top_list, cache_path=None, onto=None, pooling='mean', name_flag="all"):
  
    if cache_path and os.path.exists(cache_path):
        print(f"Loading NLP embeddings from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        if (cached_data.get('pooling') == pooling and 
            cached_data.get('key') == key and
            cached_data.get('top_list') == top_list and
            cached_data.get('name_flag') == name_flag):
            print(f"Cache loaded successfully. Shape: {cached_data['embeddings'].shape}")
            return cached_data['embeddings']
        else:
            print("Warning: Cache mismatch. Regenerating embeddings...")
    
    print(f"Generating NLP embeddings for {len(top_list)} GO terms")
    print(f"Ontology namespace: {key}")
    print(f"Pooling method: {pooling}")
    print(f"Using GO {name_flag} as context")
    
    all_embeddings = []
    
    for _tag in tqdm(top_list, desc="Processing GO terms"):
        context = ''
        
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
                    name_part = term_info['name']
                    def_part = ''
                    tag_context = term_info['def']
                    tag_contents = re.findall(r'"(.*?)"', tag_context)
                    if tag_contents:
                        def_part = tag_contents[0]
                    
                    if name_part and def_part:
                        context = f"{name_part}: {def_part}"
                    elif name_part:
                        context = name_part
                    elif def_part:
                        context = def_part
                else:
                    raise ValueError(f"Unknown name_flag: {name_flag}. Must be 'name', 'def', or 'all'")
                break
        
        if context == '':
            print(f"Warning: No context found for {_tag}, using zero vector...")
            all_embeddings.append(torch.zeros(nlp_dim).cuda())
            continue

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
        
        embed = torch.cat(last_embed, dim=0)  # [total_seq_len, nlp_dim]
        
        if pooling == 'mean':
            pooled_embed = embed.mean(dim=0)  # [nlp_dim]
        elif pooling == 'max':
            pooled_embed = embed.max(dim=0)[0]  # [nlp_dim]
        elif pooling == 'cls':
            pooled_embed = embed[0]  # [nlp_dim]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        all_embeddings.append(pooled_embed)
    
    embeddings = torch.stack(all_embeddings)  # [len(top_list), nlp_dim]
    
    print(f"NLP embedding generation completed.")
    print(f"Embedding shape: {embeddings.shape}")
    
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
    
    with open(embedding_path, 'rb') as f:
        pretrained_embeddings = pickle.load(f)
    
    embedding_dim = next(iter(pretrained_embeddings.values())).shape[0]
    
    sample_key = next(iter(pretrained_embeddings.keys()))
    has_go_prefix = sample_key.startswith('GO:')
    
    print(f"Pretrained embeddings sample key: {sample_key}")
    print(f"Label list sample: {label_list[:5]}")
    
    go_embeddings = []
    missing_terms = []
    found_count = 0
    
    for go_term in label_list:
        if has_go_prefix and not go_term.startswith('GO:'):
            lookup_term = f"GO:{go_term}"
        elif not has_go_prefix and go_term.startswith('GO:'):
            lookup_term = go_term.replace('GO:', '')
        else:
            lookup_term = go_term
        
        if lookup_term in pretrained_embeddings:
            go_embeddings.append(pretrained_embeddings[lookup_term])
            found_count += 1
        else:
            missing_terms.append(go_term)
            go_embeddings.append(np.zeros(embedding_dim))
    
    print(f"\nGO Embeddings Loading Summary:")
    print(f"  Total GO terms: {len(label_list)}")
    print(f"  Found in pretrained: {found_count}")
    print(f"  Missing: {len(missing_terms)}")
    
    if missing_terms:
        print(f"  Missing terms (first 10): {missing_terms[:10]}")
    
    go_embeddings = torch.FloatTensor(np.array(go_embeddings)).to(device)
    
    return go_embeddings