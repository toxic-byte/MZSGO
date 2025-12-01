import os
import os
import torch
import random
import numpy as np

def setup_environment():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["WANDB_DISABLED"] = "true"
    
    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed

def get_config(run_mode="full", text_mode="all",nlp_model_type="biopubmed",occ_num=0,
batch_size_train=128,batch_size_test=128,learning_rate=5e-4,epoch_num=100,patience=10,
hidden_dim=512,model="domain_pre",dropout=0.3,esm_type="esm2_t33_650M_UR50D",
embed_dim=1280,nlp_dim=2560,loss='bce'):
    config = {
        'run_mode': run_mode,
        'text_mode': text_mode,
        'nlp_model_type': nlp_model_type,
        'occ_num': occ_num,
        'nlp_path': '/e/cuiby/huggingface/hub/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract/snapshots/d673b8835373c6fa116d6d8006b33d48734e305d',
        'nlp_dim': nlp_dim,
        'embed_dim': embed_dim,
        'MAXLEN': 2048,
        'cache_dir': '/d/cuiby/paper_data/embeddings_cache',
        'output_path': 'eval/binary_model',
        'obo_path': '/d/cuiby/paper_data/Original/go-basic.obo',
        'ia_path': '/d/cuiby/paper_data/Original/IA.txt',
        'batch_size_train':batch_size_train,
        'batch_size_test': batch_size_test,
        'learning_rate': learning_rate,
        'epoch_num': epoch_num,
        'patience': patience,
        'step_size': 7,
        'gamma': 0.6,
        'projection_dim': 256,
        'dropout': dropout,
        'alpha': 0.5,  # 对比学习权重
        'temperature': 0.07,
        'hidden_dim':hidden_dim,
        'contrastive_type': 'supcon',
        'pretrain':False,
        'freeze_epochs':10,
        'projector_lr_ratio': 0.1,
        'warmup_epochs': 10,    
        'min_lr': 1e-6, 
        'min_delta': 0.0001,
        'warmup_start_lr': 1e-6,   
        'weight_decay': 1e-5, 
        'warmup_ratio': 0.1,     # warmup占总步数的比例（10%）
        'num_cycles': 0.5,       # 余弦周期数（0.5表示半个周期）
        'domain_file_path':'/d/cuiby/paper_data/domain/swissprot_domains.txt',
        'struct_dir': '/e/cuiby/data/swiss_prot_pdb', 
        'hierarchy_weight': 1,          # 层级损失权重（建议从0.05-0.2尝试）
        'hierarchy_loss_version': 'v3',   # 'v1' 或 'v2'
        'hierarchy_margin': 0.0,          # 容忍度（允许子节点比父节点高出多少）
        'hierarchy_use_log': True,        # V2版本是否使用对数空间（推荐True）
        'esm_type': esm_type,
        'cache_domain_bag': '/d/cuiby/paper_data/domain_bag',
        'graph_cache_dir': '/e/cuiby/paper/graph_data',
         'model':model,
         'loss':loss,
    }
    
    if run_mode == "sample":
        config['train_path'] = "/d/cuiby/paper_data/sequence/cafa5_train_with_struct_2.txt"
        config['test_path'] = "/d/cuiby/paper_data/sequence/cafa5_test_with_struct_2.txt"
        config['train_domain_cache'] = os.path.join(config['cache_dir'], "/d/cuiby/paper_data/domain/train_domain_features_sample.pkl")
        config['test_domain_cache'] = os.path.join(config['cache_dir'], "/d/cuiby/paper_data/domain/test_domain_features_sample.pkl")
    elif run_mode == "full":
        config['train_path'] = "/d/cuiby/paper_data/sequence/cafa5_train_with_struct.txt"
        config['test_path'] = "/d/cuiby/paper_data/sequence/cafa5_test_with_struct.txt"
        config['train_domain_cache'] = os.path.join(config['cache_dir'], "/d/cuiby/paper_data/domain/train_domain_features_full.pkl")
        config['test_domain_cache'] = os.path.join(config['cache_dir'], "/d/cuiby/paper_data/domain/test_domain_features_full.pkl")
    elif run_mode == "zero":
        config['train_path'] = "/d/cuiby/paper_data/sequence/cafa5_train_with_struct_2.txt"
        config['test_path'] = "/d/cuiby/paper_data/zero_shot/zero_shot_below30.txt"
        config['obo_path']="/d/cuiby/paper_data/Original/go-basic_2025.obo"

    if nlp_model_type == "biopubmed":
        config['nlp_path'] = '/e/cuiby/huggingface/hub/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract/snapshots/d673b8835373c6fa116d6d8006b33d48734e305d'
        config['nlp_dim'] = 768
        config['domain_text_path']='/d/cuiby/paper/pretrain/data/domain_embeddings_biopubmed.pkl'
    elif nlp_model_type == "qwen_06b":
        config['nlp_path'] = '/d/cuiby/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418'
        config['nlp_dim'] = 1024
        config['domain_text_path']='/d/cuiby/paper/pretrain/data/domain_embeddings_qwen06b.pkl'
    elif nlp_model_type == "qwen_4b":
        config['nlp_path'] = '/d/cuiby/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b'
        config['nlp_dim'] = 2560
        config['domain_text_path']='/d/cuiby/paper/pretrain/data/domain_embeddings.pkl'
    elif nlp_model_type == "biogpt":
        config['nlp_path'] = '/d/cuiby/.cache/huggingface/hub/models--microsoft--biogpt/snapshots/eb0d815e95434dc9e3b78f464e52b899bee7d923'
        config['nlp_dim'] = 1024
        config['domain_text_path']='/d/cuiby/paper/pretrain/data/domain_embeddings_biogpt.pkl'
    os.makedirs(config['cache_dir'], exist_ok=True)
    return config

    