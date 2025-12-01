from datetime import datetime
from re import S
from sklearn import preprocessing
import sys
import torch
import numpy as np

sys.path.append(r"utils")
sys.path.append(r"models")
sys.path.append(r"trainer")
from dataset import obo_graph,load_datasets,process_labels_for_ontology,create_dataloaders,compute_pos_weight,create_ontology_adjacency_matrix
from config import setup_environment, get_config
from nlp_embed import load_nlp_model,compute_nlp_embeddings_list,load_pretrained_go_embeddings
from embed import compute_esm_embeddings,load_domain_features,load_text_pretrained_domain_features,load_domain_features_with_pretrained_encoder
from trainer_domain_text import train_model_for_ontology
# from trainer_domain_text_inter import train_model_for_ontology
from util import save_results

def filter_samples_with_labels(training_labels_binary, test_labels_binary, 
                               training_sequences, test_sequences,
                               train_esm_embeddings, test_esm_embeddings,
                               train_nlp, test_nlp,
                               train_domain_features, test_domain_features,
                               train_id, test_id):
    """
    过滤出具有标签的样本
    
    Args:
        training_labels_binary: 训练集标签 (N_train, N_labels)
        test_labels_binary: 测试集标签 (N_test, N_labels)
        ... 其他特征数据
    
    Returns:
        filtered_data: 只包含有标签样本的数据字典
    """
    training_labels_binary = np.array(training_labels_binary)
    test_labels_binary = np.array(test_labels_binary)
    
    # 找出至少有一个标签的训练样本
    train_has_label = training_labels_binary.sum(axis=1) > 0
    train_indices = np.where(train_has_label)[0]
    
    # 找出至少有一个标签的测试样本
    test_has_label = test_labels_binary.sum(axis=1) > 0
    test_indices = np.where(test_has_label)[0]
    
    print(f"\n  Training samples: {len(training_sequences)} -> {len(train_indices)} (with labels)")
    print(f"  Test samples: {len(test_sequences)} -> {len(test_indices)} (with labels)")
    
    # 如果没有训练样本有标签，返回 None
    if len(train_indices) == 0:
        print(f"  WARNING: No training samples with labels for this ontology!")
        return None
    
    # 过滤训练数据
    filtered_train_sequences = [training_sequences[i] for i in train_indices]
    filtered_train_labels = training_labels_binary[train_indices]
    filtered_train_id = [train_id[i] for i in train_indices]
    
    # 处理 ESM embeddings（可能是列表或numpy数组）
    filtered_train_esm = None
    if train_esm_embeddings is not None:
        if isinstance(train_esm_embeddings, list):
            filtered_train_esm = [train_esm_embeddings[i] for i in train_indices]
        else:
            filtered_train_esm = train_esm_embeddings[train_indices]
    
    # 处理 NLP embeddings（可能是列表、numpy数组或tensor）
    filtered_train_nlp = None
    if train_nlp is not None:
        if isinstance(train_nlp, list):
            filtered_train_nlp = [train_nlp[i] for i in train_indices]
        else:
            filtered_train_nlp = train_nlp[train_indices]
    
    # 处理 Domain features（可能是列表、numpy数组或tensor）
    filtered_train_domain = None
    if train_domain_features is not None:
        if isinstance(train_domain_features, list):
            filtered_train_domain = [train_domain_features[i] for i in train_indices]
        else:
            filtered_train_domain = train_domain_features[train_indices]
    
    # 过滤测试数据
    filtered_test_sequences = [test_sequences[i] for i in test_indices]
    filtered_test_labels = test_labels_binary[test_indices]
    filtered_test_id = [test_id[i] for i in test_indices]
    
    # 处理测试集 ESM embeddings
    filtered_test_esm = None
    if test_esm_embeddings is not None:
        if isinstance(test_esm_embeddings, list):
            filtered_test_esm = [test_esm_embeddings[i] for i in test_indices]
        else:
            filtered_test_esm = test_esm_embeddings[test_indices]
    
    # 处理测试集 NLP embeddings
    filtered_test_nlp = None
    if test_nlp is not None:
        if isinstance(test_nlp, list):
            filtered_test_nlp = [test_nlp[i] for i in test_indices]
        else:
            filtered_test_nlp = test_nlp[test_indices]
    
    # 处理测试集 Domain features
    filtered_test_domain = None
    if test_domain_features is not None:
        if isinstance(test_domain_features, list):
            filtered_test_domain = [test_domain_features[i] for i in test_indices]
        else:
            filtered_test_domain = test_domain_features[test_indices]
    
    return {
        'train': {
            'sequences': filtered_train_sequences,
            'labels': filtered_train_labels,
            'ids': filtered_train_id,
            'esm_embeddings': filtered_train_esm,
            'nlp': filtered_train_nlp,
            'domain_features': filtered_train_domain
        },
        'test': {
            'sequences': filtered_test_sequences,
            'labels': filtered_test_labels,
            'ids': filtered_test_id,
            'esm_embeddings': filtered_test_esm,
            'nlp': filtered_test_nlp,
            'domain_features': filtered_test_domain
        }
    }


def main():
    seed = setup_environment()

    config = get_config(run_mode="full", text_mode="all",occ_num=0,batch_size_train=64,
    batch_size_test=64,nlp_model_type="qwen_4b",epoch_num=50,hidden_dim=512,learning_rate=1e-4,
    model='domain_text_gated',dropout=0.5,esm_type="esm2_t33_650M_UR50D",embed_dim=1280)

    # config = get_config(run_mode="sample", text_mode="all",occ_num=0,batch_size_train=64,
    # batch_size_test=64,nlp_model_type="qwen_4b",epoch_num=1,model='domain_text_gated')

    ctime = datetime.now().strftime("%Y%m%d%H%M%S")
    print('Start running date:{}'.format(ctime))
    
    nlp_tokenizer, nlp_model = None,None
    # nlp_tokenizer, nlp_model = load_nlp_model(config['nlp_path'])
    
    label_space = {
        'biological_process': [],
        'molecular_function': [],
        'cellular_component': []
    }
    enc = preprocessing.LabelEncoder()
    
    # 加载本体和IA字典
    onto, ia_dict = obo_graph(config['obo_path'], config['ia_path'])
    
    train_id, training_sequences, training_labels, test_id, test_sequences, test_labels = load_datasets(
        config, onto, label_space)
    
    train_esm_embeddings, test_esm_embeddings = compute_esm_embeddings(
        config, training_sequences, test_sequences)
    
    #     # 为训练集加载结构域特征（会fit encoder）
    # train_domain_features, domain_encoder, domain_names = load_domain_features(
    #     config['domain_file_path'], train_id, cache_file=config['train_domain_cache'])
    
    # # 为测试集加载结构域特征（使用训练集的encoder）
    # test_domain_features = load_domain_features_with_pretrained_encoder(
    #     config['domain_file_path'], test_id, domain_encoder, cache_file=config['test_domain_cache'])
    print(config['domain_text_path'])
    train_domain_features, test_domain_features = load_text_pretrained_domain_features(train_id, test_id,config['domain_text_path'])
    
    metrics_output_test = {}

    for key in label_space.keys():
        print(f"\n{'='*80}")
        print(f"Processing ontology: {key}")
        print(f"{'='*80}")
        if key=="biological_process":
            continue
        
        label_list, training_labels_binary, test_labels_binary, enc, ia_list, onto_parent, label_num = process_labels_for_ontology(
            config, key, label_space, training_labels, test_labels, onto, enc, ia_dict)
        
        # 过滤出有标签的样本
        filtered_data = filter_samples_with_labels(
            training_labels_binary, test_labels_binary,
            training_sequences, test_sequences,
            train_esm_embeddings, test_esm_embeddings,
            None, None,  # train_nlp, test_nlp
            train_domain_features, test_domain_features,
            train_id, test_id
        )
        
        # 如果该本体没有训练样本，跳过
        if filtered_data is None:
            print(f"  Skipping {key} - no training samples with labels")
            continue
        
        adj_matrix = create_ontology_adjacency_matrix(onto_parent, label_num, key, config)

        pos_weight = compute_pos_weight(filtered_data['train']['labels']).cuda()
        
        list_nlp = compute_nlp_embeddings_list(
            config, nlp_model, nlp_tokenizer, key, label_list, onto).cuda()
        
        #  #po2vec的预训练向量
        # pretrained_embedding_path = r"/e/cuiby/paper/struct_model/data/embeddings_cafa3.pkl"
        # list_nlp = load_pretrained_go_embeddings(
        #     pretrained_embedding_path,label_list
        # ).cuda()
        # config['nlp_dim']=256

        # 使用过滤后的数据创建DataLoader
        train_dataloader, test_dataloader = create_dataloaders(
            config, 
            filtered_data['train']['sequences'], 
            filtered_data['train']['labels'], 
            filtered_data['train']['esm_embeddings'], 
            filtered_data['train']['nlp'],
            filtered_data['test']['sequences'], 
            filtered_data['test']['labels'], 
            filtered_data['test']['esm_embeddings'], 
            filtered_data['test']['nlp'],
            filtered_data['train']['domain_features'], 
            filtered_data['test']['domain_features']
        )
        
        model = train_model_for_ontology(
            config, key, train_dataloader, test_dataloader, list_nlp, ia_list, ctime,
            metrics_output_test, filtered_data['train']['domain_features'], adj_matrix, pos_weight,
            training_labels_binary=filtered_data['train']['labels'],  # 使用过滤后的训练标签
            test_labels_binary=filtered_data['test']['labels'],       # 使用过滤后的测试标签
            label_list=label_list)
    
    save_results(config, metrics_output_test, seed, ctime)
    print('End running date:{}'.format(datetime.now().strftime("%Y%m%d%H%M%S")))

if __name__ == "__main__":
    main()