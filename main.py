from datetime import datetime
from re import S
from sklearn import preprocessing
import sys
import torch
import numpy as np

sys.path.append(r"utils")
from dataset import obo_graph,load_datasets,process_labels_for_ontology,create_dataloaders,compute_pos_weight,create_ontology_adjacency_matrix
from config import setup_environment, get_config
from nlp_embed import load_nlp_model,compute_nlp_embeddings_list
from embed import compute_esm_embeddings,load_text_pretrained_domain_features
from trainer import train_model_for_ontology
from util import filter_samples_with_labels,save_results

def main():
    seed = setup_environment()
    # config = get_config(run_mode="full", text_mode="all",occ_num=0,batch_size_train=64,
    # batch_size_test=64,nlp_model_type="qwen_4b",epoch_num=50,hidden_dim=512,learning_rate=1e-4,
    # model='domain_text_gated',dropout=0.5,esm_type="esm2_t33_650M_UR50D",embed_dim=1280)

    config = get_config(run_mode="sample", text_mode="all",occ_num=0,batch_size_train=64,
    batch_size_test=64,nlp_model_type="qwen_4b",epoch_num=1,model='domain_text_gated')

    ctime = datetime.now().strftime("%Y%m%d%H%M%S")
    print('Start running date:{}'.format(ctime))
    
    nlp_tokenizer, nlp_model = load_nlp_model(config['nlp_path'])
    
    label_space = {
        'biological_process': [],
        'molecular_function': [],
        'cellular_component': []
    }
    enc = preprocessing.LabelEncoder()
    
    onto, ia_dict = obo_graph(config['obo_path'], config['ia_path'])
    
    train_id, training_sequences, training_labels, test_id, test_sequences, test_labels = load_datasets(
        config, onto, label_space)
    
    train_esm_embeddings, test_esm_embeddings = compute_esm_embeddings(
        config, training_sequences, test_sequences)
    
    train_domain_features, test_domain_features = load_text_pretrained_domain_features(train_id, test_id,config['domain_text_path'])
    metrics_output_test = {}

    for key in label_space.keys():
        print(f"\n{'='*80}")
        print(f"Processing ontology: {key}")

        label_list, training_labels_binary, test_labels_binary, enc, ia_list, onto_parent, label_num = process_labels_for_ontology(
            config, key, label_space, training_labels, test_labels, onto, enc, ia_dict)
        
        filtered_data = filter_samples_with_labels(
            training_labels_binary, test_labels_binary,
            training_sequences, test_sequences,
            train_esm_embeddings, test_esm_embeddings,
            train_domain_features, test_domain_features,
            train_id, test_id
        )
        
        if filtered_data is None:
            print(f"  Skipping {key} - no training samples with labels")
            continue
        
        adj_matrix = create_ontology_adjacency_matrix(onto_parent, label_num, key, config)

        pos_weight = compute_pos_weight(filtered_data['train']['labels']).cuda()
        
        list_nlp = compute_nlp_embeddings_list(
            config, nlp_model, nlp_tokenizer, key, label_list, onto).cuda()
        
        train_dataloader, test_dataloader = create_dataloaders(
            config, 
            filtered_data['train']['sequences'], 
            filtered_data['train']['labels'], 
            filtered_data['train']['esm_embeddings'], 
            filtered_data['test']['sequences'], 
            filtered_data['test']['labels'], 
            filtered_data['test']['esm_embeddings'], 
            filtered_data['train']['domain_features'], 
            filtered_data['test']['domain_features']
        )
        
        model = train_model_for_ontology(
            config, key, train_dataloader, test_dataloader, list_nlp, ia_list, ctime,
            metrics_output_test, filtered_data['train']['domain_features'], adj_matrix, pos_weight,
            training_labels_binary=filtered_data['train']['labels'], 
            test_labels_binary=filtered_data['test']['labels'],       
            label_list=label_list)
    
    save_results(config, metrics_output_test, seed, ctime)
    print('End running date:{}'.format(datetime.now().strftime("%Y%m%d%H%M%S")))

if __name__ == "__main__":
    main()