import pickle
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

def load_text_pretrained_domain_features(
    train_id, 
    test_id,
    config,
    aggregation='mean'  # 'mean', 'max', 'sum'
):
   
    print("Loading text pretrained domain features...")
    
    if not os.path.exists(config['domain_text_path']):
        print(f"Domain embeddings file not found at {config['domain_text_path']}")
        print("Generating domain embeddings...")
        
        os.makedirs(os.path.dirname(config['domain_text_path']), exist_ok=True)
        
        extractor = DomainEmbeddingExtractor(
            model_name=config['nlp_name'],
            model_path=config.get('nlp_path', None),
            use_multi_gpu=True
        )
        
        domain_embeddings_dict = extractor.process_domain_file(
            input_file=config['entry_path'],
            output_file=config['domain_text_path'],
            batch_size=32
        )
        
        print(f"Domain embeddings generated and saved to {config['domain_text_path']}")
    else:
        with open(config['domain_text_path'], 'rb') as f:
            domain_embeddings_dict = pickle.load(f)
    
    print(f"Number of domains with embeddings: {len(domain_embeddings_dict)}")
    
    sample_domain = list(domain_embeddings_dict.keys())[0]
    embedding_dim = domain_embeddings_dict[sample_domain]['embedding'].shape[0]
    print(f"Domain embedding dimension: {embedding_dim}")
    
    protein_to_domains = {}
    with open(config['domain_file_path'], 'r') as f:
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
    print(f"  - Proteins with no valid domains: {test_no_valid_domains}/{len(test_id)} ({100*test_no_valid_domains/len(test_id):.2f}%)")
    print(f"{'='*60}\n")
    
    return train_domain_features, test_domain_features

class DomainDataset(Dataset):
    def __init__(self, domains_data):
        self.data = domains_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class DomainEmbeddingExtractor:
    def __init__(self, model_name, model_path=None, use_multi_gpu=True):
       
        self.use_multi_gpu = use_multi_gpu and torch.cuda.device_count() > 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_path and os.path.exists(model_path) and False:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        if self.use_multi_gpu:
            self.model = nn.DataParallel(self.model)
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def get_mean_pooling_embedding(self, last_hidden_state, attention_mask):
        
        attention_mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings
    
    def collate_fn(self, batch):
        texts = [item['text'] for item in batch]
        domain_ids = [item['domain_id'] for item in batch]
        
        encoded = self.tokenizer(
            texts,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )
        
        return {
            'domain_ids': domain_ids,
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def process_domain_file(self, input_file, output_file, batch_size=32):
        
        df = pd.read_csv(input_file, sep='\t', comment='#')
        
        domains_data = []
        for idx, row in df.iterrows():
            domain_id = row['ENTRY_AC']
            domain_type = row['ENTRY_TYPE']
            domain_name = row['ENTRY_NAME']
            text_description = f"{domain_type}: {domain_name}"
            
            domains_data.append({
                'domain_id': domain_id,
                'text': text_description
            })
        
        dataset = DomainDataset(domains_data)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4,  
            pin_memory=True
        )
        
        domain_embeddings = {}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing domains"):
                domain_ids = batch['domain_ids']
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state
                
                embeddings = self.get_mean_pooling_embedding(last_hidden_state, attention_mask)
                embeddings = embeddings.cpu().numpy()
                
                for i, domain_id in enumerate(domain_ids):
                    original_text = next(item['text'] for item in domains_data 
                                       if item['domain_id'] == domain_id)
                    
                    domain_embeddings[domain_id] = {
                        'description': original_text,
                        'embedding': embeddings[i]
                    }
        
        with open(output_file, 'wb') as f:
            pickle.dump(domain_embeddings, f)
        
        print(f"Saved {len(domain_embeddings)} domain embeddings to {output_file}")
        
        return domain_embeddings