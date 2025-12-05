# MZSGO: A Multimodal Zero-Shot Framework for Protein Function Prediction

Official implementation of **MZSGO** - a multimodal zero-shot framework for predicting protein Gene Ontology (GO) terms using ESM-2 embeddings, large language models, and domain features.

## Installation

```bash
git clone https://github.com/toxic-byte/MZSGO.git
cd MZSGO
pip install -r requirements.txt
```

## Quick Start

### 1. Download Resources
- **Embeddings cache**: [Google Drive](https://drive.google.com/drive/folders/1KAOMWGNiqVIhKJfaffhX5GP0B1ITsj4r) → `data/embeddings_cache/`

### 2. Prediction

```bash
# Predict specific ontology (BP/MF/CC)
python predict.py --fasta example.fasta --pred_mode mf 

#predict 指定的go
python predict.py --fasta example.fasta --go_terms go_terms.txt

# Custom zero-shot prediction
python predict.py --fasta example.fasta \
    --custom_go "protein kinase activity" \
    --custom_ontology mf 
```

### 3. Training

```bash
# Train specific ontology
python main.py --run_mode full --onto bp --epoch_num 30
```

### 4. Evaluation

```bash
#获得测试集结果
python test.py --run_mode full

# Test with zero-shot analysis
python test.py --run_mode zero
```
