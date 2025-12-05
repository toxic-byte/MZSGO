# MZSGO: A Multimodal Zero-Shot Framework for Protein Function Prediction

Official implementation of **MZSGO** - a multimodal zero-shot framework for predicting protein Gene Ontology (GO) terms using ESM-2 embeddings, large language models, and domain features.

## Features

- 🚀 **Zero-shot prediction** on unseen GO terms
- 🔬 **Multimodal architecture** combining protein sequences, GO descriptions, and domain knowledge
- 📊 **Hierarchical reasoning** using GO ontology structure
- ⚡ **Fast inference** with pre-computed embeddings

## Installation

```bash
git clone https://github.com/toxic-byte/MZSGO.git
cd MZSGO
pip install -r requirements.txt
```

## Quick Start

### 1. Download Resources

- **Pre-trained models**: [Google Drive](https://drive.google.com/file/d/1fRZYmTlPFmb6rmMSS87HoqACVm5YsNNR/view?usp=drive_link) → `ckpt/cafa5/MZSGO/`
- **Embeddings cache**: [Google Drive](https://drive.google.com/drive/folders/1KAOMWGNiqVIhKJfaffhX5GP0B1ITsj4r) → `data/embeddings_cache/`
- **ESM-2 checkpoint**: [esm2_t33_650M_UR50D.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt)

### 2. Prediction

```bash
# Predict all GO terms
python predict.py --fasta example.fasta --pred_mode all --output predictions.json

# Predict specific ontology (BP/MF/CC)
python predict.py --fasta example.fasta --pred_mode mf --output predictions.json

# Custom zero-shot prediction
python predict.py --fasta example.fasta \
    --custom_go "protein kinase activity" \
    --custom_ontology mf --output predictions.json
```

**Optimized thresholds**: BP=0.29, MF=0.46, CC=0.49

### 3. Training

```bash
# Train on all ontologies
python main.py --run_mode full --onto all --epoch_num 50

# Train specific ontology
python main.py --run_mode full --onto bp --epoch_num 50
```

### 4. Evaluation

```bash
# Test with zero-shot analysis
python test.py --run_mode full
```

## Data Sources

- **Sequences**: [UniProt](https://www.uniprot.org/)
- **Annotations**: [GOA](https://www.ebi.ac.uk/GOA/)
- **Ontology**: [GO Resource](http://geneontology.org/)
- **Structures**: [AlphaFold DB](https://alphafold.com/)

## Key Arguments

### Training (`main.py`)
- `--onto`: Ontology (`all`/`bp`/`mf`/`cc`)
- `--epoch_num`: Training epochs
- `--batch_size_train`: Batch size (default: 16)
- `--learning_rate`: Learning rate (default: 5e-4)

### Prediction (`predict.py`)
- `--fasta`: Input FASTA file
- `--pred_mode`: Prediction mode (`all`/`bp`/`mf`/`cc`)
- `--custom_go`: Custom GO description for zero-shot
- `--threshold_bp/mf/cc`: Prediction thresholds

## Output Format

```json
{
  "protein_id": {
    "predictions": [
      {
        "go_term": "GO:0005515",
        "go_name": "protein binding",
        "probability": 0.95,
        "ontology": "molecular_function"
      }
    ]
  }
}
```

## Citation

```bibtex
@article{mzsgo2024,
  title={MZSGO: A Multimodal Zero-Shot Framework for Protein Function Prediction},
  author={Your Name},
  year={2024}
}
```

## License

MIT License

## Acknowledgments

Built with [ESM](https://github.com/facebookresearch/esm), [Gene Ontology](http://geneontology.org/), [UniProt](https://www.uniprot.org/), and [AlphaFold](https://alphafold.com/).

---

📦 **Models**: [Drive](https://drive.google.com/file/d/1fRZYmTlPFmb6rmMSS87HoqACVm5YsNNR/view?usp=drive_link) | 💾 **Cache**: [Drive](https://drive.google.com/drive/folders/1KAOMWGNiqVIhKJfaffhX5GP0B1ITsj4r) | 🔗 **Repo**: [GitHub](https://github.com/toxic-byte/MZSGO)