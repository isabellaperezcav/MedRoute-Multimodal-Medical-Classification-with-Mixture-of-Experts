
# MedRouter — Multimodal Medical Classification with Mixture of Experts  

This project implements a Mixture of Experts (MoE) architecture for multimodal medical image classification, where a router automatically assigns each input to the most suitable expert without using metadata.

The core contribution is an ablation study comparing four routing mechanisms:
- Vision Transformer + Linear (Deep Learning baseline)
- Gaussian Mixture Model (GMM)
- Naive Bayes
- k-NN with FAISS

All methods operate on the same embeddings extracted from a shared ViT backbone, enabling a fair comparison.

The system supports both 2D images and 3D medical volumes through an adaptive preprocessing pipeline and includes load balancing via auxiliary loss (Switch Transformer).

A dashboard is included for real-time inference, visualization, and analysis.

---

## Overview

This project implements a **Mixture of Experts (MoE)** system for **multimodal medical image classification**, capable of processing both **2D images and 3D volumes** without any metadata.

The system automatically learns **how to route each input** to the most appropriate expert using only pixel information.

---

## Key Contributions

- Adaptive preprocessing for **2D and 3D medical data**
- Shared **Vision Transformer backbone** for feature extraction
- **Ablation study** comparing 4 routing mechanisms:
  - ViT + Linear (Deep Learning)
  - Gaussian Mixture Model (GMM)
  - Naive Bayes
  - k-NN (FAISS)
- **Expert specialization** across different medical datasets
- **Load balancing** using Switch Transformer auxiliary loss
- Interactive **dashboard for inference and visualization**

---

## Ablation Study (Core Experiment)

All routing methods are evaluated on the same embeddings:

| Router | Type | Training | Notes |
|------|------|--------|------|
| ViT + Linear | Deep Learning | Gradient-based | Baseline |
| GMM | Statistical | EM | Probabilistic clustering |
| Naive Bayes | Statistical | Analytical | Fast & lightweight |
| k-NN (FAISS) | Non-parametric | None | Memory-based |

Objective:  
**Does a Vision Transformer justify its computational cost as a router?**

---

## Project Structure

```

Proyecto2/
│
├── ablation/              # Router comparison experiments
├── dashboard/            # Streamlit/Gradio app
├── datasets/             # Raw datasets
├── embedings/            # CLS tokens (ViT outputs)
├── expertos/             # Individual expert models
├── MOE/                  # Core MoE system
│   ├── backbone.py
│   ├── moe_model.py
│   ├── preprocess.py
│   ├── router_knn.py
│   └── inference.py
│
└── transformacion_datasets/

````

---

## System Architecture

1. Input: medical image or volume (no metadata)
2. Adaptive preprocessing (2D / 3D)
3. Shared ViT backbone → CLS token
4. Router decides expert
5. Selected expert performs classification

---

## Datasets

- NIH ChestX-ray14 (2D)
- ISIC 2019 (Dermatology)
- Osteoarthritis (X-ray)
- LUNA16 (CT 3D)
- Pancreatic Cancer (CT 3D)

---

## Requirements

Install dependencies:

```bash
pip install -r requerimients.txt
````

---

## Usage

### 1. Extract embeddings (CLS tokens)

Run notebooks in:

```
embedings/
```

---

### 2. Run ablation study

```
ablation/ablation_routers_v3_CORREGIDO.ipynb
```

---

### 3. Train experts

Each expert is located in:

```
expertos/
```

---

### 4. Run MoE system

```bash
python MOE/inference.py
```

---

### 5. Launch dashboard

```bash
cd dashboard
python app.py
```

---

## Metrics

* F1 Macro (2D & 3D datasets)
* Routing Accuracy
* Load Balance ratio
* OOD Detection (entropy)

---

## Load Balancing

Auxiliary loss based on Switch Transformer:

* Prevents expert collapse
* Ensures balanced routing

---

## Outputs

* Routing accuracy comparison
* Confusion matrices
* Embedding visualizations (UMAP)
* Training curves

---

## Hardware

* 2 × GPUs (12GB VRAM each)
* FP16 + Gradient Accumulation
* Gradient Checkpointing for 3D experts

---

## Notes

* No metadata allowed (only image input)
* Router fairness ensured via shared embeddings
* High-dimensional embeddings may affect k-NN performance

---

##  Authors

- Isabella Pérez Caviedes
- Luz A. Carabalí Mulato
- Martín García Chagueza
- Nicolás Zapata Obando
