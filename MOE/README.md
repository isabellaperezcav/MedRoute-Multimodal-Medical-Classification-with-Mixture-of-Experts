# MoE — Mixture of Experts para Clasificación Médica Multimodal

Sistema MoE end-to-end que enruta automáticamente imágenes 2D y volúmenes 3D
al experto especializado correcto mediante un router k-NN con FAISS.

```
imagen/volumen
    │
    ▼
AdaptivePreprocessor    ← detecta 2D (rank=4) o 3D (rank=5)
    │                     2D: resize(224,224) + ImageNet norm
    │                     3D: HU clip + resize(64,64,64)
    ▼
SharedBackbone (ViT)    ← congelado, extrae CLS token (d=192)
    │
    ▼
KNNRouter (FAISS)       ← cosine similarity, k=5, voting
    │                     embedding → expert_id (0-4)
    ▼
Expert_k                ← arquitectura específica por dominio
    │
    ▼
Predicción final
```

---

## Estructura del proyecto

```
proyecto/
├── moe/                        ← este paquete
│   ├── __init__.py
│   ├── preprocess.py           ← preprocesador adaptativo 2D/3D
│   ├── backbone.py             ← ViT-Tiny compartido y congelado
│   ├── router_knn.py           ← router k-NN con FAISS
│   ├── experts.py              ← carga de los 5 expertos
│   ├── moe_model.py            ← clase MoEModel principal
│   ├── inference.py            ← script CLI
│   ├── utils.py                ← logging, formateo, JSON
│   └── requirements.txt
│
├── expertos/
│   ├── experto1_Xray/          ← ckpt_exp1v16_f2_best.pt
│   ├── experto2_osteo/         ← expert2_osteo_best.pth
│   ├── experto3_isic/          ← expert3_isic_best.pth
│   ├── experto4_luna/          ← LUNA-LIDCIDRI_best.pt
│   └── experto5_pancreas/      ← exp5_best.pth
│
└── embedings/
    └── Output embeddings/
        ├── all_train_Z.npy
        └── all_train_expert_y.npy
```

---

## Instalación

```bash
# 1. Clonar / ubicarse en el directorio raíz del proyecto
cd /ruta/al/proyecto

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r moe/requirements.txt

# 4. (Opcional) FAISS-GPU si tienes CUDA
# pip install faiss-gpu
```

### Verificar instalación

```python
import torch, timm, faiss, nibabel
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"FAISS: {faiss.__version__}")
```

---

## Uso rápido

### 1. Inferencia desde CLI

```bash
# Imagen 2D
python -m moe.inference --input imagen.png

# Volumen 3D NIfTI
python -m moe.inference --input scan.nii.gz

# Con GPU y FP16
python -m moe.inference --input scan.nii.gz --device cuda --fp16

# Batch (carpeta completa)
python -m moe.inference --input mis_imagenes/ --batch --output resultados.json

# Sin guardar JSON
python -m moe.inference --input imagen.png --no_save

# Desde raíz del proyecto con ruta explícita
python -m moe.inference --input datos/chest.png --base_dir /ruta/proyecto
```

### 2. Uso como librería Python

```python
from moe import MoEModel

# Inicializar (carga backbone, router y expertos)
model = MoEModel(
    base_dir = ".",          # raíz del proyecto
    device   = "cuda",       # o "cpu"
    use_fp16 = True,         # FP16 en GPU
    knn_k    = 5,
)

# Inferencia desde archivo
output, expert_id, router_info = model.predict_from_file("chest.png")

print(f"Experto: {expert_id} — {router_info['expert_name']}")
print(f"Probabilidades: {output}")

# Inferencia desde tensor
import torch
x = torch.randn(1, 3, 256, 256)     # imagen 2D cruda
output, expert_id, router_info = model.forward(x)

# Volumen 3D
x3d = torch.randn(1, 1, 128, 128, 128)
output, expert_id, router_info = model.forward(x3d)
```

### 3. Inferencia batch con logging

```python
from moe import MoEModel
from moe.utils import save_result_json, format_prediction

model = MoEModel(base_dir=".", device="cuda")

rutas = ["img1.png", "scan.nii.gz", "knee.jpg"]
resultados = []

for ruta in rutas:
    output, expert_id, router_info = model.predict_from_file(ruta)
    from moe.utils import format_prediction
    r = format_prediction(output, expert_id, router_info, elapsed_ms=0.0)
    r["input_file"] = str(ruta)
    resultados.append(r)

save_result_json(resultados, "mis_resultados.json")
```

---

## Expertos

| ID | Dataset | Arquitectura | Clases | Tarea |
|----|---------|-------------|--------|-------|
| 0 | NIH ChestX-ray14 | ConvNeXt-Tiny | 6 | Multilabel |
| 1 | ISIC 2019 | ConvNeXt-Small | 8 | Multiclase |
| 2 | Osteoartritis | EfficientNet-B0 | 5 | Multiclase (KL grade) |
| 3 | LUNA16 CT | DenseNet 3D | 2 | Binario |
| 4 | Pancreas CT | ResNet 3D | 2 | Binario |

Nota: los labels (0-4) corresponden a `EXPERT_MAP` del router entrenado en NB03
(`embeddings/-.py`). Los nombres de carpeta local (`experto2_osteo/`,
`experto3_isic/`) vienen del PDF original y no coinciden con el label (ver
`_EXPERT_CONFIG` en `experts.py`).

---

## Salida JSON

```json
{
  "timestamp": "2024-01-15T14:32:01",
  "expert_id": 0,
  "expert_name": "ChestX-ray14 (NIH)",
  "modality": "2D",
  "prediction": {
    "class_index": 2,
    "class_label": "Effusion",
    "confidence": 0.8731
  },
  "class_scores": {
    "Atelectasis": 0.0421,
    "Cardiomegaly": 0.0312,
    "Effusion": 0.8731,
    "Infiltration": 0.0201,
    "Mass": 0.0189,
    "Nodule": 0.0146
  },
  "router": {
    "k_neighbors": [0, 0, 0, 0, 0],
    "cosine_scores": [0.991, 0.988, 0.985, 0.983, 0.981],
    "vote_counts": {"0": 5},
    "router_confidence": 1.0
  },
  "inference_time_ms": 47.3,
  "input_file": "chest.png"
}
```

---

## Notas técnicas

### Router k-NN
- Usa `faiss.IndexFlatIP` con embeddings L2-normalizados (= cosine similarity exacta)
- k=5 vecinos + votación por mayoría
- **No usa metadatos ni labels de dataset en inferencia**; el routing depende únicamente del embedding CLS

### Backbone ViT
- `vit_tiny_patch16_224` de timm, pesos ImageNet, completamente congelado
- Para volúmenes 3D: slice-pooling sobre N slices centrales → promedio de CLS tokens

### Arquitecturas 3D
- `DenseNet3D` y `ResNet3D` están definidos en `experts.py` porque torchvision no incluye versiones 3D
- Se ajustan automáticamente al input `(B, 1, 64, 64, 64)`

### Carga de checkpoints heterogéneos
- `_load_checkpoint()` maneja state_dict puro, dicts con clave `model_state_dict`/`state_dict`, y modelos completos
- Usa `strict=False` para tolerar diferencias menores entre la arquitectura reconstruida y los pesos guardados

---

## Solución de problemas

**`FileNotFoundError: No se encontró checkpoint para experto X`**
→ Verifica que los archivos `.pt`/`.pth` están en las carpetas correctas.
   El sistema busca el nombre exacto primero; si no lo encuentra, usa cualquier `.pt`/`.pth` en la carpeta.

**`Keys faltantes al cargar estado`**
→ Normal con `strict=False`. Si la accuracy es la esperada, ignorar.
   Si hay problemas, verificar que la arquitectura en `experts.py` coincide con la del entrenamiento.

**`FAISS GPU no disponible`**
→ Instalar `faiss-gpu` en lugar de `faiss-cpu` y verificar CUDA toolkit.

**Memoria insuficiente en GPU**
→ Usar `--fp16` o reducir `n_slices_3d` en `MoEModel(n_slices_3d=4)`.
