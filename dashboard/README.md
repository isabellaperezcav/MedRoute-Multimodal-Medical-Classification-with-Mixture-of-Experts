# 🧬 MoE Medical Imaging Dashboard

Dashboard Streamlit para el proyecto final **Clasificación Médica con Mixture of Experts + Ablation Study del Router**.  
Curso: *Incorporar Elementos de IA — Unidad II (Bloque Visión)*

---

## 🚀 Instalación y ejecución rápida

```bash
# 1. Clona el repositorio del proyecto
cd <tu-repo>

# 2. Crea entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# 3. Instala dependencias
pip install -r requirements.txt

# 4. Corre el dashboard
streamlit run app.py
```

Abre el navegador en **http://localhost:8501**

---

## 📋 Funcionalidades obligatorias implementadas

| # | Funcionalidad | Estado |
|---|---------------|--------|
| 15 | Carga PNG / JPEG / NIfTI — detección 2D/3D automática | ✅ |
| 16 | Preprocesado transparente — dimensiones originales y adaptadas | ✅ |
| 17 | Inferencia en tiempo real — etiqueta, confianza, ms | ✅ |
| 18 | Attention Heatmap del Router ViT sobre imagen original | ✅ |
| 19 | Panel del experto activado — nombre, arq., dataset, gating score | ✅ |
| 20 | Panel ablation study — tabla comparativa 4 métodos | ✅ |
| 21 | Load Balance — gráfica f_i acumulado + alerta cociente | ✅ |
| 22 | OOD Detection — alerta por entropía del gating score | ✅ |

---

## 🔌 Cómo conectar tu modelo real

La función `mock_inference()` en `app.py` está marcada con comentarios `TODO`.  
Reemplázala con tu checkpoint entrenado:

```python
import torch
from your_moe_module import MoESystem   # tu clase

@st.cache_resource
def load_model():
    model = MoESystem(n_experts=5, d_model=192)
    ckpt = torch.load("checkpoints/best_moe.pth", map_location="cuda")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

def real_inference(img_array, router_choice):
    model = load_model()
    tensor = preprocess_to_tensor(img_array)   # tu pipeline
    with torch.no_grad(), torch.cuda.amp.autocast():
        gating_scores, expert_logits, attn_map = model(tensor, router=router_choice)
    # ... parsear y devolver dict igual que mock_inference()
```

### Attention Heatmap real (Grad-CAM / ViT attention rollout)

```python
# Opción A — ViT Attention Rollout (recomendado para el router ViT)
from timm.models.vision_transformer import VisionTransformer

def get_attention_map(model, tensor):
    attn_weights = []
    def hook(module, input, output):
        attn_weights.append(output[1])   # [B, heads, N, N]
    handles = [blk.attn.register_forward_hook(hook) for blk in model.router_vit.blocks]
    with torch.no_grad():
        model(tensor)
    for h in handles: h.remove()
    # Promediar cabezas y hacer rollout
    ...
    return attn_map  # [224, 224] numpy float32
```

---

## ⚙️ Estructura del proyecto

```
dashboard_moe/
├── app.py               ← Dashboard principal (este archivo)
├── requirements.txt     ← Dependencias
├── README.md
└── checkpoints/         ← Pon aquí tus .pth / .ckpt (crear carpeta)
    ├── best_moe.pth
    └── router_ablation/
        ├── gmm.pkl
        ├── naive_bayes.pkl
        └── faiss_index.bin
```

---

## 🎛️ Parámetros ajustables en el sidebar

| Parámetro | Descripción | Valor por defecto |
|-----------|-------------|-------------------|
| Router activo | Mecanismo de routing para inferencia | ViT + Linear |
| Umbral entropía OOD | H(g) > umbral → alerta OOD | 1.35 nats |
| Límite cociente max/min f_i | > límite → penalización | 1.30 |

---

## 📦 Para el clúster universitario (2× GPU 12 GB)

```bash
# Instala FAISS con soporte GPU
pip install faiss-gpu

# Ejecuta el dashboard en el servidor
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Accede desde tu máquina local con SSH tunnel
ssh -L 8501:localhost:8501 usuario@cluster-ip
```

---

## 📌 Seeds fijas (reproducibilidad)

```python
import torch, numpy as np, random
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
```

---

*Proyecto Final — Incorporar Elementos de IA, Unidad II · Semana 12*
