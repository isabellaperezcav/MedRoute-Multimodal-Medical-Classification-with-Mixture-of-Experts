# Cheatsheet de exposición · MedRoute

> Referencia rápida para responder preguntas del profesor. Organizado por tema, no por archivo. Cuando pregunten "¿dónde está X?" o "¿cómo hicieron Y?", busca en este doc y te pasas la respuesta en 15 segundos.

---

## 1. Pipeline completo en 5 líneas

```
imagen cruda (PNG/JPG/NIfTI/NPY)
  → AdaptivePreprocessor  (detecta 2D vs 3D por rank del tensor)
  → ViT-Tiny congelado    (extrae CLS token de 192 dimensiones)
  → kNN Router con FAISS  (IndexFlatIP, k=5, coseno con L2-norm)
  → Expert_k              (CXR14 / ISIC / Osteo / LUNA / Pancreas)
  → predicción + routing_probs + is_ood + timing_ms
```

Código que implementa esto: `MOE-2/moe_model.py` (clase `MoEModel`).

---

## 2. EDA — 6 notebooks (carpeta `notebooks/eda/`)

| # | Notebook | Propósito | Autor razonable |
|---|----------|-----------|-----------------|
| 00 | `00_cls_token_separability.ipynb` | **Prueba de concepto**: separabilidad lineal del CLS entre los 5 dominios médicos. Justifica que un router simple es suficiente. | Luz |
| 01 | `01_chestxray14.ipynb` | Exploración de CXR14: 112.120 imágenes, 14 patologías, prevalencia por clase, contaminación paciente 67,4 % en splits oficiales. | Martín |
| 02 | `02_isic2019.ipynb` | ISIC 2019: 8 clases, desbalance NV=12.875 vs DF=239, distribución de edades y áreas anatómicas. | Isabella |
| 03 | `03_osteoarthritis.ipynb` | OA de rodilla: 5 grados KL, distribución 1315/1266/765/742/678, dedup por fingerprint. | Isabella |
| 04 | `04_luna16.ipynb` | LUNA16: 888 CTs, 1.186 nódulos, ratio desbalance 10,7:1. | Nicolás |
| 05 | `05_msd_pancreas.ipynb` | PANORAMA: 1.703 CTs abdominales, 363 positivos PDAC, distribución HU. | Nicolás |

> Si preguntan **"¿por qué este recorte a 6 clases en CXR14?"** → EDA 01 + archivo `eda_previo_v21.ipynb` muestran que las 8 clases excluidas tienen prevalencia < 3 % y arrastran el F1 Macro a techo 0,37.

---

## 3. Transformaciones offline (carpeta `docs/transformaciones/`)

Referencia completa: `docs/transformaciones/datasets_transformaciones_inventario.md` (392 líneas).

### Resumen por dataset

**CXR14 (`scripts → pre_chestxray14.py`)**
- `cv2.imread(grayscale)` → validación `MIN_DIM=800` → resize 256×256 con `INTER_LINEAR` → **CLAHE** con `clipLimit=2.0`, `tileGridSize=(8,8)` → normalización `float32 [0,1]` → guarda `.npy`.
- Label: vector binario de 14 dims parseado de `Data_Entry_2017.csv` (separador `|`).

**ISIC 2019 (`pre_isic.py`)**
- **DullRazor hair removal**: grayscale → morph close 3×3 → `absdiff` → threshold=10 → dilate 1 iter → `cv2.inpaint(Telea, radius=3)`.
- Resize **lado corto = 224** con LANCZOS4.
- Guardado JPEG q95.
- *Shades of Gray* se aplica **online** con p=0,5 durante training, no offline.

**LUNA16 (`pre_embeddings.py` sección LUNA)**
- `SimpleITK.ReadImage(.mhd)` → resample isotrópico **1 mm** (`scipy.zoom order=1`) → máscara pulmonar (`seg-lungs-LUNA16`, resample NN order=0, binariza `>0.5`, fuera de pulmón = -1000) → clip HU **[-1000, 400]** → min-max a `[0,1]` → patch **64³** en centro del candidato con zero-pad → zero-centering `GLOBAL_MEAN = 0,0992`.

**Osteoartritis (`pre_modelo.py::split_osteoarthritis`)**
- Sin preprocesado de píxeles. Solo organización por clase y **dedup por fingerprint** con `similarity_threshold=0,12`, `fingerprint_size=16`.

**Pancreas (`pre_embeddings.py` sección Pancreas)**
- NIfTI → resample isotrópico 1 mm con **B-spline order=3** → clip HU **[-150, 250]** → norm `[0,1]` → resize trilineal a **64³** → **centroide desde máscara label==3** de PANORAMA → crop **48³** centrado (clamp centroide a `[24, 40]`).
- Fallback si no hay máscara: centroide geométrico `(32,32,32)`. Estrategia guardada en `centroid_strategy.txt`; si cambia, invalida caché.

**Augmentación offline de LUNA** (`create_augmented_train.py`)
- Objetivo: bajar desbalance de 10:1 a **2:1**.
- Flip 3 ejes (p=0,5 c/u) + rot 3D ±15° + zoom [0,80, 1,20] + trasl. ±4 vox + elástica (sigma [1,3], alpha [0,5], p=0,5) + ruido gauss + brillo/contraste + blur.

**Splits (todos, `pre_modelo.py`)**
- SEED=42, 80/10/10 train/val/test.
- CXR14 por `patient_id` → `StratifiedShuffleSplit`.
- ISIC por `lesion_id`.
- LUNA16 por `seriesuid`.
- Pancreas por `patient_id` con 10 % test + 5-fold `GroupKFold`.

---

## 4. Entrenamiento de los 5 expertos

| Experto | Dataset | Arquitectura | Archivo |
|---------|---------|--------------|---------|
| 0 | CXR14 | ConvNeXt-V2 Base 384 + LSE pool | `scripts/exp1_v21/train_exp1v21.py` + `post_hoc_v21.py` |
| 1 | ISIC 2019 | ConvNeXt-Small + cabeza LN+Dropout | `models/experto3_isic.ipynb` |
| 2 | Osteo KL | EfficientNet-B0 + head 5 clases | `models/experto2_osteo.ipynb` |
| 3 | LUNA16 3D | DenseNet-3D custom (6,7 M params) | pipelines previos (checkpoint cargado vía `MOE-2/experts.py`) |
| 4 | Pancreas 3D | R3D-18 (torchvision) | `models/EXP5_PANCREAS_v16.ipynb` |

> **Nombres engañosos**: `experto2_osteo` en realidad corresponde al `expert_id=2` que es Osteo; `experto3_isic` es el `expert_id=1` de ISIC. El PDF original numeraba distinto; el router ignora esos nombres y usa `EXPERT_MAP = {chest:0, isic:1, osteo:2, luna:3, panc:4}`. Documentado en `MOE-2/experts.py` y `MOE-2/experts.py:15-18`.

### CXR14 — historial (20+ versiones)

- `exp1_v7` ConvNeXt-Tiny (baseline) → v9 EfficientNet-B2 → v11 EffNetV2-S + CLAHE → v12 EffNet-B2 NS + CLAHE → v15/v16/v17/v18 iteraciones → v20 submuestreo No Finding + Fisher sampler → **v21 (final)**: ConvNeXt-V2 Base + LSE pool + LP-FT + TTA multi-escala + logit adjustment. F1 Macro = 0,4979; AUROC = 0,8064.
- **Réplica directa de Fisher** (sin LP-FT, descongelando todo desde época 1) colapsó a F1 = 0,2968 por olvido catastrófico. Documentado en `reporte_moe.md` sección "Trayectoria".

Scripts reportes CXR14: `scripts/reports/exp1_fig1_*` a `exp1_fig4_*` (trayectoria, prevalencia, Fisher weights, thresholds).

---

## 5. Embeddings + Ablation Routers

### `embeddings/NB03_embeddings_fase0_v8_CORREGIDO.ipynb`
- Extrae el CLS token de cada dataset usando el ViT-Tiny congelado (`vit_tiny_patch16_224.augreg_in21k_ft_in1k`).
- Produce los arreglos `all_train_Z.npy` (126.530, 192) y `all_train_expert_y.npy` (126.530,) con labels 0–4.
- **Detalle sucio (bueno saberlo)**: los embeddings del pancreático vienen de un pipeline anterior (`panc_train_Z.npy`) con LayerNorm pre-aplicado. El notebook los carga directamente sin re-extraer (línea 187-190). Resto de datasets se extraen nuevos.

### `embeddings/ablation_routers_v4_CORREGIDO.ipynb`
- Compara 4 routers sobre los mismos embeddings: Linear + Softmax (con Auxiliary Loss), GMM EM con 5 componentes, Naive Bayes MLE analítico, **k-NN FAISS IndexFlatIP con k=5**.
- Aplica `faiss.normalize_L2` **solo para kNN** antes del IndexFlatIP (convención SoTA: CLIP/DINOv2 normalizan antes, V-MoE/Mixtral no).
- Ganador: kNN con F1 Macro 0,9999, RA 0,9999, BA 0,9999.

### `notebooks/routing/ablation_routers.ipynb`
- Versión antigua del ablation. La autoritativa es la v4 en `embeddings/`.

---

## 6. MoE como sistema (`MOE-2/`)

| Archivo | Qué hace |
|---------|----------|
| `moe_model.py` | Clase `MoEModel` que orquesta todo. Entrada: tensor `x_raw`. Salida: `(probs, expert_id, router_info)`. |
| `preprocess.py` | `AdaptivePreprocessor`: detecta 2D/3D por `rank` del tensor. 2D → resize 224×224 + ImageNet-norm. 3D → clip HU [-1000, 400] + extrae 16 slices equidistantes. |
| `backbone.py` | `SharedBackbone` con `vit_tiny_patch16_224` congelado; para 3D promedia CLS de N slices centrales (`n_slices_3d=5` por defecto, 16 en producción). |
| `router_knn.py` | `KNNRouter`: carga `all_train_Z.npy` y `all_train_expert_y.npy`, construye `faiss.IndexFlatIP` con vectores L2-normalizados, `k=5` vecinos, agregación por votación. Devuelve `expert_id`, `cosine_scores`, `neighbor_labels`, `vote_counts`, `confidence`. |
| `experts.py` | `load_all_experts()`: reconstruye las 5 arquitecturas y carga checkpoints con `strict_load=True`. Incluye `DenseNet3D` y `ResNet3D` custom (torchvision no tiene R3D con la cabeza que usamos). |
| `inference.py` | Wrapper CLI para correr inferencia sobre un archivo. |
| `utils.py` | Helpers compartidos. |

> Si preguntan **"¿cómo detectan 2D vs 3D?"** → `rank` del tensor: 4 = 2D, 5 = 3D. Cero metadatos externos, cero nombres de archivo.

> Si preguntan **"¿y si es una imagen rara (perro, paisaje)?"** → detector OOD combinado: entropía del gating kNN + coseno mínimo del vecino más cercano. Umbrales configurables en el dashboard. El paper IEEE reporta AUROC = 0,9997 con imágenes BraTS (MRI cerebrales).

---

## 7. Scripts del MoE (`scripts/moe/`)

| Script | Función |
|--------|---------|
| `train_phase2_router.py` | Entrena el *router* lineal como candidato del ablation (no el kNN: el kNN no entrena). |
| `run_moe_eval.py` | Evalúa el MoE end-to-end sobre validación, devuelve RA, BA, F1 Macro, load balance. |
| `test_moe_system.py` | Tests unitarios: strict_load, shapes, rutas. |
| `benchmark_ood_brats.py` | Calibra el detector OOD sobre BraTS → AUROC 0,9997. |
| `bench_latency.py` | Mide latencia p50 y p95 por componente (preprocesador, ViT, router, experto). |
| `smoke_test_real_samples.py` | Prueba rápida con imágenes reales cargadas de disco. |
| `smoke_test_panc_routing.py` | Verifica que el caso específico del pancreático se enruta bien después del fix de embeddings. |

---

## 8. Dashboard (`dashboard/app.py`)

- **Streamlit app** que permite cargar una o varias imágenes (batch), ejecutar el MoE completo y visualizar: predicción, heatmap de atención, gating scores, vecinos kNN, load balance acumulado y detector OOD.
- Implementación: `accept_multiple_files=True`, loop sobre archivos con `st.progress`, tabla de resumen con `pd.DataFrame`, descarga CSV, detector OOD combinado (entropía + coseno mínimo).
- Bloqueo de OOD: si la imagen es OOD, el experto **no recibe la imagen** y no se contabiliza en el load balance (sliders configurables en la sidebar).

---

## 9. Scripts de figuras del reporte IEEE (`scripts/reports/`)

| Script | Figura que genera |
|--------|-------------------|
| `exp1_fig1_trayectoria.py` | Trayectoria F1 Macro a través de 21 versiones de CXR14. |
| `exp1_fig2_prevalencia.py` | Prevalencia por clase en CXR14. |
| `exp1_fig3_fisher_weights.py` | Pesos del muestreo geométrico de Fisher. |
| `exp1_fig4_thresholds.py` | Sweep de thresholds por clase. |
| `fig2_ablation_comparativa.py` | Comparativa 4 routers (RA, BA, F1, latencia, VRAM). |
| `fig3_curvas_moe.py` | Curvas de entrenamiento del router (20 épocas Fase 2). |
| `fig4_load_balance.py` | Evolución de $f_i$ por experto durante Fase 2. |
| `fig5_attention_heatmap.py` | Heatmaps de atención sobre 3 modalidades. |
| `fig6_cm_cxr14.py` | Matrices de confusión one-vs-rest para CXR14. |

---

## 10. Respuestas de emergencia

**"¿Por qué ganó kNN si el lineal tenía RA=1,0000?"**
→ kNN obtuvo F1 Macro 0,9999 (el más alto); cero pesos entrenables vía gradiente; actualizable sin reentrenar; interpretable por vecinos; OOD nativo por distancia. La diferencia en RA es una décima de punto.

**"¿Por qué un ViT-Tiny y no un backbone médico?"**
→ El CLS del augreg preentrenado en ImageNet-21K ya es linealmente separable entre los 5 dominios. Cambiar a backbone médico no mejora routing (ya saturado) y rehace toda la tabla de ablation. Trabajo futuro.

**"¿Por qué k=5 y no k=1?"**
→ Probamos {1, 3, 5, 7, 11}. k=1 aumenta varianza (un solo vecino decide), k=5 estabiliza, k≥7 mezcla clases en fronteras. Similar a la convención DINO (k=20 en ImageNet pero tamaño de dominio mayor).

**"¿Quién hizo las transformaciones de LUNA y Páncreas?"**
→ Nicolás, usando SimpleITK + scipy. Scripts: `pre_embeddings.py` secciones LUNA y Pancreas.

**"¿Cómo evitan data leakage en CXR14?"**
→ `MultilabelStratifiedKFold` con `patient_id` extraído del nombre del archivo (`00012345_001.png` → `00012345`). Paper Fisher 2025 documentó 67,4 % contaminación en los splits oficiales del NIH.

**"¿Por qué el pancreático usa crop 48³ y LUNA16 patches 64³?"**
→ LUNA busca nódulos (3–30 mm) y necesita contexto pulmonar → 64³. Pancreas busca el órgano completo centrado en el centroide → 48³ es suficiente y reduce memoria. Ambos tras resample isotrópico 1 mm.

**"¿Por qué Focal Loss y no oversampling?"**
→ Con 363 positivos de 1.703 casos en Pancreas, oversampling duplica y el modelo memoriza. Focal Loss pondera la clase minoritaria sin replicar muestras.

**"¿Por qué LP-FT?"**
→ La réplica directa de Fisher 2025 sin LP-FT colapsó F1 a 0,2968 por olvido catastrófico (documentado en el reporte IEEE). Congelar el backbone mientras se calienta la cabeza preserva las features aprendidas.

**"¿Por qué no GMM full?"**
→ Con 192 dimensiones y covarianza full necesita 36.864 parámetros por componente, 184.320 totales. PANORAMA con 225 muestras dispara `ConvergenceWarning` de sklearn en las 3 semillas. Usamos `covariance_type='diag'` pero aun así queda por debajo de Linear y Naive Bayes.

**"¿Cómo aseguran reproducibilidad?"**
→ Seeds fijas 42, 137, 256. `strict_load=True` en los 5 checkpoints. Commit `a39d305` referenciado en el reporte.

**"¿Qué es la Auxiliary Loss del Switch Transformer?"**
→ Pérdida adicional que penaliza cuando el router concentra tráfico en pocos expertos. α=0,02 fue el mínimo que mantenía $f_i$ en margen sin degradar RA. Aplica al router lineal, no al kNN.

---

## 11. Números clave para disparar sin dudar

- CLS dim: **192**
- k del router: **5**
- Índice FAISS: **126.530** vectores, **85 MB** RAM
- Validación: **20.132** tokens
- Seeds: **42, 137, 256**
- RA kNN: **0,9999** · F1 Macro kNN: **0,9999**
- Latencia p50 kNN: **6.582 μs** (6,6 ms)
- Latencia pipeline 2D: **27 ms** · 3D: **201 ms**
- OOD AUROC: **0,9997** · TPR: **99,49 %** · FPR: **0,46 %**
- F1 por experto: CXR14 **0,4979** · ISIC **0,8830** · Osteo **0,7987** · LUNA16 **0,9438** · Pancreas **0,6558**
- 4 de 5 expertos cumplen umbral Full (solo CXR14 queda debajo de 0,65)
