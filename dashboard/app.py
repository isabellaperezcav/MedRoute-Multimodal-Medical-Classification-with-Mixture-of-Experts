"""
Dashboard Interactivo — Sistema MoE Clasificación Médica
Proyecto Final — Incorporar Elementos de IA, Unidad II (Bloque Visión)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import io
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="MoE Medical Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Estilos CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8em; }
    .main-header p  { color: #cce3f5; margin: 4px 0 0 0; font-size: 0.95em; }

    .metric-card {
        background: #f0f6ff;
        border: 1px solid #c0d8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card h3 { margin: 0; color: #1e3a5f; font-size: 2em; }
    .metric-card p  { margin: 0; color: #555; font-size: 0.85em; }

    .expert-card {
        background: linear-gradient(135deg, #e8f4fd, #d0e9f9);
        border-left: 5px solid #2d6a9f;
        border-radius: 8px;
        padding: 14px;
        margin: 8px 0;
    }
    .expert-card h4 { color: #1e3a5f; margin: 0 0 6px 0; }
    .expert-card p  { color: #333; margin: 2px 0; font-size: 0.88em; }

    .ood-alert {
        background: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 14px;
        margin: 8px 0;
    }
    .ood-ok {
        background: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 14px;
        margin: 8px 0;
    }
    .section-title {
        font-size: 1.05em;
        font-weight: 700;
        color: #1e3a5f;
        border-bottom: 2px solid #2d6a9f;
        padding-bottom: 4px;
        margin: 16px 0 10px 0;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78em;
        font-weight: 600;
    }
    .badge-green  { background:#d4edda; color:#155724; }
    .badge-blue   { background:#cce5ff; color:#004085; }
    .badge-orange { background:#fff3cd; color:#856404; }
</style>
""", unsafe_allow_html=True)

# ─── Datos del sistema (mock hasta conectar modelos reales) ──────────────────
EXPERTS = {
    0: {"name": "Exp1 — ChestX-ray14",   "arch": "ConvNeXt-Tiny",   "dataset": "NIH ChestX-ray14",  "task": "Multilabel 14 patologías"},
    1: {"name": "Exp2 — ISIC 2019",       "arch": "EfficientNet-B3", "dataset": "ISIC 2019",          "task": "9 clases dermato."},
    2: {"name": "Exp3 — Osteoarthritis",  "arch": "VGG-16 BN",       "dataset": "OA Knee X-ray",      "task": "3 grados KL"},
    3: {"name": "Exp4 — LUNA16 CT",       "arch": "R3D-18",          "dataset": "LUNA16 / LIDC-IDRI", "task": "Nódulo pulmonar 2 clases"},
    4: {"name": "Exp5 — Pánc. Cancer",    "arch": "Swin3D-Tiny",     "dataset": "Pancreatic Cancer CT","task": "2 clases"},
}

LABELS = {
    0: ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule","Pneumonia",
        "Pneumothorax","Consolidation","Edema","Emphysema","Fibrosis","Pleural Thickening","Hernia"],
    1: ["Melanoma","Melanocytic nevus","Basal cell carcinoma","Actinic keratosis",
        "Benign keratosis","Dermatofibroma","Vascular lesion","Squamous cell carcinoma","Unknown"],
    2: ["Normal (KL 0-1)","Leve (KL 2)","Severo (KL 3-4)"],
    3: ["Sin nódulo","Con nódulo"],
    4: ["Sin tumor","Con tumor"],
}

ABLATION_DATA = {
    "Router":           ["ViT + Linear (DL)", "ViT + GMM",  "ViT + Naive Bayes", "ViT + k-NN (FAISS)"],
    "Tipo":             ["Paramétrico (grad.)","Paramétrico (EM)","Paramétrico (MLE)","No paramétrico"],
    "Routing Acc.":     [0.89, 0.83, 0.77, 0.81],
    "Latencia (ms)":    [8.2,  3.1,  0.9,  4.8],
    "VRAM (MB)":        [1500, 50,   5,    85],
    "Gradiente":        ["Sí", "No", "No", "No"],
}

OOD_THRESHOLD = 1.35   # umbral de entropía calibrado

# ─── Estado de sesión ─────────────────────────────────────────────────────────
if "f_i_history" not in st.session_state:
    st.session_state.f_i_history = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
if "n_inferences" not in st.session_state:
    st.session_state.n_inferences = 0

# ─── Funciones auxiliares ─────────────────────────────────────────────────────
def detect_modality(img_array: np.ndarray) -> str:
    return "3D" if img_array.ndim == 4 else "2D"

def adaptive_preprocess(img: Image.Image) -> tuple:
    """Simula el AdaptivePreprocessor del proyecto."""
    orig_size = img.size
    arr = np.array(img.convert("RGB"))
    arr_resized = np.array(img.convert("RGB").resize((224, 224)))
    arr_norm = (arr_resized.astype(np.float32) / 255.0 - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
    return orig_size, (224, 224), arr_norm, arr_resized

def mock_inference(img_array: np.ndarray, router_choice: str) -> dict:
    """
    ────────────────────────────────────────────────────────────────
    TODO: Reemplazar este bloque con el modelo real.
          Cargar checkpoint con torch.load() y ejecutar forward pass.

          Ejemplo:
              model = MoESystem.load_from_checkpoint("best_moe.ckpt")
              model.eval()
              with torch.no_grad():
                  gating_scores, expert_logits, attn_map = model(tensor)
    ────────────────────────────────────────────────────────────────
    """
    np.random.seed(int(time.time() * 1000) % 2**31)
    t0 = time.perf_counter()

    # Simular gating scores (5 expertos)
    raw = np.random.dirichlet(np.ones(5) * 3)
    expert_idx = int(np.argmax(raw))

    # Simular logits del experto seleccionado
    n_labels = len(LABELS[expert_idx])
    logits = np.random.randn(n_labels)
    probs  = np.exp(logits) / np.exp(logits).sum()
    pred_label = LABELS[expert_idx][int(np.argmax(probs))]
    confidence = float(np.max(probs))

    # Attention heatmap (simulado: gradiente radial + ruido)
    h, w = 224, 224
    cx, cy = np.random.randint(50, 174), np.random.randint(50, 174)
    Y, X = np.mgrid[0:h, 0:w]
    attn = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * 40**2))
    attn += np.random.rand(h, w) * 0.15
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    # Entropía del gating (OOD signal)
    entropy = float(-np.sum(raw * np.log(raw + 1e-8)))

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000

    return {
        "expert_idx":  expert_idx,
        "gating":      raw,
        "pred_label":  pred_label,
        "confidence":  confidence,
        "probs":       probs,
        "attn_map":    attn,
        "entropy":     entropy,
        "latency_ms":  latency_ms,
    }

def update_load_balance(expert_idx: int):
    n = st.session_state.n_inferences
    current = st.session_state.f_i_history
    new_counts = current * n
    new_counts[expert_idx] += 1
    st.session_state.f_i_history = new_counts / new_counts.sum()
    st.session_state.n_inferences = n + 1

def compute_load_ratio():
    fi = st.session_state.f_i_history
    return fi.max() / (fi.min() + 1e-8)

def overlay_heatmap(img_pil: Image.Image, attn: np.ndarray) -> np.ndarray:
    img_rgb  = np.array(img_pil.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
    colormap = cm.get_cmap("jet")(attn)[..., :3]
    blended  = 0.55 * img_rgb + 0.45 * colormap
    return np.clip(blended, 0, 1)

def is_nifti(file_bytes: bytes) -> bool:
    """Detecta magic bytes de NIfTI (.nii / .nii.gz)."""
    return file_bytes[:4] in [b'\x5c\x01\x00\x00', b'\x00\x00\x01\x5c']

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>🧬 MoE Medical Imaging Dashboard</h1>
  <p>Clasificación Médica con Mixture of Experts + Ablation Study del Router &nbsp;|&nbsp;
     Incorporar Elementos de IA — Unidad II (Bloque Visión)</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    router_choice = st.selectbox(
        "Router activo",
        ["ViT + Linear (DL)", "ViT + GMM", "ViT + Naive Bayes", "ViT + k-NN (FAISS)"],
        index=0,
        help="Selecciona el mecanismo de routing para la inferencia."
    )
    show_ablation = st.checkbox("Mostrar panel Ablation Study", value=True)
    show_ood      = st.checkbox("Mostrar panel OOD Detection",  value=True)
    show_balance  = st.checkbox("Mostrar panel Load Balance",   value=True)

    st.divider()
    st.markdown("**📌 Umbrales**")
    ood_threshold = st.slider("Umbral entropía OOD", 0.5, 3.0, float(OOD_THRESHOLD), 0.05)
    balance_limit = st.slider("Límite cociente max/min f_i", 1.0, 3.0, 1.30, 0.05,
                               help="Penalización al superar este cociente.")

    st.divider()
    if st.button("🔄 Reiniciar contadores de carga"):
        st.session_state.f_i_history = np.array([0.20]*5)
        st.session_state.n_inferences = 0
        st.success("Contadores reiniciados")

    st.divider()
    st.caption(f"Total inferencias: **{st.session_state.n_inferences}**")
    ratio = compute_load_ratio()
    color = "🔴" if ratio > balance_limit else "🟢"
    st.caption(f"Cociente max/min f_i: {color} **{ratio:.2f}** (límite {balance_limit:.2f})")

# ─── Columnas principales ─────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.6], gap="large")

# ══════════════════════════════════════════
# COLUMNA IZQUIERDA — Carga + Preprocesado
# ══════════════════════════════════════════
with col_left:
    st.markdown('<div class="section-title">📁 1. Carga de Imagen</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Sube una imagen médica (PNG, JPEG, NIfTI)",
        type=["png", "jpg", "jpeg", "nii"],
        label_visibility="collapsed",
    )

    # Imagen demo si no hay carga
    if uploaded_file is None:
        st.info("⬆️ Sube una imagen médica para comenzar. Se acepta PNG, JPEG y NIfTI.")
        demo_arr = np.random.randint(60, 200, (224, 224, 3), dtype=np.uint8)
        demo_arr[80:140, 80:140] = [200, 220, 240]
        img_pil = Image.fromarray(demo_arr)
        file_bytes = None
        using_demo = True
        st.caption("_Vista previa con imagen de demostración_")
    else:
        file_bytes = uploaded_file.read()
        using_demo = False

        # Detección NIfTI
        if is_nifti(file_bytes) or uploaded_file.name.endswith(".nii"):
            st.warning("🗃️ Archivo NIfTI detectado (3D). Se muestra slice central.")
            modality = "3D (NIfTI)"
            volume = np.random.randint(40, 180, (64, 64, 64), dtype=np.uint8)
            slice_2d = volume[:, :, 32]
            img_pil = Image.fromarray(slice_2d).convert("RGB")
            orig_size = (64, 64, 64)
        else:
            img_pil = Image.open(io.BytesIO(file_bytes))
            orig_size = img_pil.size
            modality = detect_modality(np.array(img_pil))

        st.image(img_pil, caption=f"Imagen cargada: {uploaded_file.name}", use_container_width=True)

    # ── Preprocesado ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔧 2. Preprocesado Transparente</div>', unsafe_allow_html=True)

    if not using_demo:
        orig_s, adapt_s, arr_norm, arr_resized = adaptive_preprocess(img_pil)
        modality_label = "3D (NIfTI)" if uploaded_file.name.endswith(".nii") else detect_modality(np.array(img_pil))

        cols_pre = st.columns(3)
        with cols_pre[0]:
            st.markdown(f"""<div class="metric-card">
                <h3>{orig_s[0]}×{orig_s[1]}</h3><p>Dimensiones originales</p></div>""",
                unsafe_allow_html=True)
        with cols_pre[1]:
            st.markdown(f"""<div class="metric-card">
                <h3>{adapt_s[0]}×{adapt_s[1]}</h3><p>Dimensiones adaptadas</p></div>""",
                unsafe_allow_html=True)
        with cols_pre[2]:
            st.markdown(f"""<div class="metric-card">
                <h3>{modality_label}</h3><p>Modalidad detectada</p></div>""",
                unsafe_allow_html=True)

        with st.expander("Ver estadísticas del tensor normalizado"):
            st.code(f"""
Forma:    {arr_norm.shape}
Media:    {arr_norm.mean():.4f}
Std:      {arr_norm.std():.4f}
Min/Max:  {arr_norm.min():.3f} / {arr_norm.max():.3f}
dtype:    float32
Norm:     ImageNet (µ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225])
            """, language="text")
    else:
        st.caption("_Carga una imagen real para ver las dimensiones de preprocesado._")

    # ── Botón de inferencia ───────────────────────────────────────────────────
    st.divider()
    run_inference = st.button("🚀 Ejecutar Inferencia", type="primary", use_container_width=True)

# ══════════════════════════════════════════
# COLUMNA DERECHA — Resultados
# ══════════════════════════════════════════
with col_right:
    if run_inference or (st.session_state.n_inferences > 0 and "last_result" in st.session_state):

        if run_inference:
            arr_input = np.array(img_pil) if not using_demo else np.random.rand(224, 224, 3)
            with st.spinner("Infiriendo..."):
                result = mock_inference(arr_input, router_choice)
            st.session_state.last_result = result
            update_load_balance(result["expert_idx"])
        else:
            result = st.session_state.last_result

        exp_info = EXPERTS[result["expert_idx"]]

        # ── 3. Inferencia en tiempo real ──────────────────────────────────────
        st.markdown('<div class="section-title">⚡ 3. Inferencia en Tiempo Real</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <h3 style="font-size:1.1em">{result['pred_label']}</h3>
                <p>Predicción</p></div>""", unsafe_allow_html=True)
        with c2:
            conf_pct = result['confidence'] * 100
            conf_color = "#28a745" if conf_pct > 70 else "#ffc107"
            st.markdown(f"""<div class="metric-card">
                <h3 style="color:{conf_color}">{conf_pct:.1f}%</h3>
                <p>Confianza</p></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <h3>{result['latency_ms']:.1f} ms</h3>
                <p>Tiempo inferencia</p></div>""", unsafe_allow_html=True)

        # ── 4. Attention Heatmap ──────────────────────────────────────────────
        st.markdown('<div class="section-title">🔥 4. Attention Heatmap — Router ViT</div>', unsafe_allow_html=True)

        overlay = overlay_heatmap(img_pil, result["attn_map"])

        fig_heat, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        fig_heat.patch.set_facecolor("#f8fafc")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])

        axes[0].imshow(img_pil.convert("RGB").resize((224,224)))
        axes[0].set_title("Original", fontsize=9, fontweight="bold")

        im = axes[1].imshow(result["attn_map"], cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Attention Map", fontsize=9, fontweight="bold")
        fig_heat.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay (α=0.45)", fontsize=9, fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig_heat, use_container_width=True)
        plt.close(fig_heat)

        # ── 5. Panel del Experto Activado ────────────────────────────────────
        st.markdown('<div class="section-title">🤖 5. Experto Activado</div>', unsafe_allow_html=True)

        gating_scores = result["gating"]
        st.markdown(f"""
        <div class="expert-card">
          <h4>🏆 {exp_info['name']}</h4>
          <p>🏗️ <strong>Arquitectura:</strong> {exp_info['arch']}</p>
          <p>📊 <strong>Dataset origen:</strong> {exp_info['dataset']}</p>
          <p>🎯 <strong>Tarea:</strong> {exp_info['task']}</p>
          <p>📈 <strong>Gating score:</strong> <code>{gating_scores[result['expert_idx']]:.4f}</code>
             &nbsp;<span class="badge badge-green">Router: {router_choice.split(' (')[0]}</span></p>
        </div>
        """, unsafe_allow_html=True)

        # Gating bar
        fig_gate = go.Figure(go.Bar(
            x=[EXPERTS[i]["name"].split("—")[1].strip() for i in range(5)],
            y=gating_scores,
            marker_color=["#2d6a9f" if i == result["expert_idx"] else "#a8c8e8" for i in range(5)],
            text=[f"{g:.3f}" for g in gating_scores],
            textposition="outside",
        ))
        fig_gate.update_layout(
            title="Gating scores (todos los expertos)",
            yaxis=dict(range=[0, 1], title="Score"),
            height=230, margin=dict(t=40, b=30, l=30, r=10),
            plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc",
        )
        st.plotly_chart(fig_gate, use_container_width=True)

    else:
        st.info("👈 Sube una imagen y presiona **Ejecutar Inferencia** para ver los resultados.")

# ══════════════════════════════════════════════════════════════════
# SECCIÓN INFERIOR — Ablation, Load Balance, OOD
# ══════════════════════════════════════════════════════════════════
st.divider()

tabs_lower = []
if show_ablation: tabs_lower.append("📋 6. Ablation Study")
if show_balance:  tabs_lower.append("⚖️ 7. Load Balance")
if show_ood:      tabs_lower.append("🚨 8. OOD Detection")

if tabs_lower:
    tabs = st.tabs(tabs_lower)
    tab_idx = 0

    # ── 6. Ablation Study ─────────────────────────────────────────────────────
    if show_ablation:
        with tabs[tab_idx]:
            st.markdown("#### Comparativa de los 4 Mecanismos de Routing")
            st.caption("Resultados sobre el mismo backbone ViT congelado — solo varía la cabeza de decisión.")

            import pandas as pd
            df_abl = pd.DataFrame(ABLATION_DATA)
            df_abl["Routing Acc."] = df_abl["Routing Acc."].map("{:.0%}".format)
            df_abl["Latencia (ms)"] = df_abl["Latencia (ms)"].map("{:.1f}".format)
            df_abl["VRAM (MB)"] = df_abl["VRAM (MB)"].map("{:,}".format)

            # Highlight ganador
            st.dataframe(
                df_abl.style.apply(
                    lambda row: ["background-color: #d4edda; font-weight:bold"]*len(row)
                    if row["Router"] == "ViT + Linear (DL)" else [""]*len(row),
                    axis=1
                ),
                use_container_width=True, hide_index=True
            )

            col_a1, col_a2 = st.columns(2)

            with col_a1:
                # Bar chart Routing Accuracy
                fig_acc = go.Figure(go.Bar(
                    x=ABLATION_DATA["Router"],
                    y=[v*100 for v in ABLATION_DATA["Routing Acc."]],
                    marker_color=["#2d6a9f","#4caf50","#ff9800","#9c27b0"],
                    text=[f"{v*100:.0f}%" for v in ABLATION_DATA["Routing Acc."]],
                    textposition="outside",
                ))
                fig_acc.add_hline(y=80, line_dash="dash", line_color="red",
                                  annotation_text="Umbral 80%")
                fig_acc.update_layout(
                    title="Routing Accuracy por método",
                    yaxis=dict(range=[0, 105], title="Accuracy (%)"),
                    height=320, margin=dict(t=45,b=40,l=40,r=10),
                    plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc",
                )
                st.plotly_chart(fig_acc, use_container_width=True)

            with col_a2:
                # Scatter latencia vs accuracy
                fig_scat = go.Figure()
                colors_s = ["#2d6a9f","#4caf50","#ff9800","#9c27b0"]
                for i, router in enumerate(ABLATION_DATA["Router"]):
                    fig_scat.add_trace(go.Scatter(
                        x=[float(ABLATION_DATA["Latencia (ms)"][i])],
                        y=[ABLATION_DATA["Routing Acc."][i]*100],
                        mode="markers+text",
                        name=router,
                        text=[router.split(" +")[0]],
                        textposition="top center",
                        marker=dict(size=16, color=colors_s[i]),
                    ))
                fig_scat.update_layout(
                    title="Latencia vs Routing Accuracy",
                    xaxis_title="Latencia (ms)", yaxis_title="Accuracy (%)",
                    height=320, margin=dict(t=45,b=40,l=40,r=10),
                    plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc",
                    showlegend=False,
                )
                st.plotly_chart(fig_scat, use_container_width=True)

            st.info("💡 **Router ganador:** ViT + Linear (DL) con 89% Routing Accuracy. "
                    "El ViT+GMM es competitivo (83%) con 30× menos VRAM y sin gradiente descendente.")
        tab_idx += 1

    # ── 7. Load Balance ───────────────────────────────────────────────────────
    if show_balance:
        with tabs[tab_idx]:
            st.markdown("#### Load Balance — Distribución Acumulada de Enrutamiento")
            fi = st.session_state.f_i_history
            ratio = compute_load_ratio()
            n_inf = st.session_state.n_inferences

            col_lb1, col_lb2 = st.columns([1.6, 1])
            with col_lb1:
                bar_colors = ["#2d6a9f" if fi[i] == fi.max() else "#a8c8e8" for i in range(5)]
                fig_lb = go.Figure(go.Bar(
                    x=[EXPERTS[i]["name"].split("—")[1].strip() for i in range(5)],
                    y=fi * 100,
                    marker_color=bar_colors,
                    text=[f"{v:.1%}" for v in fi],
                    textposition="outside",
                ))
                fig_lb.add_hline(y=20, line_dash="dot", line_color="gray",
                                 annotation_text="Balance perfecto (20%)")
                fig_lb.add_hline(y=20 * balance_limit, line_dash="dash", line_color="red",
                                 annotation_text=f"Límite ({balance_limit:.2f}× ≈ {20*balance_limit:.0f}%)")
                fig_lb.update_layout(
                    title=f"f_i acumulado — {n_inf} inferencia(s)",
                    yaxis=dict(range=[0, 60], title="f_i (%)"),
                    height=340, margin=dict(t=50,b=40,l=40,r=10),
                    plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc",
                )
                st.plotly_chart(fig_lb, use_container_width=True)

            with col_lb2:
                st.markdown("**Detalle de carga:**")
                for i in range(5):
                    pct = fi[i] * 100
                    bar_html = f"{'█' * int(pct // 4)}{'░' * (25 - int(pct // 4))}"
                    st.markdown(
                        f"**Exp{i+1}** `{bar_html}` {pct:.1f}%"
                    )
                st.divider()
                status_icon = "🔴" if ratio > balance_limit else "🟢"
                st.metric("Cociente max/min", f"{ratio:.3f}", delta=f"Límite: {balance_limit:.2f}")
                if ratio > balance_limit:
                    st.error(f"{status_icon} **PENALIZACIÓN ACTIVA** — cociente {ratio:.2f} > {balance_limit:.2f}\n\n"
                             "Ajusta α en la Auxiliary Loss.")
                else:
                    st.success(f"{status_icon} Balance dentro del límite permitido.")

                st.caption("_La penalización del 40% aplica únicamente al router ViT+Linear._")
        tab_idx += 1

    # ── 8. OOD Detection ─────────────────────────────────────────────────────
    if show_ood:
        with tabs[tab_idx]:
            st.markdown("#### OOD Detection — Detección por Entropía del Gating Score")
            st.caption(
                "La entropía H(g) del vector de gating actúa como señal de incertidumbre. "
                "Alta entropía = el router no reconoce la imagen → probable OOD."
            )

            col_o1, col_o2 = st.columns([1, 1])
            with col_o1:
                if "last_result" in st.session_state:
                    entropy = st.session_state.last_result["entropy"]
                    gating  = st.session_state.last_result["gating"]
                    is_ood  = entropy > ood_threshold
                    max_ent = np.log(5)  # entropía máxima con 5 expertos

                    st.markdown("**Última imagen procesada:**")
                    if is_ood:
                        st.markdown(f"""
                        <div class="ood-alert">
                        ⚠️ <strong>POSIBLE OOD DETECTADO</strong><br>
                        Entropía H(g) = <strong>{entropy:.4f}</strong>
                        (umbral: {ood_threshold:.2f})<br>
                        Esta imagen podría no ser una imagen médica reconocida.
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="ood-ok">
                        ✅ <strong>Imagen IN-DISTRIBUTION</strong><br>
                        Entropía H(g) = <strong>{entropy:.4f}</strong>
                        (umbral: {ood_threshold:.2f})<br>
                        El router reconoce la imagen con buena confianza.
                        </div>""", unsafe_allow_html=True)

                    # Gauge chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=entropy,
                        delta={"reference": ood_threshold, "increasing": {"color": "#dc3545"}},
                        number={"suffix": " nats", "font": {"size": 22}},
                        gauge={
                            "axis": {"range": [0, max_ent]},
                            "bar":  {"color": "#dc3545" if is_ood else "#28a745"},
                            "steps": [
                                {"range": [0, ood_threshold],   "color": "#d4edda"},
                                {"range": [ood_threshold, max_ent], "color": "#f8d7da"},
                            ],
                            "threshold": {
                                "line": {"color": "orange", "width": 3},
                                "thickness": 0.75,
                                "value": ood_threshold,
                            },
                        },
                        title={"text": "Entropía H(g)"},
                    ))
                    fig_gauge.update_layout(height=280, margin=dict(t=40,b=10,l=20,r=20),
                                            paper_bgcolor="#f8fafc")
                    st.plotly_chart(fig_gauge, use_container_width=True)
                else:
                    st.info("Ejecuta una inferencia para ver el estado OOD.")

            with col_o2:
                st.markdown("**Distribución de entropías simuladas:**")
                np.random.seed(42)
                ent_in  = np.random.beta(2, 5, 200) * np.log(5)
                ent_ood = np.random.beta(5, 2, 80)  * np.log(5)

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=ent_in, name="In-Distribution",
                    marker_color="#28a745", opacity=0.7, nbinsx=25))
                fig_hist.add_trace(go.Histogram(
                    x=ent_ood, name="OOD (no médico)",
                    marker_color="#dc3545", opacity=0.7, nbinsx=25))
                fig_hist.add_vline(x=ood_threshold, line_dash="dash", line_color="orange",
                                   annotation_text=f"Umbral {ood_threshold:.2f}")
                fig_hist.update_layout(
                    barmode="overlay",
                    title="Histograma entropía H(g) — sistema calibrado",
                    xaxis_title="Entropía (nats)", yaxis_title="Frecuencia",
                    height=300, margin=dict(t=50,b=40,l=40,r=10),
                    plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                st.caption("""
                **Fórmula:** H(g) = −∑ gᵢ · log(gᵢ)  
                **Máx. posible:** log(5) ≈ 1.609 nats (distribución uniforme)  
                **Calibración:** Ajustar el umbral con un conjunto de imágenes no médicas.
                """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>🧬 <b>MoE Medical Dashboard</b> — Proyecto Final Visión · "
    "Incorporar Elementos de IA, Unidad II · "
    "<code>TODO: Conectar modelos reales en mock_inference()</code></small></center>",
    unsafe_allow_html=True
)
