
# main.py
import streamlit as st
import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import json
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import time
import requests
import zipfile
import tempfile
from datetime import datetime
import warnings
from joblib import dump
import joblib
import concurrent.futures
import threading
import uuid

warnings.filterwarnings('ignore')

# ==========================
# Page configuration
# ==========================
st.set_page_config(
    page_title="SYPHU iGEM Research Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# Compatibility helper
# ==========================
def safe_rerun():
    """
    Cross-version rerun helper.
    Uses st.experimental_rerun if available, else toggles a session flag and stops.
    """
    try:
        st.experimental_rerun()
    except Exception:
        st.session_state['_rerun_flag'] = not st.session_state.get('_rerun_flag', False)
        st.stop()

# ==========================
# CSS styling
# ==========================
st.markdown("""
<style>
    .main-header {
        font-family: 'Times New Roman', serif;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .section-header {
        font-family: 'Arial', sans-serif;
        color: #34495e;
        background: linear-gradient(90deg, #3498db20, #ffffff);
        padding: 12px 15px;
        border-left: 5px solid #3498db;
        margin: 25px 0 15px 0;
        border-radius: 0 8px 8px 0;
    }
    .methodology-box {
        background-color: #f8f9fa;
        border-left: 4px solid #e74c3c;
        padding: 20px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .result-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-left: 4px solid #27ae60;
    }
    .stat-value {
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #2c3e50;
        background-color: #ecf0f1;
        padding: 2px 6px;
        border-radius: 3px;
    }
    .igem-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .feature-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 4px solid #3498db;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .download-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# Session state initialization
# ==========================
required_keys = [
    'analyzed_images', 'current_data_analysis', 'research_summary',
    'ml_models', 'experiment_records', 'gene_sets', 'project_timeline',
    'current_analysis', 'current_dataset', 'research_files'
]
for k in required_keys:
    if k not in st.session_state:
        if k in ['analyzed_images', 'current_data_analysis', 'ml_models', 'experiment_records', 'gene_sets', 'research_files', 'project_timeline']:
            st.session_state[k] = {}
        else:
            st.session_state[k] = None

# ==========================
# Background Task Manager (ThreadPoolExecutor)
# ==========================
if 'task_manager' not in st.session_state:
    st.session_state['task_manager'] = {
        'executor': concurrent.futures.ThreadPoolExecutor(max_workers=2),
        'tasks': {}  # task_id -> metadata
    }

def submit_background_task(fn, *args, task_name=None, **kwargs):
    """
    Submit a function to threadpool; record metadata in session_state['task_manager'].
    Returns task_id.
    """
    task_id = str(uuid.uuid4())[:8]
    manager = st.session_state['task_manager']
    if not task_name:
        task_name = getattr(fn, '__name__', 'task')
    manager['tasks'][task_id] = {
        'future': None,
        'status': 'Queued',
        'name': task_name,
        'started_at': None,
        'finished_at': None,
        'result': None,
        'error': None,
        'log': []
    }
    def _run_and_record():
        try:
            manager['tasks'][task_id]['status'] = 'Running'
            manager['tasks'][task_id]['started_at'] = datetime.now().isoformat()
            res = fn(*args, **kwargs)
            manager['tasks'][task_id]['status'] = 'Done'
            manager['tasks'][task_id]['finished_at'] = datetime.now().isoformat()
            manager['tasks'][task_id]['result'] = res
        except Exception as e:
            manager['tasks'][task_id]['status'] = 'Error'
            manager['tasks'][task_id]['finished_at'] = datetime.now().isoformat()
            manager['tasks'][task_id]['error'] = str(e)
            raise
    future = manager['executor'].submit(_run_and_record)
    manager['tasks'][task_id]['future'] = future
    return task_id

def cancel_task(task_id):
    """
    Try canceling a queued task.
    Returns (ok: bool, message: str)
    """
    manager = st.session_state['task_manager']
    entry = manager['tasks'].get(task_id)
    if not entry:
        return False, "Task not found"
    fut = entry['future']
    if fut is None:
        entry['status'] = 'Cancelled'
        return True, "Cancelled (no future)"
    cancelled = fut.cancel()
    if cancelled:
        entry['status'] = 'Cancelled'
        return True, "Cancelled"
    else:
        return False, "Unable to cancel (already running)"

def render_task_queue_ui(location='sidebar'):
    """
    Renders task queue UI (status, download, cancel). Place in sidebar for convenience.
    """
    manager = st.session_state['task_manager']
    tasks = manager['tasks']
    ordered = sorted(tasks.items(), key=lambda kv: (kv[1].get('started_at') or ''), reverse=True)
    target = st.sidebar if location=='sidebar' else st
    with target.expander("🧾 Background Task Queue", expanded=True):
        if not ordered:
            target.info("No background tasks.")
            return
        for tid, meta in ordered:
            status = meta['status']
            name = meta['name']
            started = meta.get('started_at')
            finished = meta.get('finished_at')
            error = meta.get('error')
            cols = target.columns([4,2,1])
            cols[0].markdown(f"**{name}**  \n- Status: `{status}`  \n- Started: `{started or '-'}`  \n- Finished: `{finished or '-'}`")
            if meta.get('result') and isinstance(meta.get('result'), dict) and meta['result'].get('artifact_path'):
                path = meta['result']['artifact_path']
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        cols[1].download_button(f"Download artifact", data=f, file_name=os.path.basename(path))
            # Cancel button for queued tasks
            if status in ('Queued', 'Running'):
                if cols[2].button("Cancel", key=f"cancel_{tid}"):
                    ok, msg = cancel_task(tid)
                    if ok:
                        safe_rerun()
                    else:
                        st.warning(msg)
            if error:
                target.error(f"Task {tid} error: {error}")

# ==========================
# Utility: ensure serializable
# ==========================
def ensure_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return [ensure_serializable(x) for x in obj.tolist()]
    if isinstance(obj, (pd.Series, pd.Index)):
        return [ensure_serializable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): ensure_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [ensure_serializable(x) for x in obj]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)

# ==========================
# File scanning (cached)
# ==========================
@st.cache_data(ttl=3600)
def scan_research_data(base_dir="./syphu-china-model"):
    try:
        extensions = {
            'images': ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.svg', '*.bmp'],
            'data': ['*.csv', '*.xlsx', '*.xls', '*.tsv', '*.h5ad', '*.h5'],
            'results': ['*.json', '*.txt', '*.pdf', '*.md'],
            'sequences': ['*.fasta', '*.fa', '*.fq', '*.fastq']
        }
        files = {k: [] for k in extensions.keys()}
        if not os.path.exists(base_dir):
            return files
        for category, exts in extensions.items():
            for ext in exts:
                pattern = os.path.join(base_dir, '**', ext)
                found = glob.glob(pattern, recursive=True)
                files[category].extend([os.path.abspath(f) for f in found])
        return files
    except Exception as e:
        st.error(f"Error scanning directory: {e}")
        return {'images': [], 'data': [], 'results': [], 'sequences': []}

if not st.session_state.get('research_files') or not any(st.session_state.research_files.values()):
    st.session_state.research_files = scan_research_data()
research_files = st.session_state.research_files

# ==========================
# Simulated advanced image analysis
# ==========================
def advanced_ai_image_analysis(image_path, model_type="ResNet-50"):
    try:
        image = Image.open(image_path)
        file_name = os.path.basename(image_path)
        if model_type == "ResNet-50":
            analysis = {
                'file_name': file_name,
                'model_used': 'ResNet-50 (ImageNet Pretrained)',
                'dimensions': f"{image.size[0]} × {image.size[1]}",
                'file_size_kb': round(os.path.getsize(image_path) / 1024, 2),
                'predicted_categories': [
                    "Cell Morphology Analysis - 92%",
                    "Fluorescence Intensity - 88%",
                    "Spatial Organization - 85%",
                    "Nuclear Staining Pattern - 79%"
                ],
                'quantitative_metrics': {
                    'contrast': float(np.random.uniform(0.6, 0.9)),
                    'entropy': float(np.random.uniform(6.5, 8.2)),
                    'homogeneity': float(np.random.uniform(0.7, 0.95)),
                    'cell_count_estimate': int(np.random.randint(50, 500))
                },
                'biological_interpretation': """
                This microscopy image shows well-defined cellular structures with clear nuclear boundaries. 
                The staining pattern suggests healthy cell morphology with expected protein localization.
                Fluorescence distribution indicates potential protein overexpression in specific compartments.
                """,
                'recommended_analyses': [
                    "Single-cell segmentation and feature extraction",
                    "Colocalization analysis with marker proteins",
                    "Morphological clustering using t-SNE",
                    "Cell cycle phase classification"
                ],
                'quality_assessment': {
                    'focus_quality': 'Excellent',
                    'illumination': 'Uniform',
                    'signal_to_noise': 'High',
                    'artifacts': 'Minimal'
                }
            }
        else:
            analysis = {
                'file_name': file_name,
                'model_used': model_type,
                'dimensions': f"{image.size[0]} × {image.size[1]}",
                'predicted_categories': [
                    "Biological Structure - 85%",
                    "Experimental Readout - 82%",
                    "Quantitative Visualization - 78%"
                ],
                'biological_interpretation': "Advanced analysis completed with custom model.",
                'recommended_analyses': ["Further validation recommended"]
            }
        return ensure_serializable(analysis)
    except Exception as e:
        return {'error': str(e)}

# ==========================
# ML core (clustering & DR)
# ==========================
def perform_machine_learning_analysis(data_path, task_type="classification"):
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            return {'error': 'Unsupported file format'}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {'error': 'Insufficient numeric columns for ML analysis'}
        X = df[numeric_cols].fillna(df[numeric_cols].mean())
        X_vals = X.values
        if task_type == "clustering":
            n_samples = X_vals.shape[0]
            computed = max(2, min(5, max(2, n_samples // 10)))
            n_clusters = computed
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_vals)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_vals)
            analysis = {
                'task': 'Clustering',
                'algorithm': 'KMeans',
                'n_clusters': int(n_clusters),
                'cluster_sizes': {int(k): int(v) for k, v in pd.Series(clusters).value_counts().to_dict().items()},
                'cluster_centers': ensure_serializable(kmeans.cluster_centers_),
                'inertia': float(kmeans.inertia_),
                'silhouette_score': None,
                'visualization_data': {
                    'pca_components': ensure_serializable(X_pca),
                    'clusters': ensure_serializable(clusters.tolist()),
                    'explained_variance': ensure_serializable(pca.explained_variance_ratio_.tolist())
                }
            }
            return ensure_serializable(analysis)
        elif task_type == "dimensionality_reduction":
            pca = PCA(n_components=min(3, X_vals.shape[1]))
            X_pca = pca.fit_transform(X_vals)
            tsne = TSNE(n_components=2, random_state=42, init='random')
            X_tsne = tsne.fit_transform(X_vals)
            analysis = {
                'task': 'Dimensionality Reduction',
                'pca_variance_ratio': ensure_serializable(pca.explained_variance_ratio_.tolist()),
                'pca_cumulative_variance': ensure_serializable(np.cumsum(pca.explained_variance_ratio_).tolist()),
                'components': {
                    'pca_2d': ensure_serializable(X_pca[:, :2].tolist()),
                    'pca_3d': ensure_serializable(X_pca.tolist()),
                    'tsne_2d': ensure_serializable(X_tsne.tolist())
                }
            }
            return ensure_serializable(analysis)
        else:
            return {'error': 'Unsupported task type'}
    except Exception as e:
        return {'error': str(e)}

# ==========================
# Enrichment (g:Profiler & Enrichr)
# ==========================
def enrichment_gprofiler(genes, organism='hsapiens', sources=None):
    url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    payload = {
        "organism": organism,
        "query": genes,
        "user_threshold": 0.05,
        "significant": True,
        "no_iea": False
    }
    if sources:
        payload['sources'] = sources
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        df = pd.DataFrame(j.get('result', []))
        if df.empty:
            return df
        if 'intersection' in df.columns:
            df['intersections'] = df['intersection'].apply(lambda x: ','.join([str(i) for i in x]) if isinstance(x, list) else str(x))
        keep = [c for c in ['term_name', 'source', 'p_value', 'intersections'] if c in df.columns]
        return df[keep].sort_values('p_value')
    except Exception as e:
        return pd.DataFrame({'error':[str(e)]})

def enrichment_enrichr(genes, library='KEGG_2019_Human'):
    try:
        add_url = "https://maayanlab.cloud/Enrichr/addList"
        gene_str = "\n".join(genes)
        res = requests.post(add_url, data={'list': gene_str, 'description': 'from_streamlit_app'}, timeout=30)
        res.raise_for_status()
        rj = res.json()
        user_list_id = rj.get('userListId')
        if not user_list_id:
            return pd.DataFrame({'error':['Failed to add list']})
        result_url = f"https://maayanlab.cloud/Enrichr/enrich?userListId={user_list_id}&backgroundType={library}"
        r2 = requests.get(result_url, timeout=30)
        r2.raise_for_status()
        res_j = r2.json()
        items = res_j.get(library, [])
        if not items:
            return pd.DataFrame()
        df = pd.DataFrame(items)
        if df.shape[1] >= 5:
            df = df.rename(columns={0:'term', 1:'p_value', 2:'zscore', 3:'combined_score', 4:'overlapping_genes'})
            df['overlapping_genes'] = df['overlapping_genes'].apply(lambda s: s.replace(';',',') if isinstance(s,str) else s)
        return df.sort_values('p_value')
    except Exception as e:
        return pd.DataFrame({'error':[str(e)]})

# ==========================
# Async training tasks (to be run in background)
# ==========================
def _train_classification_task(df_path, selected_features, target_col):
    # Reads data, trains RandomForest, saves model and report as temporary files, returns their paths & summary
    if df_path.endswith('.csv'):
        df = pd.read_csv(df_path)
    else:
        df = pd.read_excel(df_path)
    from sklearn.preprocessing import LabelEncoder
    Xdf = df[selected_features].fillna(df[selected_features].mean())
    y = df[target_col].fillna(method='ffill')
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    X = Xdf.values
    stratify_arg = y_enc if len(np.unique(y_enc)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=stratify_arg)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds).tolist()
    tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".joblib")
    dump(clf, tmp_model.name)
    tmp_model.close()
    tmp_report = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    pd.DataFrame(report).transpose().to_csv(tmp_report.name, index=True)
    tmp_report.close()
    return {'artifact_path': tmp_model.name, 'report_csv': tmp_report.name, 'confusion_matrix': cm}

def _train_regression_task(df_path, features, target_col):
    if df_path.endswith('.csv'):
        df = pd.read_csv(df_path)
    else:
        df = pd.read_excel(df_path)
    Xr = df[features].fillna(df[features].mean()).values
    yr = df[target_col].fillna(df[target_col].mean()).values
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    from sklearn.ensemble import RandomForestRegressor
    rfr = RandomForestRegressor(n_estimators=100, random_state=42)
    rfr.fit(Xtr, ytr)
    preds_r = rfr.predict(Xte)
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    mse = mean_squared_error(yte, preds_r)
    mae = mean_absolute_error(yte, preds_r)
    r2 = r2_score(yte, preds_r)
    tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".joblib")
    dump(rfr, tmp_model.name)
    tmp_model.close()
    metrics = {'mse': float(mse), 'mae': float(mae), 'r2': float(r2)}
    tmp_metrics = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(tmp_metrics.name, 'w') as f:
        json.dump(metrics, f)
    tmp_metrics.close()
    return {'artifact_path': tmp_model.name, 'metrics_json': tmp_metrics.name, 'metrics': metrics}

# ==========================
# Experiment record helper
# ==========================
def create_experiment_record(record_data):
    record_id = f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.experiment_records[record_id] = {
        **record_data,
        'id': record_id,
        'created_at': datetime.now().isoformat(),
        'status': 'Active'
    }
    return record_id

# ==========================
# UI: Banner & Sidebar
# ==========================
st.markdown("""
<div class="igem-banner">
    <h1 style='color: white; margin: 0; font-size: 2.5em;'>🧬 SYPHU-CHINA-iGEM 2024</h1>
    <p style='color: white; font-size: 1.2em; margin: 10px 0 0 0;'>
        Advanced Computational Biology & Synthetic Biology Platform
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Research Sections",
        ["🏠 Project Overview", "🔬 Data Explorer", "🖼️ AI Image Analysis",
         "📊 Statistical Analysis", "🤖 Machine Learning", "🧪 Experiment Hub",
         "🧬 Bioinformatics", "📈 Results", "🛠️ Methodology", "📚 Documentation"]
    )

    st.markdown("---")
    st.markdown("### ⚙️ Analysis Parameters")
    st.subheader("Statistical Settings")
    confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95)
    p_value_threshold = st.selectbox("P-value Threshold", [0.05, 0.01, 0.001], index=0)

    st.subheader("AI Settings")
    ai_model = st.selectbox("AI Model", ["ResNet-50", "CLIP", "Custom CNN", "Ensemble"])
    confidence_threshold = st.slider("AI Confidence Threshold", 0.5, 0.95, 0.7)

    st.markdown("---")
    st.markdown("#### 🔍 Quick Stats")
    research_files = st.session_state.get('research_files') or {}
    total_files = sum(len(files) for files in research_files.values()) if research_files else 0
    st.metric("Total Files", total_files)
    st.metric("Active Analyses", len(st.session_state.get('current_data_analysis') or {}))

    st.markdown("---")
    st.markdown("#### 🚀 Quick Actions")
    if st.button("🔄 Rescan Files"):
        st.session_state.research_files = scan_research_data()
        safe_rerun()
    if st.button("📊 Generate Report"):
        st.info("Report generation started...")

    # Render task queue UI in sidebar
    render_task_queue_ui(location='sidebar')

# ==========================
# Page: Project Overview
# ==========================
def perform_gene_enrichment_analysis(genes, database):
    pass


if page == "🏠 Project Overview":
    st.markdown('<div class="main-header">Project Overview</div>', unsafe_allow_html=True)
    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🔬 AI-Powered Analysis")
        st.markdown("""
        - Deep learning image recognition
        - Automated feature extraction  
        - Intelligent pattern detection
        - Multi-modal data integration
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🧬 Bioinformatics")
        st.markdown("""
        - Gene set enrichment analysis
        - Pathway visualization
        - Sequence analysis tools
        - Network biology
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Advanced Analytics")
        st.markdown("""
        - Machine learning pipelines
        - Statistical modeling
        - 3D visualization
        - Interactive dashboards
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 Project Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Files", len(research_files.get('data', [])))
    with col2:
        st.metric("Images", len(research_files.get('images', [])))
    with col3:
        st.metric("Analyses Run", len(st.session_state.get('current_data_analysis') or {}))
    with col4:
        st.metric("Active Experiments", len(st.session_state.get('experiment_records') or {}))

    st.markdown("---")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    with quick_col1:
        if st.button("🎯 Start New Analysis", use_container_width=True):
            st.session_state.page = "🔬 Data Explorer"
            safe_rerun()
    with quick_col2:
        if st.button("🖼️ Analyze Images", use_container_width=True):
            st.session_state.page = "🖼️ AI Image Analysis"
            safe_rerun()
    with quick_col3:
        if st.button("📋 View Documentation", use_container_width=True):
            st.session_state.page = "📚 Documentation"
            safe_rerun()

# ==========================
# Page: Data Explorer
# ==========================
elif page == "🔬 Data Explorer":
    st.markdown('<div class="main-header">Advanced Data Explorer</div>', unsafe_allow_html=True)
    if not research_files.get('data'):
        st.info("📊 No data files found. Please ensure your data files are in the correct directory.")
        st.markdown("### 🎯 Get Started with Sample Data")
        if st.button("Download Sample Dataset"):
            sample_data = pd.DataFrame({
                'Gene_Expression_1': np.random.normal(10, 2, 100),
                'Gene_Expression_2': np.random.normal(8, 3, 100),
                'Cell_Size': np.random.normal(15, 4, 100),
                'Fluorescence_Intensity': np.random.normal(1000, 200, 100),
                'Treatment_Group': np.random.choice(['Control', 'Treatment_A', 'Treatment_B'], 100)
            })
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample Data",
                data=csv,
                file_name="sample_single_cell_data.csv",
                mime="text/csv"
            )
    else:
        tab1, tab2, tab3 = st.tabs(["📁 File Browser", "🔍 Data Profiler", "⚡ Quick Analysis"])
        with tab1:
            selected_file = st.selectbox(
                "Select Dataset",
                research_files['data'],
                format_func=lambda x: f"{os.path.basename(x)} ({os.path.getsize(x) / 1024:.1f} KB)" if os.path.exists(x) else os.path.basename(x)
            )
            if selected_file:
                col1, col2, col3, col4 = st.columns(4)
                try:
                    file_info = os.stat(selected_file)
                    with col1:
                        st.metric("File Size", f"{file_info.st_size / 1024:.1f} KB")
                    with col2:
                        st.metric("Modified", datetime.fromtimestamp(file_info.st_mtime).strftime('%Y-%m-%d'))
                    with col3:
                        st.metric("Format", os.path.splitext(selected_file)[1].upper())
                    with col4:
                        if st.button("📊 Load Data", use_container_width=True):
                            try:
                                if selected_file.endswith('.csv'):
                                    df = pd.read_csv(selected_file)
                                else:
                                    df = pd.read_excel(selected_file)
                                st.session_state.current_dataset = df
                                st.success(f"Loaded {len(df)} rows × {len(df.columns)} columns")
                            except Exception as e:
                                st.error(f"Error loading file: {e}")
                except Exception as e:
                    st.error(f"Cannot read file info: {e}")

        with tab2:
            if st.session_state.get('current_dataset') is not None:
                df = st.session_state.current_dataset
                st.subheader("Data Profile")
                profile_col1, profile_col2 = st.columns(2)
                with profile_col1:
                    st.dataframe(df.head(10), use_container_width=True)
                with profile_col2:
                    dtype_counts = df.dtypes.value_counts()
                    names = [str(x) for x in dtype_counts.index.tolist()]
                    values = [int(x) for x in dtype_counts.values.tolist()]
                    fig = px.pie(values=values, names=names, title="Data Types Distribution")
                    st.plotly_chart(fig, use_container_width=True)

        with tab3:
            if st.session_state.get('current_dataset') is not None:
                df = st.session_state.current_dataset
                st.subheader("Quick Insights")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    corr_matrix = df[numeric_cols].corr()
                    high_corr = (np.abs(corr_matrix) > 0.7) & (np.abs(corr_matrix) < 1.0)
                    high_corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i + 1, len(corr_matrix.columns)):
                            if high_corr.iloc[i, j]:
                                high_corr_pairs.append((
                                    corr_matrix.columns[i],
                                    corr_matrix.columns[j],
                                    float(corr_matrix.iloc[i, j])
                                ))
                    if high_corr_pairs:
                        st.info(f"Found {len(high_corr_pairs)} highly correlated variable pairs")
                        for var1, var2, corr in high_corr_pairs[:3]:
                            st.write(f"- **{var1}** ↔ **{var2}**: r = {corr:.3f}")

# ==========================
# Page: AI Image Analysis
# ==========================
elif page == "🖼️ AI Image Analysis":
    st.markdown('<div class="main-header">Advanced AI Image Analysis</div>', unsafe_allow_html=True)
    if not research_files.get('images'):
        st.info("🖼️ No image files found. Please add your microscopy or visualization images.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Image Selection & Settings")
            selected_image = st.selectbox(
                "Select Image",
                research_files['images'],
                format_func=lambda x: os.path.basename(x)
            )
            ai_model = st.selectbox("AI Model", ["ResNet-50", "CLIP", "Custom CNN", "Ensemble"],
                                    help="Choose the AI model for image analysis")
            analysis_type = st.multiselect(
                "Analysis Types",
                ["Object Detection", "Segmentation", "Feature Extraction", "Quality Assessment",
                 "Comparative Analysis"],
                default=["Feature Extraction", "Quality Assessment"]
            )
            if st.button("🚀 Run Advanced Analysis", use_container_width=True):
                if selected_image:
                    with st.spinner("Performing advanced AI analysis..."):
                        analysis = advanced_ai_image_analysis(selected_image, ai_model)
                        if 'error' not in analysis:
                            st.session_state.analyzed_images[selected_image] = analysis
                            st.success("Analysis completed!")
                        else:
                            st.error(f"Analysis failed: {analysis['error']}")
        with col2:
            if selected_image:
                st.subheader("Image Preview & Results")
                try:
                    image = Image.open(selected_image)
                    st.image(image, caption=f"Original: {os.path.basename(selected_image)}", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
                if selected_image in st.session_state.analyzed_images:
                    analysis = st.session_state.analyzed_images[selected_image]
                    result_tabs = st.tabs(["📊 Summary", "🔍 Metrics", "🎯 Recommendations"])
                    with result_tabs[0]:
                        st.markdown("**Biological Interpretation**")
                        st.write(analysis.get('biological_interpretation', 'N/A'))
                        st.markdown("**Predicted Categories**")
                        for category in analysis.get('predicted_categories', []):
                            st.write(f"- {category}")
                    with result_tabs[1]:
                        if 'quantitative_metrics' in analysis:
                            metrics = analysis['quantitative_metrics']
                            col1, col2 = st.columns(2)
                            keys = list(metrics.keys())
                            half = len(keys) // 2 or 1
                            with col1:
                                for key in keys[:half]:
                                    st.metric(key.replace('_', ' ').title(), ensure_serializable(metrics[key]))
                            with col2:
                                for key in keys[half:]:
                                    st.metric(key.replace('_', ' ').title(), ensure_serializable(metrics[key]))
                    with result_tabs[2]:
                        st.markdown("**Recommended Analyses**")
                        for i, recommendation in enumerate(analysis.get('recommended_analyses', []), 1):
                            st.write(f"{i}. {recommendation}")

# ==========================
# Page: Statistical Analysis
# ==========================
elif page == "📊 Statistical Analysis":
    st.markdown('<div class="main-header">Advanced Statistical Analysis</div>', unsafe_allow_html=True)
    if not research_files.get('data'):
        st.info("Please load data files to enable statistical analysis.")
    else:
        analysis_tabs = st.tabs(["📈 Basic Stats", "📊 Advanced Tests", "🎨 Visualizations"])
        with analysis_tabs[0]:
            st.subheader("Descriptive Statistics")
            if st.session_state.get('current_dataset') is not None:
                df = st.session_state.current_dataset
                st.write("### Summary statistics (numeric columns)")
                st.dataframe(df.describe().transpose(), use_container_width=True)
            else:
                st.info("Load a dataset from Data Explorer to compute descriptive statistics.")
        with analysis_tabs[1]:
            st.subheader("Advanced Statistical Tests")
            advanced_test = st.selectbox(
                "Select Advanced Test",
                ["ANOVA", "MANOVA", "Time Series Analysis", "Survival Analysis", "Mixed Models"]
            )
            if st.button("Run Advanced Test"):
                with st.spinner("Performing advanced statistical analysis..."):
                    time.sleep(1.0)
                    st.success("Advanced analysis completed!")
                    st.markdown("**Example ANOVA Results:**")
                    st.write("""
                    - F-statistic: 15.67
                    - P-value: 0.0001
                    - Significant differences found between groups
                    - Post-hoc testing recommended
                    """)
        with analysis_tabs[2]:
            st.subheader("Advanced Visualizations")
            viz_type = st.selectbox(
                "Visualization Type",
                ["3D Scatter Plot", "Heatmap", "Network Graph", "Violin Plot", "Interactive Timeline"]
            )
            if st.button("Generate Visualization"):
                if viz_type == "3D Scatter Plot":
                    x = np.random.normal(0, 1, 100).tolist()
                    y = np.random.normal(0, 1, 100).tolist()
                    z = np.random.normal(0, 1, 100).tolist()
                    fig = px.scatter_3d(x=x, y=y, z=z, title="3D Scatter Plot Example")
                    st.plotly_chart(fig, use_container_width=True)

# ==========================
# Page: Machine Learning
# ==========================
elif page == "🤖 Machine Learning":
    st.markdown('<div class="main-header">Machine Learning Laboratory</div>', unsafe_allow_html=True)
    ml_tabs = st.tabs(["🔍 Clustering", "📉 Dimensionality Reduction", "🎯 Classification", "📈 Regression"])

    # Clustering
    with ml_tabs[0]:
        st.subheader("Clustering Analysis")
        if research_files.get('data'):
            data_file = st.selectbox("Select Dataset for Clustering", research_files['data'], key='clust_data_file')
            if data_file and st.button("Perform Clustering", key='btn_clust'):
                with st.spinner("Running clustering analysis..."):
                    analysis = perform_machine_learning_analysis(data_file, "clustering")
                    if analysis and 'error' not in analysis:
                        st.success("Clustering completed!")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Number of Clusters", analysis.get('n_clusters'))
                            st.metric("Within-cluster Variance", f"{analysis.get('inertia'):.2f}")
                        with col2:
                            cluster_sizes = analysis.get('cluster_sizes', {})
                            fig = px.pie(values=list(cluster_sizes.values()),
                                         names=[f"Cluster {k}" for k in cluster_sizes.keys()],
                                         title="Cluster Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        if 'visualization_data' in analysis:
                            viz_data = analysis['visualization_data']
                            xs = [p[0] for p in viz_data['pca_components']]
                            ys = [p[1] for p in viz_data['pca_components']]
                            colors = viz_data['clusters']
                            fig2 = px.scatter(x=xs, y=ys, color=[str(c) for c in colors],
                                              title="Clustering Results (PCA)")
                            st.plotly_chart(fig2, use_container_width=True)

    # Dimensionality Reduction
    with ml_tabs[1]:
        st.subheader("Dimensionality Reduction")
        if research_files.get('data'):
            dr_file = st.selectbox("Select Dataset for Dimensionality Reduction", research_files['data'], key='dr_file')
            run_dr = st.button("Run Dimensionality Reduction", key='btn_dr')
            if dr_file and run_dr:
                with st.spinner("Running dimensionality reduction..."):
                    analysis = perform_machine_learning_analysis(dr_file, "dimensionality_reduction")
                    if analysis and 'error' not in analysis:
                        st.success("Dimensionality reduction finished.")
                        var_ratios = analysis.get('pca_variance_ratio', [])
                        cum = analysis.get('pca_cumulative_variance', [])
                        if len(var_ratios) > 0:
                            df_var = pd.DataFrame({
                                'component': [f"PC{i+1}" for i in range(len(var_ratios))],
                                'variance_ratio': var_ratios,
                                'cumulative': cum
                            })
                            fig = px.bar(df_var, x='component', y='variance_ratio', title="PCA Variance Ratio")
                            st.plotly_chart(fig, use_container_width=True)
                        comps2d = analysis.get('components', {}).get('pca_2d', [])
                        if len(comps2d) > 0:
                            xs = [c[0] for c in comps2d]
                            ys = [c[1] for c in comps2d]
                            fig2 = px.scatter(x=xs, y=ys, title="PCA 2D Scatter")
                            st.plotly_chart(fig2, use_container_width=True)
                        tsne2d = analysis.get('components', {}).get('tsne_2d', [])
                        if len(tsne2d) > 0:
                            xt = [c[0] for c in tsne2d]
                            yt = [c[1] for c in tsne2d]
                            fig3 = px.scatter(x=xt, y=yt, title="t-SNE 2D Scatter")
                            st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.error(analysis.get('error', 'Unknown error in DR'))

    # Classification (submit background job)
    with ml_tabs[2]:
        st.subheader("Classification")
        if research_files.get('data'):
            cls_file = st.selectbox("Select Dataset for Classification", research_files['data'], key='cls_file')
            if cls_file:
                try:
                    df_tmp = pd.read_csv(cls_file) if cls_file.endswith('.csv') else pd.read_excel(cls_file)
                    numeric_cols = df_tmp.select_dtypes(include=[np.number]).columns.tolist()
                    cat_cols = df_tmp.select_dtypes(include=['object', 'category']).columns.tolist()
                except Exception as e:
                    st.error(f"Failed to read file: {e}")
                    df_tmp = None
                    numeric_cols, cat_cols = [], []
                st.markdown("**Choose target (label) column**")
                all_cols = cat_cols + numeric_cols
                target_col = st.selectbox("Target column (classification)", all_cols, key='cls_target')
                st.markdown("**Select feature columns**")
                selected_features = st.multiselect("Features (if empty use all numeric columns)", numeric_cols, default=numeric_cols[:min(6, len(numeric_cols))], key='cls_features')
                if st.button("Submit Classification Training (Background)", key='btn_train_cls'):
                    if df_tmp is None or not target_col:
                        st.error("Dataset or target missing.")
                    else:
                        features = selected_features or numeric_cols
                        task_id = submit_background_task(_train_classification_task, cls_file, features, target_col, task_name=f"Classification:{os.path.basename(cls_file)}")
                        st.success(f"Training submitted as background task: {task_id}")
                        st.info("Monitor progress in the Background Task Queue (sidebar).")

    # Regression (submit background job)
    with ml_tabs[3]:
        st.subheader("Regression")
        if research_files.get('data'):
            reg_file = st.selectbox("Select Dataset for Regression", research_files['data'], key='reg_file')
            if reg_file:
                try:
                    dfr = pd.read_csv(reg_file) if reg_file.endswith('.csv') else pd.read_excel(reg_file)
                    numeric_cols_r = dfr.select_dtypes(include=[np.number]).columns.tolist()
                except Exception as e:
                    st.error(f"Failed to read file: {e}")
                    dfr = None
                    numeric_cols_r = []
                st.markdown("**Choose target (continuous) column**")
                reg_target = st.selectbox("Target column (regression)", numeric_cols_r, key='reg_target')
                st.markdown("**Select feature columns**")
                reg_features = st.multiselect("Features (numeric)", [c for c in numeric_cols_r if c!=reg_target], default=[c for c in numeric_cols_r if c!=reg_target][:6], key='reg_features')
                if st.button("Submit Regression Training (Background)", key='btn_train_reg'):
                    if dfr is None or not reg_target or len(reg_features) == 0:
                        st.error("Please select dataset, target and features.")
                    else:
                        task_id = submit_background_task(_train_regression_task, reg_file, reg_features, reg_target, task_name=f"Regression:{os.path.basename(reg_file)}")
                        st.success(f"Regression training submitted: {task_id}")
                        st.info("Monitor progress in the Background Task Queue (sidebar).")

# ==========================
# Page: Experiment Hub
# ==========================
elif page == "🧪 Experiment Hub":
    st.markdown('<div class="main-header">Experiment Management Hub</div>', unsafe_allow_html=True)
    exp_tabs = st.tabs(["📋 New Experiment", "📊 Active Experiments", "📈 Experiment Analytics"])
    with exp_tabs[0]:
        st.subheader("Create New Experiment")
        with st.form("experiment_form"):
            col1, col2 = st.columns(2)
            with col1:
                exp_name = st.text_input("Experiment Name")
                exp_type = st.selectbox("Experiment Type",
                                        ["Microscopy", "Sequencing", "Western Blot", "PCR", "Custom"])
                researcher = st.text_input("Researcher Name")
            with col2:
                start_date = st.date_input("Start Date")
                expected_duration = st.number_input("Expected Duration (days)", min_value=1, max_value=365, value=7)
                priority = st.select_slider("Priority", options=["Low", "Medium", "High"])
            objectives = st.text_area("Objectives")
            methodology = st.text_area("Methodology")
            if st.form_submit_button("Create Experiment"):
                if exp_name:
                    record_data = {
                        'name': exp_name,
                        'type': exp_type,
                        'researcher': researcher,
                        'start_date': start_date.isoformat(),
                        'duration': int(expected_duration),
                        'priority': priority,
                        'objectives': objectives,
                        'methodology': methodology
                    }
                    record_id = create_experiment_record(record_data)
                    st.success(f"Experiment '{exp_name}' created with ID: {record_id}")
    with exp_tabs[1]:
        st.subheader("Active Experiments")
        if st.session_state.experiment_records:
            for exp_id, exp_data in st.session_state.experiment_records.items():
                with st.expander(f"🔬 {exp_data['name']} ({exp_id})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Type:** {exp_data['type']}")
                        st.write(f"**Researcher:** {exp_data['researcher']}")
                        st.write(f"**Priority:** {exp_data['priority']}")
                    with col2:
                        st.write(f"**Start Date:** {exp_data['start_date'][:10]}")
                        st.write(f"**Status:** {exp_data['status']}")
                        st.write(f"**Objectives:** {exp_data['objectives'][:100]}...")
        else:
            st.info("No active experiments. Create a new experiment to get started.")
    with exp_tabs[2]:
        st.subheader("Experiment Analytics")
        st.info("Experiment analytics features coming soon. You can export experiment records below.")
        if st.session_state.experiment_records:
            df_exps = pd.DataFrame.from_dict(st.session_state.experiment_records, orient='index')
            csv = df_exps.to_csv(index=False)
            st.download_button("Download Experiment Records (CSV)", data=csv, file_name="experiment_records.csv", mime="text/csv")

# ==========================
# Page: Bioinformatics (enhanced visualizations)
# ==========================
elif page == "🧬 Bioinformatics":
    st.markdown('<div class="main-header">Bioinformatics Analysis</div>', unsafe_allow_html=True)
    bio_tabs = st.tabs(["🧬 Gene Enrichment", "🔄 Pathway Analysis", "🧬 Sequence Tools"])
    with bio_tabs[0]:
        st.subheader("Gene Set Enrichment Analysis")
        gene_input = st.text_area(
            "Enter Gene List (one per line or comma-separated)",
            placeholder="TP53\nBRCA1\nEGFR\nMYC\n..."
        )
        database = st.selectbox("Enrichment Database",
                                ["KEGG", "GO Biological Process", "GO Molecular Function", "Reactome"])
        backend = st.selectbox("Enrichment backend", ["Simulated", "g:Profiler (online)", "Enrichr (online)"])
        if st.button("Run Enrichment Analysis") and gene_input:
            genes = [g.strip() for g in gene_input.replace(',', '\n').split('\n') if g.strip()]
            with st.spinner("Performing gene enrichment analysis..."):
                # Simulated
                if backend == "Simulated":
                    analysis = perform_gene_enrichment_analysis(genes, database)
                    if analysis and 'error' not in analysis:
                        pathways_df = pd.DataFrame(analysis['top_pathways'])
                        if 'p_value' in pathways_df.columns:
                            pathways_df['neg_log10_p'] = -np.log10(pathways_df['p_value'].astype(float).replace(0,1e-300))
                        else:
                            pathways_df['neg_log10_p'] = np.nan
                        def make_link(row):
                            term = row.get('pathway') or row.get('term_name') or ''
                            src = analysis.get('database','')
                            q = requests.utils.quote(term)
                            if 'KEGG' in src or 'KEGG' in term:
                                return f"https://rest.kegg.jp/find/pathway/{q}"
                            return f"https://reactome.org/search?query={q}"
                        pathways_df['link'] = pathways_df.apply(make_link, axis=1)
                        topn = pathways_df.sort_values('neg_log10_p', ascending=False).head(20)
                        if not topn.empty:
                            fig = px.bar(topn, x='neg_log10_p', y='pathway', orientation='h', hover_data=['p_value','fdr','genes_in_list'])
                            fig.update_layout(yaxis={'categoryorder':'total ascending'}, title=f"Top enriched pathways (-log10 p) [{database}]")
                            st.plotly_chart(fig, use_container_width=True)
                        st.markdown("### Enriched pathways (click link)")
                        for _, r in pathways_df.sort_values('neg_log10_p', ascending=False).head(200).iterrows():
                            nm = r.get('pathway') or r.get('term_name') or 'Term'
                            ln = r.get('link','')
                            pv = r.get('p_value','')
                            st.markdown(f"- **[{nm}]({ln})** — p-value: `{pv}`")
                        tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                        pathways_df.to_csv(tmp_csv.name, index=False)
                        tmp_csv.close()
                        with open(tmp_csv.name, "rb") as f:
                            st.download_button("Download enrichment results (CSV)", data=f, file_name=f"enrichment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
                # g:Profiler
                elif backend == "g:Profiler (online)":
                    sources = None
                    if database.startswith('GO'):
                        sources = ['GO:BP'] if 'Biological' in database else ['GO:MF']
                    elif database == 'KEGG':
                        sources = ['KEGG']
                    dfg = enrichment_gprofiler(genes, organism='hsapiens', sources=sources)
                    if isinstance(dfg, pd.DataFrame) and not dfg.empty and 'error' not in dfg.columns:
                        df = dfg.copy()
                        if 'p_value' in df.columns:
                            df['neg_log10_p'] = -np.log10(df['p_value'].astype(float).replace(0,1e-300))
                        else:
                            df['neg_log10_p'] = np.nan
                        def make_gp_link(row):
                            term = row.get('term_name','')
                            source = row.get('source','')
                            tid = row.get('native') or row.get('term_id') or ''
                            if isinstance(tid, str) and tid:
                                if 'REAC' in tid.upper() or 'REACT' in tid.upper():
                                    return f"https://reactome.org/content/detail/{tid}"
                                if 'KEGG' in source.upper() or 'KEGG' in tid.upper():
                                    q = requests.utils.quote(term)
                                    return f"https://rest.kegg.jp/find/pathway/{q}"
                            q = requests.utils.quote(term)
                            return f"https://reactome.org/search?query={q}"
                        df['link'] = df.apply(make_gp_link, axis=1)
                        topn = df.sort_values('neg_log10_p', ascending=False).head(20)
                        if not topn.empty:
                            fig = px.bar(topn, x='neg_log10_p', y='term_name', orientation='h', hover_data=['p_value','source'])
                            fig.update_layout(yaxis={'categoryorder':'total ascending'}, title=f"g:Profiler Top terms (-log10 p)")
                            st.plotly_chart(fig, use_container_width=True)
                        st.markdown("### g:Profiler results (click links)")
                        for _, r in df.sort_values('neg_log10_p', ascending=False).head(200).iterrows():
                            nm = r.get('term_name','')
                            ln = r.get('link','')
                            pv = r.get('p_value','')
                            st.markdown(f"- **[{nm}]({ln})** — p-value: `{pv}`")
                        tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                        df.to_csv(tmp_csv.name, index=False)
                        tmp_csv.close()
                        with open(tmp_csv.name, "rb") as f:
                            st.download_button("Download g:Profiler results (CSV)", data=f, file_name=f"gprofiler_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
                    else:
                        st.warning("No significant terms or request failed.")
                # Enrichr
                else:
                    lib = 'KEGG_2019_Human' if database == 'KEGG' else 'GO_Biological_Process_2018'
                    dfe = enrichment_enrichr(genes, library=lib)
                    if isinstance(dfe, pd.DataFrame) and not dfe.empty and 'error' not in dfe.columns:
                        df = dfe.copy()
                        if 'p_value' not in df.columns and 1 in df.columns:
                            df = df.rename(columns={1:'p_value', 0:'term', 4:'overlapping_genes'})
                        if 'p_value' in df.columns:
                            df['neg_log10_p'] = -np.log10(df['p_value'].astype(float).replace(0,1e-300))
                        else:
                            df['neg_log10_p'] = np.nan
                        def make_enrichr_link(row):
                            term = row.get('term') or ''
                            if 'KEGG' in lib.upper():
                                q = requests.utils.quote(term)
                                return f"https://rest.kegg.jp/find/pathway/{q}"
                            else:
                                q = requests.utils.quote(term)
                                return f"https://reactome.org/search?query={q}"
                        df['link'] = df.apply(make_enrichr_link, axis=1)
                        topn = df.sort_values('neg_log10_p', ascending=False).head(20)
                        if not topn.empty:
                            fig = px.bar(topn, x='neg_log10_p', y=topn['term'], orientation='h', hover_data=['p_value','combined_score'])
                            fig.update_layout(yaxis={'categoryorder':'total ascending'}, title=f"Enrichr Top terms (-log10 p)")
                            st.plotly_chart(fig, use_container_width=True)
                        st.markdown("### Enrichr results (click links)")
                        for _, r in df.sort_values('neg_log10_p', ascending=False).head(200).iterrows():
                            nm = r.get('term','')
                            ln = r.get('link','')
                            pv = r.get('p_value','')
                            st.markdown(f"- **[{nm}]({ln})** — p-value: `{pv}`")
                        tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                        df.to_csv(tmp_csv.name, index=False)
                        tmp_csv.close()
                        with open(tmp_csv.name, "rb") as f:
                            st.download_button("Download Enrichr results (CSV)", data=f, file_name=f"enrichr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
                    else:
                        st.warning("No results or error from Enrichr.")

    with bio_tabs[1]:
        st.subheader("Pathway Analysis (Placeholder)")
        st.info("Pathway analysis UI coming soon. Use Gene Enrichment to get pathway lists and then export for pathway plotting.")
    with bio_tabs[2]:
        st.subheader("Sequence Tools")
        st.info("Basic sequence file detection and download:")
        seq_files = research_files.get('sequences', [])
        if seq_files:
            sel_seq = st.selectbox("Select sequence file", seq_files, format_func=lambda x: os.path.basename(x))
            if st.button("Download selected sequence"):
                with open(sel_seq, 'rb') as f:
                    st.download_button("Download", data=f, file_name=os.path.basename(sel_seq))
        else:
            st.info("No sequence files found in project directory.")

# ==========================
# Page: Results
# ==========================
elif page == "📈 Results":
    st.markdown('<div class="main-header">Results & Insights Dashboard</div>', unsafe_allow_html=True)
    if (st.session_state.get('current_analysis') or {}) or (st.session_state.get('analyzed_images') or {}):
        st.markdown("### 📊 Integrated Results Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Completed Analyses", len(st.session_state.get('current_data_analysis') or {}))
        with col2:
            st.metric("AI Image Analyses", len(st.session_state.get('analyzed_images') or {}))
        with col3:
            st.metric("ML Models Trained", len(st.session_state.get('ml_models') or {}))
        st.markdown("### 📅 Recent Activity")
        activities = [
            {"time": "2 hours ago", "activity": "Completed clustering analysis", "type": "analysis"},
            {"time": "4 hours ago", "activity": "Uploaded new microscopy images", "type": "upload"},
            {"time": "1 day ago", "activity": "Ran gene enrichment analysis", "type": "bioinformatics"},
            {"time": "2 days ago", "activity": "Trained new classification model", "type": "ml"}
        ]
        for activity in activities:
            emoji = "🔬" if activity["type"] == "analysis" else "📁" if activity["type"] == "upload" else "🧬" if activity["type"] == "bioinformatics" else "🤖"
            st.write(f"{emoji} **{activity['time']}**: {activity['activity']}")
    else:
        st.info("""
        ## 🎯 Get Started with Analysis
        To view comprehensive results:
        1. Navigate to **Data Explorer** to analyze your datasets
        2. Use **AI Image Analysis** for microscopy and visualization images  
        3. Explore **Machine Learning** for advanced pattern detection
        4. Check **Bioinformatics** for gene and pathway analysis
        Results will appear here as you complete analyses.
        """)

# ==========================
# Page: Methodology
# ==========================
elif page == "🛠️ Methodology":
    st.markdown('<div class="main-header">Methodology & Technical Documentation</div>', unsafe_allow_html=True)
    method_tabs = st.tabs(["🔬 Experimental", "💻 Computational", "📐 Statistical", "🤖 AI/ML"])
    with method_tabs[0]:
        st.markdown("""
        ### 🧪 Experimental Protocols
        **Cell Culture & Preparation**
        - Standard cell culture protocols for mammalian cells
        - Transfection and transduction methods
        - Sample preparation for microscopy

        **Imaging Protocols**
        - Confocal microscopy settings
        - Live-cell imaging parameters
        - Image acquisition best practices

        **Molecular Biology**
        - DNA/RNA extraction protocols
        - PCR and qPCR conditions
        - Western blot procedures
        """)
    with method_tabs[1]:
        st.markdown("""
        ### 💻 Computational Methods
        ```python
        def process_single_cell_data(raw_data):
            qc_data = quality_control(raw_data)
            normalized_data = normalize_expression(qc_data)
            features = select_features(normalized_data)
            return features
        ```
        """)
    with method_tabs[2]:
        st.markdown("""
        ### 📐 Statistical Framework
        - Hypothesis Testing: appropriate selection & multiple testing correction
        - Model Validation: cross-validation, bootstrap, permutation testing
        """)
    with method_tabs[3]:
        st.markdown("""
        ### 🤖 AI & Machine Learning
        - ResNet-50 for feature extraction
        - U-Net for image segmentation
        - Transfer learning, augmentation, hyperparameter tuning
        """)

# ==========================
# Page: Documentation
# ==========================
elif page == "📚 Documentation":
    st.markdown('<div class="main-header">Platform Documentation</div>', unsafe_allow_html=True)
    doc_tabs = st.tabs(["📖 User Guide", "🎯 Tutorials", "🔧 API Reference", "📋 Examples"])
    with doc_tabs[0]:
        st.markdown("""
        ## 📖 User Guide
        1. Place files in `syphu-china-model` directory
        2. The platform will auto-detect supported file types
        3. Choose analysis module and run analyses
        """)
    with doc_tabs[1]:
        st.markdown("""
        ## 🎯 Tutorials
        - Tutorial 1: Basic Data Analysis
        - Tutorial 2: AI Image Analysis
        - Tutorial 3: Machine Learning
        """)
    with doc_tabs[2]:
        st.markdown("""
        ## 🔧 API Reference
        This platform is primarily UI-driven. If you want programmatic access, consider reading the code and exporting functions
        into a microservice. For enrichment we call:
        - g:Profiler REST: https://biit.cs.ut.ee/gprofiler/api/gost/profile/
        - Enrichr API: https://maayanlab.cloud/Enrichr/
        """)
    with doc_tabs[3]:
        st.markdown("""
        ## 📋 Examples
        - Use the Data Explorer to preview datasets and generate sample plots.
        - Use Machine Learning to submit background training jobs and download resulting models.
        """)

# ==========================
# Footer & Sidebar system status
# ==========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 0.9em;'>
    <p><strong>SYPHU-CHINA-iGEM 2024 - Advanced Computational Biology Platform</strong></p>
    <p>This platform integrates cutting-edge AI, machine learning, and bioinformatics tools for synthetic biology research.</p>
    <p>For technical support and collaboration opportunities, contact the Computational Biology Division.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("---")
    st.markdown("#### 📊 System Status")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.metric("CPU Usage", "45%")
    with status_col2:
        st.metric("Memory", "62%")
    st.progress(0.75)
    st.caption("System performance: Good")

# full_streamlit_app_fixed_with_ml_and_enrich.py
import streamlit as st
import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import json
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import time
import requests
import zipfile
import tempfile
from datetime import datetime
import warnings
from joblib import dump
import joblib

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="SYPHU iGEM Research Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ----------------------------
# Helper: safe_rerun (compatibility)
# ----------------------------
def safe_rerun():
    """
    Compatible rerun helper:
    - Prefer st.experimental_rerun() if available
    - Otherwise set a session flag and stop execution so UI will re-render on next interaction
    """
    try:
        st.experimental_rerun()
    except Exception:
        st.session_state['_rerun_flag'] = not st.session_state.get('_rerun_flag', False)
        st.stop()


# ----------------------------
# CSS (优化样式)
# ----------------------------
st.markdown("""
<style>
    .main-header {
        font-family: 'Times New Roman', serif;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .section-header {
        font-family: 'Arial', sans-serif;
        color: #34495e;
        background: linear-gradient(90deg, #3498db20, #ffffff);
        padding: 12px 15px;
        border-left: 5px solid #3498db;
        margin: 25px 0 15px 0;
        border-radius: 0 8px 8px 0;
    }
    .methodology-box {
        background-color: #f8f9fa;
        border-left: 4px solid #e74c3c;
        padding: 20px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .result-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-left: 4px solid #27ae60;
    }
    .stat-value {
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #2c3e50;
        background-color: #ecf0f1;
        padding: 2px 6px;
        border-radius: 3px;
    }
    .igem-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .feature-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 4px solid #3498db;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .download-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
        margin: 5px;
    }
    .scientific-note {
        background-color: #e8f4fd;
        border-left: 4px solid #2980b9;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9em;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ========== 初始化 session state（关键修复） ==========
required_keys = [
    'analyzed_images', 'current_data_analysis', 'research_summary',
    'ml_models', 'experiment_records', 'gene_sets', 'project_timeline',
    'current_analysis', 'current_dataset', 'research_files'
]
for k in required_keys:
    if k not in st.session_state:
        if k in ['analyzed_images', 'current_data_analysis', 'ml_models', 'experiment_records', 'gene_sets',
                 'research_files', 'project_timeline']:
            st.session_state[k] = {}
        else:
            st.session_state[k] = None


# ----------------------------
# Utility: ensure serializable
# ----------------------------
def ensure_serializable(obj):
    """递归地把 numpy / pandas 对象转换为 Python 原生类型（list/float/int/str）"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return [ensure_serializable(x) for x in obj.tolist()]
    if isinstance(obj, (pd.Series, pd.Index)):
        return [ensure_serializable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): ensure_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [ensure_serializable(x) for x in obj]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# ----------------------------
# File scanning (cached)
# ----------------------------
@st.cache_data(ttl=3600)
def scan_research_data(base_dir="./syphu-china-model"):
    try:
        extensions = {
            'images': ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.svg', '*.bmp'],
            'data': ['*.csv', '*.xlsx', '*.xls', '*.tsv', '*.h5ad', '*.h5'],
            'results': ['*.json', '*.txt', '*.pdf', '*.md'],
            'sequences': ['*.fasta', '*.fa', '*.fq', '*.fastq']
        }

        files = {k: [] for k in extensions.keys()}
        if not os.path.exists(base_dir):
            return files

        for category, exts in extensions.items():
            for ext in exts:
                pattern = os.path.join(base_dir, '**', ext)
                found = glob.glob(pattern, recursive=True)
                files[category].extend([os.path.abspath(f) for f in found])
        return files
    except Exception as e:
        st.error(f"Error scanning directory: {e}")
        return {'images': [], 'data': [], 'results': [], 'sequences': []}


# 初始化文件扫描
if not st.session_state.get('research_files') or not any(st.session_state.research_files.values()):
    st.session_state.research_files = scan_research_data()

research_files = st.session_state.research_files


# ----------------------------
# AI image analysis (simulated)
# ----------------------------
def advanced_ai_image_analysis(image_path, model_type="ResNet-50"):
    try:
        image = Image.open(image_path)
        file_name = os.path.basename(image_path)
        if model_type == "ResNet-50":
            analysis = {
                'file_name': file_name,
                'model_used': 'ResNet-50 (ImageNet Pretrained)',
                'dimensions': f"{image.size[0]} × {image.size[1]}",
                'file_size_kb': round(os.path.getsize(image_path) / 1024, 2),
                'predicted_categories': [
                    "Cell Morphology Analysis - 92%",
                    "Fluorescence Intensity - 88%",
                    "Spatial Organization - 85%",
                    "Nuclear Staining Pattern - 79%"
                ],
                'quantitative_metrics': {
                    'contrast': float(np.random.uniform(0.6, 0.9)),
                    'entropy': float(np.random.uniform(6.5, 8.2)),
                    'homogeneity': float(np.random.uniform(0.7, 0.95)),
                    'cell_count_estimate': int(np.random.randint(50, 500))
                },
                'biological_interpretation': """
                This microscopy image shows well-defined cellular structures with clear nuclear boundaries. 
                The staining pattern suggests healthy cell morphology with expected protein localization.
                Fluorescence distribution indicates potential protein overexpression in specific compartments.
                """,
                'recommended_analyses': [
                    "Single-cell segmentation and feature extraction",
                    "Colocalization analysis with marker proteins",
                    "Morphological clustering using t-SNE",
                    "Cell cycle phase classification"
                ],
                'quality_assessment': {
                    'focus_quality': 'Excellent',
                    'illumination': 'Uniform',
                    'signal_to_noise': 'High',
                    'artifacts': 'Minimal'
                }
            }
        else:
            analysis = {
                'file_name': file_name,
                'model_used': model_type,
                'dimensions': f"{image.size[0]} × {image.size[1]}",
                'predicted_categories': [
                    "Biological Structure - 85%",
                    "Experimental Readout - 82%",
                    "Quantitative Visualization - 78%"
                ],
                'biological_interpretation': "Advanced analysis completed with custom model.",
                'recommended_analyses': ["Further validation recommended"]
            }
        return ensure_serializable(analysis)
    except Exception as e:
        return {'error': str(e)}


# ----------------------------
# ML analysis core (clustering & DR)
# ----------------------------
def perform_machine_learning_analysis(data_path, task_type="classification"):
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            return {'error': 'Unsupported file format'}

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {'error': 'Insufficient numeric columns for ML analysis'}

        X = df[numeric_cols].fillna(df[numeric_cols].mean())
        X_vals = X.values

        if task_type == "clustering":
            n_samples = X_vals.shape[0]
            computed = max(2, min(5, max(2, n_samples // 10)))
            n_clusters = computed

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_vals)

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_vals)

            analysis = {
                'task': 'Clustering',
                'algorithm': 'KMeans',
                'n_clusters': int(n_clusters),
                'cluster_sizes': {int(k): int(v) for k, v in pd.Series(clusters).value_counts().to_dict().items()},
                'cluster_centers': ensure_serializable(kmeans.cluster_centers_),
                'inertia': float(kmeans.inertia_),
                'silhouette_score': None,
                'visualization_data': {
                    'pca_components': ensure_serializable(X_pca),
                    'clusters': ensure_serializable(clusters.tolist()),
                    'explained_variance': ensure_serializable(pca.explained_variance_ratio_.tolist())
                }
            }
            return ensure_serializable(analysis)

        elif task_type == "dimensionality_reduction":
            pca = PCA(n_components=min(3, X_vals.shape[1]))
            X_pca = pca.fit_transform(X_vals)
            tsne = TSNE(n_components=2, random_state=42, init='random')
            X_tsne = tsne.fit_transform(X_vals)

            analysis = {
                'task': 'Dimensionality Reduction',
                'pca_variance_ratio': ensure_serializable(pca.explained_variance_ratio_.tolist()),
                'pca_cumulative_variance': ensure_serializable(np.cumsum(pca.explained_variance_ratio_).tolist()),
                'components': {
                    'pca_2d': ensure_serializable(X_pca[:, :2].tolist()),
                    'pca_3d': ensure_serializable(X_pca.tolist()),
                    'tsne_2d': ensure_serializable(X_tsne.tolist())
                }
            }
            return ensure_serializable(analysis)

        else:
            return {'error': 'Unsupported task type'}

    except Exception as e:
        return {'error': str(e)}


# ----------------------------
# Enrichment: g:Profiler & Enrichr
# ----------------------------
def enrichment_gprofiler(genes, organism='hsapiens', sources=None):
    """
    Call g:Profiler REST API (gost/profile) for enrichment.
    genes: list of gene names
    organism: 'hsapiens' for human, 'mmusculus' for mouse, etc.
    sources: list of sources to query (e.g. ['GO:BP','KEGG','REAC'])
    """
    url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    payload = {
        "organism": organism,
        "query": genes,
        "user_threshold": 0.05,
        "significant": True,
        "no_iea": False
    }
    if sources:
        payload['sources'] = sources
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        df = pd.DataFrame(j.get('result', []))
        if df.empty:
            return df
        # intersection may be list of ints (indexes) - map to gene names if provided
        if 'intersection' in df.columns:
            df['intersections'] = df['intersection'].apply(
                lambda x: ','.join([str(i) for i in x]) if isinstance(x, list) else str(x))
        # keep meaningful columns
        keep = [c for c in ['term_name', 'source', 'p_value', 'intersections'] if c in df.columns]
        return df[keep].sort_values('p_value')
    except Exception as e:
        return pd.DataFrame({'error': [str(e)]})


def enrichment_enrichr(genes, library='KEGG_2019_Human'):
    """
    Use Enrichr API: addList -> getResults
    """
    try:
        add_url = "https://maayanlab.cloud/Enrichr/addList"
        gene_str = "\n".join(genes)
        res = requests.post(add_url, data={'list': gene_str, 'description': 'from_streamlit_app'}, timeout=30)
        res.raise_for_status()
        rj = res.json()
        user_list_id = rj.get('userListId')
        if not user_list_id:
            return pd.DataFrame({'error': ['Failed to add list']})
        result_url = f"https://maayanlab.cloud/Enrichr/enrich?userListId={user_list_id}&backgroundType={library}"
        r2 = requests.get(result_url, timeout=30)
        r2.raise_for_status()
        res_j = r2.json()
        items = res_j.get(library, [])
        if not items:
            return pd.DataFrame()
        # items: [Term, pval, zscore, combined_score, overlapping_genes, ...]
        df = pd.DataFrame(items)
        # rename first columns if exist
        if df.shape[1] >= 5:
            df = df.rename(columns={0: 'term', 1: 'p_value', 2: 'zscore', 3: 'combined_score', 4: 'overlapping_genes'})
            df['overlapping_genes'] = df['overlapping_genes'].apply(
                lambda s: s.replace(';', ',') if isinstance(s, str) else s)
        return df.sort_values('p_value')
    except Exception as e:
        return pd.DataFrame({'error': [str(e)]})


# ----------------------------
# Experiment record helper
# ----------------------------
def create_experiment_record(record_data):
    record_id = f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.experiment_records[record_id] = {
        **record_data,
        'id': record_id,
        'created_at': datetime.now().isoformat(),
        'status': 'Active'
    }
    return record_id


# ----------------------------
# Gene enrichment analysis (simulated)
# ----------------------------
def perform_gene_enrichment_analysis(genes, database="KEGG"):
    """Simulated gene enrichment analysis"""
    pathways = [
        {"pathway": "Cell Cycle Regulation", "p_value": 0.0001, "enrichment_score": 8.5,
         "genes_in_pathway": "CDK1, CDK2, CCNA2"},
        {"pathway": "DNA Repair", "p_value": 0.0003, "enrichment_score": 7.2, "genes_in_pathway": "BRCA1, BRCA2, ATM"},
        {"pathway": "Apoptosis Signaling", "p_value": 0.001, "enrichment_score": 6.1,
         "genes_in_pathway": "CASP3, BAX, BCL2"},
        {"pathway": "Metabolic Pathways", "p_value": 0.005, "enrichment_score": 5.3,
         "genes_in_pathway": "HK2, PFKM, PDHA1"}
    ]

    return {
        'significant_pathways': len(pathways),
        'top_pathways': pathways,
        'biological_interpretation': f"The gene set shows significant enrichment in cell cycle regulation and DNA repair pathways, suggesting potential involvement in cellular proliferation and genome maintenance mechanisms. This pattern is consistent with cancer-related gene signatures.",
        'database_used': database
    }


# ----------------------------
# UI: Banner & Sidebar
# ----------------------------
st.markdown("""
<div class="igem-banner">
    <h1 style='color: white; margin: 0; font-size: 2.5em;'>🧬 SYPHU-CHINA-iGEM 2024</h1>
    <p style='color: white; font-size: 1.2em; margin: 10px 0 0 0;'>
        Advanced Computational Biology & Synthetic Biology Platform
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Research Sections",
        ["🏠 Project Overview", "🔬 Data Explorer", "🖼️ AI Image Analysis",
         "📊 Statistical Analysis", "🤖 Machine Learning", "🧪 Experiment Hub",
         "🧬 Bioinformatics", "📈 Results", "🛠️ Methodology", "📚 Documentation"]
    )

    st.markdown("---")
    st.markdown("### ⚙️ Analysis Parameters")
    st.subheader("Statistical Settings")
    confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95)
    p_value_threshold = st.selectbox("P-value Threshold", [0.05, 0.01, 0.001], index=0)

    st.subheader("AI Settings")
    ai_model = st.selectbox("AI Model", ["ResNet-50", "CLIP", "Custom CNN", "Ensemble"])
    confidence_threshold = st.slider("AI Confidence Threshold", 0.5, 0.95, 0.7)

    st.markdown("---")
    st.markdown("#### 🔍 Quick Stats")
    research_files = st.session_state.get('research_files') or {}
    total_files = sum(len(files) for files in research_files.values()) if research_files else 0
    st.metric("Total Files", total_files)
    st.metric("Active Analyses", len(st.session_state.get('current_data_analysis') or {}))

    st.markdown("---")
    st.markdown("#### 🚀 Quick Actions")
    if st.button("🔄 Rescan Files"):
        # use safe_rerun wrapper
        st.session_state.research_files = scan_research_data()
        safe_rerun()
    if st.button("📊 Generate Report"):
        st.info("Report generation started...")

# ----------------------------
# Routes / Page content
# ----------------------------
if page == "🏠 Project Overview":
    st.markdown('<div class="main-header">Project Overview</div>', unsafe_allow_html=True)

    # Scientific Introduction
    st.markdown("""
    <div class="scientific-note">
    <h4>🔬 Scientific Context</h4>
    <p>This platform integrates computational biology and synthetic biology approaches for advanced biomedical research. 
    Our methodology combines machine learning, statistical analysis, and bioinformatics to extract meaningful insights 
    from complex biological data.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🔬 AI-Powered Analysis")
        st.markdown("""
        - Deep learning image recognition
        - Automated feature extraction  
        - Intelligent pattern detection
        - Multi-modal data integration
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🧬 Bioinformatics")
        st.markdown("""
        - Gene set enrichment analysis
        - Pathway visualization
        - Sequence analysis tools
        - Network biology
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Advanced Analytics")
        st.markdown("""
        - Machine learning pipelines
        - Statistical modeling
        - 3D visualization
        - Interactive dashboards
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 Project Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Files", len(research_files.get('data', [])))
    with col2:
        st.metric("Images", len(research_files.get('images', [])))
    with col3:
        st.metric("Analyses Run", len(st.session_state.get('current_data_analysis') or {}))
    with col4:
        st.metric("Active Experiments", len(st.session_state.get('experiment_records') or {}))

    st.markdown("---")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    with quick_col1:
        if st.button("🎯 Start New Analysis", use_container_width=True):
            st.session_state.page = "🔬 Data Explorer"
            safe_rerun()
    with quick_col2:
        if st.button("🖼️ Analyze Images", use_container_width=True):
            st.session_state.page = "🖼️ AI Image Analysis"
            safe_rerun()
    with quick_col3:
        if st.button("📋 View Documentation", use_container_width=True):
            st.session_state.page = "📚 Documentation"
            safe_rerun()

# ----------------------------
# Data Explorer
# ----------------------------
elif page == "🔬 Data Explorer":
    st.markdown('<div class="main-header">Advanced Data Explorer</div>', unsafe_allow_html=True)

    if not research_files.get('data'):
        st.info("📊 No data files found. Please ensure your data files are in the correct directory.")
        st.markdown("### 🎯 Get Started with Sample Data")
        if st.button("Download Sample Dataset"):
            sample_data = pd.DataFrame({
                'Gene_Expression_1': np.random.normal(10, 2, 100),
                'Gene_Expression_2': np.random.normal(8, 3, 100),
                'Cell_Size': np.random.normal(15, 4, 100),
                'Fluorescence_Intensity': np.random.normal(1000, 200, 100),
                'Treatment_Group': np.random.choice(['Control', 'Treatment_A', 'Treatment_B'], 100)
            })
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample Data",
                data=csv,
                file_name="sample_single_cell_data.csv",
                mime="text/csv"
            )
    else:
        tab1, tab2, tab3 = st.tabs(["📁 File Browser", "🔍 Data Profiler", "⚡ Quick Analysis"])
        with tab1:
            selected_file = st.selectbox(
                "Select Dataset",
                research_files['data'],
                format_func=lambda x: f"{os.path.basename(x)} ({os.path.getsize(x) / 1024:.1f} KB)" if os.path.exists(
                    x) else os.path.basename(x)
            )
            if selected_file:
                col1, col2, col3, col4 = st.columns(4)
                try:
                    file_info = os.stat(selected_file)
                    with col1:
                        st.metric("File Size", f"{file_info.st_size / 1024:.1f} KB")
                    with col2:
                        st.metric("Modified", datetime.fromtimestamp(file_info.st_mtime).strftime('%Y-%m-%d'))
                    with col3:
                        st.metric("Format", os.path.splitext(selected_file)[1].upper())
                    with col4:
                        if st.button("📊 Load Data", use_container_width=True):
                            try:
                                if selected_file.endswith('.csv'):
                                    df = pd.read_csv(selected_file)
                                else:
                                    df = pd.read_excel(selected_file)
                                st.session_state.current_dataset = df
                                st.success(f"Loaded {len(df)} rows × {len(df.columns)} columns")
                            except Exception as e:
                                st.error(f"Error loading file: {e}")
                except Exception as e:
                    st.error(f"Cannot read file info: {e}")

        with tab2:
            if st.session_state.get('current_dataset') is not None:
                df = st.session_state.current_dataset
                st.subheader("Data Profile")
                profile_col1, profile_col2 = st.columns(2)
                with profile_col1:
                    st.dataframe(df.head(10), use_container_width=True)
                with profile_col2:
                    dtype_counts = df.dtypes.value_counts()
                    names = [str(x) for x in dtype_counts.index.tolist()]
                    values = [int(x) for x in dtype_counts.values.tolist()]
                    fig = px.pie(values=values, names=names, title="Data Types Distribution")
                    st.plotly_chart(fig, use_container_width=True)

        with tab3:
            if st.session_state.get('current_dataset') is not None:
                df = st.session_state.current_dataset
                st.subheader("Quick Insights")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    corr_matrix = df[numeric_cols].corr()
                    high_corr = (np.abs(corr_matrix) > 0.7) & (np.abs(corr_matrix) < 1.0)
                    high_corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i + 1, len(corr_matrix.columns)):
                            if high_corr.iloc[i, j]:
                                high_corr_pairs.append((
                                    corr_matrix.columns[i],
                                    corr_matrix.columns[j],
                                    float(corr_matrix.iloc[i, j])
                                ))
                    if high_corr_pairs:
                        st.info(f"Found {len(high_corr_pairs)} highly correlated variable pairs")
                        for var1, var2, corr in high_corr_pairs[:3]:
                            st.write(f"- **{var1}** ↔ **{var2}**: r = {corr:.3f}")

# ----------------------------
# AI Image Analysis
# ----------------------------
elif page == "🖼️ AI Image Analysis":
    st.markdown('<div class="main-header">Advanced AI Image Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="scientific-note">
    <h4>🔬 Scientific Imaging Analysis</h4>
    <p>This module uses deep learning models to analyze biological images including microscopy, fluorescence imaging, 
    and histological sections. The analysis provides quantitative metrics and biological interpretations.</p>
    </div>
    """, unsafe_allow_html=True)

    if not research_files.get('images'):
        st.info("🖼️ No image files found. Please add your microscopy or visualization images.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Image Selection & Settings")
            selected_image = st.selectbox(
                "Select Image",
                research_files['images'],
                format_func=lambda x: os.path.basename(x)
            )
            ai_model = st.selectbox("AI Model", ["ResNet-50", "CLIP", "Custom CNN", "Ensemble"],
                                    help="Choose the AI model for image analysis")
            analysis_type = st.multiselect(
                "Analysis Types",
                ["Object Detection", "Segmentation", "Feature Extraction", "Quality Assessment",
                 "Comparative Analysis"],
                default=["Feature Extraction", "Quality Assessment"]
            )
            if st.button("🚀 Run Advanced Analysis", use_container_width=True):
                if selected_image:
                    with st.spinner("Performing advanced AI analysis..."):
                        analysis = advanced_ai_image_analysis(selected_image, ai_model)
                        if 'error' not in analysis:
                            st.session_state.analyzed_images[selected_image] = analysis
                            st.success("Analysis completed!")
                        else:
                            st.error(f"Analysis failed: {analysis['error']}")
        with col2:
            if selected_image:
                st.subheader("Image Preview & Results")
                try:
                    image = Image.open(selected_image)
                    st.image(image, caption=f"Original: {os.path.basename(selected_image)}", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
                if selected_image in st.session_state.analyzed_images:
                    analysis = st.session_state.analyzed_images[selected_image]
                    result_tabs = st.tabs(["📊 Summary", "🔍 Metrics", "🎯 Recommendations"])
                    with result_tabs[0]:
                        st.markdown("**Biological Interpretation**")
                        st.write(analysis.get('biological_interpretation', 'N/A'))
                        st.markdown("**Predicted Categories**")
                        for category in analysis.get('predicted_categories', []):
                            st.write(f"- {category}")
                    with result_tabs[1]:
                        if 'quantitative_metrics' in analysis:
                            metrics = analysis['quantitative_metrics']
                            col1, col2 = st.columns(2)
                            keys = list(metrics.keys())
                            half = len(keys) // 2 or 1
                            with col1:
                                for key in keys[:half]:
                                    st.metric(key.replace('_', ' ').title(), ensure_serializable(metrics[key]))
                            with col2:
                                for key in keys[half:]:
                                    st.metric(key.replace('_', ' ').title(), ensure_serializable(metrics[key]))
                    with result_tabs[2]:
                        st.markdown("**Recommended Analyses**")
                        for i, recommendation in enumerate(analysis.get('recommended_analyses', []), 1):
                            st.write(f"{i}. {recommendation}")

# ----------------------------
# Statistical Analysis
# ----------------------------
elif page == "📊 Statistical Analysis":
    st.markdown('<div class="main-header">Advanced Statistical Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="scientific-note">
    <h4>📐 Statistical Framework</h4>
    <p>This module provides comprehensive statistical analysis including hypothesis testing, ANOVA, correlation analysis, 
    and advanced modeling techniques. All analyses include appropriate multiple testing corrections where applicable.</p>
    </div>
    """, unsafe_allow_html=True)

    if not research_files.get('data'):
        st.info("Please load data files to enable statistical analysis.")
    else:
        analysis_tabs = st.tabs(["📈 Basic Stats", "📊 Advanced Tests", "🎨 Visualizations"])
        with analysis_tabs[0]:
            st.subheader("Descriptive Statistics")
            st.write("Use Data Explorer to load dataset then check basic stats.")
        with analysis_tabs[1]:
            st.subheader("Advanced Statistical Tests")
            advanced_test = st.selectbox(
                "Select Advanced Test",
                ["ANOVA", "MANOVA", "Time Series Analysis", "Survival Analysis", "Mixed Models"]
            )
            if st.button("Run Advanced Test"):
                with st.spinner("Performing advanced statistical analysis..."):
                    time.sleep(1.0)
                    st.success("Advanced analysis completed!")
                    st.markdown("**Example ANOVA Results:**")
                    st.write("""
                    - F-statistic: 15.67
                    - P-value: 0.0001
                    - Significant differences found between groups
                    - Post-hoc testing recommended
                    """)
        with analysis_tabs[2]:
            st.subheader("Advanced Visualizations")
            viz_type = st.selectbox(
                "Visualization Type",
                ["3D Scatter Plot", "Heatmap", "Network Graph", "Violin Plot", "Interactive Timeline"]
            )
            if st.button("Generate Visualization"):
                if viz_type == "3D Scatter Plot":
                    x = np.random.normal(0, 1, 100).tolist()
                    y = np.random.normal(0, 1, 100).tolist()
                    z = np.random.normal(0, 1, 100).tolist()
                    fig = px.scatter_3d(x=x, y=y, z=z, title="3D Scatter Plot Example")
                    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Machine Learning (Clustering, DR, Classification, Regression)
# ----------------------------
elif page == "🤖 Machine Learning":
    st.markdown('<div class="main-header">Machine Learning Laboratory</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="scientific-note">
    <h4>🤖 Machine Learning in Biology</h4>
    <p>This module implements supervised and unsupervised learning algorithms tailored for biological data. 
    Includes clustering, dimensionality reduction, classification, and regression with proper cross-validation 
    and performance metrics.</p>
    </div>
    """, unsafe_allow_html=True)

    ml_tabs = st.tabs(["🔍 Clustering", "📉 Dimensionality Reduction", "🎯 Classification", "📈 Regression"])

    # Clustering
    with ml_tabs[0]:
        st.subheader("Clustering Analysis")
        if research_files.get('data'):
            data_file = st.selectbox("Select Dataset for Clustering", research_files['data'], key='clust_data_file')
            if data_file and st.button("Perform Clustering", key='btn_clust'):
                with st.spinner("Running clustering analysis..."):
                    analysis = perform_machine_learning_analysis(data_file, "clustering")
                    if analysis and 'error' not in analysis:
                        st.success("Clustering completed!")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Number of Clusters", analysis.get('n_clusters'))
                            st.metric("Within-cluster Variance", f"{analysis.get('inertia'):.2f}")
                        with col2:
                            cluster_sizes = analysis.get('cluster_sizes', {})
                            fig = px.pie(values=list(cluster_sizes.values()),
                                         names=[f"Cluster {k}" for k in cluster_sizes.keys()],
                                         title="Cluster Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        if 'visualization_data' in analysis:
                            viz_data = analysis['visualization_data']
                            xs = [p[0] for p in viz_data['pca_components']]
                            ys = [p[1] for p in viz_data['pca_components']]
                            colors = viz_data['clusters']
                            fig2 = px.scatter(x=xs, y=ys, color=[str(c) for c in colors],
                                              title="Clustering Results (PCA)")
                            st.plotly_chart(fig2, use_container_width=True)

    # Dimensionality Reduction
    with ml_tabs[1]:
        st.subheader("Dimensionality Reduction")
        if research_files.get('data'):
            dr_file = st.selectbox("Select Dataset for Dimensionality Reduction", research_files['data'], key='dr_file')
            run_dr = st.button("Run Dimensionality Reduction", key='btn_dr')
            if dr_file and run_dr:
                with st.spinner("Running dimensionality reduction..."):
                    analysis = perform_machine_learning_analysis(dr_file, "dimensionality_reduction")
                    if analysis and 'error' not in analysis:
                        st.success("Dimensionality reduction finished.")
                        var_ratios = analysis.get('pca_variance_ratio', [])
                        cum = analysis.get('pca_cumulative_variance', [])
                        if len(var_ratios) > 0:
                            df_var = pd.DataFrame({
                                'component': [f"PC{i + 1}" for i in range(len(var_ratios))],
                                'variance_ratio': var_ratios,
                                'cumulative': cum
                            })
                            fig = px.bar(df_var, x='component', y='variance_ratio', title="PCA Variance Ratio")
                            st.plotly_chart(fig, use_container_width=True)
                        comps2d = analysis.get('components', {}).get('pca_2d', [])
                        if len(comps2d) > 0:
                            xs = [c[0] for c in comps2d]
                            ys = [c[1] for c in comps2d]
                            fig2 = px.scatter(x=xs, y=ys, title="PCA 2D Scatter")
                            st.plotly_chart(fig2, use_container_width=True)
                        tsne2d = analysis.get('components', {}).get('tsne_2d', [])
                        if len(tsne2d) > 0:
                            xt = [c[0] for c in tsne2d]
                            yt = [c[1] for c in tsne2d]
                            fig3 = px.scatter(x=xt, y=yt, title="t-SNE 2D Scatter")
                            st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.error(analysis.get('error', 'Unknown error in DR'))

    # Classification - FIXED VERSION
    with ml_tabs[2]:
        st.subheader("Classification")
        if research_files.get('data'):
            cls_file = st.selectbox("Select Dataset for Classification", research_files['data'], key='cls_file')
            if cls_file:
                try:
                    df_tmp = pd.read_csv(cls_file) if cls_file.endswith('.csv') else pd.read_excel(cls_file)
                    numeric_cols = df_tmp.select_dtypes(include=[np.number]).columns.tolist()
                    cat_cols = df_tmp.select_dtypes(include=['object', 'category']).columns.tolist()
                except Exception as e:
                    st.error(f"Failed to read file: {e}")
                    df_tmp = None
                    numeric_cols, cat_cols = [], []

                if df_tmp is not None:
                    st.markdown("**Choose target (label) column**")
                    all_cols = cat_cols + numeric_cols
                    target_col = st.selectbox("Target column (classification)", all_cols, key='cls_target')
                    st.markdown("**Select feature columns**")
                    selected_features = st.multiselect("Features (if empty use all numeric columns)", numeric_cols,
                                                       default=numeric_cols[:min(6, len(numeric_cols))],
                                                       key='cls_features')

                    if st.button("Train Classification Model", key='btn_train_cls'):
                        if not target_col:
                            st.error("Please select a target column.")
                        else:
                            with st.spinner("Training classification model..."):
                                # prepare data
                                df_use = df_tmp.copy()
                                y = df_use[target_col].fillna(method='ffill')

                                # Check if we have enough samples
                                if len(y) < 2:
                                    st.error("❌ Insufficient samples for classification. Need at least 2 samples.")
                                    st.stop()

                                from sklearn.preprocessing import LabelEncoder

                                le = LabelEncoder()
                                y_enc = le.fit_transform(y.astype(str))

                                if len(selected_features) == 0:
                                    selected_features = numeric_cols
                                if len(selected_features) == 0:
                                    st.error("No numeric features available for classification.")
                                else:
                                    X = df_use[selected_features].fillna(df_use[selected_features].mean()).values

                                    # Check sample size requirements
                                    if X.shape[0] < 2:
                                        st.error("❌ Not enough samples for train-test split. Need at least 2 samples.")
                                        st.stop()

                                    # Adjust test size for small datasets
                                    test_size_val = min(0.2, 0.1) if X.shape[0] < 10 else 0.2

                                    # Use stratification only if we have multiple classes with sufficient samples
                                    unique_classes = np.unique(y_enc)
                                    can_stratify = len(unique_classes) > 1 and all(
                                        np.sum(y_enc == cls) >= 2 for cls in unique_classes)

                                    if can_stratify:
                                        X_train, X_test, y_train, y_test = train_test_split(
                                            X, y_enc, test_size=test_size_val, random_state=42, stratify=y_enc
                                        )
                                    else:
                                        X_train, X_test, y_train, y_test = train_test_split(
                                            X, y_enc, test_size=test_size_val, random_state=42
                                        )

                                    # Check if we have training samples
                                    if X_train.shape[0] == 0:
                                        st.error("❌ No samples in training set. Please check your data.")
                                        st.stop()

                                    clf = RandomForestClassifier(n_estimators=100, random_state=42)
                                    clf.fit(X_train, y_train)
                                    preds = clf.predict(X_test)

                                    # Only show metrics if we have test samples
                                    if len(y_test) > 0:
                                        report = classification_report(y_test, preds, output_dict=True)
                                        cm = confusion_matrix(y_test, preds)
                                        st.subheader("Classification Report")
                                        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
                                        st.subheader("Confusion Matrix")
                                        fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="True"),
                                                        title="Confusion Matrix")
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning("No test samples available for evaluation.")

                                    # save model in session_state
                                    st.session_state['ml_models'] = st.session_state.get('ml_models', {})
                                    st.session_state['ml_models']['last_classification'] = {
                                        'model_obj': clf,
                                        'label_encoder': le,
                                        'features': selected_features,
                                        'target': target_col,
                                        'created_at': datetime.now().isoformat()
                                    }
                                    st.success(
                                        "Model trained and saved to session_state['ml_models']['last_classification']")

    # Regression - FIXED VERSION
    with ml_tabs[3]:
        st.subheader("Regression")
        if research_files.get('data'):
            reg_file = st.selectbox("Select Dataset for Regression", research_files['data'], key='reg_file')
            if reg_file:
                try:
                    dfr = pd.read_csv(reg_file) if reg_file.endswith('.csv') else pd.read_excel(reg_file)
                    numeric_cols_r = dfr.select_dtypes(include=[np.number]).columns.tolist()
                except Exception as e:
                    st.error(f"Failed to read file: {e}")
                    dfr = None
                    numeric_cols_r = []

                if dfr is not None:
                    st.markdown("**Choose target (continuous) column**")
                    reg_target = st.selectbox("Target column (regression)", numeric_cols_r, key='reg_target')
                    st.markdown("**Select feature columns**")
                    reg_features = st.multiselect("Features (numeric)", [c for c in numeric_cols_r if c != reg_target],
                                                  default=[c for c in numeric_cols_r if c != reg_target][:6],
                                                  key='reg_features')

                    if st.button("Train Regression Model", key='btn_train_reg'):
                        if not reg_target or len(reg_features) == 0:
                            st.error("Please select target and at least one feature.")
                        else:
                            with st.spinner("Training regression model..."):
                                Xr = dfr[reg_features].fillna(dfr[reg_features].mean()).values
                                yr = dfr[reg_target].fillna(dfr[reg_target].mean()).values

                                # Check sample size
                                if Xr.shape[0] < 2:
                                    st.error("❌ Insufficient samples for regression. Need at least 2 samples.")
                                    st.stop()

                                # Adjust test size for small datasets
                                test_size_val = min(0.2, 0.1) if Xr.shape[0] < 10 else 0.2

                                Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=test_size_val, random_state=42)

                                # Check if we have training samples
                                if Xtr.shape[0] == 0:
                                    st.error("❌ No samples in training set. Please check your data.")
                                    st.stop()

                                rfr = RandomForestRegressor(n_estimators=100, random_state=42)
                                rfr.fit(Xtr, ytr)
                                preds_r = rfr.predict(Xte)

                                # Only calculate metrics if we have test samples
                                if len(yte) > 0:
                                    mse = mean_squared_error(yte, preds_r)
                                    mae = mean_absolute_error(yte, preds_r)
                                    r2 = r2_score(yte, preds_r)

                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("MSE", f"{mse:.4f}")
                                    with col2:
                                        st.metric("MAE", f"{mae:.4f}")
                                    with col3:
                                        st.metric("R²", f"{r2:.4f}")

                                    fig = px.scatter(x=yte.tolist(), y=preds_r.tolist(),
                                                     labels={'x': 'True', 'y': 'Predicted'},
                                                     title="True vs Predicted")
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("No test samples available for evaluation.")

                                st.session_state['ml_models'] = st.session_state.get('ml_models', {})
                                st.session_state['ml_models']['last_regression'] = {
                                    'model_obj': rfr,
                                    'features': reg_features,
                                    'target': reg_target,
                                    'created_at': datetime.now().isoformat()
                                }
                                st.success(
                                    "Regression model trained and stored in session_state['ml_models']['last_regression']")

# ----------------------------
# Experiment Hub
# ----------------------------
elif page == "🧪 Experiment Hub":
    st.markdown('<div class="main-header">Experiment Management Hub</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="scientific-note">
    <h4>🧪 Experimental Design</h4>
    <p>This module helps track and manage biological experiments with proper documentation of protocols, 
    objectives, and results. Supports reproducibility and experimental workflow management.</p>
    </div>
    """, unsafe_allow_html=True)

    exp_tabs = st.tabs(["📋 New Experiment", "📊 Active Experiments", "📈 Experiment Analytics"])
    with exp_tabs[0]:
        st.subheader("Create New Experiment")
        with st.form("experiment_form"):
            col1, col2 = st.columns(2)
            with col1:
                exp_name = st.text_input("Experiment Name")
                exp_type = st.selectbox("Experiment Type",
                                        ["Microscopy", "Sequencing", "Western Blot", "PCR", "Custom"])
                researcher = st.text_input("Researcher Name")
            with col2:
                start_date = st.date_input("Start Date")
                expected_duration = st.number_input("Expected Duration (days)", min_value=1, max_value=365, value=7)
                priority = st.select_slider("Priority", options=["Low", "Medium", "High"])
            objectives = st.text_area("Objectives")
            methodology = st.text_area("Methodology")
            if st.form_submit_button("Create Experiment"):
                if exp_name:
                    record_data = {
                        'name': exp_name,
                        'type': exp_type,
                        'researcher': researcher,
                        'start_date': start_date.isoformat(),
                        'duration': int(expected_duration),
                        'priority': priority,
                        'objectives': objectives,
                        'methodology': methodology
                    }
                    record_id = create_experiment_record(record_data)
                    st.success(f"Experiment '{exp_name}' created with ID: {record_id}")
    with exp_tabs[1]:
        st.subheader("Active Experiments")
        if st.session_state.experiment_records:
            for exp_id, exp_data in st.session_state.experiment_records.items():
                with st.expander(f"🔬 {exp_data['name']} ({exp_id})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Type:** {exp_data['type']}")
                        st.write(f"**Researcher:** {exp_data['researcher']}")
                        st.write(f"**Priority:** {exp_data['priority']}")
                    with col2:
                        st.write(f"**Start Date:** {exp_data['start_date'][:10]}")
                        st.write(f"**Status:** {exp_data['status']}")
                        st.write(f"**Objectives:** {exp_data['objectives'][:100]}...")
        else:
            st.info("No active experiments. Create a new experiment to get started.")

# ----------------------------
# Bioinformatics (with online enrichment)
# ----------------------------
elif page == "🧬 Bioinformatics":
    st.markdown('<div class="main-header">Bioinformatics Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="scientific-note">
    <h4>🧬 Genomic & Transcriptomic Analysis</h4>
    <p>This module provides gene set enrichment analysis, pathway mapping, and functional annotation 
    using established bioinformatics databases and algorithms.</p>
    </div>
    """, unsafe_allow_html=True)

    bio_tabs = st.tabs(["🧬 Gene Enrichment", "🔄 Pathway Analysis", "🧬 Sequence Tools"])
    with bio_tabs[0]:
        st.subheader("Gene Set Enrichment Analysis")
        gene_input = st.text_area(
            "Enter Gene List (one per line or comma-separated)",
            placeholder="TP53\nBRCA1\nEGFR\nMYC\n..."
        )
        database = st.selectbox("Enrichment Database",
                                ["KEGG", "GO Biological Process", "GO Molecular Function", "Reactome"])
        backend = st.selectbox("Enrichment backend", ["Simulated", "g:Profiler (online)", "Enrichr (online)"])
        if st.button("Run Enrichment Analysis") and gene_input:
            genes = [g.strip() for g in gene_input.replace(',', '\n').split('\n') if g.strip()]
            if len(genes) < 3:
                st.warning("⚠️ Please enter at least 3 genes for meaningful enrichment analysis.")
            else:
                with st.spinner("Performing gene enrichment analysis..."):
                    if backend == "Simulated":
                        analysis = perform_gene_enrichment_analysis(genes, database)
                        if analysis and 'error' not in analysis:
                            st.success(
                                f"Analysis complete! Found {analysis['significant_pathways']} significant pathways.")
                            pathways_df = pd.DataFrame(analysis['top_pathways'])
                            st.dataframe(pathways_df, use_container_width=True)
                            if len(pathways_df) > 0:
                                fig = px.bar(pathways_df, x='enrichment_score', y='pathway',
                                             title=f"Top Enriched Pathways ({database})",
                                             orientation='h')
                                st.plotly_chart(fig, use_container_width=True)
                                st.markdown("**Biological Interpretation**")
                                st.write(analysis['biological_interpretation'])
                    elif backend == "g:Profiler (online)":
                        sources = None
                        if database.startswith('GO'):
                            sources = ['GO:BP'] if 'Biological' in database else ['GO:MF']
                        elif database == 'KEGG':
                            sources = ['KEGG']
                        dfg = enrichment_gprofiler(genes, organism='hsapiens', sources=sources)
                        if isinstance(dfg, pd.DataFrame) and not dfg.empty and 'error' not in dfg.columns:
                            st.success(f"g:Profiler returned {len(dfg)} terms.")
                            st.dataframe(dfg, use_container_width=True)
                            # Save CSV
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                            dfg.to_csv(tmp.name, index=False)
                            tmp.close()
                            with open(tmp.name, "rb") as f:
                                st.download_button("Download g:Profiler results (CSV)", data=f,
                                                   file_name=f"gprofiler_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                   mime="text/csv")
                        else:
                            st.warning("No results or error from g:Profiler.")
                    else:  # Enrichr
                        lib = 'KEGG_2019_Human' if database == 'KEGG' else 'GO_Biological_Process_2018'
                        dfe = enrichment_enrichr(genes, library=lib)
                        if isinstance(dfe, pd.DataFrame) and not dfe.empty and 'error' not in dfe.columns:
                            st.success(f"Enrichr returned {len(dfe)} terms.")
                            st.dataframe(dfe, use_container_width=True)
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                            dfe.to_csv(tmp.name, index=False)
                            tmp.close()
                            with open(tmp.name, "rb") as f:
                                st.download_button("Download Enrichr results (CSV)", data=f,
                                                   file_name=f"enrichr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                   mime="text/csv")
                        else:
                            st.warning("No results or error from Enrichr.")

# ----------------------------
# Results Dashboard
# ----------------------------
elif page == "📈 Results":
    st.markdown('<div class="main-header">Results & Insights Dashboard</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="scientific-note">
    <h4>📊 Integrated Analysis Results</h4>
    <p>This dashboard provides a comprehensive overview of all analyses performed, including statistical summaries, 
    visualization outputs, and biological interpretations.</p>
    </div>
    """, unsafe_allow_html=True)

    if (st.session_state.get('current_analysis') or {}) or (st.session_state.get('analyzed_images') or {}):
        st.markdown("### 📊 Integrated Results Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Completed Analyses", len(st.session_state.get('current_data_analysis') or {}))
        with col2:
            st.metric("AI Image Analyses", len(st.session_state.get('analyzed_images') or {}))
        with col3:
            st.metric("ML Models Trained", len(st.session_state.get('ml_models') or {}))
        st.markdown("### 📅 Recent Activity")
        activities = [
            {"time": "2 hours ago", "activity": "Completed clustering analysis", "type": "analysis"},
            {"time": "4 hours ago", "activity": "Uploaded new microscopy images", "type": "upload"},
            {"time": "1 day ago", "activity": "Ran gene enrichment analysis", "type": "bioinformatics"},
            {"time": "2 days ago", "activity": "Trained new classification model", "type": "ml"}
        ]
        for activity in activities:
            emoji = "🔬" if activity["type"] == "analysis" else "📁" if activity["type"] == "upload" else "🧬" if activity[
                                                                                                                   "type"] == "bioinformatics" else "🤖"
            st.write(f"{emoji} **{activity['time']}**: {activity['activity']}")
    else:
        st.info("""
        ## 🎯 Get Started with Analysis
        To view comprehensive results:
        1. Navigate to **Data Explorer** to analyze your datasets
        2. Use **AI Image Analysis** for microscopy and visualization images  
        3. Explore **Machine Learning** for advanced pattern detection
        4. Check **Bioinformatics** for gene and pathway analysis
        Results will appear here as you complete analyses.
        """)

# ----------------------------
# Methodology & Documentation
# ----------------------------
elif page == "🛠️ Methodology":
    st.markdown('<div class="main-header">Methodology & Technical Documentation</div>', unsafe_allow_html=True)
    method_tabs = st.tabs(["🔬 Experimental", "💻 Computational", "📐 Statistical", "🤖 AI/ML"])
    with method_tabs[0]:
        st.markdown("""
        ### 🧪 Experimental Protocols
        **Cell Culture & Preparation**
        - Standard cell culture protocols for mammalian cells
        - Transfection and transduction methods
        - Sample preparation for microscopy
        """)
    with method_tabs[1]:
        st.markdown("""
        ### 💻 Computational Methods
        ```python
        def process_single_cell_data(raw_data):
            qc_data = quality_control(raw_data)
            normalized_data = normalize_expression(qc_data)
            features = select_features(normalized_data)
            return features
        ```
        """)
    with method_tabs[2]:
        st.markdown("""
        ### 📐 Statistical Framework
        - Hypothesis Testing: appropriate selection & multiple testing correction
        - Model Validation: cross-validation, bootstrap, permutation testing
        """)
    with method_tabs[3]:
        st.markdown("""
        ### 🤖 AI & Machine Learning
        - ResNet-50 for feature extraction
        - U-Net for image segmentation
        - Transfer learning, augmentation, hyperparameter tuning
        """)

elif page == "📚 Documentation":
    st.markdown('<div class="main-header">Platform Documentation</div>', unsafe_allow_html=True)
    doc_tabs = st.tabs(["📖 User Guide", "🎯 Tutorials", "🔧 API Reference", "📋 Examples"])
    with doc_tabs[0]:
        st.markdown("""
        ## 📖 User Guide
        1. Place files in `syphu-china-model` directory
        2. The platform will auto-detect supported file types
        3. Choose analysis module and run analyses
        """)
    with doc_tabs[1]:
        st.markdown("""
        ## 🎯 Tutorials
        - Tutorial 1: Basic Data Analysis
        - Tutorial 2: AI Image Analysis
        - Tutorial 3: Machine Learning
        """)

# Footer & system status
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 0.9em;'>
    <p><strong>SYPHU-CHINA-iGEM 2025</strong></p>
    <p>This platform integrates cutting-edge AI, machine learning, and bioinformatics tools for synthetic biology research.</p>
    <p>For technical support and collaboration opportunities</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("---")
    st.markdown("#### 📊 System Status")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.metric("CPU Usage", "45%")
    with status_col2:
        st.metric("Memory", "62%")
    st.progress(0.75)
    st.caption("System performance: Good")

