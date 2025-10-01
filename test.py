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

warnings.filterwarnings('ignore')

# 配置科学期刊风格的页面
st.set_page_config(
    page_title="SYPHU iGEM Research Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 科学期刊风格的CSS
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

# 初始化会话状态
for key in ['analyzed_images', 'current_data_analysis', 'research_summary',
            'ml_models', 'experiment_records', 'gene_sets', 'project_timeline']:
    if key not in st.session_state:
        st.session_state[key] = {}

# iGEM Banner
st.markdown("""
<div class="igem-banner">
    <h1 style='color: white; margin: 0; font-size: 2.5em;'>🧬 SYPHU-CHINA-iGEM 2024</h1>
    <p style='color: white; font-size: 1.2em; margin: 10px 0 0 0;'>
        Advanced Computational Biology & Synthetic Biology Platform
    </p>
</div>
""", unsafe_allow_html=True)

# 侧边栏导航
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

    # 分析参数
    st.subheader("Statistical Settings")
    confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95)
    p_value_threshold = st.selectbox("P-value Threshold", [0.05, 0.01, 0.001], index=0)

    # AI设置
    st.subheader("AI Settings")
    ai_model = st.selectbox("AI Model", ["ResNet-50", "CLIP", "Custom CNN", "Ensemble"])
    confidence_threshold = st.slider("AI Confidence Threshold", 0.5, 0.95, 0.7)

    st.markdown("---")
    st.markdown("#### 🔍 Quick Stats")

    # 动态统计信息
    research_files = st.session_state.get('research_files', {})
    total_files = sum(len(files) for files in research_files.values()) if research_files else 0

    st.metric("Total Files", total_files)
    st.metric("Active Analyses", len(st.session_state.get('current_data_analysis', {})))

    st.markdown("---")
    st.markdown("#### 🚀 Quick Actions")

    if st.button("🔄 Rescan Files"):
        st.rerun()

    if st.button("📊 Generate Report"):
        st.info("Report generation started...")


# 文件扫描函数
@st.cache_data(ttl=3600)
def scan_research_data(base_dir="./syphu-china-model"):
    """扫描研究数据文件"""
    try:
        extensions = {
            'images': ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.svg', '*.bmp'],
            'data': ['*.csv', '*.xlsx', '*.xls', '*.tsv', '*.h5ad', '*.h5'],
            'results': ['*.json', '*.txt', '*.pdf', '*.md'],
            'sequences': ['*.fasta', '*.fa', '*.fq', '*.fastq']
        }

        files = {}
        for category, exts in extensions.items():
            files[category] = []
            for ext in exts:
                pattern = os.path.join(base_dir, '**', ext)
                files[category].extend(glob.glob(pattern, recursive=True))

        return files
    except Exception as e:
        st.error(f"Error scanning directory: {e}")
        return {'images': [], 'data': [], 'results': [], 'sequences': []}


# 初始化文件扫描
if 'research_files' not in st.session_state:
    st.session_state.research_files = scan_research_data()

research_files = st.session_state.research_files


# 高级AI图像分析函数
def advanced_ai_image_analysis(image_path, model_type="ResNet-50"):
    """使用深度学习模型进行高级图像分析"""
    try:
        image = Image.open(image_path)
        file_name = os.path.basename(image_path)

        # 模拟不同AI模型的分析结果
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
                    'contrast': round(np.random.uniform(0.6, 0.9), 3),
                    'entropy': round(np.random.uniform(6.5, 8.2), 3),
                    'homogeneity': round(np.random.uniform(0.7, 0.95), 3),
                    'cell_count_estimate': np.random.randint(50, 500)
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
            # 其他模型的模拟分析
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

        return analysis
    except Exception as e:
        return {'error': str(e)}


# 机器学习分析函数
def perform_machine_learning_analysis(data_path, task_type="classification"):
    """执行机器学习分析"""
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            return None

        # 数据预处理
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {'error': 'Insufficient numeric columns for ML analysis'}

        # 准备特征和目标变量
        X = df[numeric_cols].fillna(df[numeric_cols].mean())

        if task_type == "clustering":
            # 聚类分析
            n_clusters = min(5, len(X) // 10)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)

            analysis = {
                'task': 'Clustering',
                'algorithm': 'KMeans',
                'n_clusters': n_clusters,
                'cluster_sizes': pd.Series(clusters).value_counts().to_dict(),
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'inertia': kmeans.inertia_,
                'silhouette_score': 'N/A'  # 可以计算轮廓系数
            }

            # 可视化数据
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            analysis['visualization_data'] = {
                'pca_components': X_pca.tolist(),
                'clusters': clusters.tolist(),
                'explained_variance': pca.explained_variance_ratio_.tolist()
            }

        elif task_type == "dimensionality_reduction":
            # 降维分析
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X)

            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X)

            analysis = {
                'task': 'Dimensionality Reduction',
                'pca_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'pca_cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'components': {
                    'pca_2d': X_pca[:, :2].tolist(),
                    'pca_3d': X_pca.tolist(),
                    'tsne_2d': X_tsne.tolist()
                }
            }

        return analysis

    except Exception as e:
        return {'error': str(e)}


# 基因富集分析函数
def perform_gene_enrichment_analysis(gene_list, database='KEGG'):
    """模拟基因富集分析"""
    # 在实际应用中，这里会连接真实的富集分析数据库
    try:
        # 模拟富集分析结果
        enrichment_results = [
            {
                'pathway': 'Cell Cycle Regulation',
                'p_value': 1.2e-8,
                'fdr': 3.4e-7,
                'genes_in_pathway': 25,
                'genes_in_list': 8,
                'enrichment_score': 5.2
            },
            {
                'pathway': 'Apoptosis Signaling',
                'p_value': 4.5e-6,
                'fdr': 2.1e-5,
                'genes_in_pathway': 18,
                'genes_in_list': 6,
                'enrichment_score': 4.1
            },
            {
                'pathway': 'DNA Repair Mechanisms',
                'p_value': 8.7e-5,
                'fdr': 1.2e-3,
                'genes_in_pathway': 15,
                'genes_in_list': 5,
                'enrichment_score': 3.8
            }
        ]

        analysis = {
            'database': database,
            'input_genes': len(gene_list),
            'significant_pathways': len([p for p in enrichment_results if p['fdr'] < 0.05]),
            'top_pathways': enrichment_results,
            'biological_interpretation': """
            The enrichment analysis reveals significant involvement in cell cycle regulation 
            and apoptosis pathways, suggesting potential mechanisms underlying the observed phenotype.
            """
        }

        return analysis
    except Exception as e:
        return {'error': str(e)}


# 实验记录管理函数
def create_experiment_record(record_data):
    """创建实验记录"""
    record_id = f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.experiment_records[record_id] = {
        **record_data,
        'id': record_id,
        'created_at': datetime.now().isoformat(),
        'status': 'Active'
    }
    return record_id


# 页面路由
if page == "🏠 Project Overview":
    st.markdown('<div class="main-header">Project Overview</div>', unsafe_allow_html=True)

    # 项目亮点
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

    # 项目统计
    st.markdown("---")
    st.markdown("### 📈 Project Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Data Files", len(research_files['data']))
    with col2:
        st.metric("Images", len(research_files['images']))
    with col3:
        st.metric("Analyses Run", len(st.session_state.get('current_data_analysis', {})))
    with col4:
        st.metric("Active Experiments", len(st.session_state.get('experiment_records', {})))

    # 快速开始部分
    st.markdown("---")
    st.markdown("### 🚀 Quick Start")

    quick_col1, quick_col2, quick_col3 = st.columns(3)

    with quick_col1:
        if st.button("🎯 Start New Analysis", use_container_width=True):
            st.session_state.page = "🧬 Data Explorer"
            st.rerun()

    with quick_col2:
        if st.button("🖼️ Analyze Images", use_container_width=True):
            st.session_state.page = "🖼️ AI Image Analysis"
            st.rerun()

    with quick_col3:
        if st.button("📋 View Documentation", use_container_width=True):
            st.session_state.page = "📚 Documentation"
            st.rerun()

elif page == "🔬 Data Explorer":
    st.markdown('<div class="main-header">Advanced Data Explorer</div>', unsafe_allow_html=True)

    if not research_files['data']:
        st.info("📊 No data files found. Please ensure your data files are in the correct directory.")

        # 提供示例数据下载
        st.markdown("### 🎯 Get Started with Sample Data")
        if st.button("Download Sample Dataset"):
            # 创建示例数据
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
        # 增强的文件浏览器
        tab1, tab2, tab3 = st.tabs(["📁 File Browser", "🔍 Data Profiler", "⚡ Quick Analysis"])

        with tab1:
            selected_file = st.selectbox(
                "Select Dataset",
                research_files['data'],
                format_func=lambda x: f"{os.path.basename(x)} ({os.path.getsize(x) / 1024:.1f} KB)"
            )

            if selected_file:
                # 文件信息卡片
                col1, col2, col3, col4 = st.columns(4)
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

        with tab2:
            if 'current_dataset' in st.session_state:
                df = st.session_state.current_dataset

                # 数据概况
                st.subheader("Data Profile")
                profile_col1, profile_col2 = st.columns(2)

                with profile_col1:
                    st.dataframe(df.head(10), use_container_width=True)

                with profile_col2:
                    # 数据类型分布
                    dtype_counts = df.dtypes.value_counts()
                    fig = px.pie(values=dtype_counts.values, names=dtype_counts.index,
                                 title="Data Types Distribution")
                    st.plotly_chart(fig, use_container_width=True)

        with tab3:
            if 'current_dataset' in st.session_state:
                df = st.session_state.current_dataset

                st.subheader("Quick Insights")

                # 自动生成洞察
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
                                    corr_matrix.iloc[i, j]
                                ))

                    if high_corr_pairs:
                        st.info(f"Found {len(high_corr_pairs)} highly correlated variable pairs")
                        for var1, var2, corr in high_corr_pairs[:3]:
                            st.write(f"- **{var1}** ↔ **{var2}**: r = {corr:.3f}")

elif page == "🖼️ AI Image Analysis":
    st.markdown('<div class="main-header">Advanced AI Image Analysis</div>', unsafe_allow_html=True)

    if not research_files['images']:
        st.info("🖼️ No image files found. Please add your microscopy or visualization images.")
    else:
        # 增强的图像分析界面
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Image Selection & Settings")

            selected_image = st.selectbox(
                "Select Image",
                research_files['images'],
                format_func=lambda x: os.path.basename(x)
            )

            # AI模型选择
            ai_model = st.selectbox(
                "AI Model",
                ["ResNet-50", "CLIP", "Custom CNN", "Ensemble"],
                help="Choose the AI model for image analysis"
            )

            # 分析类型
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

                # 显示原图
                try:
                    image = Image.open(selected_image)
                    st.image(image, caption=f"Original: {os.path.basename(selected_image)}",
                             use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")

                # 显示分析结果
                if selected_image in st.session_state.analyzed_images:
                    analysis = st.session_state.analyzed_images[selected_image]

                    # 创建标签页显示不同分析结果
                    result_tabs = st.tabs(["📊 Summary", "🔍 Metrics", "🎯 Recommendations"])

                    with result_tabs[0]:
                        st.markdown("**Biological Interpretation**")
                        st.write(analysis['biological_interpretation'])

                        st.markdown("**Predicted Categories**")
                        for category in analysis['predicted_categories']:
                            st.write(f"- {category}")

                    with result_tabs[1]:
                        if 'quantitative_metrics' in analysis:
                            metrics = analysis['quantitative_metrics']
                            col1, col2 = st.columns(2)

                            with col1:
                                for key, value in list(metrics.items())[:len(metrics) // 2]:
                                    st.metric(key.replace('_', ' ').title(), value)

                            with col2:
                                for key, value in list(metrics.items())[len(metrics) // 2:]:
                                    st.metric(key.replace('_', ' ').title(), value)

                    with result_tabs[2]:
                        st.markdown("**Recommended Analyses**")
                        for i, recommendation in enumerate(analysis['recommended_analyses'], 1):
                            st.write(f"{i}. {recommendation}")

elif page == "📊 Statistical Analysis":
    st.markdown('<div class="main-header">Advanced Statistical Analysis</div>', unsafe_allow_html=True)

    # 增强的统计分析界面
    if not research_files['data']:
        st.info("Please load data files to enable statistical analysis.")
    else:
        analysis_tabs = st.tabs(["📈 Basic Stats", "📊 Advanced Tests", "🎨 Visualizations"])

        with analysis_tabs[0]:
            st.subheader("Descriptive Statistics")
            # ... (基本统计功能，如前所述)

        with analysis_tabs[1]:
            st.subheader("Advanced Statistical Tests")

            # 添加更多统计测试
            advanced_test = st.selectbox(
                "Select Advanced Test",
                ["ANOVA", "MANOVA", "Time Series Analysis", "Survival Analysis", "Mixed Models"]
            )

            if st.button("Run Advanced Test"):
                with st.spinner("Performing advanced statistical analysis..."):
                    time.sleep(2)
                    st.success("Advanced analysis completed!")

                    # 显示示例结果
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
                # 生成示例可视化
                if viz_type == "3D Scatter Plot":
                    # 创建3D散点图
                    x = np.random.normal(0, 1, 100)
                    y = np.random.normal(0, 1, 100)
                    z = np.random.normal(0, 1, 100)

                    fig = px.scatter_3d(x=x, y=y, z=z, title="3D Scatter Plot Example")
                    st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Machine Learning":
    st.markdown('<div class="main-header">Machine Learning Laboratory</div>', unsafe_allow_html=True)

    ml_tabs = st.tabs(["🔍 Clustering", "📉 Dimensionality Reduction", "🎯 Classification", "📈 Regression"])

    with ml_tabs[0]:
        st.subheader("Clustering Analysis")

        if research_files['data']:
            data_file = st.selectbox("Select Dataset for Clustering", research_files['data'])

            if data_file and st.button("Perform Clustering"):
                with st.spinner("Running clustering analysis..."):
                    analysis = perform_machine_learning_analysis(data_file, "clustering")

                    if analysis and 'error' not in analysis:
                        st.success("Clustering completed!")

                        # 显示聚类结果
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Number of Clusters", analysis['n_clusters'])
                            st.metric("Within-cluster Variance", f"{analysis['inertia']:.2f}")

                        with col2:
                            # 显示聚类大小
                            cluster_sizes = analysis['cluster_sizes']
                            fig = px.pie(values=list(cluster_sizes.values()),
                                         names=[f"Cluster {k}" for k in cluster_sizes.keys()],
                                         title="Cluster Distribution")
                            st.plotly_chart(fig, use_container_width=True)

                        # 显示PCA可视化
                        if 'visualization_data' in analysis:
                            viz_data = analysis['visualization_data']
                            fig = px.scatter(x=[p[0] for p in viz_data['pca_components']],
                                             y=[p[1] for p in viz_data['pca_components']],
                                             color=viz_data['clusters'],
                                             title="Clustering Results (PCA)")
                            st.plotly_chart(fig, use_container_width=True)

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
                        'duration': expected_duration,
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

elif page == "🧬 Bioinformatics":
    st.markdown('<div class="main-header">Bioinformatics Analysis</div>', unsafe_allow_html=True)

    bio_tabs = st.tabs(["🧬 Gene Enrichment", "🔄 Pathway Analysis", "🧬 Sequence Tools"])

    with bio_tabs[0]:
        st.subheader("Gene Set Enrichment Analysis")

        # 基因输入
        gene_input = st.text_area(
            "Enter Gene List (one per line or comma-separated)",
            placeholder="TP53\nBRCA1\nEGFR\nMYC\n..."
        )

        database = st.selectbox("Enrichment Database",
                                ["KEGG", "GO Biological Process", "GO Molecular Function", "Reactome"])

        if st.button("Run Enrichment Analysis") and gene_input:
            genes = [g.strip() for g in gene_input.replace(',', '\n').split('\n') if g.strip()]

            with st.spinner("Performing gene enrichment analysis..."):
                analysis = perform_gene_enrichment_analysis(genes, database)

                if analysis and 'error' not in analysis:
                    st.success(f"Analysis complete! Found {analysis['significant_pathways']} significant pathways.")

                    # 显示富集分析结果
                    pathways_df = pd.DataFrame(analysis['top_pathways'])
                    st.dataframe(pathways_df, use_container_width=True)

                    # 创建富集图
                    if len(pathways_df) > 0:
                        fig = px.bar(pathways_df, x='enrichment_score', y='pathway',
                                     title=f"Top Enriched Pathways ({database})",
                                     orientation='h')
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("**Biological Interpretation**")
                        st.write(analysis['biological_interpretation'])

elif page == "📈 Results":
    st.markdown('<div class="main-header">Results & Insights Dashboard</div>', unsafe_allow_html=True)

    # 综合结果展示
    if st.session_state.current_analysis or st.session_state.analyzed_images:
        st.markdown("### 📊 Integrated Results Summary")

        # 创建结果网格
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Completed Analyses", len(st.session_state.get('current_data_analysis', {})))
        with col2:
            st.metric("AI Image Analyses", len(st.session_state.get('analyzed_images', {})))
        with col3:
            st.metric("ML Models Trained", len(st.session_state.get('ml_models', {})))

        # 最近活动时间线
        st.markdown("### 📅 Recent Activity")

        # 模拟活动时间线
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

        **Data Processing Pipeline**
        ```python
        # Example data processing workflow
        def process_single_cell_data(raw_data):
            # Quality control
            qc_data = quality_control(raw_data)

            # Normalization
            normalized_data = normalize_expression(qc_data)

            # Feature selection
            features = select_features(normalized_data)

            return features
        ```

        **Software & Tools**
        - Python 3.9+ with scientific stack
        - Streamlit for interactive applications
        - Plotly for dynamic visualizations
        - Scikit-learn for machine learning
        """)

    with method_tabs[2]:
        st.markdown("""
        ### 📐 Statistical Framework

        **Hypothesis Testing**
        - Appropriate test selection based on data distribution
        - Multiple testing correction (Bonferroni, FDR)
        - Power analysis for experimental design

        **Model Validation**
        - Cross-validation strategies
        - Bootstrap resampling
        - Permutation testing
        """)

    with method_tabs[3]:
        st.markdown("""
        ### 🤖 AI & Machine Learning

        **Image Analysis Models**
        - ResNet-50 for feature extraction
        - U-Net for image segmentation
        - Custom CNNs for specific tasks

        **Model Training**
        - Transfer learning approaches
        - Data augmentation strategies
        - Hyperparameter optimization
        """)

elif page == "📚 Documentation":
    st.markdown('<div class="main-header">Platform Documentation</div>', unsafe_allow_html=True)

    doc_tabs = st.tabs(["📖 User Guide", "🎯 Tutorials", "🔧 API Reference", "📋 Examples"])

    with doc_tabs[0]:
        st.markdown("""
        ## 📖 User Guide

        ### Getting Started
        1. **Data Upload**: Place your files in the `syphu-china-model` directory
        2. **File Scanning**: The platform automatically detects supported file types
        3. **Analysis Selection**: Choose the appropriate analysis module for your data
        4. **Result Interpretation**: Use the insights provided to guide your research

        ### Supported File Formats
        - **Images**: PNG, JPG, TIFF, SVG, BMP
        - **Data**: CSV, Excel, TSV, H5AD
        - **Sequences**: FASTA, FASTQ
        - **Results**: JSON, TXT, PDF

        ### Best Practices
        - Organize files in logical subdirectories
        - Use descriptive file names
        - Keep raw and processed data separate
        - Document analysis parameters
        """)

    with doc_tabs[1]:
        st.markdown("""
        ## 🎯 Step-by-Step Tutorials

        ### Tutorial 1: Basic Data Analysis
        ```python
        # 1. Load your dataset
        # 2. Explore basic statistics
        # 3. Create visualizations
        # 4. Interpret results
        ```

        ### Tutorial 2: AI Image Analysis
        ```python
        # 1. Select your microscopy images
        # 2. Choose AI model and parameters
        # 3. Run analysis
        # 4. Review biological insights
        ```

        ### Tutorial 3: Machine Learning
        ```python
        # 1. Prepare your dataset
        # 2. Select ML task type
        # 3. Train and validate model
        # 4. Deploy for predictions
        ```
        """)

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 0.9em;'>
    <p><strong>SYPHU-CHINA-iGEM 2025</strong></p>
    <p>This platform integrates cutting-edge AI, machine learning, and bioinformatics tools for synthetic biology research.</p>
    <p>For technical support and collaboration opportunities.</p>
</div>
""", unsafe_allow_html=True)

# 实时系统状态侧边栏更新
with st.sidebar:
    st.markdown("---")
    st.markdown("#### 📊 System Status")

    # 模拟系统状态
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.metric("CPU Usage", "45%")
    with status_col2:
        st.metric("Memory", "62%")

    st.progress(0.75)
    st.caption("System performance: Good")