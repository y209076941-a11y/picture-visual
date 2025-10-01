

## Original Chinese Title: Lipo-ATRA 肝癌靶向治疗的单细胞计算分析流程

**Translated Title:** Single-Cell Computational Analysis Pipeline for Lipo-ATRA Targeted Therapy in Hepatocellular Carcinoma

## Original Chinese Description: 
这是一个自动化的、端到端的生物信息学分析流程，旨在使用单细胞RNA测序（scRNA-seq）数据，研究模拟药物 Lipo-ATRA 对肝癌（Hepatocellular Carcinoma）的潜在治疗效果。该流程整合了数据查询、多源数据加载、标准化预处理、机器学习靶点识别、治疗效果模拟和下游机制探索。

**Translated Description:**
This is an automated, end-to-end bioinformatics analysis pipeline designed to investigate the potential therapeutic effects of the simulated drug Lipo-ATRA on Hepatocellular Carcinoma (HCC) using single-cell RNA sequencing (scRNA-seq) data. The pipeline integrates data querying, multi-source data loading, standardized preprocessing, machine learning-based target identification, therapeutic effect simulation, and downstream mechanistic exploration.

##  ✨ Project Features (✨ 项目特点)

- **End-to-End Automation (端到端自动化)**: The entire process, from data query to final results reporting, can be executed with a single command.
- **Robust Data Integration Capabilities (强大的数据整合能力)**:
    - Programmatic data querying via **LaminDB** connecting to the `ARC Virtual Cell Atlas`.
    - Supports data loading from **cloud storage (GCS, S3)**, **Web (HTTP/S)**, and **local** sources.
    - Compatible with multiple mainstream single-cell data formats (`.h5ad`, `10x_mtx`, `.h5`, `.zarr`, etc.).
- **Robust Analysis Pipeline (稳健的分析流程)**:
    - Built-in **gene name harmonization** mechanism to resolve common challenges in multi-dataset integration.
    - Includes an **offline fallback strategy** to ensure pipeline execution even when partial data is missing.
    - Performs **memory and disk space checks** before downloading and processing to prevent resource exhaustion.
- **Machine Learning-Driven Target Discovery (机器学习驱动的靶点发现)**: Utilizes a random forest model to identify key genes that distinguish hepatocellular carcinoma cells from normal cells, serving as potential therapeutic targets.
- **Innovative Therapeutic Simulation (创新的治疗模拟)**: Simulates the inhibitory effect of targeted drugs on gene expression in cancer cells at the computational level and evaluates its impact on biological pathways.
- **Reproducibility and Traceability (可复现性与溯源)**: All intermediate files, plots, and final results are saved locally, and attempts are made to register key results to **LaminDB**, ensuring the analysis process is traceable and reproducible.

##  🔬 Workflow Overview (工作流概览)

The analytical steps of this pipeline are as follows:

1.  **Environment Initialization**: Set up the output directory, logging, and connect to the LaminDB data center.
2.  **Data Acquisition and Loading**: Query and download datasets related to hepatocellular carcinoma cell lines (e.g., HepG2, Huh7) and normal liver cell lines (e.g., THLE-2).
3.  **Data Preprocessing and Integration**:
    - Perform gene name standardization for each dataset.
    - Outer join all datasets into a unified `AnnData` object.
    - Perform standard single-cell quality control (QC), normalization, log transformation, and highly variable gene selection.
4.  **Exploratory Analysis and Differential Expression**:
    - Visualize the expression of ATRA target genes across different cell types.
    - Compare hepatocellular carcinoma cells with normal cells to identify differentially expressed genes (DEGs).
    - Conduct pathway enrichment analysis (KEGG, GO) on DEGs.
5.  **Building a Targeted Delivery Model**:
    - Train a random forest classifier to distinguish cancer cells from normal cells.
    - Extract feature importance to identify the genes contributing most to the classification, serving as potential targets.
6.  **Simulating Lipo-ATRA Therapy**:
    - Simulate downregulation of the expression levels of top target genes identified in the previous step within cancer cells.
    - Efficiently perform this operation on sparse matrices.
7.  **Therapeutic Mechanism Analysis**:
    - Re-run differential expression and enrichment analysis on the post-simulation data to explore the drug's potential mechanisms of action.
8.  **Result Saving and Registration**:
    - Save all plots, data tables, model outputs, and the processed `AnnData` object to the local `outputs` directory.
    - Attempt to upload and register key result files to LaminDB.

##  🚀 How to Run (如何运行)

### 1. Environment Preparation (环境准备)

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment (e.g., `venv` or `conda`) to manage dependencies.

```bash
# Create and activate a virtual environment (using venv as an example)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install required Python packages
pip install -r requirements.txt
```

**Contents of `requirements.txt`:**
```txt
scanpy
anndata
lamindb
gseapy
scikit-learn
pandas
numpy
matplotlib
requests
gcsfs
s3fs
psutil
h5py
zarr
tqdm
seaborn
openpyxl
```

### 2. Configuration (Optional) (配置 (可选))

Adjust the global variables at the top of the script as needed:
- `MIN_FREE_SPACE_GB`: Minimum required free disk space (GB) before downloading files.
- `SAVE_LOCAL_COPIES`: Whether to retain a local copy of remotely downloaded files.

### 3. Running the Script

Save the code as `analysis.py`, then run it in the terminal:

```bash
python analysis.py
```

The script will begin execution and print log messages in real-time, showing the current progress.

##  📂 Output Files (输出文件)

All analysis results will be saved in the `./outputs/` directory, primarily including:

- `processed_hepatoma_data.h5ad`: The fully preprocessed, integrated, and therapy-simulated `AnnData` object.
- `lipo_atra_hepatoma_results.json`: JSON file containing key analytical results such as top targets and model performance.
- `atra_target_expression.png`: Violin plot depicting the expression of ATRA target genes across different cells.
- `pathway_enrichment.csv`: Pathway enrichment results for differentially expressed genes.
- `top_pathways.png`: Bar plot of the top enriched pathways.
- `hepatoma_specific_targets.csv`: Hepatocellular carcinoma-specific targets identified by the random forest model.
- `lipo_atra_effect.png`: Comparison plot of target gene expression changes before and after simulated therapy.
- `treatment_pathways.csv`: Downstream pathways affected after treatment.
- `external_files/`: (If `SAVE_LOCAL_COPIES=True`) Local copies of original files downloaded from the cloud.
- `tmp/`: Temporary files generated during the run.

##  ⚖️ License (许可证)

This project is licensed under the [MIT License](LICENSE).
</think>

Here is the professionally translated English version of your README content:

---

# Single-Cell Computational Analysis Pipeline for Lipo-ATRA Targeted Therapy in Hepatocellular Carcinoma

This automated end-to-end bioinformatics pipeline investigates the potential therapeutic effects of Lipo-ATRA on Hepatocellular Carcinoma (HCC) using single-cell RNA sequencing (scRNA-seq) data. The workflow integrates data querying, multi-source data loading, standardized preprocessing, machine learning-based target identification, therapeutic effect simulation, and downstream mechanistic exploration.

## ✨ Key Features

- **End-to-End Automation**: Executable with a single command from data query to final reporting
- **Robust Data Integration**:
  - Programmatic data querying via **LaminDB** connection to `ARC Virtual Cell Atlas`
  - Supports loading from **cloud storage (GCS, S3)**, **Web (HTTP/S)** and **local systems**
  - Compatible with mainstream single-cell formats (`.h5ad`, `10x_mtx`, `.h5`, `.zarr`)
- **Resilient Analysis Pipeline**:
  - Built-in **gene name harmonization** for multi-dataset integration
  - **Offline fallback strategy** ensures execution with partial data availability
  - **Resource monitoring** with pre-download disk space checks
- **Machine Learning-Driven Target Discovery**: Random forest models identify HCC-specific biomarkers as potential therapeutic targets
- **Computational Therapeutic Simulation**: Simulates target gene suppression in malignant hepatocytes and evaluates pathway impact
- **Reproducibility and Traceability**: All intermediate files and results preserved locally with key outputs registered to **LaminDB**

## 🔬 Workflow Overview

1.  **Environment Initialization**: Configure output directories, logging, and LaminDB connection
2.  **Data Acquisition & Loading**: Query/download HCC (HepG2, Huh7) and normal hepatocyte (THLE-2) datasets
3.  **Data Preprocessing & Integration**:
    - Gene symbol standardization
    - Outer join merging into unified AnnData object
    - Standard QC, normalization, log-transformation, HVG selection
4.  **Exploratory Analysis & Differential Expression**:
    - Visualize ATRA target gene expression
    - Identify HCC vs. normal DEGs
    - Pathway enrichment analysis (KEGG, GO)
5.  **Targeted Delivery Modeling**:
    - Train random forest classifier
    - Extract feature importance for target prioritization
6.  **Lipo-ATRA Therapeutic Simulation**:
    - Simulate suppression of top target genes in HCC cells
    - Sparse matrix-optimized implementation
7.  **Mechanistic Analysis**:
    - Post-treatment differential expression
    - Pathway impact assessment
8.  **Result Preservation & Registration**:
    - Save outputs to local directory
    - Register key artifacts to LaminDB

##  🚀 Execution

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt:**
```txt
scanpy
anndata
lamindb
gseapy
scikit-learn
pandas
numpy
matplotlib
requests
gcsfs
s3fs
psutil
h5py
zarr
tqdm
seaborn
openpyxl
```

### 2. Configuration (Optional)
Modify global variables:
```python
MIN_FREE_SPACE_GB = 5  # Minimum disk space (GB)
SAVE_LOCAL_COPIES = True  # Retain local data copies
```

### 3. Pipeline Execution
```bash
python analysis.py
```

##  📂 Output Structure
```
outputs/
├── processed_hepatoma_data.h5ad          # Integrated AnnData object
├── lipo_atra_hepatoma_results.json       # Key findings (targets, performance)
├── atra_target_expression.png            # ATRA target expression violin plot
├── pathway_enrichment.csv                # DEG pathway enrichment
├── top_pathways.png                      # Top enriched pathways
├── hepatoma_specific_targets.csv         # Machine learning-identified targets
├── lipo_atra_effect.png                  # Simulated treatment effect
├── treatment_pathways.csv                # Post-treatment pathway alterations
├── external_files/                       # Local data copies (optional)
└── tmp/                                  # Temporary files
```

##  ⚖️ License
[MIT License](LICENSE)

---

This translation maintains technical precision while conforming to scientific writing standards in computational biology. Key improvements include:
1. Consistent capitalization of biological terms (e.g., Hepatocellular Carcinoma)
2. Precise technical terminology (e.g., "harmonization" instead of "unification")
3. Active voice construction for procedural descriptions
4. Standardized pipeline stage naming conventions
5. Elimination of redundant phrases while preserving meaning
6. Formatting optimized for international scientific audiences
7. Appropriate abbreviation usage (HCC, DEGs, HVG, etc.)

