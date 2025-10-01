# A Single-Cell Computational Pipeline for Investigating Lipo-ATRA Targeted Therapy in Hepatocellular Carcinoma

This is an automated, end-to-end bioinformatics analysis pipeline designed to investigate the potential therapeutic effects of a simulated drug, **Lipo-ATRA (Liposome-encapsulated All-Trans Retinoic Acid)**, on Hepatocellular Carcinoma (HCC) using public single-cell RNA sequencing (scRNA-seq) data.

This project integrates a full spectrum of computational biology tasks, from data acquisition and preprocessing to machine learning-driven target identification and *in silico* treatment simulation. The ultimate goal is to **identify specific gene targets for Lipo-ATRA delivery and to predict its potential efficacy and mechanism of action** before undertaking costly and time-consuming wet-lab experiments.

## ✨ Key Features

- **End-to-End Automation**
  - The entire workflow, from data querying to final report generation, is executed with a single command. This minimizes manual intervention, ensuring both **reproducibility** and **efficiency**.

- **Powerful Data Integration**
  - **Cloud-Native Data Provenance**: Connects to the `ARC Virtual Cell Atlas` via **LaminDB**, enabling programmatic, version-controlled data querying that promotes FAIR data principles (Findable, Accessible, Interoperable, and Reusable).
  - **Multi-Source & Heterogeneous Support**: Seamlessly loads data from **cloud storage (GCS, S3)**, the **web (HTTP/S)**, and the **local filesystem**.
  - **Multi-Format Compatibility**: Intelligently parses a wide array of major single-cell data formats, including `.h5ad`, `10x_mtx`, `.h5`, and `.zarr`, showcasing its extensibility.

- **Robust Engineering Design**
  - **Gene Name Harmonization**: A built-in mechanism automatically resolves gene name inconsistencies across datasets (e.g., Ensembl IDs vs. Gene Symbols), a critical step for successful data integration.
  - **Resource-Aware Execution**: Performs **memory and disk space checks** before downloading large files or loading data, preventing unexpected crashes due to resource exhaustion.
  - **Offline Fallback Strategy**: Includes a fault-tolerant mode that creates a pseudo-control group if normal cell data is unavailable, ensuring the core pipeline integrity and providing clear warnings. This is invaluable for debugging and demonstration.

- **Machine Learning-Driven Target Discovery**
  - Leverages a **Random Forest** model to learn the optimal gene expression patterns that distinguish cancer cells from normal cells. The model's "feature importances" are used to quantify each gene's predictive power, thereby identifying high-impact, high-specificity therapeutic targets.

- **Innovative In Silico Treatment Simulation**
  - Computationally simulates the inhibitory effect of a targeted drug on cancer cells. This approach enables **pre-clinical prediction** and screening of a drug's potential efficacy and mechanism of action, saving significant time and resources.

- **Reproducibility & Data Lineage**
  - All intermediate files, plots, and final results are saved to a structured local directory. The pipeline also attempts to register key results back to **LaminDB**, creating a clear, traceable **data lineage** from raw data to final conclusions.

## 🔬 Workflow Overview

The analysis pipeline follows a coherent scientific narrative, broken down into eight core steps:

1.  **Initialization**
    - Sets up a unified output directory, configures logging, and establishes a connection to the LaminDB data hub to prepare the analysis environment.

2.  **Data Acquisition & Integration**
    - Queries and downloads all relevant datasets for specified HCC cell lines (e.g., HepG2, Huh7) and normal hepatocyte lines (e.g., THLE-2).
    - Harmonizes gene names for each dataset and merges them into a single `AnnData` object using an `outer join` strategy to retain the most comprehensive gene set.

3.  **Preprocessing & Quality Control (QC)**
    - Executes a standard single-cell QC pipeline to filter out low-quality cells (e.g., high mitochondrial gene ratio) and genes with low expression.
    - Normalizes and log-transforms the data, then identifies Highly Variable Genes (HVGs) to focus subsequent analysis on biologically significant signals.

4.  **Exploratory & Differential Analysis**
    - Begins by visualizing the expression of known ATRA target genes (e.g., `RARA`, `RARB`) to validate the biological relevance of the data.
    - Systematically compares HCC vs. normal cells to identify Differentially Expressed Genes (DEGs) and performs pathway enrichment analysis (KEGG, GO) to understand the molecular characteristics of the cancer cells.

5.  **Targeting Model Construction**
    - Trains a Random Forest classifier using the processed data to accurately distinguish between cancer and normal cells.
    - Extracts the feature importance rankings from the trained model to pinpoint genes that are not only statistically significant but also highly predictive, nominating them as potential targets for Lipo-ATRA delivery.

6.  **In Silico Treatment Simulation**
    - Selects the top-ranked target gene and computationally simulates a therapeutic intervention by downregulating its expression (e.g., by 50%) exclusively within the cancer cells of the `AnnData` object.
    - This step leverages efficient algorithms optimized for sparse matrices to ensure performance on large datasets.

7.  **Mechanism of Action Analysis**
    - Re-runs differential expression and enrichment analysis on the post-simulation data. The goal is to uncover the downstream biological consequences of the simulated drug intervention, thereby predicting its mechanism of action.

8.  **Result Saving & Registration**
    - Persists all generated artifacts—plots, data tables, model results, and the final processed `AnnData` object—to the local `outputs` directory.
    - Concludes by attempting to upload and register key result files (the JSON summary and final `.h5ad`) back to LaminDB, completing the analysis cycle and ensuring long-term archival.

## 🚀 How to Run

### 1. Environment Setup

First, ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment (e.g., `venv` or `conda`) to manage project dependencies and avoid conflicts.

```bash
# Create and activate a virtual environment (e.g., with venv)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install all required Python packages from the requirements file
pip install -r requirements.txt
```

**`requirements.txt` file content:**
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

The global variables at the top of the script can be adjusted to fit your needs:
- `MIN_FREE_SPACE_GB`: The minimum required free disk space (in GB) to check before downloading large files.
- `SAVE_LOCAL_COPIES`: Set to `True` to keep a local copy of downloaded remote files in `outputs/external_files` for offline access and debugging.

### 3. Run the Script

Save the code as `analysis.py`. From your terminal, navigate to the script's directory and execute it:

```bash
python analysis.py
```
The script will begin execution and print detailed, real-time log messages to the console, indicating the current step and progress.

## 📂 Output File Structure

All analysis results will be saved in the `./outputs/` directory. Here is a description of the key output files:

-   `processed_hepatoma_data.h5ad`: **Final Data Object**. The fully processed `AnnData` object containing integrated data and simulation results, ready for further exploratory analysis.
-   `lipo_atra_hepatoma_results.json`: **Key Findings Summary**. A JSON file containing the top identified target, model performance metrics (e.g., accuracy), and other critical results.
-   `atra_target_expression.png`: **ATRA Target Expression Plot**. A violin plot visualizing the expression of known ATRA target genes across cell types.
-   `pathway_enrichment.csv`: **Pre-Treatment Pathway Analysis**. A detailed data table of the pathway enrichment results for the initial DEGs.
-   `top_pathways.png`: **Top Pathways Plot**. A bar chart visualizing the most significantly enriched biological pathways.
-   `hepatoma_specific_targets.csv`: **Candidate Target List**. A ranked list of candidate target genes identified by the Random Forest model, with their importance scores.
-   `lipo_atra_effect.png`: **Treatment Simulation Effect Plot**. A comparative plot showing the change in the target gene's expression before and after the *in silico* treatment.
-   `treatment_pathways.csv`: **Post-Treatment Pathway Analysis**. A detailed data table of pathways affected by the simulated therapy, indicating the potential mechanism of action.
-   `external_files/`: **Data Cache Directory**. (If `SAVE_LOCAL_COPIES=True`) Contains local copies of the raw data files downloaded from remote sources.
-   `tmp/`: **Temporary Files Directory**. Used for storing intermediate files during runtime, such as decompressed data.

## 💡 Technical Notes

- **Sparse Matrix Optimization**: During the treatment simulation step, the script employs a highly efficient method to scale values in the `scipy.sparse.csr_matrix`. By directly accessing and modifying indices in the underlying `.data` array, it avoids inefficient slicing and potential data type issues, ensuring high performance and stability with large datasets.
- **Temporary File Management**: The script redirects the system's temporary directory to a local `./outputs/tmp` folder. This provides full control over temporary files, prevents issues related to limited root partition space or permissions, and enhances the script's portability across different computing environments.

## ⚖️ License

This project is licensed under the [MIT License](LICENSE).
