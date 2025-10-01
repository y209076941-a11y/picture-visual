# pages/7_🧬_Bioinformatics.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import io
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Path Configuration and Module Import
# ============================================================================

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from components.sidebar import render_sidebar
    from components.headers import render_page_header, render_section_header, render_info_box
    from utils.data_manager import DataManager
except ImportError as e:
    logger.error(f"Module import failed: {e}")
    st.error(f"⚠️ Critical module import error: {e}")

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Bioinformatics - SYPHU iGEM",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Constants and Configuration
# ============================================================================

# Genetic code (standard)
CODON_TABLE = {
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*',
    'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W'
}

# Amino acid properties
AA_PROPERTIES = {
    'hydrophobic': set('AILMFWV'),
    'polar': set('STNQ'),
    'positive': set('KRH'),
    'negative': set('DE'),
    'aromatic': set('FYW'),
    'small': set('AGST')
}

# Pathway databases
PATHWAY_DATABASES = {
    "KEGG": {
        "categories": [
            "Cell cycle", "DNA replication", "p53 signaling pathway",
            "MAPK signaling pathway", "PI3K-Akt signaling pathway",
            "Wnt signaling pathway", "TGF-beta signaling pathway",
            "Apoptosis", "Autophagy", "mTOR signaling pathway"
        ],
        "url": "https://www.genome.jp/kegg/"
    },
    "GO Biological Process": {
        "categories": [
            "cell division", "DNA repair", "signal transduction",
            "regulation of transcription", "protein phosphorylation",
            "cell migration", "apoptotic process", "immune response"
        ],
        "url": "http://geneontology.org/"
    },
    "GO Molecular Function": {
        "categories": [
            "protein binding", "DNA binding", "ATP binding",
            "kinase activity", "transcription factor activity",
            "enzyme binding", "metal ion binding"
        ],
        "url": "http://geneontology.org/"
    },
    "Reactome": {
        "categories": [
            "Cell Cycle", "DNA Repair", "Signal Transduction",
            "Gene Expression", "Metabolism", "Immune System"
        ],
        "url": "https://reactome.org/"
    }
}

# Color schemes
NUCLEOTIDE_COLORS = {
    'A': '#FF6B6B', 'T': '#4ECDC4', 'G': '#45B7D1', 'C': '#96CEB4',
    'U': '#4ECDC4', 'N': '#BDC3C7'
}

AA_COLORS = {
    'A': '#FF6B6B', 'R': '#4ECDC4', 'N': '#45B7D1', 'D': '#96CEB4',
    'C': '#FECA57', 'E': '#FF9FF3', 'Q': '#54A0FF', 'G': '#5F27CD',
    'H': '#00D2D3', 'I': '#FF9F43', 'L': '#10AC84', 'K': '#EE5A24',
    'M': '#A3CB38', 'F': '#C4E538', 'P': '#009432', 'S': '#0652DD',
    'T': '#9980FA', 'W': '#B53471', 'Y': '#ED4C67', 'V': '#F79F1F'
}


# ============================================================================
# Sequence Analysis Functions
# ============================================================================

def clean_sequence(sequence: str) -> str:
    """
    Remove whitespace and non-alphabetic characters from sequence.

    Parameters
    ----------
    sequence : str
        Raw sequence string.

    Returns
    -------
    str
        Cleaned uppercase sequence.
    """
    return re.sub(r'[^A-Za-z]', '', sequence.upper())


def detect_sequence_type(sequence: str) -> str:
    """
    Automatically detect sequence type (DNA/RNA/Protein).

    Parameters
    ----------
    sequence : str
        Input sequence.

    Returns
    -------
    str
        Detected type: 'DNA', 'RNA', 'Protein', or 'Unknown'.

    Notes
    -----
    Detection is based on character composition:
    - DNA: Only A, T, C, G, N
    - RNA: Only A, U, C, G, N
    - Protein: Standard amino acid codes
    """
    if not sequence:
        return "Unknown"

    sequence = sequence.upper()

    # Check for DNA
    if all(c in 'ATCGN' for c in sequence):
        return "DNA"

    # Check for RNA
    if all(c in 'AUCGN' for c in sequence):
        return "RNA"

    # Check for protein
    if all(c in 'ACDEFGHIKLMNPQRSTVWY*X' for c in sequence):
        return "Protein"

    return "Unknown"


def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content percentage."""
    if not sequence:
        return 0.0
    g_count = sequence.count('G')
    c_count = sequence.count('C')
    return (g_count + c_count) / len(sequence) * 100


def calculate_molecular_weight_dna(sequence: str) -> float:
    """
    Calculate DNA molecular weight (kDa).

    Uses average molecular weights:
    A: 313.2, T: 304.2, G: 329.2, C: 289.2
    """
    weights = {'A': 313.2, 'T': 304.2, 'G': 329.2, 'C': 289.2, 'N': 309.0}
    total = sum(weights.get(base, 309.0) for base in sequence)
    return total / 1000  # Convert to kDa


def calculate_melting_temperature(sequence: str) -> float:
    """
    Estimate DNA melting temperature using basic formula.

    For sequences < 14 nt: Tm = 2(A+T) + 4(G+C)
    For longer sequences: Tm = 64.9 + 41(G+C-16.4)/(A+T+G+C)
    """
    if len(sequence) < 14:
        return 2 * (sequence.count('A') + sequence.count('T')) + \
            4 * (sequence.count('G') + sequence.count('C'))
    else:
        gc_content = calculate_gc_content(sequence) / 100
        return 64.9 + 41 * (gc_content - 0.164)


def reverse_complement(sequence: str) -> str:
    """Calculate reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base, base) for base in reversed(sequence))


def translate_dna(sequence: str, reading_frame: int = 0) -> str:
    """
    Translate DNA sequence to protein.

    Parameters
    ----------
    sequence : str
        DNA sequence.
    reading_frame : int, optional
        Reading frame (0, 1, or 2), default 0.

    Returns
    -------
    str
        Translated protein sequence.
    """
    protein = []
    for i in range(reading_frame, len(sequence) - 2, 3):
        codon = sequence[i:i + 3]
        if len(codon) == 3:
            protein.append(CODON_TABLE.get(codon, 'X'))
    return ''.join(protein)


def calculate_protein_properties(sequence: str) -> Dict[str, Any]:
    """
    Calculate various protein properties.

    Returns dictionary with:
    - molecular_weight: MW in kDa
    - isoelectric_point: estimated pI
    - hydrophobicity: GRAVY score
    - instability_index: protein stability
    """
    # Simplified amino acid weights
    aa_weights = {
        'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
        'E': 147.1, 'Q': 146.2, 'G': 75.1, 'H': 155.2, 'I': 131.2,
        'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
        'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
    }

    # Calculate MW
    mw = sum(aa_weights.get(aa, 110) for aa in sequence) / 1000

    # Calculate hydrophobicity (simplified GRAVY)
    hydropathy = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    gravy = sum(hydropathy.get(aa, 0) for aa in sequence) / len(sequence) if sequence else 0

    return {
        'molecular_weight': mw,
        'length': len(sequence),
        'hydrophobicity': gravy,
        'charge': sequence.count('K') + sequence.count('R') - sequence.count('D') - sequence.count('E')
    }


def find_orfs(sequence: str, min_length: int = 100) -> List[Dict[str, Any]]:
    """
    Find open reading frames in DNA sequence.

    Parameters
    ----------
    sequence : str
        DNA sequence.
    min_length : int, optional
        Minimum ORF length in nucleotides, default 100.

    Returns
    -------
    List[Dict[str, Any]]
        List of ORFs with start, end, frame, and protein sequence.
    """
    orfs = []

    # Search in all 6 frames (3 forward, 3 reverse)
    for strand, seq in [('+', sequence), ('-', reverse_complement(sequence))]:
        for frame in range(3):
            for i in range(frame, len(seq) - 2, 3):
                codon = seq[i:i + 3]
                if codon == 'ATG':  # Start codon
                    for j in range(i + 3, len(seq) - 2, 3):
                        stop_codon = seq[j:j + 3]
                        if stop_codon in ['TAA', 'TAG', 'TGA']:
                            orf_length = j - i + 3
                            if orf_length >= min_length:
                                protein = translate_dna(seq[i:j + 3])
                                orfs.append({
                                    'start': i + 1,
                                    'end': j + 3,
                                    'strand': strand,
                                    'frame': frame + 1,
                                    'length': orf_length,
                                    'protein': protein,
                                    'protein_length': len(protein)
                                })
                            break

    return sorted(orfs, key=lambda x: x['length'], reverse=True)


# ============================================================================
# Gene Enrichment Analysis Functions
# ============================================================================

def simulate_enrichment_analysis(
        gene_list: List[str],
        database: str,
        organism: str,
        pvalue_cutoff: float,
        min_overlap: int,
        max_display: int
) -> List[Dict[str, Any]]:
    """
    Simulate gene enrichment analysis.

    Parameters
    ----------
    gene_list : List[str]
        List of gene symbols.
    database : str
        Pathway database name.
    organism : str
        Organism name.
    pvalue_cutoff : float
        P-value threshold for significance.
    min_overlap : int
        Minimum number of overlapping genes.
    max_display : int
        Maximum number of pathways to display.

    Returns
    -------
    List[Dict[str, Any]]
        List of enriched pathways with statistics.

    Notes
    -----
    This is a demonstration implementation. For production use,
    integrate with actual enrichment analysis tools like:
    - gseapy (Python GSEA)
    - enrichr API
    - clusterProfiler (via rpy2)
    """
    pathways = PATHWAY_DATABASES.get(database, {}).get('categories', [])
    results = []

    np.random.seed(42)  # For reproducibility

    for pathway in pathways[:max_display * 2]:  # Generate more, filter later
        # Simulate p-value (exponential distribution)
        p_value = np.random.exponential(0.02)

        if p_value > pvalue_cutoff:
            continue

        # Simulate overlap
        overlap_count = np.random.randint(min_overlap, min(len(gene_list), 15))
        if overlap_count < min_overlap:
            continue

        # Select random genes
        overlap_genes = np.random.choice(gene_list, overlap_count, replace=False).tolist()

        # Calculate adjusted p-value (Bonferroni)
        adj_pvalue = min(p_value * len(pathways), 1.0)

        # Calculate fold enrichment
        fold_enrichment = np.random.uniform(2.0, 10.0)

        results.append({
            'pathway': pathway,
            'p_value': p_value,
            'adj_pvalue': adj_pvalue,
            'genes_count': overlap_count,
            'genes_ratio': f"{overlap_count}/{len(gene_list)}",
            'overlap_genes': overlap_genes,
            'fold_enrichment': fold_enrichment,
            'neg_log10_p': -np.log10(p_value)
        })

    # Sort by p-value
    results.sort(key=lambda x: x['p_value'])

    return results[:max_display]


def plot_enrichment_dotplot(results: pd.DataFrame) -> go.Figure:
    """
    Create enrichment dot plot.

    Parameters
    ----------
    results : pd.DataFrame
        Enrichment results.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results['fold_enrichment'],
        y=results['pathway'],
        mode='markers',
        marker=dict(
            size=results['genes_count'] * 3,
            color=results['neg_log10_p'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="-log10(p-value)"),
            line=dict(color='white', width=1)
        ),
        text=[f"Pathway: {p}<br>Genes: {c}<br>P-value: {pv:.2e}<br>Fold Enrichment: {fe:.2f}"
              for p, c, pv, fe in zip(results['pathway'], results['genes_count'],
                                      results['p_value'], results['fold_enrichment'])],
        hovertemplate='%{text}<extra></extra>'
    ))

    fig.update_layout(
        title="Gene Enrichment Analysis",
        xaxis_title="Fold Enrichment",
        yaxis_title="",
        yaxis=dict(categoryorder='total ascending'),
        height=max(400, len(results) * 30),
        showlegend=False,
        font=dict(family="Arial, sans-serif", size=12)
    )

    return fig


# ============================================================================
# FASTA Parsing Functions
# ============================================================================

def parse_fasta(content: str) -> List[Dict[str, str]]:
    """
    Parse FASTA format sequences.

    Parameters
    ----------
    content : str
        FASTA formatted text.

    Returns
    -------
    List[Dict[str, str]]
        List of sequence dictionaries with id, description, and sequence.
    """
    sequences = []
    current_id = None
    current_desc = ""
    current_seq = []

    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('>'):
            # Save previous sequence
            if current_id is not None:
                sequences.append({
                    'id': current_id,
                    'description': current_desc,
                    'sequence': ''.join(current_seq)
                })
            # Start new sequence
            parts = line[1:].split(maxsplit=1)
            current_id = parts[0]
            current_desc = parts[1] if len(parts) > 1 else parts[0]
            current_seq = []
        elif line:
            current_seq.append(line)

    # Save last sequence
    if current_id is not None:
        sequences.append({
            'id': current_id,
            'description': current_desc,
            'sequence': ''.join(current_seq)
        })

    return sequences


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_sequence(sequence: str, seq_type: str, max_length: int = 500) -> go.Figure:
    """
    Visualize sequence with color-coded bases/amino acids.

    Parameters
    ----------
    sequence : str
        Sequence to visualize.
    seq_type : str
        Type of sequence ('DNA', 'RNA', or 'Protein').
    max_length : int, optional
        Maximum sequence length to display, default 500.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
        title_suffix = f" (first {max_length} residues)"
    else:
        title_suffix = ""

    colors = NUCLEOTIDE_COLORS if seq_type in ['DNA', 'RNA'] else AA_COLORS

    fig = go.Figure()

    for i, residue in enumerate(sequence):
        color = colors.get(residue, '#BDC3C7')
        fig.add_trace(go.Bar(
            x=[i + 1],
            y=[1],
            marker_color=color,
            hovertext=f"Position {i + 1}: {residue}",
            hoverinfo='text',
            showlegend=False,
            width=1
        ))

    fig.update_layout(
        title=f"{seq_type} Sequence Visualization{title_suffix}",
        xaxis_title="Position",
        yaxis=dict(visible=False),
        height=250,
        showlegend=False,
        font=dict(family="Arial, sans-serif", size=12),
        bargap=0
    )

    return fig


def plot_sequence_composition(sequence: str, seq_type: str) -> go.Figure:
    """Create pie chart of sequence composition."""
    if seq_type == "DNA":
        bases = ['A', 'T', 'G', 'C', 'N']
    elif seq_type == "RNA":
        bases = ['A', 'U', 'G', 'C', 'N']
    else:
        bases = list('ACDEFGHIKLMNPQRSTVWY')

    counts = {base: sequence.count(base) for base in bases if sequence.count(base) > 0}

    if not counts:
        return None

    fig = px.pie(
        values=list(counts.values()),
        names=list(counts.keys()),
        title=f"{seq_type} Sequence Composition",
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(font=dict(family="Arial, sans-serif", size=12))

    return fig


# ============================================================================
# Main Page Rendering
# ============================================================================

def main():
    """Main function to render Bioinformatics page."""

    render_sidebar()

    render_page_header(
        title="Bioinformatics Analysis",
        icon="🧬",
        subtitle="Computational biology tools for sequence and pathway analysis"
    )

    # Feature overview
    render_feature_overview()

    # Main tabs
    render_analysis_tabs()


def render_feature_overview():
    """Display overview of available features."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 1.5rem; border-radius: 10px; text-align: center;'>
            <h3 style='margin: 0; font-size: 1.2rem;'>🧬 Gene Enrichment</h3>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;'>
            Pathway and functional enrichment
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white; padding: 1.5rem; border-radius: 10px; text-align: center;'>
            <h3 style='margin: 0; font-size: 1.2rem;'>📝 Sequence Analysis</h3>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;'>
            DNA/RNA/Protein tools
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    color: white; padding: 1.5rem; border-radius: 10px; text-align: center;'>
            <h3 style='margin: 0; font-size: 1.2rem;'>🔍 Sequence Alignment</h3>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;'>
            Multiple sequence comparison
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
                    color: white; padding: 1.5rem; border-radius: 10px; text-align: center;'>
            <h3 style='margin: 0; font-size: 1.2rem;'>📊 Genome Visualization</h3>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;'>
            Genomic feature mapping
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")


def render_analysis_tabs():
    """Render main analysis tabs."""

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧬 Gene Enrichment",
        "📝 Sequence Analysis",
        "🔍 Sequence Alignment",
        "📊 Genome Browser"
    ])

    with tab1:
        render_enrichment_tab()

    with tab2:
        render_sequence_analysis_tab()

    with tab3:
        render_alignment_tab()

    with tab4:
        render_genome_browser_tab()


# ============================================================================
# Tab 1: Gene Enrichment Analysis (continued)
# ============================================================================

def render_enrichment_tab():
    """Render gene enrichment analysis interface."""

    render_section_header("Gene Set Enrichment Analysis", "🧬")

    render_info_box(
        content="""
        **About Gene Enrichment Analysis:**

        Identify biological pathways and functional categories that are overrepresented
        in your gene list. This analysis helps interpret large-scale experimental results
        by connecting genes to known biological processes.

        **Statistical Method:** Hypergeometric test with multiple testing correction
        (Bonferroni or FDR)

        **Note:** This is a demonstration implementation. For publication-quality analysis,
        use established tools like DAVID, gseapy, or clusterProfiler.
        """,
        box_type="info",
        title="Gene Enrichment Analysis"
    )

    # Input method selection
    input_method = st.radio(
        "Gene List Input Method",
        ["Manual Input", "File Upload", "From Dataset"],
        horizontal=True
    )

    gene_list = []

    if input_method == "Manual Input":
        gene_input = st.text_area(
            "Enter Gene Symbols (one per line)",
            placeholder="TP53\nBRCA1\nEGFR\nMYC\nAKT1\nVEGFA\nPTEN\nKRAS\n...",
            height=200,
            help="Enter official gene symbols, one per line"
        )
        if gene_input.strip():
            gene_list = [g.strip().upper() for g in gene_input.split('\n') if g.strip()]

    elif input_method == "File Upload":
        uploaded_file = st.file_uploader(
            "Upload Gene List File",
            type=['txt', 'csv', 'tsv'],
            help="Text file with one gene per line, or CSV with gene column"
        )
        if uploaded_file:
            content = uploaded_file.getvalue().decode("utf-8")
            gene_list = [g.strip().upper() for g in content.split('\n') if g.strip()]

    elif input_method == "From Dataset":
        if DataManager.validate_dataset():
            df = st.session_state.current_dataset
            gene_col = st.selectbox("Select Gene Column", df.columns)
            if gene_col:
                gene_list = df[gene_col].dropna().astype(str).str.upper().unique().tolist()
                st.info(f"Extracted {len(gene_list)} unique genes from '{gene_col}'")
        else:
            st.warning("No dataset loaded. Please load data in Data Management Hub.")

    # Display gene list statistics
    if gene_list:
        st.success(f"✅ {len(gene_list)} genes loaded")

        with st.expander("View Gene List", expanded=False):
            # Display genes in 3 columns
            cols = st.columns(3)
            genes_per_col = (len(gene_list) + 2) // 3
            for i, col in enumerate(cols):
                with col:
                    start = i * genes_per_col
                    end = min((i + 1) * genes_per_col, len(gene_list))
                    for gene in gene_list[start:end]:
                        st.code(gene, language=None)

    # Analysis parameters
    st.markdown("---")
    st.markdown("#### Analysis Parameters")

    col1, col2 = st.columns(2)

    with col1:
        database = st.selectbox(
            "Pathway Database",
            list(PATHWAY_DATABASES.keys()),
            help="Select biological pathway database"
        )

        organism = st.selectbox(
            "Organism",
            [
                "Homo sapiens (Human)",
                "Mus musculus (Mouse)",
                "Rattus norvegicus (Rat)",
                "Drosophila melanogaster (Fruit fly)",
                "Caenorhabditis elegans (Worm)",
                "Saccharomyces cerevisiae (Yeast)",
                "Escherichia coli (E. coli)"
            ]
        )

    with col2:
        pvalue_cutoff = st.slider(
            "P-value Threshold",
            0.001, 0.05, 0.05, 0.001,
            help="Significance threshold (typically 0.05)"
        )

        correction_method = st.selectbox(
            "Multiple Testing Correction",
            ["Bonferroni", "FDR (Benjamini-Hochberg)", "None"]
        )

        min_overlap = st.slider(
            "Minimum Gene Overlap",
            2, 10, 3,
            help="Minimum genes required in pathway"
        )

        max_display = st.slider(
            "Maximum Pathways to Display",
            5, 50, 20
        )

    # Run analysis button
    if st.button("🚀 Run Enrichment Analysis", type="primary", use_container_width=True):
        if not gene_list:
            st.error("⚠️ Please provide a gene list")
        else:
            with st.spinner("Performing enrichment analysis..."):
                results = simulate_enrichment_analysis(
                    gene_list, database, organism,
                    pvalue_cutoff, min_overlap, max_display
                )

                if results:
                    st.success(f"✅ Found {len(results)} significantly enriched pathways")

                    # Display results
                    display_enrichment_results(results)

                    # Store results
                    store_enrichment_results(gene_list, results, database, organism)
                else:
                    st.warning("No significantly enriched pathways found. Try adjusting parameters.")


def display_enrichment_results(results: List[Dict[str, Any]]):
    """Display enrichment analysis results with visualizations."""

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Significant Pathways", len(results))
    with col2:
        avg_pval = df_results['p_value'].mean()
        st.metric("Mean P-value", f"{avg_pval:.2e}")
    with col3:
        avg_genes = df_results['genes_count'].mean()
        st.metric("Avg Genes/Pathway", f"{avg_genes:.1f}")
    with col4:
        best_pval = df_results['p_value'].min()
        st.metric("Best P-value", f"{best_pval:.2e}")

    st.markdown("---")

    # Visualizations
    col1, col2 = st.columns([2, 1])

    with col1:
        # Dot plot
        fig = plot_enrichment_dotplot(df_results.head(15))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pathway categories pie chart
        fig_pie = px.pie(
            df_results.head(10),
            values='genes_count',
            names='pathway',
            title="Top 10 Pathways by Gene Count"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Detailed results table
    with st.expander("📋 Detailed Results Table", expanded=True):
        display_df = df_results[[
            'pathway', 'p_value', 'adj_pvalue',
            'genes_count', 'genes_ratio', 'fold_enrichment'
        ]].copy()

        display_df['p_value'] = display_df['p_value'].apply(lambda x: f"{x:.2e}")
        display_df['adj_pvalue'] = display_df['adj_pvalue'].apply(lambda x: f"{x:.2e}")
        display_df['fold_enrichment'] = display_df['fold_enrichment'].apply(lambda x: f"{x:.2f}")

        st.dataframe(display_df, use_container_width=True, height=400)

    # Gene-pathway network
    st.markdown("---")
    st.markdown("#### Pathway-Gene Associations")

    selected_pathway = st.selectbox(
        "Select pathway to view associated genes",
        df_results['pathway'].tolist()
    )

    for result in results:
        if result['pathway'] == selected_pathway:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Pathway:** {result['pathway']}")
                st.markdown(f"**P-value:** {result['p_value']:.2e}")
                st.markdown(f"**Adjusted P-value:** {result['adj_pvalue']:.2e}")
                st.markdown(f"**Fold Enrichment:** {result['fold_enrichment']:.2f}×")

            with col2:
                st.markdown(f"**Genes in Pathway:** {result['genes_count']}")
                st.markdown(f"**Gene Ratio:** {result['genes_ratio']}")
                st.markdown("**Overlapping Genes:**")
                for gene in result['overlap_genes']:
                    st.code(gene, language=None)
            break

    # Download results
    st.markdown("---")
    csv = df_results.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name=f"enrichment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def store_enrichment_results(
        gene_list: List[str],
        results: List[Dict[str, Any]],
        database: str,
        organism: str
):
    """Store enrichment results in session state."""

    if 'bioinformatics_results' not in st.session_state:
        st.session_state.bioinformatics_results = []

    analysis_record = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'Gene Enrichment',
        'gene_count': len(gene_list),
        'database': database,
        'organism': organism,
        'significant_pathways': len(results),
        'results': results[:10]  # Store top 10
    }

    st.session_state.bioinformatics_results.append(analysis_record)


# ============================================================================
# Tab 2: Sequence Analysis
# ============================================================================

def render_sequence_analysis_tab():
    """Render sequence analysis tools interface."""

    render_section_header("Sequence Analysis Tools", "📝")

    render_info_box(
        content="""
        **Sequence Analysis Capabilities:**

        - DNA/RNA/Protein sequence analysis
        - GC content and molecular weight calculation
        - Reverse complement and translation
        - Open reading frame (ORF) detection
        - Sequence pattern searching
        - Composition analysis and visualization
        """,
        box_type="info",
        title="Available Tools"
    )

    # Sequence input
    sequence_input = st.text_area(
        "Enter Biological Sequence",
        placeholder="ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG...",
        height=200,
        help="DNA, RNA, or Protein sequence (automatic detection)"
    )

    if not sequence_input.strip():
        st.info("Enter a sequence to begin analysis")
        return

    # Process sequence
    sequence = clean_sequence(sequence_input)
    seq_type = detect_sequence_type(sequence)

    if seq_type == "Unknown":
        st.error("⚠️ Unable to determine sequence type. Please check input.")
        return

    st.success(
        f"✅ Detected as **{seq_type}** sequence | Length: **{len(sequence)}** {'bp' if seq_type in ['DNA', 'RNA'] else 'aa'}")

    # Display sequence with line breaks
    with st.expander("View Sequence", expanded=False):
        formatted_seq = '\n'.join([sequence[i:i + 60] for i in range(0, len(sequence), 60)])
        st.code(formatted_seq, language=None)

    # Analysis tabs
    analysis_tabs = st.tabs([
        "📊 Basic Properties",
        "🔧 Sequence Operations",
        "🔍 Pattern Search",
        "📈 Visualization"
    ])

    with analysis_tabs[0]:
        render_basic_properties(sequence, seq_type)

    with analysis_tabs[1]:
        render_sequence_operations(sequence, seq_type)

    with analysis_tabs[2]:
        render_pattern_search(sequence)

    with analysis_tabs[3]:
        render_sequence_visualization(sequence, seq_type)


def render_basic_properties(sequence: str, seq_type: str):
    """Display basic sequence properties."""

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Length", f"{len(sequence)}")

    with col2:
        if seq_type in ["DNA", "RNA"]:
            gc = calculate_gc_content(sequence)
            st.metric("GC Content", f"{gc:.1f}%")

    with col3:
        if seq_type == "DNA":
            mw = calculate_molecular_weight_dna(sequence)
            st.metric("Molecular Weight", f"{mw:.1f} kDa")

    with col4:
        if seq_type == "DNA":
            tm = calculate_melting_temperature(sequence)
            st.metric("Tm (estimated)", f"{tm:.1f}°C")

    # Additional properties for proteins
    if seq_type == "Protein":
        st.markdown("---")
        st.markdown("#### Protein Properties")

        props = calculate_protein_properties(sequence)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Molecular Weight", f"{props['molecular_weight']:.1f} kDa")
        with col2:
            st.metric("Length", f"{props['length']} aa")
        with col3:
            st.metric("Net Charge", props['charge'])
        with col4:
            st.metric("Hydrophobicity (GRAVY)", f"{props['hydrophobicity']:.2f}")


def render_sequence_operations(sequence: str, seq_type: str):
    """Render sequence manipulation operations."""

    col1, col2 = st.columns(2)

    with col1:
        if seq_type == "DNA":
            st.markdown("#### DNA Operations")

            if st.button("🔄 Reverse Complement", use_container_width=True):
                rev_comp = reverse_complement(sequence)
                st.text_area("Reverse Complement", rev_comp, height=150)

            if st.button("🧬 Translate to Protein", use_container_width=True):
                reading_frame = st.selectbox("Reading Frame", [0, 1, 2])
                protein = translate_dna(sequence, reading_frame)
                st.text_area("Protein Sequence", protein, height=150)

                # Show protein properties
                if protein:
                    props = calculate_protein_properties(protein)
                    st.info(f"Protein: {len(protein)} aa, MW: {props['molecular_weight']:.1f} kDa")

            if st.button("🔍 Find ORFs", use_container_width=True):
                min_length = st.slider("Minimum ORF length (nt)", 50, 300, 100)
                orfs = find_orfs(sequence, min_length)

                if orfs:
                    st.success(f"Found {len(orfs)} ORFs")
                    for i, orf in enumerate(orfs[:10], 1):
                        with st.expander(
                                f"ORF {i}: {orf['start']}-{orf['end']} ({orf['strand']}) - {orf['length']} bp"):
                            st.write(f"**Frame:** {orf['frame']}")
                            st.write(f"**Protein Length:** {orf['protein_length']} aa")
                            st.code(orf['protein'], language=None)
                else:
                    st.info("No ORFs found with specified parameters")

    with col2:
        st.markdown("#### Composition Analysis")

        if st.button("📊 Show Composition", use_container_width=True):
            fig = plot_sequence_composition(sequence, seq_type)
            if fig:
                st.plotly_chart(fig, use_container_width=True)


def render_pattern_search(sequence: str):
    """Render pattern search interface."""

    st.markdown("#### Pattern Search")

    pattern = st.text_input(
        "Search Pattern (regex supported)",
        placeholder="ATG|GTG|TTG",
        help="Use regex for advanced patterns. Examples: ATG (exact), [AG]TG (A or G), ATG{2,} (ATG repeated)"
    )

    if pattern and st.button("🔍 Search", use_container_width=True):
        try:
            matches = re.finditer(pattern, sequence, re.IGNORECASE)
            results = [(m.start() + 1, m.end(), m.group()) for m in matches]

            if results:
                st.success(f"Found {len(results)} match(es)")

                # Display results
                df_matches = pd.DataFrame(results, columns=['Start', 'End', 'Sequence'])
                st.dataframe(df_matches, use_container_width=True)
            else:
                st.info("No matches found")
        except re.error as e:
            st.error(f"Invalid regex pattern: {e}")


def render_sequence_visualization(sequence: str, seq_type: str):
    """Render sequence visualization."""

    if len(sequence) > 500:
        st.warning(f"Sequence length ({len(sequence)}) exceeds visualization limit (500). Showing first 500 residues.")

    fig = visualize_sequence(sequence, seq_type)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Tab 3: Multiple Sequence Alignment
# ============================================================================

def render_alignment_tab():
    """Render multiple sequence alignment interface."""

    render_section_header("Multiple Sequence Alignment", "🔍")

    render_info_box(
        content="""
        **Multiple Sequence Alignment:**

        Upload FASTA-formatted sequences for comparative analysis.
        Identifies conserved regions, mutations, and phylogenetic relationships.

        **Note:** This is a demonstration. For production alignment, use:
        - Clustal Omega
        - MUSCLE
        - MAFFT
        - T-Coffee
        """,
        box_type="info"
    )

    uploaded_files = st.file_uploader(
        "Upload FASTA Files",
        type=['fasta', 'fa', 'fna', 'faa'],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Upload FASTA files to begin alignment")
        return

    # Parse sequences
    sequences = []
    for file in uploaded_files:
        content = file.getvalue().decode("utf-8")
        seqs = parse_fasta(content)
        sequences.extend(seqs)
        st.success(f"✅ {file.name}: {len(seqs)} sequence(s)")

    if sequences:
        st.success(f"Total: {len(sequences)} sequences loaded")

        # Sequence information
        seq_info = pd.DataFrame([
            {
                'ID': seq['id'],
                'Description': seq['description'][:50],
                'Length': len(seq['sequence']),
                'Type': detect_sequence_type(seq['sequence'])
            }
            for seq in sequences
        ])

        st.dataframe(seq_info, use_container_width=True)

        # Alignment button
        if st.button("🔬 Run Alignment (Demo)", type="primary"):
            st.info("Alignment feature requires integration with actual alignment tools (Clustal Omega, MUSCLE, etc.)")


# ============================================================================
# Tab 4: Genome Browser
# ============================================================================

def render_genome_browser_tab():
    """Render genome visualization interface."""

    render_section_header("Genome Visualization", "📊")

    st.info("Interactive genome browser for visualizing genomic features, annotations, and expression data.")

    # This would integrate with actual genomic data
    st.markdown("**Feature**: Coming soon - integrate with genomic databases and annotation files")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
