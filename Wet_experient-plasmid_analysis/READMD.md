# Enhanced Carotenoid Synthesis Plasmid Analyzer

## Overview

The Enhanced Carotenoid Synthesis Plasmid Analyzer is a comprehensive bioinformatics tool designed for advanced analysis of genetic constructs, with specialized capabilities for carotenoid synthesis pathway plasmids. This integrated Python-based platform provides professional-grade sequence characterization, genetic element detection, restriction analysis, ORF prediction, and multi-format visualization capabilities.

## Scientific Background

Carotenoid biosynthesis represents a critical metabolic pathway in both natural biological systems and synthetic biology applications. The pathway involves several key enzymes: phytone synthase (crtB), phytone desaturase (crtI), lycopene cyclase (crtY), and geranylgeranyl pyrophosphate synthase (crtE). Efficient analysis of plasmids containing these genetic elements requires specialized tools that can handle the complexity of synthetic genetic constructs while providing biologically relevant insights.

This analyzer addresses the growing need for comprehensive plasmid characterization in metabolic engineering projects, particularly those focused on carotenoid production in various host organisms including E. coli, yeast, and plant systems.

## Technical Architecture

### Core Dependencies
- **Biopython**: Core sequence handling and biological computations
- **Matplotlib/Seaborn**: Scientific visualization and plotting
- **NumPy/Pandas**: Numerical computation and data manipulation
- **Plotly**: Interactive visualization components
- **Upsetplot**: Set visualization for genetic elements
- **Pypinyin**: Chinese character handling for bilingual reporting
- **Chardet**: Character encoding detection for file parsing

### Optional Dependencies
- **DNA Features Viewer**: Advanced plasmid map visualization
- **NetworkX**: Network analysis for enzyme interactions
- **Pysankey2**: Sankey diagram generation

## Installation

### Basic Installation
```bash
pip install biopython matplotlib seaborn numpy pandas plotly upsetplot pypinyin chardet
```

### Full Installation with Optional Features
```bash
pip install biopython matplotlib seaborn numpy pandas plotly upsetplot pypinyin chardet dna-features-viewer networkx pysankey2
```

### Verification Installation
```bash
# Test basic functionality
python -c "from Bio.Seq import Seq; print(Seq('ATCG').reverse_complement())"

# Test visualization components
python -c "import matplotlib.pyplot as plt; plt.plot([1,2,3],[4,5,6]); print('Matplotlib working')"
```

## Detailed Usage

### Basic Command Line Usage
```bash
python plasmid_analysis_integrated_full.py
```

### Advanced Programmatic Usage
```python
from plasmid_analysis_integrated_full import EnhancedCarotenoidPlasmidAnalyzer

# Initialize with custom parameters
analyzer = EnhancedCarotenoidPlasmidAnalyzer(
    file_paths=["plasmid1.fasta", "plasmid2.gb"],
    project_name="My Carotenoid Project",
    # Custom enzyme configurations
    EcoRI=Restriction.EcoRI,
    BamHI=Restriction.BamHI,
    # Add other enzymes as needed
)

# Run specific analysis components
analyzer.read_sequences()
results = analyzer.analysis_results

# Generate specific report components
analyzer.generate_publication_grade_plots("output_directory", "en")
```

### Configuration Options

#### Genetic Element Pattern Customization
```python
# Custom genetic element patterns
custom_patterns = {
    'my_gene': [
        r'ATG[ATCG]{50,200}TAA|TAG|TGA',  # Regex pattern
        'Description of my custom gene'    # Human-readable description
    ],
    'custom_promoter': [
        r'TTGACA[ATCGN]{15,25}TATAAT',
        'Custom promoter sequence'
    ]
}

# Integrate with analyzer
analyzer.gene_patterns.update(custom_patterns)
```

#### Restriction Enzyme Configuration
```python
from Bio.Restriction import *

# Custom enzyme set
custom_enzymes = {
    'EcoRI': EcoRI,
    'BamHI': BamHI,
    'HindIII': HindIII,
    'XhoI': XhoI,
    'SalI': SalI,
    'NdeI': NdeI,
    'NotI': NotI,
    # Add additional enzymes as needed
}

analyzer.construction_enzymes.update(custom_enzymes)
```

## Input File Format Specifications

### Supported Formats
1. **FASTA** (.fasta, .fa)
   - Standard nucleotide sequence format
   - Supports multi-sequence files (analyzes first sequence)
   - Handles ambiguous nucleotides (N, R, Y, S, W, K, M, B, D, H, V)

2. **GenBank** (.gb, .genbank)
   - Maintains annotation information
   - Extracts sequence features when available
   - Preserves metadata for reporting

3. **EMBL** (.embl)
   - European Molecular Biology Laboratory format
   - Similar feature extraction to GenBank

4. **Raw Text** (.txt, .dna, no extension)
   - Automatic sequence extraction from text
   - Handles various encodings (UTF-8, Latin-1, auto-detected)
   - Filters non-sequence characters intelligently

### File Encoding Handling
The analyzer automatically detects and handles multiple text encodings:
- UTF-8 (recommended)
- Latin-1 (fallback)
- ASCII
- Automatic detection via chardet library

## Analysis Modules Detailed Specification

### 1. Sequence Statistics Engine

#### Implementation Details
```python
# GC Content Calculation
def gc_fraction(seq):
    seq = seq.upper()
    gc_count = seq.count('G') + seq.count('C')
    return gc_count / len(seq) if len(seq) > 0 else 0

# Molecular Weight Calculation
try:
    mw = molecular_weight(seq, 'DNA')
except Exception:
    unambiguous_seq = re.sub(r'[^ATGC]', '', seq)
    mw = molecular_weight(unambiguous_seq, 'DNA') if unambiguous_seq else 0.0
```

#### Output Metrics
- **Sequence Length**: Base pair count
- **GC Content**: Percentage with two decimal precision
- **Molecular Weight**: Daltons (Da)
- **Base Composition**: Absolute counts and percentages
- **Ambiguous Base Handling**: Reporting and filtering

### 2. Genetic Element Detection System

#### Pattern Matching Algorithm
```python
def detect_genetic_elements(self, seq):
    elements = defaultdict(list)
    for name, (pattern, desc) in self.gene_patterns.items():
        try:
            for match in re.finditer(pattern, seq, re.IGNORECASE):
                elements[name].append({
                    'start': match.start(),
                    'end': match.end(),
                    'length': match.end() - match.start(),
                    'description': desc
                })
        except re.error:
            continue
    return dict(elements)
```

#### Default Genetic Elements Detected

##### Carotenoid Pathway Genes
- **crtE**: Geranylgeranyl pyrophosphate synthase
  - Pattern: `r'ATG[ATCGN]{850,950}T(?:AA|AG|GA)'`
  - Length: ~900 bp expected

- **crtB**: Phytone synthase
  - Pattern: `r'ATG[ATCGN]{900,1000}T(?:AA|AG|GA)'`
  - Length: ~950 bp expected

- **crtI**: Phytone desaturase
  - Pattern: `r'ATG[ATCGN]{1400,1500}T(?:AA|AG|GA)'`
  - Length: ~1450 bp expected

- **crtY**: Lycopene cyclase
  - Pattern: `r'ATG[ATCGN]{1100,1200}T(?:AA|AG|GA)'`
  - Length: ~1150 bp expected

##### Selection Markers
- **ampR**: Ampicillin resistance (beta-lactamase)
- **kanR**: Kanamycin resistance (aminoglycoside phosphotransferase)
- **tetR**: Tetracycline resistance (efflux pump)
- **cmR**: Chloramphenicol resistance (chloramphenicol acetyltransferase)

##### Replication Origins
- **ori_pBR322**: pBR322 derived origin
- **ori_pUC**: pUC plasmid origin
- **ori_ColE1**: ColE1-type origin

##### Regulatory Elements
- **trc_promoter**: Hybrid trp/lac promoter
- **T7_promoter**: T7 bacteriophage promoter
- **lac_promoter**: Lactose operon promoter
- **lac_operator**: Lac repressor binding site
- **RBS_strong**: Strong ribosome binding site
- **T7_terminator**: T7 transcription terminator
- **rrnB_T1_terminator**: E. coli ribosomal RNA terminator

### 3. Restriction Analysis Module

#### Enzyme Site Prediction
```python
def analyze_restriction_sites(self, seq):
    bio_seq = Seq(seq)
    sites = {}
    all_sites = {}
    for enzyme_name, enzyme_class in self.construction_enzymes.items():
        try:
            cut_sites = enzyme_class().search(bio_seq)
            if cut_sites:
                sites[enzyme_name] = cut_sites
                all_sites[enzyme_name] = {'count': len(cut_sites), 'positions': cut_sites}
        except Exception:
            continue
    mcs = self.find_multiple_cloning_site(seq, sites)
    return {'sites': sites, 'all_sites': all_sites, 'multiple_cloning_site': mcs}
```

#### Multiple Cloning Site (MCS) Detection
```python
def find_multiple_cloning_site(self, seq, sites):
    all_positions = sorted([pos for pos_list in sites.values() for pos in pos_list])
    if not all_positions:
        return None
    
    # Sliding window analysis for MCS detection
    best_region = {'start': 0, 'end': 0, 'site_count': 0, 'sites': []}
    window_size = 100  # Standard MCS size range
    
    for i, start_pos in enumerate(all_positions):
        count = 0
        current_end = start_pos
        for j in range(i, len(all_positions)):
            if all_positions[j] - start_pos < window_size:
                count += 1
                current_end = all_positions[j]
            else:
                break
                
        if count > best_region['site_count']:
            best_region.update({'site_count': count, 'start': start_pos, 'end': current_end})
    
    # Minimum threshold for MCS identification
    if best_region['site_count'] >= 3:
        best_region['sites'] = sorted(
            [(enzyme, pos) for enzyme, positions in sites.items() for pos in positions
             if best_region['start'] <= pos <= best_region['end']],
            key=lambda x: x[1]
        )
        return best_region
    return None
```

### 4. ORF Prediction and Protein Analysis

#### Six-Frame Translation Implementation
```python
def analyze_orfs_and_proteins(self, seq, min_prot_len=50):
    orfs = []
    codon_counts = Counter()
    bio_seq = Seq(seq)

    for strand, nuc in [(+1, bio_seq), (-1, bio_seq.reverse_complement())]:
        for frame in range(3):
            frame_seq = nuc[frame:]
            trans = frame_seq.translate(to_stop=False)
            trans_len = len(trans)
            aa_start = 0
            while aa_start < trans_len:
                aa_stop = trans.find("*", aa_start)
                if aa_stop == -1:
                    aa_stop = trans_len
                prot_seq = str(trans[aa_start:aa_stop])
                if len(prot_seq) >= min_prot_len:
                    # Filter out sequences with ambiguous amino acids
                    if any(aa in prot_seq for aa in ['X', 'B', 'Z']):
                        aa_start = aa_stop + 1
                        continue
                    
                    # Calculate positions considering frame and strand
                    start = frame + aa_start * 3
                    end = frame + aa_stop * 3 + 3
                    if strand == -1:
                        start, end = len(seq) - end, len(seq) - start
                    
                    try:
                        # Protein analysis
                        p_analysis = ProteinAnalysis(prot_seq)
                        
                        # Codon Adaptation Index calculation
                        orf_dna_seq = str(frame_seq[aa_start * 3: aa_stop * 3])
                        cai_val = 0.0
                        try:
                            if self.cai_calculator:
                                cai_val = self.cai_calculator.cai_for_gene(orf_dna_seq)
                        except Exception:
                            cai_val = 0.0

                        orf_data = {
                            'id': f"ORF_{len(orfs) + 1}",
                            'protein': prot_seq,
                            'start': start,
                            'end': end,
                            'strand': strand,
                            'length_aa': len(prot_seq),
                            'mw_da': p_analysis.molecular_weight(),
                            'pI': p_analysis.isoelectric_point(),
                            'instability': p_analysis.instability_index(),
                            'cai': cai_val
                        }
                        orfs.append(orf_data)
                        
                        # Codon usage analysis
                        codon_seq = str(frame_seq[aa_start * 3: aa_stop * 3])
                        for i in range(0, len(codon_seq), 3):
                            codon = codon_seq[i:i + 3]
                            if len(codon) == 3:
                                codon_counts[codon] += 1
                    except Exception:
                        pass
                aa_start = aa_stop + 1
    return {'orfs': orfs, 'codon_usage': dict(codon_counts)}
```

#### Protein Property Calculations
- **Molecular Weight**: Calculated using Biopython's molecular_weight method
- **Isoelectric Point (pI)**: Computed using the Bjellqvist method
- **Instability Index**: Prediction of protein stability (index < 40 considered stable)
- **Codon Adaptation Index (CAI)**: Measure of codon usage optimality

## Visualization System

### Publication-Quality Plots

#### Nature-Style Formatting Specifications
```python
def set_nature_style():
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['ytick.minor.width'] = 0.5
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
```

#### Custom Color Schemes
```python
SCIENCE_COLORS = {
    'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'pastel': ['#a1c9f4', '#ffb482', '#8de5a1', '#ff9f9b', '#d0bbff',
               '#debb9b', '#fab0e4', '#cfcfcf', '#fffea3', '#b9f2f0'],
    'diverging': ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087',
                  '#f95d6a', '#ff7c43', '#ffa600']
}
```

### Visualization Types Generated

#### 1. GC Skew Analysis
- Cumulative GC skew calculation
- Origin and terminus prediction
- Window-based analysis (default: 1000bp windows)

#### 2. Codon Usage Heatmaps
- Relative Synonymous Codon Usage (RSCU) calculation
- Amino acid-grouped visualization
- Publication-ready formatting

#### 3. Plasmid Maps
- Circular DNA representation
- Feature annotation with color coding
- Restriction site marking

#### 4. Sequence Comparison Plots
- GC content comparison across plasmids
- Length distribution visualization
- Element composition analysis

#### 5. Advanced Visualizations
- UpSet plots for genetic element comparison
- Network graphs for enzyme interactions
- 3D interactive plots for multi-parameter visualization
- Metabolic pathway mapping

## Output System

### Bilingual Reporting Structure

#### Directory Organization
```
Project_Name_报告_YYYYMMDD_HHMMSS/
├── Report_EN/                          # English reports
│   ├── Comprehensive_Analysis_Report.txt
│   ├── Restriction_Enzyme_Analysis/
│   │   ├── Plasmid1_Restriction_Enzyme_Analysis_Report.txt
│   │   └── Plasmid2_Restriction_Enzyme_Analysis_Report.txt
│   ├── Publication_Grade_Plots/
│   │   ├── GC_Content_Comparison.png
│   │   ├── Sequence_Length_Comparison.png
│   │   ├── Nature_GC_Skew_*.png
│   │   ├── Nature_Codon_Usage_RSCU.png
│   │   ├── Nature_Element_Distribution.png
│   │   ├── ORF_Length_Histogram.png
│   │   ├── ORF_Length_GC_Scatter.png
│   │   └── ORF_Property_Distribution.png
│   ├── Advanced_Analysis/
│   │   ├── Genetic_Element_Comparison_UpSet.png
│   │   ├── Enzyme_Plasmid_Network.png
│   │   └── Pathway_Coverage_Map.png
│   └── Interactive_Plots/
│       └── interactive_3d_plot.html
├── 报告_ZH/                            # Chinese reports (mirrored structure)
└── analysis_results_YYYYMMDD_HHMMSS.pkl  # Serialized analysis data
```

### Report Content Specifications

#### Comprehensive Analysis Report
- Sequence statistics and metrics
- Detected genetic elements with positions
- Summary information for quick overview

#### Restriction Analysis Reports
- Enzyme cutting sites with positions
- Multiple Cloning Site identification
- Digest planning information

#### ORF and Protein Reports
- Predicted open reading frames
- Protein characteristics and properties
- Codon usage statistics

## Performance Considerations

### Memory Management
- Optimized for plasmid-sized sequences (<20kb)
- Streaming file reading for large sequences
- Efficient data structures for sequence storage

### Processing Optimization
- Multiprocessing support for large datasets
- Caching of intermediate results
- Incremental analysis capabilities

### Large-Scale Analysis
For batch processing of multiple plasmids:

```python
# Batch processing example
import os
from concurrent.futures import ProcessPoolExecutor

def process_plasmid(file_path):
    analyzer = EnhancedCarotenoidPlasmidAnalyzer([file_path])
    if analyzer.run_complete_analysis():
        return analyzer.analysis_results
    return None

# Process all files in a directory
plasmid_dir = "path/to/plasmids"
file_paths = [os.path.join(plasmid_dir, f) for f in os.listdir(plasmid_dir) 
              if f.endswith(('.fasta', '.gb', '.embl', '.dna'))]

# Parallel processing
with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_plasmid, file_paths))
```

## Validation and Quality Control

### Sequence Quality Checks
- Ambiguous base detection and reporting
- Sequence length validation
- Format compatibility checking

### Analysis Validation
- Cross-validation with known plasmid sequences
- Comparison with established tools
- Manual verification of key findings

### Error Handling
- Comprehensive exception handling
- Detailed error logging
- Graceful degradation of features

## Extension and Customization

### Adding New Genetic Elements
```python
# Example: Adding a new fluorescent protein
new_elements = {
    'gfp': [
        r'ATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCCAATTCTTGTTGAATTAGATGGTGATGTTAATGGGCACAAATTTTCTGTCAGTGGAGAGGGTGAAGGTGATGCTACATACGGAAAGCTTACCCTTAAATTTATTTGCACTACTGGAAAACTACCTGTTCCATGGCCAACACTTGTCACTACTTTCTCTTATGGTGTTCAATGCTTTTCCCGTTATCCGGATCATATGAAACGGCATGACTTTTTCAAGAGTGCCATGCCCGAAGGTTATGTACAGGAACGCACTATATCTTTCAAAGATGACGGGAACTACAAGACGCGTGCTGAAGTCAAGTTTGAAGGTGATACCCTTGTTAATCGTATCGAGTTAAAAGGTATTGATTTTAAAGAAGATGGAAACATTCTCGGACACAAACTCGAGTACAACTATAACTCACACAATGTATACATCACGGCAGACAAACAAAAGAATGGAATCAAAGCTAACTTCAAAATTCGCCACAACATTGAAGATGGATCCGTTCAACTAGCAGACCATTATCAACAAAATACTCCAATTGGCGATGGCCCTGTCCTTTTACCAGACAACCATTACCTGTCGACACAATCTGCCCTTTCGAAAGATCCCAACGAAAAGCGTGACCACATGGTCCTTCTTGAGTTTGTAACTGCTGCTGGGATTACACATGGCATGGATGAGCTCTACAAATAA',
        'Green Fluorescent Protein (GFP)'
    ]
}

analyzer.gene_patterns.update(new_elements)
```

### Custom Visualization Templates
```python
# Custom plot styling
def set_custom_style():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.0
    # Additional custom settings
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **Memory Errors with Large Sequences**
   - Use 64-bit Python installation
   - Increase system swap space
   - Process sequences individually rather than in batch

2. **Encoding Problems with Text Files**
   - Specify encoding explicitly when possible
   - Use UTF-8 formatted files
   - Check file contents with hex editor if necessary

3. **Missing Dependencies**
   - Use the provided requirements.txt
   - Install optional dependencies as needed
   - Verify installation with test commands

4. **Visualization Rendering Issues**
   - Set appropriate matplotlib backend
   - Ensure proper display configuration
   - Use non-interactive backends for server environments

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run analysis with debug output
analyzer = EnhancedCarotenoidPlasmidAnalyzer(file_paths)
analyzer.run_complete_analysis()
```

## Citation and Attribution

When using this tool in publications, please cite:

```bibtex
@software{CarotenoidPlasmidAnalyzer2023,
  title = {Enhanced Carotenoid Synthesis Plasmid Analyzer},
  author = {Xianpu China iGEM Team},
  year = {2023},
  url = {https://gitlab.igem.org/xianpu/syphu-china-model},
  version = {1.0},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

## Support and Community

### Issue Reporting
- Use the GitLab issue tracker
- Include example files when possible
- Provide detailed error messages

### Contribution Guidelines
- Fork the repository
- Create feature branches
- Submit pull requests with tests
- Follow PEP 8 coding standards

### Community Resources
- Documentation wiki
- Example datasets
- Tutorial notebooks
- Frequently Asked Questions

## License Information

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Biopython development team for core bioinformatics capabilities
- Matplotlib and Seaborn teams for visualization tools
- iGEM community for testing and feedback
- Contributing developers and researchers

## Version History

- **1.0.0** (Current): Initial release with core functionality
- **0.9.0**: Beta release with bilingual reporting
- **0.8.0**: Alpha release with basic analysis capabilities

## Future Development Roadmap

### Planned Features
- CRISPR guide RNA design integration
- Protein structure prediction interface
- Metabolic flux analysis coupling
- Machine learning-based element prediction
- Cloud-based processing capabilities
- Additional language support
- Enhanced interactive reports
- API for web service integration

### Optimization Goals
- Improved memory efficiency
- Faster processing algorithms
- Enhanced parallel processing
- Better large-sequence handling
- Extended format support

---

**Note**: This tool is designed for research purposes and should be validated with experimental data for critical applications. Always verify computational predictions with laboratory experiments before making biological conclusions.