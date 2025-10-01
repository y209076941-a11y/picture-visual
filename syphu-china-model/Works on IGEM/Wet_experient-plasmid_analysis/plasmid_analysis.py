#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plasmid_analysis_integrated_full.py
完整整合版 — 类胡萝卜素合成质粒分析器
包含:
 - 序列读取 (fasta / genbank / embl / 文本提取)
 - 基本统计 (长度, GC, 碱基组成, 分子量)
 - 遗传元件检测 (可自定义模式)
 - 限制酶分析 & MCS 检测
 - ORF 预测与蛋白质性质分析 (分子量, pI, 不稳定指数)
 - 出版级与交互式图表 (桑基、组成、GC对比、长度对比、质粒图谱、密码子热图、ORF属性分布、同源性热图、GC Skew、UpSet、网络图、3D交互图、通路图、CAI)
 - 双语报告 (中文 + 英文)
Usage:
    python plasmid_analysis_integrated_full.py
    或在脚本末尾修改 file_paths 列表并运行
"""
import os
import re
import warnings
from collections import defaultdict, Counter
from datetime import datetime

# 科学计算与可视化
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无显示环境可用（批处理）
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ========== 安全文件名 + pypinyin 支持 + 日志配置 ==========
import logging
import builtins
from datetime import datetime as _dt
try:
    from pypinyin import lazy_pinyin
except Exception:
    lazy_pinyin = None

import re as _re
def safe_filename(name: str) -> str:
    """
    将文件名中的中文或特殊字符替换为可读的 ASCII/下划线。
    - 若安装了 pypinyin，会将中文字符转换为拼音拼接（更可读）
    - 保留英文字母、数字、连字符、下划线和点
    """
    if not isinstance(name, str):
        name = str(name)
    base, ext = os.path.splitext(name)
    if lazy_pinyin:
        converted = []
        for ch in base:
            if '\u4e00' <= ch <= '\u9fff':
                try:
                    p = ''.join(lazy_pinyin(ch))
                    converted.append(p)
                except Exception:
                    converted.append(ch)
            else:
                converted.append(ch)
        base = ''.join(converted)
    base_safe = _re.sub(r"[^\w\-.]", "_", base)
    ext_safe = _re.sub(r"[^\w\-.]", "_", ext)
    safe = (base_safe + ext_safe).strip("_")
    return safe or "file"

# 日志文件名（使用 safe_filename），含时间戳
_log_ts = _dt.now().strftime("%Y%m%d_%H%M%S")
_log_filename = safe_filename(f"run_log_{_log_ts}.txt")

# 初始化 logging：控制台 + 文件
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_filename, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# wrapper: print -> logging.info + console
_original_print = builtins.print
def _print_and_log(*args, **kwargs):
    try:
        _original_print(*args, **kwargs)
    except Exception:
        pass
    try:
        text = " ".join(str(a) for a in args)
        logger.info(text)
    except Exception:
        pass
builtins.print = _print_and_log

# 可选增强库（按需导入）
try:
    import seaborn as sns
except Exception:
    sns = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

try:
    import networkx as nx
except Exception:
    nx = None

try:
    from upsetplot import from_contents, UpSet
except Exception:
    from_contents = None
    UpSet = None

try:
    from pysankey2 import Sankey
except Exception:
    Sankey = None

# Biopython
try:
    from Bio import SeqIO, Align
    from Bio.Seq import Seq
    from Bio.SeqUtils import molecular_weight, GC
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from Bio.Restriction import *
    from Bio.SeqUtils.CodonUsage import CodonAdaptationIndex
except Exception as e:
    raise ImportError("需要安装 biopython: pip install biopython") from e

# DNA features drawing (可选)
try:
    from dna_features_viewer import GraphicFeature, GraphicRecord
except Exception:
    GraphicFeature = None
    GraphicRecord = None

# 编码检测
try:
    import chardet
except Exception:
    chardet = None

warnings.filterwarnings('ignore')
# 设置Nature风格的全局参数
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
    # 可选：使用灰度颜色
    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.gray(np.linspace(0, 1, 10)))

# 然后在生成图表前调用
set_nature_style()
# Matplotlib 中文字体自动查找与样式设置
def setup_matplotlib_chinese_font():
    font_names = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'PingFang SC', 'WenQuanYi Micro Hei']
    for font_name in font_names:
        try:
            matplotlib.font_manager.findfont(font_name, fallback_to_default=False)
            print(f"✓ 找到并设置中文字体: {font_name}")
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            if sns:
                try:
                    sns.set_theme(font=font_name)
                except Exception:
                    pass
            return True
        except Exception:
            continue
    print("⚠ 警告: 未找到任何可用的中文字体。图表中的中文可能显示为方块。")
    print("  请尝试安装 'SimHei', 'Microsoft YaHei' 或 'WenQuanYi Micro Hei' 等字体。")
    return False

setup_matplotlib_chinese_font()
set_nature_style()  # 添加这行plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# 自定义颜色方案
SCIENCE_COLORS = {
    'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'pastel': ['#a1c9f4', '#ffb482', '#8de5a1', '#ff9f9b', '#d0bbff',
               '#debb9b', '#fab0e4', '#cfcfcf', '#fffea3', '#b9f2f0'],
    'diverging': ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087',
                  '#f95d6a', '#ff7c43', '#ffa600']
}

def create_custom_colormap(n_colors=10, cmap_name='science_primary'):
    if cmap_name == 'science_primary':
        colors = SCIENCE_COLORS['primary'][:n_colors]
    elif cmap_name == 'science_pastel':
        colors = SCIENCE_COLORS['pastel'][:n_colors]
    elif cmap_name == 'science_diverging':
        colors = SCIENCE_COLORS['diverging'][:n_colors]
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, n_colors))
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=n_colors)

# -----------------------------------
# 视觉化工具类 (简明)
# -----------------------------------
class AdvancedScientificVisualizer:
    def __init__(self):
        self.color_palette = SCIENCE_COLORS['primary']
        self.pastel_palette = SCIENCE_COLORS['pastel']

    def create_sankey_bubble_plot(self, **kwargs):
        source = kwargs.get('source')
        target = kwargs.get('target')
        value = kwargs.get('value')
        save_path = kwargs.get('save_path', "sankey_bubble.png")
        title = kwargs.get('title', "Sankey-Bubble Plot")

        # 根据 source, target, value 绘图
        print(f"生成 Sankey-Bubble 图: {save_path}")

    def create_volcano_plot(self, data_df, pvalue_col, logfc_col, gene_col,
                            pvalue_thresh=0.05, logfc_thresh=1.0, title='Volcano Plot',
                            save_path=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        data_df = data_df.copy()
        data_df['neg_log10_pvalue'] = -np.log10(np.clip(data_df[pvalue_col], 1e-300, None))
        data_df['significance'] = 'Not Significant'
        data_df.loc[(data_df[pvalue_col] < pvalue_thresh) &
                    (data_df[logfc_col].abs() < logfc_thresh), 'significance'] = 'Significant only'
        data_df.loc[(data_df[pvalue_col] >= pvalue_thresh) &
                    (data_df[logfc_col].abs() >= logfc_thresh), 'significance'] = 'FC only'
        data_df.loc[(data_df[pvalue_col] < pvalue_thresh) &
                    (data_df[logfc_col].abs() >= logfc_thresh), 'significance'] = 'Both'

        categories = data_df['significance'].unique()
        colors = self.color_palette[:len(categories)]
        for i, category in enumerate(categories):
            subset = data_df[data_df['significance'] == category]
            ax.scatter(subset[logfc_col], subset['neg_log10_pvalue'],
                       c=[colors[i]], label=category, alpha=0.7, s=30)

        ax.axhline(-np.log10(pvalue_thresh), color='grey', linestyle='--', alpha=0.8)
        ax.axvline(-logfc_thresh, color='grey', linestyle='--', alpha=0.8)
        ax.axvline(logfc_thresh, color='grey', linestyle='--', alpha=0.8)

        ax.set_xlabel('Log2 Fold Change', fontsize=14)
        ax.set_ylabel('-Log10 P-value', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 火山图已保存: {os.path.basename(save_path)}")
        plt.close(fig)
        return fig

    # 其它方法与之前示例相同：heatmap, pca, 3d, sankey...
    # 为简洁起见，这里保留你需要的图像生成逻辑（已在主类中实现）。
    # ... (你可以把更多通用图表函数放在这里)

# -----------------------------------
# 主分析器类
# -----------------------------------
# 在脚本开头，导入其他模块之后添加：
def gc_fraction(seq):
    """计算 DNA 序列的 GC 含量比例"""
    seq = seq.upper()
    gc_count = seq.count('G') + seq.count('C')
    return gc_count / len(seq) if len(seq) > 0 else 0


class EnhancedCarotenoidPlasmidAnalyzer:
    def _generate_nature_style_gc_skew(self, pub_dir, loc_text):
        """生成Nature风格的GC偏斜图"""
        for name, res in self.analysis_results.items():
            seq = res['sequence']
            if not seq:
                continue

            window_size = 1000
            gc_skews = []
            positions = []

            for i in range(0, len(seq) - window_size, window_size):
                window = seq[i:i + window_size]
                g_count = window.count('G')
                c_count = window.count('C')
                if g_count + c_count > 0:
                    gc_skew = (g_count - c_count) / (g_count + c_count)
                else:
                    gc_skew = 0
                gc_skews.append(gc_skew)
                positions.append(i + window_size // 2)

            fig, ax = plt.subplots(figsize=(3.54, 2.0))  # Nature单栏宽度
            ax.plot(positions, gc_skews, color='#1f77b4', linewidth=0.8)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.set_xlabel('Position (bp)')
            ax.set_ylabel('GC skew')
            ax.set_title(f'GC Skew: {name}')

            # Nature风格调整
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            save_path = os.path.join(pub_dir, f"Nature_GC_Skew_{name}.png")
            plt.savefig(save_path, dpi=600)
            plt.close(fig)
        # 增加别名
        _generate_base_composition_radar_plot = self._generate_composition_data_and_plot
    def _generate_nature_style_codon_usage(self, pub_dir, loc_text):
        """生成Nature风格的密码子使用热图"""
        total_codon_counts = Counter()
        for res in self.analysis_results.values():
            total_codon_counts.update(res['orfs_analysis']['codon_usage'])

        if not total_codon_counts:
            return

        # 密码子到氨基酸的映射
        codon_map = {
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

        # 计算相对同义密码子使用频率 (RSCU)
        aa_codon_counts = defaultdict(Counter)
        for codon, count in total_codon_counts.items():
            aa = codon_map.get(codon)
            if aa:
                aa_codon_counts[aa][codon] = count

        rscu_values = {}
        for aa, codon_counts in aa_codon_counts.items():
            total = sum(codon_counts.values())
            num_synonymous = len(codon_counts)
            for codon, count in codon_counts.items():
                rscu = (count / total) * num_synonymous
                rscu_values[codon] = rscu

        # 创建数据矩阵
        amino_acids = sorted(aa_codon_counts.keys())
        codons = []
        for aa in amino_acids:
            codons.extend(sorted(aa_codon_counts[aa].keys()))

        data = []
        for aa in amino_acids:
            row = []
            for codon in codons:
                if codon_map.get(codon) == aa:
                    row.append(rscu_values.get(codon, 0))
                else:
                    row.append(np.nan)
            data.append(row)

        data = np.array(data)

        # 绘制热图
        fig, ax = plt.subplots(figsize=(7.2, 4.0))  # Nature双栏宽度
        im = ax.imshow(data, cmap='viridis', aspect='auto')

        # 设置刻度
        ax.set_xticks(range(len(codons)))
        ax.set_xticklabels(codons, rotation=90, fontsize=4)
        ax.set_yticks(range(len(amino_acids)))
        ax.set_yticklabels(amino_acids, fontsize=6)

        # 添加网格线
        ax.set_xticks(np.arange(-0.5, len(codons), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(amino_acids), 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('RSCU', fontsize=6)
        cbar.ax.tick_params(labelsize=5)

        ax.set_title('Relative Synonymous Codon Usage (RSCU)', fontsize=7)

        save_path = os.path.join(pub_dir, "Nature_Codon_Usage_RSCU.png")
        plt.savefig(save_path, dpi=600)
        plt.close(fig)

    def _generate_nature_style_element_distribution(self, pub_dir, loc_text, plasmid=None):
        """生成 Nature 风格的元件分布图"""
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import os

        element_data = []
        for name, res in self.analysis_results.items():
            for element_type, elements in res['genetic_elements'].items():
                for element in elements:
                    element_data.append({
                        'plasmid': name,
                        'element': element_type,
                        'start': element['start'],
                        'end': element['end'],
                        'length': element['length']
                    })

        if not element_data:
            print("⚠️ 警告: 没有元件数据，跳过 Nature 风格图生成。")
            return

        df = pd.DataFrame(element_data)

        # 创建图形
        fig, ax = plt.subplots(figsize=(7.2, max(3.0, len(df['plasmid'].unique()) * 0.4)))  # 高度根据质粒数量动态调整

        # 为每个质粒创建轨道
        plasmids = df['plasmid'].unique()
        y_positions = {p: i for i, p in enumerate(plasmids)}  # 修正：使用 plasmids

        # 绘制每个元件
        colors = plt.cm.Set3(np.linspace(0, 1, len(df['element'].unique())))
        color_map = {elem: colors[i] for i, elem in enumerate(df['element'].unique())}

        for _, row in df.iterrows():
            ax.plot([row['start'], row['end']],
                    [y_positions[row['plasmid']], y_positions[row['plasmid']]],
                    color=color_map[row['element']],
                    linewidth=5,
                    solid_capstyle='butt')

        # 设置y轴
        ax.set_yticks(range(len(plasmids)))
        ax.set_yticklabels(plasmids)
        ax.set_xlabel(loc_text.get('position', 'Position (bp)'))
        ax.set_ylabel(loc_text.get('plasmid', 'Plasmid'))

        # 添加图例
        legend_elements = [mpatches.Patch(color=color_map[elem], label=elem)
                           for elem in df['element'].unique()]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.set_title(loc_text.get('element_distribution', 'Genetic Element Distribution Across Plasmids'), fontsize=7)

        save_path = os.path.join(pub_dir, "Nature_Element_Distribution.png")
        plt.savefig(save_path, dpi=600)
        plt.close(fig)
        print(f"✓ Nature 风格质粒元件分布图已保存: {save_path}")

    def __init__(self, file_paths, project_name="类胡萝卜素合成质粒分析", EcoRI=None, XhoI=None, XbaI=None, BamHI=None,
                 SalI=None, SpeI=None, KpnI=None, HindIII=None, SacI=None, NotI=None, NdeI=None, NcoI=None):
        self.file_paths = file_paths
        self.sequences = {}         # filename -> SeqRecord
        self.analysis_results = {}  # filename -> analysis dict
        self.visualizer = AdvancedScientificVisualizer()
        self.loc = self._get_localization_dict(project_name)
        self.color_palette = SCIENCE_COLORS['primary']
        self.pastel_palette = SCIENCE_COLORS['pastel']

        # CAI 计算器（尝试）
        try:
            self.cai_calculator = CodonAdaptationIndex()
        except Exception:
            self.cai_calculator = None

        # gene patterns (扩展版)
        self.gene_patterns = {
            'crtE': [r'ATG[ATCGN]{850,950}T(?:AA|AG|GA)', 'GGPP synthase (crtE-like)'],
            'crtB': [r'ATG[ATCGN]{900,1000}T(?:AA|AG|GA)', 'Phytoene synthase (crtB-like)'],
            'crtI': [r'ATG[ATCGN]{1400,1500}T(?:AA|AG|GA)', 'Phytoene desaturase (crtI-like)'],
            'crtY': [r'ATG[ATCGN]{1100,1200}T(?:AA|AG|GA)', 'Lycopene cyclase (crtY-like)'],
            'ampR': [r'ATGAGTATTCAACATTTCCGTGTCG', 'Ampicillin resistance (bla-like)'],
            'kanR': [r'ATGAGCCATATTCAACGGGAAACGT', 'Kanamycin resistance (aph-like)'],
            'tetR': [r'ATGTCTAGATTAGATAAAAGTAAAG', 'Tetracycline resistance (tetA-like)'],
            'cmR': [r'ATGACCACCGAGATCGAGCAG', 'Chloramphenicol resistance (cat-like)'],
            'ori_pBR322': [r'GATCTTTTTAATTAAGGATCCG', 'Origin of replication (pBR322-like)'],
            'ori_pUC': [r'GTTTTGGCGCGAGCGG', 'Origin of replication (pUC-like)'],
            'ori_ColE1': [r'GTGGCTAACTCACATTAATTGCGT', 'Origin of replication (ColE1-like)'],
            'trc_promoter': [r'TTGACA[ATCGN]{15,20}TATAAT', 'trc promoter'],
            'T7_promoter': [r'TAATACGACTCACTATAGGG', 'T7 promoter'],
            'lac_promoter': [r'GACACCATCGAATGGCGCAAAAC', 'lac promoter'],
            'lac_operator': [r'AATTGTGAGCGGATAACAATT', 'lac operator'],
            'RBS_strong': [r'AAGGAGGT[ATCGN]{5,10}ATG', 'Strong Ribosome Binding Site'],
            'T7_terminator': [r'GCTAGTTATTGCTCAGCGG', 'T7 terminator'],
            'rrnB_T1_terminator': [r'AAATAAAAAA[ATCGN]{5,10}GCTTATCA', 'rrnB T1 terminator']
        }

        self.construction_enzymes = {
            'EcoRI': EcoRI, 'BamHI': BamHI, 'HindIII': HindIII,
            'XhoI': XhoI, 'SalI': SalI, 'KpnI': KpnI, 'SacI': SacI,
            'XbaI': XbaI, 'SpeI': SpeI, 'NotI': NotI, 'NdeI': NdeI, 'NcoI': NcoI
        }

    def _get_localization_dict(self, project_name_zh):
        return {
            'zh': {
                'project_name': project_name_zh, 'report_dir_suffix': "报告_ZH",
                'comprehensive_report_title': "综合分析报告",
                'restriction_report_title': "限制酶分析报告",
                'homology_report_title': "序列同源性比较报告",
                'stats_report_title': "统计分析报告",
                'orf_analysis_title': "ORF与蛋白质分析",
                'codon_usage_title': "密码子使用频率",
                'sankey_bubble_title': "质粒-元件关联及表达强度分析",
                'bubble_x_label': "模拟表达强度",
                'bubble_color_label': "功能重要性评分",
                'composition_bar_title': "质粒元件组成丰度图",
                'volcano_plot_title': "差异分析火山图",
                'pca_plot_title': "主成分分析图",
                'heatmap_title': "热图分析",
                'dir_publication_plots': "出版级图表",
                'dir_advanced_analysis': "高级分析",
                'dir_interactive_plots': "交互式图表",
                'dir_re': "限制酶分析",
                'generated_at': "生成时间",
                'sequence': "序列",
                'length': "长度",
                'gc_content': "GC含量",
                'molecular_weight': "分子量",
                'base_composition': "碱基组成",
                'detected_elements': "检测到的遗传元件",
                'no_elements': "未检测到任何元件。",
                'restriction_analysis': "限制酶分析",
                'enzyme_summary': "共 {count} 种酶有切点。总切点数: {total}",
                'sites': "个切点",
                'positions': "位置",
                'mcs_title': "推测的多克隆位点 (MCS)",
                'mcs_position': "位置",
                'mcs_site_count': "切点数量",
                'mcs_enzymes': "包含的酶和位点",
                'orf_id': "ORF ID",
                'protein_seq': "蛋白质序列",
                'protein_len': "长度(aa)",
                'protein_mw': "分子量(Da)",
                'isoelectric_point': "等电点(pI)",
                'instability_index': "不稳定指数",
                'stable': "稳定",
                'unstable': "不稳定",
                'no_orfs_found': "未找到长度大于50个氨基酸的ORF。"
            },
            'en': {
                'project_name': "Carotenoid_Synthesis_Plasmid_Analysis",
                'report_dir_suffix': "Report_EN",
                'comprehensive_report_title': "Comprehensive_Analysis_Report",
                'restriction_report_title': "Restriction_Enzyme_Analysis_Report",
                'homology_report_title': "Sequence_Homology_Comparison_Report",
                'stats_report_title': "Statistical_Analysis_Report",
                'orf_analysis_title': "ORF_and_Protein_Analysis",
                'codon_usage_title': "Codon_Usage_Frequency",
                'sankey_bubble_title': "Plasmid-Element Association and Expression Analysis",
                'bubble_x_label': "Simulated Expression Strength",
                'bubble_color_label': "Functional Importance Score",
                'composition_bar_title': "Plasmid Element Composition Abundance",
                'volcano_plot_title': "Differential Analysis Volcano Plot",
                'pca_plot_title': "Principal Component Analysis Plot",
                'heatmap_title': "Heatmap Analysis",
                'dir_publication_plots': "Publication_Grade_Plots",
                'dir_advanced_analysis': "Advanced_Analysis",
                'dir_interactive_plots': "Interactive_Plots",
                'dir_re': "Restriction_Enzyme_Analysis",
                'generated_at': "Generated at",
                'sequence': "Sequence",
                'length': "Length",
                'gc_content': "GC Content",
                'molecular_weight': "Molecular Weight",
                'base_composition': "Base Composition",
                'detected_elements': "Detected Genetic Elements",
                'no_elements': "No elements detected.",
                'restriction_analysis': "Restriction Analysis",
                'enzyme_summary': "{count} enzymes have cut sites. Total cuts: {total}",
                'sites': "sites",
                'positions': "Positions",
                'mcs_title': "Predicted Multiple Cloning Site (MCS)",
                'mcs_position': "Position",
                'mcs_site_count': "Site Count",
                'mcs_enzymes': "Included Enzymes and Sites",
                'orf_id': "ORF ID",
                'protein_seq': "Protein Sequence",
                'protein_len': "Length(aa)",
                'protein_mw': "MW(Da)",
                'isoelectric_point': "Isoelectric Point(pI)",
                'instability_index': "Instability Index",
                'stable': "Stable",
                'unstable': "Unstable",
                'no_orfs_found': "No ORFs longer than 50 amino acids were found."
            }
        }

    # -------------------- 读取序列 --------------------
    def read_sequences(self):
        print("正在读取序列文件...")
        successful_reads = 0
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                print(f"警告: 文件不存在: {file_path}")
                continue
            file_name = os.path.basename(file_path)
            try:
                for fmt in ['fasta', 'genbank', 'embl', 'gb']:
                    try:
                        records = list(SeqIO.parse(file_path, fmt))
                        if records and len(records[0].seq) > 100:
                            self.sequences[file_name] = records[0]
                            print(f"✓ 成功: '{file_name}' 以 '{fmt}' 格式读取 (长度: {len(records[0].seq)} bp)。")
                            successful_reads += 1
                            break
                    except Exception:
                        continue
                else:
                    print(f"信息: '{file_name}' 非标准格式，尝试作为纯文本进行解析...")
                    with open(file_path, 'rb') as f:
                        raw_data = f.read()
                    detected_encoding = None
                    if chardet:
                        detected_encoding = chardet.detect(raw_data).get('encoding')
                    detected_encoding = detected_encoding or 'utf-8'
                    try:
                        content = raw_data.decode(detected_encoding, errors='replace')
                    except Exception:
                        content = raw_data.decode('latin-1', errors='replace')
                    dna_parts = [re.sub(r'[^ATCGNKSYWMRVBDH]', '', line.strip().upper())
                                 for line in content.splitlines()
                                 if line.strip() and not line.strip().startswith(('>', ';', '#', '//', 'LOCUS'))]
                    full_sequence = "".join(dna_parts)
                    if len(full_sequence) > 100:
                        record = SeqIO.SeqRecord(Seq(full_sequence), id=file_name,
                                                 description=f"From text (encoding:{detected_encoding})")
                        self.sequences[file_name] = record
                        print(f"✓ 成功: 从 '{file_name}' 文本中提取到序列 (长度: {len(full_sequence)} bp)。")
                        successful_reads += 1
                    else:
                        print(f"✗ 失败: 从 '{file_name}' 提取的序列过短 ({len(full_sequence)} bp)。")
            except Exception as e:
                print(f"✗ 处理文件 '{file_path}' 时发生错误: {e}")
        return successful_reads > 0

    # -------------------- 基础分析 --------------------
    def analyze_sequence_basic(self, seq_record, name):
        seq = str(seq_record.seq).upper()
        if not seq:
            return None
        try:
            mw = molecular_weight(seq, 'DNA')
        except Exception:
            unambiguous_seq = re.sub(r'[^ATGC]', '', seq)
            mw = molecular_weight(unambiguous_seq, 'DNA') if unambiguous_seq else 0.0

        result = {
            'sequence': seq,
            'length': len(seq),
            'a_count': seq.count('A'),
            't_count': seq.count('T'),
            'g_count': seq.count('G'),
            'c_count': seq.count('C'),
            'gc_content': gc_fraction(seq) * 100,
            'molecular_weight': mw,
            'genetic_elements': self.detect_genetic_elements(seq),
            'restriction_analysis': self.analyze_restriction_sites(seq),
            'orfs_analysis': self.analyze_orfs_and_proteins(seq)
        }
        return result

    # -------------------- 遗传元件检测 --------------------
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

    # -------------------- 限制酶分析 --------------------
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

    def find_multiple_cloning_site(self, seq, sites):
        all_positions = sorted([pos for pos_list in sites.values() for pos in pos_list])
        if not all_positions:
            return None
        best_region = {'start': 0, 'end': 0, 'site_count': 0, 'sites': []}
        window_size = 100
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
        if best_region['site_count'] >= 3:
            best_region['sites'] = sorted(
                [(enzyme, pos) for enzyme, positions in sites.items() for pos in positions
                 if best_region['start'] <= pos <= best_region['end']],
                key=lambda x: x[1]
            )
            return best_region
        return None

    # -------------------- ORF 与蛋白质分析 --------------------
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
                        if any(aa in prot_seq for aa in ['X', 'B', 'Z']):
                            aa_start = aa_stop + 1
                            continue
                        start = frame + aa_start * 3
                        end = frame + aa_stop * 3 + 3
                        if strand == -1:
                            start, end = len(seq) - end, len(seq) - start
                        try:
                            p_analysis = ProteinAnalysis(prot_seq)
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
                            codon_seq = str(frame_seq[aa_start * 3: aa_stop * 3])
                            for i in range(0, len(codon_seq), 3):
                                codon = codon_seq[i:i + 3]
                                if len(codon) == 3:
                                    codon_counts[codon] += 1
                        except Exception:
                            pass
                    aa_start = aa_stop + 1
        return {'orfs': orfs, 'codon_usage': dict(codon_counts)}

    # -------------------- 运行完整分析 --------------------
    def run_complete_analysis(self):
        if not self.read_sequences():
            print("错误：无法读取任何有效的序列文件，分析终止。")
            return False
        print("\n正在分析各个序列...")
        for name, seq_record in self.sequences.items():
            print(f"分析 {name}...")
            result = self.analyze_sequence_basic(seq_record, name)
            if result:
                self.analysis_results[name] = result
                print(f"✓ {name} 分析完成。")
            else:
                print(f"✗ {name} 分析失败。")

        if not self.analysis_results:
            print("错误：没有可供分析的有效序列数据。")
            return False

        self.generate_all_reports_bilingual()
        return True

    # -------------------- 报告生成 --------------------
    def generate_all_reports_bilingual(self, pub_dir=None):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_project_name = self.loc['zh']['project_name']
        report_root = f"{base_project_name}_报告_{timestamp}"
        os.makedirs(report_root, exist_ok=True)

        print(f"✓ 报告根目录: {report_root}")

        for lang in ['zh', 'en']:
            loc_text = self.loc[lang]
            report_dir = os.path.join(report_root, loc_text['report_dir_suffix'])
            os.makedirs(report_dir, exist_ok=True)
            pub_dir_local = os.path.join(report_dir, loc_text['dir_publication_plots'])
            os.makedirs(pub_dir_local, exist_ok=True)

            print(f"\n--- 正在生成 {lang.upper()} 版本报告到: {report_dir} ---")
            self.generate_text_report(report_dir, lang)
            self.generate_restriction_report(report_dir, lang)
            self.generate_orf_protein_report(report_dir, lang)
            self.generate_publication_grade_plots(report_dir, lang)

            self._generate_composition_data_and_plot(pub_dir_local, loc_text)
            self._generate_orf_length_histogram(pub_dir_local, loc_text)
            self._generate_orf_gc_scatter(pub_dir_local, loc_text)

        # 保存 analysis_results.pkl
        try:
            import pickle
            pkl_name = safe_filename(f"analysis_results_{timestamp}.pkl")
            pkl_path = os.path.join(report_root, pkl_name)
            with open(pkl_path, "wb") as pf:
                pickle.dump(self.analysis_results, pf)
            print(f"✓ analysis_results 已保存到报告根目录: {pkl_path}")
        except Exception as e:
            print(f"⚠ 保存 analysis_results.pkl 失败: {e}")

    def generate_text_report(self, report_dir, language):
        loc_text = self.loc[language]
        report_path = os.path.join(report_dir, f"{loc_text['comprehensive_report_title']}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{loc_text['project_name']} {loc_text['comprehensive_report_title']}\n")
            f.write(f"{loc_text['generated_at']}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 60}\n\n")
            for name, res in self.analysis_results.items():
                f.write(f"--- {loc_text['sequence']}: {name} ---\n")
                f.write(f"{loc_text['length']}: {res['length']} bp\n")
                f.write(f"{loc_text['gc_content']}: {res['gc_content']:.2f}%\n")
                f.write(f"{loc_text['molecular_weight']}: {res['molecular_weight']:.2f} Da\n")
                f.write(f"{loc_text['base_composition']}: A:{res['a_count']}, T:{res['t_count']}, G:{res['g_count']}, C:{res['c_count']}\n\n")
                f.write(f"{loc_text['detected_elements']}:\n")
                if not res['genetic_elements']:
                    f.write(f"  {loc_text['no_elements']}\n")
                else:
                    for el_name, el_list in res['genetic_elements'].items():
                        f.write(f"  - {el_name} ({el_list[0]['description']}): {len(el_list)} found\n")
                f.write(f"\n{loc_text['restriction_analysis']}:\n")
                re_sites = res['restriction_analysis']['all_sites']
                unique_enzymes = [e for e, i in re_sites.items() if i.get('count', 0) > 0]
                total = sum(i.get('count', 0) for i in re_sites.values())
                f.write(f"  {loc_text['enzyme_summary'].format(count=len(unique_enzymes), total=total)}\n\n{'=' * 60}\n\n")
        print(f"✓ 文本报告已保存: {report_path}")

    def _generate_orf_length_histogram(self, pub_dir, loc_text):
        """生成ORF长度直方图"""
        all_orfs = []
        for name, res in self.analysis_results.items():
            for orf in res['orfs_analysis']['orfs']:
                all_orfs.append(orf['length_aa'])

        if not all_orfs:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(all_orfs, bins=30, color=self.color_palette[0], alpha=0.7)
        ax.set_xlabel('ORF Length (aa)')
        ax.set_ylabel('Frequency')
        ax.set_title('ORF Length Distribution')

        save_path = os.path.join(pub_dir, "ORF_Length_Histogram.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _generate_orf_gc_scatter(self, pub_dir, loc_text):
        """生成ORF GC含量散点图"""
        data = []
        for name, res in self.analysis_results.items():
            for orf in res['orfs_analysis']['orfs']:
                orf_seq = res['sequence'][orf['start']:orf['end']]
                gc_content = GC(orf_seq) if orf_seq else 0
                data.append({
                    'length': orf['length_aa'],
                    'gc_content': gc_content,
                    'plasmid': name
                })

        if not data:
            return

        df = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (name, group) in enumerate(df.groupby('plasmid')):
            ax.scatter(group['length'], group['gc_content'],
                       color=self.color_palette[i % len(self.color_palette)],
                       label=name, alpha=0.6)

        ax.set_xlabel('ORF Length (aa)')
        ax.set_ylabel('GC Content (%)')
        ax.set_title('ORF Length vs GC Content')
        ax.legend()

        save_path = os.path.join(pub_dir, "ORF_Length_GC_Scatter.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    def generate_restriction_report(self, report_dir, language):
        loc_text = self.loc[language]
        re_dir = os.path.join(report_dir, loc_text['dir_re'])
        os.makedirs(re_dir, exist_ok=True)
        for name, res in self.analysis_results.items():
            path = os.path.join(re_dir, f"{name}_{loc_text['restriction_report_title']}.txt")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(f"{name} - {loc_text['restriction_report_title']}\n{'=' * 40}\n\n")
                re_sites, mcs = res['restriction_analysis']['all_sites'], res['restriction_analysis']['multiple_cloning_site']
                sorted_sites = sorted(re_sites.items(), key=lambda item: item[1].get('count', 0), reverse=True)
                for enzyme, info in sorted_sites:
                    if info.get('count', 0) > 0:
                        f.write(f"{enzyme}: {info['count']} {loc_text['sites']}\n  {loc_text['positions']}: {', '.join(map(str, info['positions']))}\n")
                if mcs:
                    f.write(f"\n{loc_text['mcs_title']}:\n")
                    f.write(f"  {loc_text['mcs_position']}: {mcs['start']}-{mcs['end']}\n")
                    f.write(f"  {loc_text['mcs_site_count']}: {mcs['site_count']}\n")
                    f.write(f"  {loc_text['mcs_enzymes']}:\n")
                    for enzyme, pos in mcs['sites']:
                        f.write(f"    - {enzyme} at {pos}\n")
        print(f"✓ 限制酶分析报告已保存到: {re_dir}")

    def generate_orf_protein_report(self, report_dir, language):
        loc_text = self.loc[language]
        for name, res in self.analysis_results.items():
            report_path = os.path.join(report_dir, f"{name}_{loc_text['orf_analysis_title']}.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"--- {name}: {loc_text['orf_analysis_title']} ---\n\n")
                orfs = res['orfs_analysis']['orfs']
                if not orfs:
                    f.write(loc_text['no_orfs_found'] + "\n")
                    continue
                for orf in sorted(orfs, key=lambda x: x['start']):
                    f.write(f"{loc_text['orf_id']}: {orf['id']} ({orf['start']}-{orf['end']}, Strand: {orf['strand']})\n")
                    f.write(f"  {loc_text['protein_len']}: {orf['length_aa']}\n")
                    f.write(f"  {loc_text['protein_mw']}: {orf['mw_da']:.2f}\n")
                    f.write(f"  {loc_text['isoelectric_point']}: {orf['pI']:.2f}\n")
                    stability = loc_text['stable'] if orf['instability'] < 40 else loc_text['unstable']
                    f.write(f"  {loc_text['instability_index']}: {orf['instability']:.2f} ({stability})\n")
                    if 'cai' in orf:
                        f.write(f"  Codon Adaptation Index (vs reference): {orf['cai']:.3f}\n")
                    f.write(f"  {loc_text['protein_seq']}: {orf['protein']}\n\n")
                f.write(f"\n--- {loc_text['codon_usage_title']} ---\n")
                codons = res['orfs_analysis']['codon_usage']
                if codons:
                    total_codons = sum(codons.values())
                    for codon, count in sorted(codons.items()):
                        freq = (count / total_codons) * 100
                        f.write(f"  {codon}: {count} ({freq:.2f}%)\n")
            print(f"✓ ORF报告已保存: {os.path.basename(report_path)}")

    # -------------------- 出版级图表与高级图表 --------------------
    def generate_publication_grade_plots(self, report_dir, language):
        loc_text = self.loc[language]
        pub_dir = os.path.join(report_dir, loc_text['dir_publication_plots'])
        os.makedirs(pub_dir, exist_ok=True)

        # 原有的图表生成
        self._generate_sankey_bubble_data_and_plot(pub_dir, loc_text)
        self._generate_composition_data_and_plot(pub_dir, loc_text)
        self._generate_gc_comparison_plot(pub_dir, loc_text)
        self._generate_length_comparison_plot(pub_dir, loc_text)
        self._generate_plasmid_maps(pub_dir, loc_text)
        self._generate_codon_usage_heatmap(pub_dir, loc_text)
        self._generate_orf_property_distribution_plots(pub_dir, loc_text)
        self._generate_sequence_homology_heatmap(pub_dir, loc_text)
        self._generate_gc_skew_plots(pub_dir, loc_text)
        self._generate_element_upset_plot(pub_dir, loc_text)
        self._generate_enzyme_network_graph(pub_dir, loc_text)
        self._generate_enhanced_interactive_3d_plot(pub_dir, loc_text)
        self._generate_pathway_map(pub_dir, loc_text)

        # 新增Nature风格图表
        self._generate_nature_style_gc_skew(pub_dir, loc_text)
        self._generate_nature_style_codon_usage(pub_dir, loc_text)
        self._generate_nature_style_element_distribution(pub_dir, loc_text)
    # 以下方法实现与前面示例一致：sankey, composition, gc compare, length compare,
    # plasmid maps (使用 dna_features_viewer), codon heatmap, orf property violins,
    # sequence homology heatmap (pairwise aligner), gc skew, upset plot, enzyme network,
    # interactive 3D plot, pathway map.
    #
    # 为了可读性，下面省略重复注释但保留实现（确保在保存前创建目录）。
    def _generate_sankey_bubble_data_and_plot(self, pub_dir, loc_text):
        sankey_data = []
        importance_map = {'crt': 5, 'amp': 3, 'kan': 3, 'ori': 2, 'promoter': 4}
        for plasmid_name, results in self.analysis_results.items():
            for element_name, element_list in results['genetic_elements'].items():
                if not element_list:
                    continue
                avg_length = np.mean([e['length'] for e in element_list])
                simulated_expression = avg_length / 1000
                importance_score = next((v for k, v in importance_map.items() if k in element_name), 1)
                sankey_data.append({
                    'Source': plasmid_name,
                    'Target': element_name,
                    'Value': len(element_list),
                    'BubbleX': simulated_expression,
                    'BubbleColor': importance_score
                })
        df_sankey = pd.DataFrame(sankey_data)
        if df_sankey.empty:
            print("信息: 无足够数据绘制桑基图。")
            return
        try:
            bubble_df = pd.DataFrame({
                'x': df_sankey.groupby('Target')['BubbleX'].mean(),
                'size': df_sankey.groupby('Target')['Value'].sum(),
                'color': df_sankey.groupby('Target')['BubbleColor'].mean()
            })
            save_path = os.path.join(pub_dir, f"{loc_text['sankey_bubble_title']}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Use visualizer (which will save)
            self.visualizer.create_sankey_bubble_plot(
                source=df_sankey['Source'],
                target=df_sankey['Target'],
                value=df_sankey['Value'],
                bubble_data=bubble_df,
                title=loc_text['sankey_bubble_title'],
                save_path=save_path
            )
        except Exception:
            save_path = os.path.join(pub_dir, f"{loc_text['sankey_bubble_title']}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.visualizer.create_sankey_bubble_plot(
                source=df_sankey['Source'],
                target=df_sankey['Target'],
                value=df_sankey['Value'],
                title=loc_text['sankey_bubble_title'],
                save_path=save_path
            )

    def _generate_composition_data_and_plot(self, pub_dir, loc_text):
        composition_data = {}
        for plasmid_name, results in self.analysis_results.items():
            total_length = results['length']
            comp = defaultdict(float)
            for element_name, element_list in results['genetic_elements'].items():
                total_element_len = sum(e['length'] for e in element_list)
                if 'crt' in element_name:
                    comp['Carotenoid Genes'] += total_element_len
                elif 'amp' in element_name or 'kan' in element_name:
                    comp['Resistance Genes'] += total_element_len
                else:
                    comp['Regulatory/Other'] += total_element_len
            coded_length = sum(comp.values())
            comp['Backbone'] = max(0, total_length - coded_length)
            composition_data[plasmid_name] = {k: (v / total_length) for k, v in comp.items()}
        df_comp = pd.DataFrame(composition_data).T.fillna(0)
        if df_comp.empty:
            print("信息: 无足够数据绘制组成图。")
            return
        fig, ax = plt.subplots(figsize=(max(8, len(df_comp) * 1.2), 6))
        colors = self.color_palette
        df_comp.plot(kind='bar', stacked=True, color=colors[:len(df_comp.columns)], ax=ax, width=0.7)
        ax.set_title(loc_text['composition_bar_title'], fontsize=16)
        ax.set_xlabel("Plasmid", fontsize=12)
        ax.set_ylabel("Composition (%)", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Element Type', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        save_path = os.path.join(pub_dir, f"{loc_text['composition_bar_title']}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 元件组成图已保存: {os.path.basename(save_path)}")
        plt.close(fig)
        # 增加别名
        _generate_base_composition_radar_plot = self._generate_composition_data_and_plot

    def _generate_gc_comparison_plot(self, pub_dir, loc_text):
        gc_data = {name: res['gc_content'] for name, res in self.analysis_results.items()}
        if len(gc_data) <= 1:
            print("信息: 样本少于2，跳过 GC 比较图。")
            return
        fig, ax = plt.subplots(figsize=(max(8, len(gc_data) * 1.2), 6))
        categories = list(gc_data.keys()); values = list(gc_data.values()); x_pos = np.arange(len(categories))
        bars = ax.bar(x_pos, values, color=self.color_palette[:len(categories)])
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5, f'{values[i]:.2f}%', ha='center', va='bottom')
        ax.set_xlabel(loc_text['sequence'], fontsize=12)
        ax.set_ylabel(loc_text['gc_content'], fontsize=12)
        ax.set_title("GC Content Comparison", fontsize=16)
        ax.set_xticks(x_pos); ax.set_xticklabels(categories, fontsize=10, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        save_path = os.path.join(pub_dir, "GC_Content_Comparison.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ GC含量比较图已保存: {os.path.basename(save_path)}")
        plt.close(fig)

    def _generate_length_comparison_plot(self, pub_dir, loc_text):
        length_data = {name: res['length'] / 1000 for name, res in self.analysis_results.items()}
        if len(length_data) <= 1:
            print("信息: 样本少于2，跳过长度比较图。")
            return
        fig, ax = plt.subplots(figsize=(max(8, len(length_data) * 1.2), 6))
        categories = list(length_data.keys()); values = list(length_data.values()); x_pos = np.arange(len(categories))
        bars = ax.bar(x_pos, values, color=self.pastel_palette[:len(categories)])
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.05, f'{values[i]:.2f} kbp', ha='center', va='bottom')
        ax.set_xlabel(loc_text['sequence'], fontsize=12)
        ax.set_ylabel("Length (kbp)", fontsize=12)
        ax.set_title("Sequence Length Comparison", fontsize=16)
        ax.set_xticks(x_pos); ax.set_xticklabels(categories, fontsize=10, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        save_path = os.path.join(pub_dir, "Sequence_Length_Comparison.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 序列长度比较图已保存: {os.path.basename(save_path)}")
        plt.close(fig)

    def _generate_plasmid_maps(self, pub_dir, loc_text):
        if GraphicRecord is None or GraphicFeature is None:
            print("警告: dna_features_viewer 未安装，跳过质粒图谱绘制。")
            return
        print("正在生成质粒图谱...")
        for name, res in self.analysis_results.items():
            features = []
            element_colors = {'crt': '#ff7f0e', 'amp': '#d62728', 'kan': '#d62728', 'ori': '#9467bd', 'promoter': '#2ca02c'}
            for el_name, el_list in res['genetic_elements'].items():
                color_key = next((k for k in element_colors if k in el_name.lower()), 'grey')
                color = element_colors.get(color_key, '#7f7f7f')
                for element in el_list:
                    features.append(GraphicFeature(start=element['start'], end=element['end'], strand=+1,
                                                   color=color, label=el_name))
            for orf in res['orfs_analysis']['orfs']:
                is_known_element = False
                for el_list in res['genetic_elements'].values():
                    for el in el_list:
                        if max(orf['start'], el['start']) < min(orf['end'], el['end']):
                            is_known_element = True; break
                    if is_known_element: break
                if not is_known_element:
                    features.append(GraphicFeature(start=orf['start'], end=orf['end'], strand=orf['strand'],
                                                   color="#a1c9f4", label=f"ORF ({orf['length_aa']}aa)"))
            if not features:
                print(f"警告: {name} 未找到可供可视化的特征。")
                continue
            record = GraphicRecord(sequence_length=res['length'], features=features)
            fig, ax = plt.subplots(figsize=(15, 2))
            record.plot(ax=ax, with_ruler=True)
            ax.set_title(f"Plasmid Map: {name}", fontsize=14)
            save_path = os.path.join(pub_dir, f"Plasmid_Map_{name}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"✓ 质粒图谱已保存: {os.path.basename(save_path)}")
            plt.close(fig)

    def _generate_codon_usage_heatmap(self, pub_dir, loc_text):
        print("正在生成密码子使用频率热图...")
        total_codon_counts = Counter()
        for res in self.analysis_results.values():
            total_codon_counts.update(res['orfs_analysis']['codon_usage'])
        if not total_codon_counts:
            print("警告: 未找到密码子数据，无法生成热图。")
            return
        codon_map = {
            'ATA':'I','ATC':'I','ATT':'I','ATG':'M','ACA':'T','ACC':'T','ACG':'T','ACT':'T',
            'AAC':'N','AAT':'N','AAA':'K','AAG':'K','AGC':'S','AGT':'S','AGA':'R','AGG':'R',
            'CTA':'L','CTC':'L','CTG':'L','CTT':'L','CCA':'P','CCC':'P','CCG':'P','CCT':'P',
            'CAC':'H','CAT':'H','CAA':'Q','CAG':'Q','CGA':'R','CGC':'R','CGG':'R','CGT':'R',
            'GTA':'V','GTC':'V','GTG':'V','GTT':'V','GCA':'A','GCC':'A','GCG':'A','GCT':'A',
            'GAC':'D','GAT':'D','GAA':'E','GAG':'E','GGA':'G','GGC':'G','GGG':'G','GGT':'G',
            'TCA':'S','TCC':'S','TCG':'S','TCT':'S','TTC':'F','TTT':'F','TTA':'L','TTG':'L',
            'TAC':'Y','TAT':'Y','TAA':'_','TAG':'_','TGC':'C','TGT':'C','TGA':'_','TGG':'W',
        }
        aa_codon_freq = defaultdict(dict)
        aa_codon_total = Counter()
        for codon, count in total_codon_counts.items():
            aa = codon_map.get(codon)
            if aa:
                aa_codon_total[aa] += count
        for codon, count in total_codon_counts.items():
            aa = codon_map.get(codon)
            if aa and aa_codon_total[aa] > 0:
                freq = count / aa_codon_total[aa]
                aa_codon_freq[aa][codon] = freq
        df_codon = pd.DataFrame.from_dict(aa_codon_freq).T.fillna(0).sort_index()
        if sns is None:
            print("警告: seaborn 未安装，跳过密码子热图。")
            return
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.heatmap(df_codon, annot=True, fmt=".2f", cmap="viridis", linewidths=.5, ax=ax)
        ax.set_title(loc_text['codon_usage_title'], fontsize=16)
        ax.set_xlabel("Codon", fontsize=12)
        ax.set_ylabel("Amino Acid", fontsize=12)
        save_path = os.path.join(pub_dir, f"{loc_text['codon_usage_title']}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✓ 密码子使用热图已保存: {os.path.basename(save_path)}")
        plt.close(fig)

    def _generate_orf_property_distribution_plots(self, pub_dir, loc_text):
        print("正在生成ORF属性分布图...")
        all_orfs = []
        for name, res in self.analysis_results.items():
            for orf in res['orfs_analysis']['orfs']:
                orf_data = orf.copy(); orf_data['plasmid'] = name; all_orfs.append(orf_data)
        if not all_orfs:
            print("警告: 未找到ORF数据，无法生成分布图。")
            return
        if sns is None:
            print("警告: seaborn 未安装，跳过 ORF 分布图。")
            return
        df_orfs = pd.DataFrame(all_orfs)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Distribution of Predicted ORF Properties', fontsize=18)
        sns.violinplot(ax=axes[0, 0], data=df_orfs, y='length_aa', color=self.color_palette[0]); axes[0,0].set_title('Protein Length (aa)')
        sns.violinplot(ax=axes[0, 1], data=df_orfs, y='mw_da', color=self.color_palette[1]); axes[0,1].set_title('Molecular Weight (Da)')
        sns.violinplot(ax=axes[1, 0], data=df_orfs, y='pI', color=self.color_palette[2]); axes[1,0].set_title('Isoelectric Point (pI)')
        sns.violinplot(ax=axes[1, 1], data=df_orfs, y='instability', color=self.color_palette[3]); axes[1,1].axhline(40, color='r', linestyle='--', label='Stability Threshold (40)'); axes[1,1].set_title('Instability Index'); axes[1,1].legend()
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(pub_dir, "ORF_Property_Distribution.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✓ ORF属性分布图已保存: {os.path.basename(save_path)}")
        plt.close(fig)

    def _generate_sequence_homology_heatmap(self, pub_dir, loc_text):
        print("正在生成序列同源性比较热图 (这可能需要一些时间)...")
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        names = list(self.sequences.keys())
        num_seqs = len(names)
        if num_seqs < 2:
            print("信息: 只有一个序列，跳过同源性比较。")
            return
        identity_matrix = np.zeros((num_seqs, num_seqs))
        for i in range(num_seqs):
            for j in range(i, num_seqs):
                if i == j:
                    identity_matrix[i, j] = 1.0
                else:
                    seq1 = self.sequences[names[i]].seq
                    seq2 = self.sequences[names[j]].seq
                    score = aligner.score(seq1, seq2)
                    identity = score / min(len(seq1), len(seq2)) if min(len(seq1), len(seq2))>0 else 0
                    identity = max(0, min(1, identity))
                    identity_matrix[i, j] = identity; identity_matrix[j, i] = identity
        df_identity = pd.DataFrame(identity_matrix, index=names, columns=names)
        if sns is None:
            print("警告: seaborn 未安装，跳过同源性热图。")
            return
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_identity, annot=True, fmt=".2%", cmap="YlGnBu", ax=ax)
        ax.set_title("Sequence Pairwise Identity Matrix", fontsize=16)
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
        save_path = os.path.join(pub_dir, "Sequence_Homology_Heatmap.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✓ 序列同源性热图已保存: {os.path.basename(save_path)}")
        plt.close(fig)

    def _generate_gc_skew_plots(self, pub_dir, loc_text):
        print("正在生成 GC Skew 分析图...")
        for name, res in self.analysis_results.items():
            seq = res['sequence']
            if not seq:
                continue
            cumulative_skew = [0]
            for i in range(len(seq)):
                base = seq[i:i+1]
                g = 1 if base=='G' else 0
                c = 1 if base=='C' else 0
                if g + c > 0:
                    skew = (g - c) / (g + c)
                else:
                    skew = 0
                cumulative_skew.append(cumulative_skew[-1] + skew)
            fig, ax = plt.subplots(figsize=(15, 4))
            ax.plot(range(len(cumulative_skew)), cumulative_skew, color='#2f4b7c')
            max_skew_idx = int(np.argmax(cumulative_skew))
            min_skew_idx = int(np.argmin(cumulative_skew))
            ax.axvline(x=max_skew_idx, color='green', linestyle='--', label=f'Ori (approx): {max_skew_idx} bp')
            ax.axvline(x=min_skew_idx, color='red', linestyle='--', label=f'Ter (approx): {min_skew_idx} bp')
            ax.set_title(f"GC Skew Cumulative: {name}", fontsize=14)
            ax.set_xlabel("Sequence position (bp)", fontsize=12)
            ax.set_ylabel("Cumulative GC Skew", fontsize=12)
            ax.grid(True, alpha=0.3); ax.legend()
            save_path = os.path.join(pub_dir, f"GC_Skew_{name}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"✓ GC Skew 图已保存: {os.path.basename(save_path)}")
            plt.close(fig)

    def _generate_element_upset_plot(self, pub_dir, loc_text):
        if from_contents is None or UpSet is None:
            print("警告: upsetplot 未安装，跳过 UpSet 图。")
            return
        if len(self.analysis_results) < 2:
            print("信息: 样本不足，跳过 UpSet 图。")
            return
        print("正在生成遗传元件 UpSet 图...")
        element_contents = defaultdict(list)
        for name, res in self.analysis_results.items():
            unique_elements = set(res['genetic_elements'].keys())
            for element in unique_elements:
                element_contents[element].append(name)
        if not element_contents:
            print("警告: 未找到遗传元件，跳过 UpSet 图。")
            return
        upset_data = from_contents(element_contents)
        fig = plt.figure(figsize=(12, 7))
        UpSet(upset_data, subset_size='count', show_counts=True, sort_by='cardinality').plot(fig=fig)
        plt.suptitle("Genetic Element Comparison (UpSet)", fontsize=16)
        save_path = os.path.join(pub_dir, "Genetic_Element_Comparison_UpSet.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✓ 遗传元件 UpSet 图已保存: {os.path.basename(save_path)}")
        plt.close(fig)

    def _generate_enzyme_network_graph(self, pub_dir, loc_text):
        if nx is None:
            print("警告: networkx 未安装，跳过酶网络图。")
            return
        print("正在生成酶切点关联网络图...")
        G = nx.Graph()
        for name, res in self.analysis_results.items():
            G.add_node(name, type='plasmid')
            re_sites = res['restriction_analysis']['all_sites']
            for enzyme, info in re_sites.items():
                if info.get('count', 0) > 0:
                    G.add_node(enzyme, type='enzyme')
                    G.add_edge(name, enzyme, weight=info['count'])
        enzymes = [n for n,d in G.nodes(data=True) if d.get('type')=='enzyme']
        if not enzymes:
            print("警告: 未找到任何限制酶切点，跳过网络图。")
            return
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=0.8, iterations=50)
        plasmid_nodes = [n for n,d in G.nodes(data=True) if d.get('type')=='plasmid']
        enzyme_nodes = [n for n,d in G.nodes(data=True) if d.get('type')=='enzyme']
        nx.draw_networkx_nodes(G, pos, nodelist=plasmid_nodes, node_color='#ff7f0e', node_size=3000, label='plasmid')
        nx.draw_networkx_nodes(G, pos, nodelist=enzyme_nodes, node_color='#1f77b4', node_size=1000, label='enzyme')
        edges = G.edges(data=True)
        weights = [d['weight'] for u,v,d in edges]
        nx.draw_networkx_edges(G, pos, width=[w*0.5 for w in weights], alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=10)
        plt.title("Plasmid - Restriction Enzyme Network", fontsize=16)
        plt.legend(); plt.box(False)
        save_path = os.path.join(pub_dir, "Enzyme_Plasmid_Network.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✓ 酶切点网络图已保存: {os.path.basename(save_path)}")
        plt.close()

    def _generate_enhanced_interactive_3d_plot(self, report_dir, loc_text):
        if px is None:
            print("警告: plotly 未安装，跳过增强交互式3D图。")
            return
        int_dir = os.path.join(report_dir, loc_text['dir_interactive_plots'])
        os.makedirs(int_dir, exist_ok=True)
        print("正在生成增强互动 3D 图...")
        data = {'x': [], 'y': [], 'z': [], 'color': [], 'size': [], 'label': []}
        for name, res in self.analysis_results.items():
            data['x'].append(res['length'])
            data['y'].append(res['gc_content'])
            data['z'].append(len(res['orfs_analysis']['orfs']))
            data['color'].append(len(res['genetic_elements']))
            data['size'].append(sum([len(v) for v in res['restriction_analysis']['all_sites'].values()]))
            data['label'].append(name)
        df = pd.DataFrame(data)
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='color', size='size', hover_name='label', title='Plasmid 3D Overview')
        save_path = os.path.join(int_dir, "interactive_3d_plot.html")
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"✓ 交互式 3D 图已保存: {os.path.basename(save_path)}")
        except Exception as e:
            print(f"保存 3D html 失败: {e}")

    def _generate_pathway_map(self, pub_dir, loc_text):
        print("正在生成代谢通路图...")
        pathway = {
            'GGPP': ('crtE', 'Phytoene'),
            'Phytoene': ('crtB', 'Lycopene'),
            'Lycopene': ('crtI', 'Neurosporene/Lycopene'),
            'Lycopene_cycle': ('crtY', 'β-Carotene')
        }
        num_plasmids = len(self.analysis_results)
        if num_plasmids == 0: return
        fig, axes = plt.subplots(num_plasmids, 1, figsize=(10, 2 * num_plasmids), squeeze=False)
        fig.suptitle('Carotenoid Synthesis Pathway Coverage', fontsize=16)
        for i, (name, res) in enumerate(self.analysis_results.items()):
            ax = axes[i, 0]
            ax.set_title(name)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 2)
            ax.axis('off')
            ax.text(1, 1, 'GGPP', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="lightblue"))
            ax.text(4, 1, 'Phytoene', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="lightblue"))
            ax.text(7, 1, 'Lycopene', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="lightblue"))
            ax.text(9.5, 1, 'β-Carotene', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="lightblue"))
            present_genes = set(res['genetic_elements'].keys())
            color_e = 'green' if 'crtE' in present_genes else 'red'
            ax.arrow(1.8, 1, 1.4, 0, head_width=0.1, head_length=0.2, fc=color_e, ec=color_e)
            ax.text(2.5, 1.2, 'crtE', ha='center', color=color_e)
            color_b = 'green' if 'crtB' in present_genes else 'red'
            ax.arrow(4.8, 1, 1.4, 0, head_width=0.1, head_length=0.2, fc=color_b, ec=color_b)
            ax.text(5.5, 1.2, 'crtB', ha='center', color=color_b)
            color_i = 'green' if 'crtI' in present_genes else 'red'
            ax.text(5.5, 0.8, 'crtI', ha='center', color=color_i)
            color_y = 'green' if 'crtY' in present_genes else 'red'
            ax.arrow(7.8, 1, 1.4, 0, head_width=0.1, head_length=0.2, fc=color_y, ec=color_y)
            ax.text(8.5, 1.2, 'crtY', ha='center', color=color_y)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(pub_dir, "Pathway_Coverage_Map.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✓ 代谢通路图已保存: {os.path.basename(save_path)}")
        plt.close(fig)

# -------------------- 脚本直接运行入口 --------------------
if __name__ == "__main__":
    # 在这里修改为你的文件路径列表（示例）
    file_paths = [
        r"C:\Users\Administrator\Desktop\Wet_experient-DNA\融合β胡萝卜素质粒(1)(1).dna",
        r"C:\Users\Administrator\Desktop\Wet_experient-DNA\融合PCR.dna",
        r"C:\Users\Administrator\Desktop\Wet_experient-DNA\融合pcr(1).dna",
        r"C:\Users\Administrator\Desktop\Wet_experient-DNA\融合ATRA质粒.dna",
        r"C:\Users\Administrator\Desktop\Wet_experient-DNA\线性化的21trc.dna",
    ]

    if not file_paths:
        print("提示: 请在脚本末尾将 file_paths 列表填入你的序列文件路径，然后重新运行脚本。")
    else:
        analyzer = EnhancedCarotenoidPlasmidAnalyzer(file_paths)
        success = analyzer.run_complete_analysis()
        if success:
            print("全部分析完成。输出文件夹已生成。")
        else:
            print("分析未完成，请检查输入文件与依赖项。")
