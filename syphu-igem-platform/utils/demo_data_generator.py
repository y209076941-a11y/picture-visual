# utils/demo_data_generator.py
"""
示例数据生成器
为在线演示创建精简数据集
"""

from pathlib import Path
import shutil
import random


class DemoDataGenerator:
    """生成用于部署的示例数据集"""

    @staticmethod
    def create_demo_dataset(
            source_dir: Path,
            demo_dir: Path,
            samples_per_type: int = 3
    ):
        """
        从完整数据集中抽取示例文件

        Parameters
        ----------
        source_dir : Path
            原始完整数据目录
        demo_dir : Path
            演示数据输出目录
        samples_per_type : int
            每种文件类型抽取的数量
        """

        demo_dir.mkdir(parents=True, exist_ok=True)

        file_categories = {
            'data': ['.csv', '.xlsx', '.tsv'],
            'images': ['.png', '.jpg', '.tiff'],
            'sequences': ['.fasta', '.fastq'],
            'results': ['.json', '.html']
        }

        for category, extensions in file_categories.items():
            category_dir = demo_dir / category
            category_dir.mkdir(exist_ok=True)

            # 收集该类型的所有文件
            files = []
            for ext in extensions:
                files.extend(source_dir.rglob(f'*{ext}'))

            # 随机抽取样本
            if files:
                samples = random.sample(
                    files,
                    min(samples_per_type, len(files))
                )

                for file in samples:
                    # 复制到演示目录
                    dest = category_dir / file.name
                    shutil.copy2(file, dest)
                    print(f"✓ Copied: {file.name} ({file.stat().st_size / 1024:.1f} KB)")
