"""准备部署用示例数据集"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.demo_data_generator import DemoDataGenerator
import pandas as pd

def create_readme(demo_dir, file_count, total_size):
    readme_content = f"""# Demo Dataset for SYPHU-CHINA iGEM Platform

## 数据统计
- 文件总数: {file_count}
- 总大小: {total_size / 1024 / 1024:.2f} MB
- 更新时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
⚠️ 这是演示数据集,非完整研究数据
"""
    (demo_dir / "README.md").write_text(readme_content, encoding='utf-8')

def main():
    SOURCE_DIR = Path(r"C:\Users\Administrator\PycharmProjects\picture\syphu-igem-platform\syphu-china-model")
    DEMO_DIR = project_root / "demo_data"
    
    print("=" * 70)
    print("生成演示数据集")
    print("=" * 70)
    print(f"源目录: {SOURCE_DIR}")
    print(f"目标目录: {DEMO_DIR}\n")
    
    if not SOURCE_DIR.exists():
        print(f"错误: 源目录不存在: {SOURCE_DIR}")
        return
    
    if DEMO_DIR.exists():
        import shutil
        shutil.rmtree(DEMO_DIR)
    
    DemoDataGenerator.create_demo_dataset(
        source_dir=SOURCE_DIR,
        demo_dir=DEMO_DIR,
        samples_per_type=3
    )
    
    total_size = 0
    file_count = 0
    print("\n数据统计:")
    
    for category_dir in DEMO_DIR.iterdir():
        if category_dir.is_dir():
            files = list(category_dir.iterdir())
            size = sum(f.stat().st_size for f in files if f.is_file())
            total_size += size
            file_count += len(files)
            print(f"  {category_dir.name:12s}: {len(files):2d} 个文件 ({size / 1024 / 1024:.2f} MB)")
    
    print(f"\n总计: {file_count} 个文件 ({total_size / 1024 / 1024:.2f} MB)")
    create_readme(DEMO_DIR, file_count, total_size)
    print("\n✅ 完成!")

if __name__ == "__main__":
    main()
