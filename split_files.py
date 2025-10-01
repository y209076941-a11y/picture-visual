"""

import os
import argparse
def split_file(file_path, chunk_size=90*1024*1024):  # 90MB chunks
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        return False
    
    file_size = os.path.getsize(file_path)
    output_dir = os.path.dirname(file_path)
    print(f"📁 分割文件: {file_name}")
    
    num_chunks = (file_size + chunk_size - 1) // chunk_size
    
    # 生成 MD5 校验和
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
    
    # 分割文件
    with open(file_path, 'rb') as f:
        for i in range(num_chunks):
            chunk_file = f"{file_path}.part{i+1:03d}"
                chunk_data = f.read(chunk_size)
                chunk_f.write(chunk_data)
            chunk_size_mb = len(chunk_data) / (1024*1024)
            print(f"  创建: {os.path.basename(chunk_file)} ({chunk_size_mb:.2f} MB)")
    # 创建校验文件
        f.write(f"filename={file_name}\n")
        f.write(f"size={file_size}\n")
        f.write(f"md5={file_md5}\n")
        f.write(f"parts={num_chunks}\n")
    
    print(f"✅ 分割完成! 创建了 {num_chunks} 个部分文件")
    print(f"📋 校验文件: {os.path.basename(checksum_file)}")
    return True

def main():
    parser = argparse.ArgumentParser(description='分割大文件为多个小文件')
    parser.add_argument('file_path', help='要分割的文件路径')
                       help='每个分块的大小(MB)，默认90MB')
    
    args = parser.parse_args()
    
    chunk_size_bytes = args.chunk_size * 1024 * 1024

    main()
[200~cat > split_files.py << 'EOF'
文件分割工具
将大文件分割成多个小于 100MB 的部分
"""
import os
import argparse
import hashlib

def split_file(file_path, chunk_size=90*1024*1024):  # 90MB chunks
    """分割文件为多个部分"""
    if not os.path.exists(file_path):
        return False
    
    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)
    output_dir = os.path.dirname(file_path)
    print(f"📁 分割文件: {file_name}")
    print(f"📊 文件大小: {file_size / (1024*1024):.2f} MB")
    
    num_chunks = (file_size + chunk_size - 1) // chunk_size
    
    md5_hash = hashlib.md5()
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    
    # 分割文件
        for i in range(num_chunks):
            chunk_file = f"{file_path}.part{i+1:03d}"
            with open(chunk_file, 'wb') as chunk_f:
                chunk_data = f.read(chunk_size)
            
            print(f"  创建: {os.path.basename(chunk_file)} ({chunk_size_mb:.2f} MB)")
    
    checksum_file = f"{file_path}.checksum"
        f.write(f"filename={file_name}\n")
        f.write(f"md5={file_md5}\n")
    
    

    parser = argparse.ArgumentParser(description='分割大文件为多个小文件')
    parser.add_argument('file_path', help='要分割的文件路径')
    parser.add_argument('--chunk-size', type=int, default=90, 
    
    
    chunk_size_bytes = args.chunk_size * 1024 * 1024
    split_file(args.file_path, chunk_size_bytes)

if __name__ == "__main__":
