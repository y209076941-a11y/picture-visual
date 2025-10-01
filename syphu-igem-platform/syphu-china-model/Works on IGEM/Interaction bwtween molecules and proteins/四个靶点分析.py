from google.cloud import storage
import os

# 配置参数
bucket_name = "arc-ctc-tahoe100"  # 存储桶名称（固定）
source_path = "path/to/your/file.h5ad.gz"  # 替换为实际文件路径
output_dir = "/your/target/folder"  # 替换为本地目标目录（如 ~/Downloads/tahoe_data）

# 自动创建目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 初始化客户端（自动使用浏览器登录的账户）
client = storage.Client()

# 下载文件
try:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_path)
    destination_path = os.path.join(output_dir, os.path.basename(source_path))

    blob.download_to_filename(destination_path)
    print(f"✅ 文件已保存到: {destination_path}")

except Exception as e:
    print(f"❌ 下载失败: {str(e)}")
    if "403" in str(e):
        print("请检查：1) 项目权限 2) 存储桶名称是否正确")
