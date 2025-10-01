import os
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# 文件夹路径
folder_path = r'C:\Users\Administrator\PycharmProjects\igem1\igem工作\CTD'
output_dir = os.path.join(folder_path, 'processed_results')
os.makedirs(output_dir, exist_ok=True)


def read_xml(file_path):
    """改进的XML读取函数，处理XML文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'xml')
        records = []
        for item in soup.find_all('Record'):
            record = {}
            for child in item.children:
                if child.name:
                    record[child.name] = child.text
            records.append(record)
        return pd.DataFrame(records)
    except Exception as e:
        print(f"处理XML文件 {os.path.basename(file_path)} 时出错: {str(e)}")
        return None


def safe_read_csv(file_path):
    """改进的CSV读取函数，处理分隔符问题"""
    try:
        # 尝试自动检测分隔符
        with open(file_path, 'r', encoding='utf-8') as f:
            first_lines = [f.readline() for _ in range(5)]  # 读取前5行判断格式

        # 检查可能的引号问题
        quote_char = '"' if any('"' in line for line in first_lines) else None

        # 尝试常见分隔符
        for sep in [',', '\t', ';', '|']:
            try:
                df = pd.read_csv(file_path, sep=sep, quotechar=quote_char,
                                 engine='python', on_bad_lines='warn')
                if len(df.columns) > 1:  # 确保成功分割
                    return df
            except:
                continue

        # 如果所有分隔符都失败，尝试固定宽度
        try:
            return pd.read_fwf(file_path)
        except:
            print(f"无法确定文件 {os.path.basename(file_path)} 的分隔符")
            return None
    except Exception as e:
        print(f"处理CSV文件 {os.path.basename(file_path)} 时出错: {str(e)}")
        return None


# 获取文件夹中所有文件
all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# 读取所有文件数据
all_data = []
for file in tqdm(all_files, desc="处理文件中"):
    file_path = os.path.join(folder_path, file)

    try:
        if file == 'EXO.csv':
            # 特殊处理EXO.csv文件
            try:
                data = pd.read_csv(file_path)
                if 'source_file' not in data.columns:
                    data['source_file'] = file
                all_data.append(data)
                continue
            except Exception as e:
                print(f"处理文件 {file} 时出错: {str(e)}")
                continue

        if file.endswith('.xml'):
            data = read_xml(file_path)
        elif file.endswith(('.xlsx', '.xls')):
            try:
                data = pd.read_excel(file_path, engine='openpyxl')
            except:
                # 如果openpyxl失败，尝试其他引擎
                try:
                    data = pd.read_excel(file_path, engine='xlrd')
                except Exception as e:
                    print(f"处理Excel文件 {file} 时出错: {str(e)}")
                    continue
        elif file.endswith(('.csv', '.tsv', '.txt')):
            data = safe_read_csv(file_path)
        else:
            print(f"不支持的文件类型: {file}")
            continue

        if data is not None and not data.empty:
            if 'source_file' not in data.columns:
                data['source_file'] = file
            all_data.append(data)
            print(f"成功读取: {file} (共 {data.shape[0]} 行, {data.shape[1]} 列)")
    except Exception as e:
        print(f"处理文件 {file} 时出错: {str(e)}")

# 合并所有数据
if all_data:
    combined_data = pd.concat(all_data, ignore_index=True)
    print("\n数据合并完成:")
    print(combined_data.head())
    print(f"\n总共读取了 {len(all_files)} 个文件，成功合并 {len(all_data)} 个文件，共 {len(combined_data)} 条记录")

    # 保存合并后的数据
    save_path = os.path.join(output_dir, 'combined_data.csv')
    combined_data.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n已保存合并数据到: {save_path}")

    # 保存处理日志
    with open(os.path.join(output_dir, 'processing_log.txt'), 'w', encoding='utf-8') as f:
        f.write(f"成功处理文件: {len(all_data)}/{len(all_files)}\n")
        f.write(f"总记录数: {len(combined_data)}\n")
else:
    print("没有有效数据可供分析")

print("\n处理完成。输出目录:", output_dir)
