import kagglehub

import kagglehub
import shutil
import os


target_dir = "/Users/xiongjian/models/datasets"


cache_path = kagglehub.dataset_download("pankajjsh06/ibm-watson-marketing-customer-value-data")

# 3. 如果目标目录不存在则创建
if not os.path.exists(target_dir):
    os.makedirs(target_dir)


for filename in os.listdir(cache_path):
    source_file = os.path.join(cache_path, filename)
    target_file = os.path.join(target_dir, filename)
    shutil.copy(source_file, target_file)

print(f"✅ 数据集已成功保存至: {os.path.abspath(target_dir)}")