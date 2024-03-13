import os

# 设置包含你的epoch图像的文件夹路径
folder_path = "epoch_plots"

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件名是否以'1.png'结尾
    if filename.endswith("1.png"):
        # 构造新的文件名（将1替换为0）
        new_filename = filename[:-5] + "0.png"
        # 构造完整的旧文件和新文件路径
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)
        # 重命名文件
        os.rename(old_file, new_file)
        print(f"Renamed {old_file} to {new_file}")
