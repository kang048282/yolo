import os
import glob

def replace_class_id_in_folder(folder_path, old_class_id, new_class_id):
    """将指定文件夹中所有YOLO格式txt文件里的特定class-id替换为新的class-id"""
    # 获取文件夹中所有txt文件
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    # 统计信息
    total_files = len(txt_files)
    modified_files = 0
    modified_lines = 0
    
    print(f"找到 {total_files} 个txt文件")
    
    # 处理每个文件
    for txt_file in txt_files:
        file_modified = False
        lines = []
        
        # 读取文件内容
        with open(txt_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:  # YOLO格式至少有5个值（class_id + 4个坐标值）
                    if parts[0] == str(old_class_id):
                        parts[0] = str(new_class_id)
                        line = ' '.join(parts) + '\n'
                        modified_lines += 1
                        file_modified = True
                lines.append(line)
        
        # 如果文件被修改，写回文件
        if file_modified:
            with open(txt_file, 'w') as f:
                f.writelines(lines)
            modified_files += 1
    
    print(f"处理完成！修改了 {modified_files} 个文件中的 {modified_lines} 行")
    print(f"将class-id从 {old_class_id} 替换为 {new_class_id}")

# 使用示例
if __name__ == "__main__":
    folder_path = "E:/yolov/cross5/labels"  # 直接指定文件夹路径
    old_class_id = 2  # 要替换的class-id
    new_class_id = 0  # 替换后的class-id
    
    if os.path.isdir(folder_path):
        replace_class_id_in_folder(folder_path, old_class_id, new_class_id)
    else:
        print(f"错误：{folder_path} 不是有效的文件夹路径")