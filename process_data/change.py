import os
import glob

def rename_txt_files(root_folder):
    """
    递归遍历文件夹，重命名所有包含'gtFine_polygons'的txt文件
    将'gtFine_polygons'替换为'leftImg8bit'
    """
    # 统计重命名的文件数量
    renamed_count = 0
    
    # 使用glob递归查找所有txt文件
    pattern = os.path.join(root_folder, '**', '*.txt')
    txt_files = glob.glob(pattern, recursive=True)
    
    print(f"找到 {len(txt_files)} 个txt文件")
    
    for file_path in txt_files:
        # 获取文件名（不包含路径）
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        
        # 检查文件名是否包含'gtFine_polygons'
        if 'gtFine_polygons' in file_name:
            # 替换文件名中的'gtFine_polygons'为'leftImg8bit'
            new_file_name = file_name.replace('gtFine_polygons', 'leftImg8bit')
            new_file_path = os.path.join(file_dir, new_file_name)
            
            try:
                # 重命名文件
                os.rename(file_path, new_file_path)
                print(f"重命名: {file_name} -> {new_file_name}")
                renamed_count += 1
            except Exception as e:
                print(f"重命名失败 {file_path}: {e}")
    
    print(f"\n总共重命名了 {renamed_count} 个文件")

def main():
    # 请修改这里的路径为你要处理的文件夹路径
    root_folder = input("请输入要处理的文件夹路径: ").strip()
    
    # 检查路径是否存在
    if not os.path.exists(root_folder):
        print(f"错误: 路径 '{root_folder}' 不存在")
        return
    
    if not os.path.isdir(root_folder):
        print(f"错误: '{root_folder}' 不是一个文件夹")
        return
    
    print(f"开始处理文件夹: {root_folder}")
    
    # 确认操作
    confirm = input("确定要重命名文件吗？(y/n): ").strip().lower()
    if confirm != 'y':
        print("操作已取消")
        return
    
    # 执行重命名
    rename_txt_files(root_folder)

if __name__ == "__main__":
    main()