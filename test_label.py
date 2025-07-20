import os
from pathlib import Path


def check_dataset(data_path):
    """检查数据集完整性"""

    # 检查验证集
    val_images_dir = Path(data_path) / 'images' / 'val' / 'aachen'
    val_labels_dir = Path(data_path) / 'labels' / 'val' / 'aachen'

    print(f"验证集图像目录: {val_images_dir}")
    print(f"验证集标签目录: {val_labels_dir}")

    # 统计文件数量
    image_files = list(val_images_dir.glob('*.jpg')) + list(val_images_dir.glob('*.png'))
    label_files = list(val_labels_dir.glob('*.txt'))

    print(f"验证集图像数量: {len(image_files)}")
    print(f"验证集标签数量: {len(label_files)}")

    # 检查标签文件内容
    empty_labels = 0
    for label_file in label_files[:5]:  # 检查前5个标签文件
        with open(label_file, 'r') as f:
            content = f.read().strip()
            if not content:
                empty_labels += 1
                print(f"空标签文件: {label_file}")
            else:
                print(f"标签文件 {label_file.name} 内容: {content[:50]}...")

    print(f"空标签文件数量: {empty_labels}")


# 运行检查
check_dataset('E:/yolov/leftImg8bit')