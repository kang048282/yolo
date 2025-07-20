# 智能交通目标检测与跟踪项目

本项目基于 Ultralytics YOLO 系列模型，结合 Deep SORT 实现多目标检测、分割与跟踪，适用于智能交通场景下的行为分析与预警。支持自定义数据集训练、模型评估、视频检测与分割等功能。

## 目录结构

- `train.py`：目标检测模型训练脚本
- `train_seg.py`：分割模型训练脚本
- `video_detection.py`：视频检测与跟踪脚本
- `evaluation_model.py`：模型评估脚本
- `yolov_warning.py`：交通行为分析与预警主控脚本
- `cross4.yaml`、`data.yaml`、`data_road.yaml`：数据集配置文件 测试数据集配置、目标检测数据集配置、目标分割数据配置集
- `yolo11n.pt`、`yolov8n.pt`、`yolov8n-seg.pt`：预训练模型权重
- `deep_sort_pytorch/`：Deep SORT 跟踪模块
- `cross/`：自定义数据集或相关资源
- `evaluation_results_*/`：模型（根据不同类）评估结果图表输出目录
- `proccess_data/`：数据预处理与格式转换相关脚本
- `RoadANDSidewalk_segmentation/`：道路与人行道分割的训练模型
- `train_model/`：不同模型训练得到的训练文件
- `validation_results/`：模型验证与评估结果和图表
## 环境依赖

推荐使用 Conda 环境：

```sh
conda env create -f environment.yml
conda activate yolov
# 或使用 pip
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型

#### 目标检测训练
> 具体参数和用法可参考 `train.py` 脚本内注释

```sh
python train.py
```

#### 分割模型训练
```sh
python train_seg.py
```

### 2. 视频目标检测

需要在 `video_detection.py` 中修改以下路径:

- **模型路径**: `model = YOLO('train_model/16b_1280_origin_yolov8/best.pt')`
- **视频路径**: `video_path = "cross.mp4"`
- **标签路径**: `labels_dir = "cross/labels"`

```sh
python video_detection.py
```

### 3. 模型评估

在 `evaluation_model.py` 中需修改:
- evaluate_model 中的模型路径
- 结果输出路径

```sh
python evaluation_model.py
```

### 4. 视频检测与行为分析

在 `yolov_warning.py` 中配置以下参数:

```python
config = {
    # 分割模型相关配置
    'segmentation_model_path': 'RoadANDSidewalk_segmentation/RoadANDSidewalk_segmentation3/weights/best.pt',
    'segmentation_data_config': 'data_road.yaml',
    
    # 目标检测模型配置
    'detection_model_path': 'train_model/16b_1280_origin_yolov8/best.pt',
    'deepsort_config': 'deep_sort_pytorch/configs/deep_sort.yaml',
    
    # 视频路径配置
    'input_video': 'cross.mp4',
    'output_video': 'yolo_warning/intelligent_traffic_analysis_results.mp4'
}
```

运行分析:
```sh
python yolov_warning.py
```


## 数据集格式

### YOLO 目标检测数据集格式

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.png
│   │   └── image2.png
│   └── val/
│       ├── image3.png
│       └── image4.png
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   └── image2.txt
│   └── val/
│       ├── image3.txt
│       └── image4.txt
```

标签文件格式 (*.txt):
```
<class_id> <x_center> <y_center> <width> <height>
```
- class_id: 目标类别索引 (从0开始)
- x_center, y_center: 边界框中心点坐标 (均为归一化值，相对值0-1)
- width, height: 边界框宽度和高度 (均为归一化值，相对值0-1)

### YOLO 语义分割数据集格式

结构和检测一致

标签文件格式 (*.txt):
```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ... <xn> <yn>
```
- class_id: 目标类别索引 (从0开始)
- xn yn: 多边形坐标 (均为归一化值，相对值0-1)

配置文件格式 (data.yaml):
```yaml
path: ./dataset  # 数据集根目录
train: images/train  # 训练集图片目录
val: images/val      # 验证集图片目录

# 类别名称
names:
  0: road
  1: sidewalk

# 可选配置
nc: 2  # 类别数量
test:  # 测试集目录(可选)
```

数据集配置可参考项目中的 [cross4.yaml](cross4.yaml) 或 [data.yaml](data.yaml) 配置文件。

测试视频的标签，即cross目录中的标签，是我用DarkLabel标注的。

## 参考资料

- [Ultralytics YOLO 官方文档](https://docs.ultralytics.com/)
- [Deep SORT 论文与实现](https://github.com/nwojke/deep_sort)
- [ultralytics](https://github.com/ultralytics)

## 致谢

感谢 Ultralytics 团队及开源社区的支持与贡献。
