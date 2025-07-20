import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
  # 加载分割模型而不是检测模型
  model = YOLO('yolov8n-seg.pt')  # 使用预训练的分割模型
  # 或者从头开始训练
  # model = YOLO('ultralytics/cfg/models/v8/yolov8n-seg.yaml')
  
  results = model.train(
    data='data_road.yaml',  # 使用您的道路分割数据集
    epochs=200,  # 训练轮次
    batch=16,  # 批量大小
    imgsz=640,  # 训练图像尺寸
    workers=8,  # 加载数据的工作线程数
    device=0,  # 指定训练的计算设备，无nvidia显卡则改为 'cpu'
    optimizer='SGD',  # 训练使用优化器
    amp=True,  # 自动混合精度训练
    cache=False,  # 是否缓存数据集
    project='RoadANDSidewalk_segmentation',  # 指定项目根目录
    name='RoadANDSidewalk_segmentation'  # 指定实验名称
)