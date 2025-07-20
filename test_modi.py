from ultralytics import YOLO

# 加载您的模型
model = YOLO("train_model/64b_640_conv2_yolov11n/best.pt")

# 如果需要，先进行模型融合
if not model.model.is_fused():
    print("模型未融合，正在进行融合...")
    model.model.fuse()
    print("融合完成")

# 使用验证数据集验证模型
results = model.val(data="data.yaml")

# 打印验证结果
print(f"验证结果: {results}")
