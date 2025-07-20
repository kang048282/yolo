import cv2

from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.pt")

# 视频处理
cap = cv2.VideoCapture("video.mp4")

# 创建ByteTracker实例
from ultralytics.cfg import get_cfg
from ultralytics.trackers.byte_tracker import BYTETracker

# 加载默认配置
tracker_config = get_cfg("trackers/bytetrack.yaml")
tracker = BYTETracker(tracker_config, frame_rate=30)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 使用YOLO进行检测
    results = model(frame, verbose=False)

    # 获取检测结果
    boxes = results[0].boxes

    # 筛选高置信度的检测结果
    high_conf_boxes = boxes[boxes.conf > 0.5]

    if len(high_conf_boxes) > 0:
        # 转换为ByteTracker需要的格式
        dets = high_conf_boxes.xyxy.cpu().numpy()  # 边界框坐标
        scores = high_conf_boxes.conf.cpu().numpy()  # 置信度
        classes = high_conf_boxes.cls.cpu().numpy()  # 类别

        # 使用ByteTracker进行跟踪
        tracks = tracker.update(dets, scores, classes, frame)

        # 绘制跟踪结果
        for track in tracks:
            xyxy = track[:4]
            track_id = track[4]
            cls_id = track[5]

            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {int(track_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
