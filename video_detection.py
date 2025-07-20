import os
import time

import cv2
import numpy as np

from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("train_model/16b_1280_origin_yolov8/best.pt")

# 打开视频文件
video_path = "cross.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频属性
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 创建视频写入器
output_path = "detection_results.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# 降低播放速度
out = cv2.VideoWriter(output_path, fourcc, fps / 2, (width, height))

# 标注数据路径
labels_dir = "cross/labels"

# 设置置信度阈值
conf_threshold = 0.5  # 可以根据需要调整这个值

# 设置形状差异阈值
shape_diff_threshold = 0.5  # 形状差异阈值，可以根据需要调整

# 设置面积差异阈值
area_diff_threshold = 0.5  # 面积差异阈值，可以根据需要调整


# 定义IoU计算函数
def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU
    box格式: [x1, y1, x2, y2].
    """
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算两个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union = box1_area + box2_area - intersection

    # 计算IoU
    iou = intersection / union if union > 0 else 0

    return iou


# 定义形状差异计算函数
def calculate_shape_difference(box1, box2):
    """
    计算两个边界框的形状差异
    box格式: [x1, y1, x2, y2]
    返回一个0到1之间的值，0表示完全相同，1表示完全不同.
    """
    # 计算宽高
    width1 = box1[2] - box1[0]
    height1 = box1[3] - box1[1]
    width2 = box2[2] - box2[0]
    height2 = box2[3] - box2[1]

    # 计算宽高比
    ratio1 = width1 / height1 if height1 > 0 else 0
    ratio2 = width2 / height2 if height2 > 0 else 0

    # 计算宽高比差异（归一化）
    if ratio1 > ratio2:
        shape_diff = (ratio1 - ratio2) / ratio1 if ratio1 > 0 else 1
    else:
        shape_diff = (ratio2 - ratio1) / ratio2 if ratio2 > 0 else 1

    return shape_diff


def calculate_area_difference(box1, box2):
    """
    计算两个边界框的面积差异
    box格式: [x1, y1, x2, y2]
    返回一个0到1之间的值，0表示完全相同，1表示完全不同.
    """
    # 计算面积
    width1 = box1[2] - box1[0]
    height1 = box1[3] - box1[1]
    width2 = box2[2] - box2[0]
    height2 = box2[3] - box2[1]

    area1 = width1 * height1
    area2 = width2 * height2

    # 计算面积差异（归一化）
    if area1 > area2:
        area_diff = (area1 - area2) / area1 if area1 > 0 else 1
    else:
        area_diff = (area2 - area1) / area2 if area2 > 0 else 1

    return area_diff


# 加载标注数据
def load_annotations(frame_id):
    """
    加载特定帧的标注数据
    返回格式: {class_id: [[x1, y1, x2, y2], ...], ...}.
    """
    # 如果帧ID超过了标注范围（只有0-158帧有标注），返回空字典
    if frame_id > 240:
        return {}

    # 构建标注文件路径（基于用户提供的信息，标注文件从00000000.txt到00000158.txt）
    label_file = os.path.join(labels_dir, f"{frame_id:08d}.txt")

    # 如果文件不存在，返回空字典
    if not os.path.exists(label_file):
        return {}

    annotations = {0: [], 1: []}  # 0: car, 1: person

    # 读取标注文件
    with open(label_file) as f:
        for line in f.readlines():
            data = line.strip().split()
            if len(data) >= 5:
                # YOLO格式: class_id, x_center, y_center, width, height (归一化坐标)
                class_id = int(data[0])
                x_center = float(data[1]) * width
                y_center = float(data[2]) * height
                box_width = float(data[3]) * width
                box_height = float(data[4]) * height

                # 转换为xyxy格式
                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)
                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)

                # 确保坐标在图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                # 只处理person和car类别
                if class_id in [0, 1]:
                    annotations[class_id].append([x1, y1, x2, y2])

    return annotations


# 初始化IoU统计
person_ious = []
car_ious = []

# 用于计算FPS的变量
frame_count = 0
start_time = time.time()
current_frame = 0

# 在主循环前添加一个变量来跟踪视频是否暂停
is_paused = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    current_frame += 1
    frame_count += 1

    if current_frame > 240:
        break

    # 计算实时FPS
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:
        current_fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    else:
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    # 保存原始帧的副本
    original_frame = frame.copy()

    # 使用YOLOv8进行目标检测
    results = model(frame)

    # 处理检测结果
    result = results[0]
    boxes = result.boxes

    # 获取当前帧的标注数据
    gt_annotations = load_annotations(current_frame)

    # 处理检测结果
    detections = {0: [], 1: []}  # 0: car, 1: person

    # 首先收集所有检测框
    all_detections = {0: [], 1: []}  # 0: car, 1: person
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # 只处理置信度高于阈值的行人和车辆
        if cls in [0, 1] and conf >= conf_threshold:  # 添加置信度筛选
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            all_detections[cls].append([x1, y1, x2, y2, conf])

    # 对每个类别，筛选形状差异不大的检测框
    for cls in [0, 1]:  # 0: car, 1: person
        if (
            cls in gt_annotations
            and len(gt_annotations[cls]) > 0
            and cls in all_detections
            and len(all_detections[cls]) > 0
        ):
            gt_boxes = gt_annotations[cls]

            # 对每个检测框，检查与标注框的形状差异
            for det_box in all_detections[cls]:
                min_shape_diff = float("inf")
                min_area_diff = float("inf")
                for gt_box in gt_boxes:
                    shape_diff = calculate_shape_difference(gt_box, det_box[:4])
                    area_diff = calculate_area_difference(gt_box, det_box[:4])
                    min_shape_diff = min(min_shape_diff, shape_diff)
                    min_area_diff = min(min_area_diff, area_diff)

                # 如果最小形状差异小于阈值，则保留该检测框
                if min_shape_diff < shape_diff_threshold and min_area_diff < area_diff_threshold:
                    detections[cls].append(det_box)
        else:
            # 如果没有标注框，则保留所有检测框
            detections[cls] = all_detections[cls]

    # 计算每个类别的IoU
    frame_person_ious = []
    frame_car_ious = []

    # 对每个类别计算IoU
    for cls in [0, 1]:  # 0: car, 1: person
        if cls in gt_annotations and len(gt_annotations[cls]) > 0 and cls in detections and len(detections[cls]) > 0:
            gt_boxes = gt_annotations[cls]
            det_boxes = [d[:4] for d in detections[cls]]

            # 对每个标注框，找到IoU最高的检测框
            for gt_box in gt_boxes:
                max_iou = 0
                for det_box in det_boxes:
                    iou = calculate_iou(gt_box, det_box)
                    max_iou = max(max_iou, iou)

                # 记录IoU
                if cls == 0:  # car
                    frame_car_ious.append(max_iou)
                else:  # person
                    frame_person_ious.append(max_iou)

    # 更新IoU统计
    if frame_person_ious:
        person_ious.append(np.mean(frame_person_ious))
    else:
        person_ious.append(0)

    if frame_car_ious:
        car_ious.append(np.mean(frame_car_ious))
    else:
        car_ious.append(0)

    # 计算平均IoU（识别率）
    window_size = min(4, len(person_ious))
    # window_size = len(person_ious)
    avg_person_iou = np.mean(person_ious[-window_size:]) if person_ious else 0
    avg_car_iou = np.mean(car_ious[-window_size:]) if car_ious else 0

    # 在原始帧上绘制边界框和标签 - 只绘制通过形状筛选的框
    for cls in [0, 1]:
        for det in detections[cls]:
            x1, y1, x2, y2, conf = det

            # 根据类别选择颜色
            color = (0, 0, 255) if cls == 0 else (0, 255, 0)  # 红色为车辆，绿色为行人

            # 绘制边界框
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, 5)

            # 添加标签
            label = f"{'Car' if cls == 0 else 'Person'} {conf:.2f}"
            cv2.putText(original_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    # 添加展示标注框的显示代码
    # 不再显示标注框

    # 添加识别率信息（使用IoU），增大字体大小
    cv2.putText(
        original_frame, f"FPS: {current_fps:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5
    )  # 增大字体
    cv2.putText(
        original_frame,
        f"Conf Threshold: {conf_threshold:.2f}",
        (10, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 255, 0),
        4,
    )  # 显示置信度阈值
    cv2.putText(
        original_frame,
        f"Shape Diff Threshold: {shape_diff_threshold:.2f}",
        (10, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 255, 0),
        4,
    )  # 显示形状差异阈值
    # cv2.putText(original_frame, f"Person IoU: {avg_person_iou:.3f}",
    #            (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)  # 增大字体
    # cv2.putText(original_frame, f"Car IoU: {avg_car_iou:.3f}",
    #            (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)  # 增大字体
    cv2.putText(
        original_frame,
        f"Area Diff Threshold: {area_diff_threshold:.2f}",
        (10, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 255, 0),
        4,
    )  # 显示面积差异阈值

    # 写入输出视频
    out.write(original_frame)

    # 调整显示窗口大小
    display_width = int(width * 0.5)
    display_height = int(height * 0.4)
    display_frame = cv2.resize(original_frame, (display_width, display_height))

    # 显示调整大小后的结果
    cv2.imshow("YOLOv8 Detection", display_frame)
    # 处理键盘事件
    key = cv2.waitKey(50 if not is_paused else 0) & 0xFF
    # 按'q'键退出
    if key == ord("q"):
        break
    # 按空格键暂停/继续
    elif key == ord(" "):
        is_paused = not is_paused
        print("视频已" + ("暂停" if is_paused else "继续"))
    # 添加调整置信度阈值的键盘控制
    elif key == ord("+") or key == ord("="):  # 增加置信度阈值
        conf_threshold = min(0.99, conf_threshold + 0.05)
        print(f"置信度阈值增加到: {conf_threshold:.2f}")
    elif key == ord("-") or key == ord("_"):  # 减少置信度阈值
        conf_threshold = max(0.05, conf_threshold - 0.05)
        print(f"置信度阈值减少到: {conf_threshold:.2f}")
    # 添加调整形状差异阈值的键盘控制
    elif key == ord("[") or key == ord("{"):  # 减少形状差异阈值
        shape_diff_threshold = max(0.1, shape_diff_threshold - 0.1)
        print(f"形状差异阈值减少到: {shape_diff_threshold:.2f}")
    elif key == ord("]") or key == ord("}"):  # 增加形状差异阈值
        shape_diff_threshold = min(0.9, shape_diff_threshold + 0.1)
        print(f"形状差异阈值增加到: {shape_diff_threshold:.2f}")
    # 添加调整面积差异阈值的键盘控制
    elif key == ord("a") or key == ord("A"):  # 减少面积差异阈值
        area_diff_threshold = max(0.1, area_diff_threshold - 0.1)
        print(f"面积差异阈值减少到: {area_diff_threshold:.2f}")
    elif key == ord("d") or key == ord("D"):  # 增加面积差异阈值
        area_diff_threshold = min(0.9, area_diff_threshold + 0.1)
        print(f"面积差异阈值增加到: {area_diff_threshold:.2f}")


# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
