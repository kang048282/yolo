import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
import argparse

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='使用YOLO模型检测视频中的行人和车辆')
    parser.add_argument('--model', type=str, default='train_model/16b_1280_origin_yolov8/best.pt', help='模型路径，例如：yolo11n.pt')
    parser.add_argument('--video', type=str, default='crash.mp4', help='视频文件路径')
    parser.add_argument('--output', type=str, default='detection_results_PANDC.mp4', help='输出视频路径')
    args = parser.parse_args()
    
    # 加载模型
    print(f"正在加载模型: {args.model}")
    model = YOLO(args.model)
    
    # 打开视频文件
    print(f"正在打开视频: {args.video}")
    cap = cv2.VideoCapture(args.video)
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps/2, (width, height))
    
    # 用于计算FPS的变量
    frame_count = 0
    start_time = time.time()
    current_frame = 0
    
    # 在主循环前添加一个变量来跟踪视频是否暂停
    is_paused = False
    
    print(f"开始处理视频，按'q'退出，空格键暂停/继续")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        current_frame += 1
        frame_count += 1
        
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
        
        # 使用YOLO模型进行目标检测
        results = model(frame)
        
        # 处理检测结果
        result = results[0]
        boxes = result.boxes
        
        # 处理检测结果
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # 只处理行人和车辆
            if cls in [0, 1]:  # 0: car, 1: person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 根据类别选择颜色
                color = (0, 0, 255) if cls == 0 else (0, 255, 0)  # 红色为车辆，绿色为行人
                
                # 绘制边界框
                cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, 2)
                
                # 添加标签
                label = f"{'Car' if cls == 0 else 'Person'}"
                cv2.putText(original_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 添加FPS信息
        cv2.putText(original_frame, f"FPS: {current_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # 写入输出视频
        out.write(original_frame)
        
        # 调整显示窗口大小
        display_width = min(1280, width)
        display_height = int(height * (display_width / width))
        display_frame = cv2.resize(original_frame, (display_width, display_height))
        
        # 显示调整大小后的结果
        cv2.imshow("YOLO检测", display_frame)
        
        # 处理键盘事件
        key = cv2.waitKey(1 if not is_paused else 0) & 0xFF
        
        # 按'q'键退出
        if key == ord('q'):
            break
        # 按空格键暂停/继续
        elif key == ord(' '):
            is_paused = not is_paused
            print("视频已" + ("暂停" if is_paused else "继续"))
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"视频处理完成，结果保存至: {args.output}")

if __name__ == "__main__":
    main()