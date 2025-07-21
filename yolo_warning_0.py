import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time
import json

class AlertLevel(Enum):
    """预警级别枚举"""
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"

@dataclass
class DetectionResult:
    """检测结果数据类"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

@dataclass
class SegmentationResult:
    """分割结果数据类"""
    mask: np.ndarray
    class_id: int
    class_name: str
    confidence: float

@dataclass
class BehaviorAlert:
    """行为预警数据类"""
    alert_type: str
    level: AlertLevel
    description: str
    position: Tuple[int, int]
    timestamp: float
    involved_objects: List[int]  # track_ids
    frame_counter: int = 0  # 添加帧计数器，用于控制显示持续时间

class SegmentationManager:
    """分割模型管理器"""
    
    def __init__(self, model_path: str, data_config: str):
        self.model = YOLO(model_path)
        self.load_config(data_config)
        
    def load_config(self, config_path: str):
        """加载数据配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.class_names = self.config['names']
        
    def segment_frame(self, frame: np.ndarray) -> List[SegmentationResult]:
        """对帧进行分割"""
        results = self.model(frame, task='segment')
        segmentation_results = []
        
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes
            
            for i, mask in enumerate(masks):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                class_name = self.class_names[cls_id]
                
                segmentation_results.append(SegmentationResult(
                    mask=mask,
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf
                ))
                
        return segmentation_results
    
    def create_overlay(self, frame: np.ndarray, seg_results: List[SegmentationResult]) -> np.ndarray:
        """创建分割叠加层 - 增强道路和人行道显示"""
        overlay = frame.copy()
        
        # 优化的颜色映射 - 更鲜明的颜色
        colors = {
            0: (0, 200, 0),    # road - 更亮的绿色
            1: (255, 100, 0),  # sidewalk - 橙色（更容易区分）
        }
        
        # 类别名称映射
        class_names = {
            0: "Road",
            1: "Sidewalk"
        }
        
        for seg_result in seg_results:
            if seg_result.confidence > 0.3:  # 降低阈值以显示更多区域
                mask = seg_result.mask
                color = colors.get(seg_result.class_id, (128, 128, 128))
                class_name = class_names.get(seg_result.class_id, "Unknown")
                
                # 调整mask尺寸以匹配frame
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_bool = mask_resized > 0.3  # 降低阈值
                
                # 应用更明显的半透明叠加
                overlay[mask_bool] = cv2.addWeighted(
                    overlay[mask_bool], 0.6,  # 降低原图权重
                    np.full_like(overlay[mask_bool], color), 0.4, 0  # 增加叠加权重
                )
                
                # 添加轮廓线以更清晰地显示边界
                contours, _ = cv2.findContours(
                    (mask_resized > 0.3).astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, color, 2)
                
        return overlay

class ObjectTracker:
    """目标检测和跟踪管理器"""
    
    def __init__(self, detection_model_path: str, deepsort_config: str):
        self.detection_model = YOLO(detection_model_path)
        self.setup_deepsort(deepsort_config)
        self.track_history = {}
        self.target_classes = [0, 1]  # person, car
        
    def setup_deepsort(self, config_path: str):
        """设置DeepSORT"""
        cfg = get_config()
        cfg.merge_from_file(config_path)
        self.deepsort = DeepSort(
            cfg.DEEPSORT.REID_CKPT,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=True
        )
        
    def detect_and_track(self, frame: np.ndarray) -> List[DetectionResult]:
        """检测和跟踪目标"""
        results = self.detection_model(frame)
        detections = results[0].boxes
        
        # 准备DeepSORT输入
        bbox_xywh = []
        confidences = []
        class_ids = []
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cls_id = int(det.cls[0])
            conf = float(det.conf[0])
            
            if cls_id in self.target_classes and conf > 0.3:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                
                bbox_xywh.append([center_x, center_y, w, h])
                confidences.append(conf)
                class_ids.append(cls_id)
        
        detection_results = []
        
        if len(bbox_xywh) > 0:
            bbox_xywh = np.array(bbox_xywh)
            confidences = np.array(confidences)
            
            # DeepSORT跟踪
            outputs = self.deepsort.update(bbox_xywh, confidences, class_ids, frame)
            
            for output in outputs:
                x1, y1, x2, y2, track_id, cls_id = output
                cls_id = int(cls_id)
                class_name = self.detection_model.names[cls_id]
                
                detection_results.append(DetectionResult(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidences[0],  # 简化处理
                    class_id=cls_id,
                    class_name=class_name,
                    track_id=track_id
                ))
                
                # 更新跟踪历史
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                self.track_history[track_id].append(center)
                
                # 限制轨迹长度
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)
                    
        return detection_results

class BehaviorAnalyzer:
    """行为分析器"""
    
    def __init__(self):
        self.crosswalk_zones = []  # 人行横道区域
        self.intersection_center = None  # 十字路口中心
        self.speed_threshold = 50  # 速度阈值(像素/帧)
        self.proximity_threshold = 100  # 接近阈值(像素)
        
    def set_intersection_zones(self, crosswalk_zones: List[Tuple], intersection_center: Tuple):
        """设置十字路口区域"""
        self.crosswalk_zones = crosswalk_zones
        self.intersection_center = intersection_center
        
    def analyze_behavior(self, detections: List[DetectionResult], 
                        seg_results: List[SegmentationResult],
                        track_history: Dict) -> List[BehaviorAlert]:
        """分析行为并生成预警"""
        alerts = []
        current_time = time.time()
        
        # 分析每个检测到的目标
        for detection in detections:
            if detection.track_id is None:
                continue
                
            # 获取目标位置和历史轨迹
            x1, y1, x2, y2 = detection.bbox
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            history = track_history.get(detection.track_id, [])
            
            # 1. 速度分析
            speed_alert = self._analyze_speed(detection, history, current_time)
            if speed_alert:
                alerts.append(speed_alert)
                
            # 2. 区域违规分析
            zone_alert = self._analyze_zone_violation(detection, seg_results, center)
            if zone_alert:
                alerts.append(zone_alert)
                
            # 3. 碰撞风险分析
            collision_alerts = self._analyze_collision_risk(detection, detections, center)
            alerts.extend(collision_alerts)
            
        return alerts
    
    def _analyze_speed(self, detection: DetectionResult, history: List, timestamp: float) -> Optional[BehaviorAlert]:
        """分析速度异常"""
        if len(history) < 5:
            return None
            
        # 计算最近5帧的平均速度
        recent_points = history[-5:]
        total_distance = 0
        for i in range(1, len(recent_points)):
            dx = recent_points[i][0] - recent_points[i-1][0]
            dy = recent_points[i][1] - recent_points[i-1][1]
            total_distance += np.sqrt(dx*dx + dy*dy)
            
        avg_speed = total_distance / 4  # 4个间隔
        
        if avg_speed > self.speed_threshold:
            return BehaviorAlert(
                alert_type="Overspeeding",
                level=AlertLevel.WARNING,
                description=f"{detection.class_name}Overspeeding，speed: {avg_speed:.1f}Pixels per frame",
                position=history[-1],
                timestamp=timestamp,
                involved_objects=[detection.track_id]
            )
        return None
    
    def _analyze_zone_violation(self, detection: DetectionResult, 
                               seg_results: List[SegmentationResult], 
                               center: Tuple) -> Optional[BehaviorAlert]:
        """分析区域违规"""
        # 检查行人是否在车道上
        if detection.class_name == "person":
            for seg_result in seg_results:
                if seg_result.class_name == "road":
                    mask_resized = cv2.resize(seg_result.mask, (1920, 1080))  # 假设视频尺寸
                    if center[1] < mask_resized.shape[0] and center[0] < mask_resized.shape[1]:
                        if mask_resized[center[1], center[0]] > 0.5:
                            return BehaviorAlert(
                                alert_type="PedestrianViolation",
                                level=AlertLevel.DANGER,
                                description="Pedestrian enter lane region",
                                position=center,
                                timestamp=time.time(),
                                involved_objects=[detection.track_id]
                            )
        
        # 检查车辆是否在人行道上
        elif detection.class_name in ["car", "truck", "bus"]:
            for seg_result in seg_results:
                if seg_result.class_name == "sidewalk":
                    mask_resized = cv2.resize(seg_result.mask, (1920, 1080))
                    if center[1] < mask_resized.shape[0] and center[0] < mask_resized.shape[1]:
                        if mask_resized[center[1], center[0]] > 0.5:
                            return BehaviorAlert(
                                alert_type="VehicleViolation",
                                level=AlertLevel.DANGER,
                                description="Vehicle enter sidewalk region",
                                position=center,
                                timestamp=time.time(),
                                involved_objects=[detection.track_id]
                            )
        return None
    
    def _analyze_collision_risk(self, detection: DetectionResult, 
                               all_detections: List[DetectionResult], 
                               center: Tuple) -> List[BehaviorAlert]:
        """分析碰撞风险"""
        alerts = []
        
        for other_detection in all_detections:
            if (other_detection.track_id == detection.track_id or 
                other_detection.track_id is None):
                continue
                
            other_x1, other_y1, other_x2, other_y2 = other_detection.bbox
            other_center = ((other_x1 + other_x2) // 2, (other_y1 + other_y2) // 2)
            
            # 计算距离
            distance = np.sqrt((center[0] - other_center[0])**2 + 
                             (center[1] - other_center[1])**2)
            
            # 如果是行人和车辆接近
            if (distance < self.proximity_threshold and 
                ((detection.class_name == "person" and other_detection.class_name in ["car", "truck", "bus"]) or
                 (detection.class_name in ["car", "truck", "bus"] and other_detection.class_name == "person"))):
                
                alerts.append(BehaviorAlert(
                    alert_type="CrashDanger",
                    level=AlertLevel.WARNING,
                    description=f"Pedestrian and car are too close: {distance:.1f}Pixels",
                    position=center,
                    timestamp=time.time(),
                    involved_objects=[detection.track_id, other_detection.track_id]
                ))
                
        return alerts

class AlertSystem:
    """预警系统"""
    
    def __init__(self):
        self.alert_history = []
        self.alert_cooldown = {}  # 防止重复预警
        self.cooldown_time = 3.0  # 3秒冷却时间
        self.persistent_alerts = []  # 持续显示的预警
        self.alert_display_duration = 6  # 预警显示持续帧数

    def process_alerts(self, alerts: List[BehaviorAlert]) -> List[BehaviorAlert]:
        """处理预警信息"""
        current_time = time.time()
        valid_alerts = []
        
        # 处理新的预警
        for alert in alerts:
            # 检查冷却时间
            alert_key = f"{alert.alert_type}_{alert.involved_objects}"
            last_alert_time = self.alert_cooldown.get(alert_key, 0)
            
            if current_time - last_alert_time > self.cooldown_time:
                # 添加帧计数器到预警对象
                alert.frame_counter = self.alert_display_duration
                valid_alerts.append(alert)
                self.alert_cooldown[alert_key] = current_time
                self.alert_history.append(alert)
                # 添加到持续显示列表
                self.persistent_alerts.append(alert)
        
        # 更新持续显示的预警列表
        self._update_persistent_alerts()
        
        # 限制历史记录长度
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
            
        return self.get_current_display_alerts()
    
    def _update_persistent_alerts(self):
        """更新持续显示的预警列表"""
        # 减少每个预警的帧计数器
        for alert in self.persistent_alerts[:]:
            if hasattr(alert, 'frame_counter'):
                alert.frame_counter -= 1
                if alert.frame_counter <= 0:
                    self.persistent_alerts.remove(alert)
    
    def get_current_display_alerts(self) -> List[BehaviorAlert]:
        """获取当前应该显示的预警"""
        return [alert for alert in self.persistent_alerts if hasattr(alert, 'frame_counter') and alert.frame_counter > 0]
    
    def get_statistics(self) -> Dict:
        """获取预警统计信息"""
        if not self.alert_history:
            return {}
            
        alert_counts = {}
        level_counts = {}
        
        for alert in self.alert_history:
            alert_counts[alert.alert_type] = alert_counts.get(alert.alert_type, 0) + 1
            level_counts[alert.level.value] = level_counts.get(alert.level.value, 0) + 1
            
        return {
            "total_alerts": len(self.alert_history),
            "alert_types": alert_counts,
            "alert_levels": level_counts,
            "current_displaying": len(self.persistent_alerts)
        }

class IntelligentTrafficAnalyzer:
    """智能交通分析主控制器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_components()
        self.setup_video()
        self.frame_count = 0
        
    def setup_components(self):
        """初始化各个组件"""
        # 初始化分割管理器
        self.seg_manager = SegmentationManager(
            self.config['segmentation_model_path'],
            self.config['segmentation_data_config']
        )
        
        # 初始化目标跟踪器
        self.object_tracker = ObjectTracker(
            self.config['detection_model_path'],
            self.config['deepsort_config']
        )
        
        # 初始化行为分析器
        self.behavior_analyzer = BehaviorAnalyzer()
        
        # 初始化预警系统
        self.alert_system = AlertSystem()
        
    def setup_video(self):
        """设置视频输入输出"""
        self.cap = cv2.VideoCapture(self.config['input_video'])
        assert self.cap.isOpened(), "无法打开视频文件"
        
        # 获取视频属性
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)  # 保存原始帧率
        
        # 设置降低的帧率（可以调整这个值）
        self.target_fps = 1  # 降低到15帧每秒，您可以根据需要调整
        self.fps = self.target_fps
        # 计算跳帧间隔
        self.frame_skip = max(1, int(self.original_fps / self.target_fps))
        
        # 设置输出视频（使用降低的帧率）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            self.config['output_video'], fourcc, self.target_fps, (self.width, self.height)
        )
        
        # 播放速度控制
        self.playback_delay = int(1000 / self.target_fps)  # 根据目标帧率计算延迟

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧"""
        # 1. 分割处理
        seg_results = self.seg_manager.segment_frame(frame)
        
        # 2. 目标检测和跟踪
        detections = self.object_tracker.detect_and_track(frame)
        
        # 3. 行为分析
        alerts = self.behavior_analyzer.analyze_behavior(
            detections, seg_results, self.object_tracker.track_history
        )
        
        # 4. 预警处理
        valid_alerts = self.alert_system.process_alerts(alerts)
        
        # 5. 可视化
        result_frame = self.visualize_results(
            frame, seg_results, detections, valid_alerts
        )
        
        return result_frame
    
    def visualize_results(self, frame: np.ndarray, 
                         seg_results: List[SegmentationResult],
                         detections: List[DetectionResult],
                         alerts: List[BehaviorAlert]) -> np.ndarray:
        """可视化结果"""
        # 创建分割叠加层
        result_frame = self.seg_manager.create_overlay(frame, seg_results)
        
        # 绘制检测和跟踪结果
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # 选择颜色
            color = (0, 255, 0) if detection.class_name == "person" else (255, 0, 0)
            
            # 绘制边界框
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{detection.class_name}-{detection.track_id}"
            cv2.putText(result_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 绘制轨迹
            if detection.track_id in self.object_tracker.track_history:
                history = self.object_tracker.track_history[detection.track_id]
                for i in range(1, len(history)):
                    if i % 2 == 0:
                        cv2.line(result_frame, history[i-1], history[i], color, 1)
        
        # 添加图例显示
        legend_y = 80
        cv2.putText(result_frame, "Legend:", (self.width - 200, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 道路图例
        cv2.rectangle(result_frame, (self.width - 190, legend_y + 10), 
                     (self.width - 170, legend_y + 25), (0, 200, 0), -1)
        cv2.putText(result_frame, "Road", (self.width - 160, legend_y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 人行道图例
        cv2.rectangle(result_frame, (self.width - 190, legend_y + 35), 
                     (self.width - 170, legend_y + 50), (255, 100, 0), -1)
        cv2.putText(result_frame, "Sidewalk", (self.width - 160, legend_y + 47), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示分割统计信息
        road_count = sum(1 for seg in seg_results if seg.class_id == 0 and seg.confidence > 0.3)
        sidewalk_count = sum(1 for seg in seg_results if seg.class_id == 1 and seg.confidence > 0.3)
        
        cv2.putText(result_frame, f"Road regions: {road_count}", (10, self.height - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        cv2.putText(result_frame, f"Sidewalk regions: {sidewalk_count}", (10, self.height - 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        # 绘制预警信息
        alert_y = 50
        for alert in alerts:
            alert_color = {
                AlertLevel.SAFE: (0, 255, 0),
                AlertLevel.WARNING: (0, 255, 255),
                AlertLevel.DANGER: (0, 0, 255)
            }[alert.level]
            
            alert_text = f"[{alert.level.value}] {alert.description}"
            cv2.putText(result_frame, alert_text, (10, alert_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)
            alert_y += 25
            
            # 在预警位置绘制标记
            cv2.circle(result_frame, alert.position, 10, alert_color, -1)
        
        # 显示统计信息
        stats = self.alert_system.get_statistics()
        if stats:
            stats_text = f"total alert: {stats.get('total_alerts', 0)}"
            cv2.putText(result_frame, stats_text, (10, self.height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示帧率
        cv2.putText(result_frame, f"FPS: {int(self.fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return result_frame
    
    def run(self):
        """运行主循环"""
        print("开始智能交通分析...")
        paused = False  # 添加暂停状态变量
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
                
            self.frame_count += 1
            
            # 处理帧
            result_frame = self.process_frame(frame)
            
            # 写入输出视频
            self.out.write(result_frame)
            
            # 显示结果
            cv2.imshow("智能交通分析系统", result_frame)
            
            # 按键控制
            key = cv2.waitKey(50 if not paused else 0) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):  # 空格键暂停/恢复
                paused = not paused
                if paused:
                    print("视频已暂停，按空格键继续...")
                else:
                    print("视频继续播放...")
            elif key == ord('s'):  # 保存当前帧
                cv2.imwrite(f"frame_{self.frame_count}.jpg", result_frame)
                print(f"已保存帧 {self.frame_count}")
            
            # 每100帧输出一次统计信息
            if self.frame_count % 100 == 0:
                stats = self.alert_system.get_statistics()
                print(f"处理帧数: {self.frame_count}, 预警统计: {stats}")
        
        self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        
        # 保存最终统计报告
        final_stats = self.alert_system.get_statistics()
        with open('yolo_warning/traffic_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump({
                'total_frames': self.frame_count,
                'alert_statistics': final_stats,
                'alert_history': [{
                    'type': alert.alert_type,
                    'level': alert.level.value,
                    'description': alert.description,
                    'timestamp': alert.timestamp
                } for alert in self.alert_system.alert_history]
            }, f, ensure_ascii=False, indent=2)
        
        print("分析完成，报告已保存到 traffic_analysis_report.json")

def main():
    """主函数"""
    # 配置参数
    config = {
        'segmentation_model_path': 'RoadANDSidewalk_segmentation/RoadANDSidewalk_segmentation3/weights/best.pt',  # 您的分割模型路径
        'segmentation_data_config': 'data_road.yaml',
        'detection_model_path': 'train_model/16b_1280_origin_yolov8/best.pt',
        'deepsort_config': 'deep_sort_pytorch/configs/deep_sort.yaml',
        'input_video': 'cross.mp4',
        'output_video': 'yolo_warning/intelligent_traffic_analysis_results.mp4'
    }
    
    # 创建并运行分析器
    analyzer = IntelligentTrafficAnalyzer(config)
    analyzer.run()

if __name__ == "__main__":
    main()