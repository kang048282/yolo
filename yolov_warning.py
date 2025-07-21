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
    velocity: Optional[Tuple[float, float]] = None  # 添加速度向量 (vx, vy)
    acceleration: Optional[Tuple[float, float]] = None  # 添加加速度向量 (ax, ay)

@dataclass
class SegmentationResult:
    """分割结果数据类"""
    mask: np.ndarray
    class_id: int
    class_name: str
    confidence: float

@dataclass
class SafetyMetrics:
    """Traffic safety assessment metrics data class"""
    ttc: Optional[float] = None  # Time to Collision
    pet: Optional[float] = None  # Post Encroachment Time
    dr: Optional[float] = None  # Deceleration Rate
    ma: Optional[float] = None  # Motion Adaptation
    severity_score: Optional[float] = None  # Comprehensive severity score

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
    safety_metrics: Optional[SafetyMetrics] = None  # 添加安全评估指标

class SegmentationManager:
    """分割模型管理器"""
    
    def __init__(self, model_path: str, data_config: str = None):
        self.model = YOLO(model_path)
        if data_config:
            self.load_config(data_config)
        else:
            # 使用默认的分割类别
            self.class_names = {0: "road", 1: "sidewalk"}
        
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
                
                # 添加错误处理，如果类别ID不在字典中，使用一个默认名称
                if cls_id in self.class_names:
                    class_name = self.class_names[cls_id]
                else:
                    class_name = f"unknown_{cls_id}"
                
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
        """检测和跟踪目标，并计算速度和加速度"""
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
                
                # 计算中心点
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # 计算速度和加速度
                velocity = (0, 0)
                acceleration = (0, 0)
                
                if track_id in self.track_history and len(self.track_history[track_id]) >= 2:
                    prev_center = self.track_history[track_id][-1]
                    velocity = (center[0] - prev_center[0], center[1] - prev_center[1])
                    
                    if len(self.track_history[track_id]) >= 3:
                        prev_velocity = (prev_center[0] - self.track_history[track_id][-2][0],
                                        prev_center[1] - self.track_history[track_id][-2][1])
                        acceleration = (velocity[0] - prev_velocity[0], velocity[1] - prev_velocity[1])
                
                # 找到对应的置信度
                conf_for_this_detection = conf  # 使用当前检测的置信度
                
                detection_results.append(DetectionResult(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf_for_this_detection,
                    class_id=cls_id,
                    class_name=class_name,
                    track_id=track_id,
                    velocity=velocity,
                    acceleration=acceleration
                ))
                
                # 更新跟踪历史
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                
                self.track_history[track_id].append(center)
                
                # 限制轨迹长度
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)
                    
        return detection_results

class BehaviorAnalyzer:
    """行为分析器"""
    
    def __init__(self):
        """初始化行为分析器"""
        self.crosswalk_zones = []  # 人行横道区域列表
        self.intersection_center = None  # 十字路口中心点
        self.conflict_areas = []  # 冲突区域列表
        self.speed_threshold = 20  # 速度阈值(像素/帧)
        self.proximity_threshold = 100  # 接近阈值(像素)
        self.ttc_threshold = 2.0  # TTC阈值(秒)，低于此值视为危险
        self.pet_threshold = 2.0  # PET阈值(秒)，低于此值视为危险
        self.fps = 30  # 视频帧率，用于时间计算
        
    def set_intersection_zones(self, crosswalk_zones: List[Tuple] = None, intersection_center: Tuple = None, conflict_areas: List[Tuple] = None):
        """设置十字路口区域和冲突区域，所有参数都是可选的"""
        self.crosswalk_zones = crosswalk_zones if crosswalk_zones else []
        self.intersection_center = intersection_center
        self.conflict_areas = conflict_areas if conflict_areas else []
        
    def analyze_behavior(self, detections: List[DetectionResult], 
                        seg_results: List[SegmentationResult],
                        track_history: Dict) -> List[BehaviorAlert]:
        """分析行为并生成预警"""
        alerts = []
        current_time = time.time()
        
        # 计算所有检测对象之间的安全指标
        safety_metrics = self._calculate_safety_metrics(detections)
        
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
                
            # 3. 碰撞风险分析 (增强版)
            collision_alerts = self._analyze_collision_risk(detection, detections, center, safety_metrics)
            alerts.extend(collision_alerts)
            
            # 4. 行为异常分析
            behavior_alert = self._analyze_abnormal_behavior(detection, history)
            if behavior_alert:
                alerts.append(behavior_alert)
            
        return alerts
    
    def _calculate_safety_metrics(self, detections: List[DetectionResult]) -> Dict:
        """计算所有检测对象之间的安全评估指标"""
        safety_metrics = {}
        
        for i, det1 in enumerate(detections):
            if det1.track_id is None or det1.velocity is None:
                continue
                
            for j, det2 in enumerate(detections[i+1:], i+1):
                if det2.track_id is None or det2.velocity is None:
                    continue
                    
                # 只关注行人-车辆交互
                if not ((det1.class_name == "person" and det2.class_name in ["car", "truck", "bus"]) or
                        (det2.class_name == "person" and det1.class_name in ["car", "truck", "bus"])):
                    continue
                
                # 计算TTC (Time to Collision)
                ttc = self._calculate_ttc(det1, det2)
                
                # 计算PET (Post Encroachment Time)
                pet = self._calculate_pet(det1, det2)
                
                # 计算减速率 (Deceleration Rate)
                dr = self._calculate_deceleration_rate(det1, det2)
                
                # 计算运动适应性 (Motion Adaptation)
                ma = self._calculate_motion_adaptation(det1, det2)
                
                # 计算综合严重程度评分
                severity = self._calculate_severity_score(ttc, pet, dr, ma)
                
                # 存储计算结果
                pair_id = (det1.track_id, det2.track_id)
                safety_metrics[pair_id] = SafetyMetrics(
                    ttc=ttc,
                    pet=pet,
                    dr=dr,
                    ma=ma,
                    severity_score=severity
                )
        
        return safety_metrics
    
    def _calculate_ttc(self, det1: DetectionResult, det2: DetectionResult) -> Optional[float]:
        """计算TTC (Time to Collision)
        
        TTC是在当前速度和方向保持不变的情况下，两个道路使用者碰撞所需的时间。
        TTC = d / |v_rel|，其中d是两个物体之间的距离，v_rel是相对速度。
        """
        # 获取中心点和速度
        x1, y1, x2, y2 = det1.bbox
        center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        v1 = det1.velocity
        
        x1, y1, x2, y2 = det2.bbox
        center2 = ((x1 + x2) / 2, (y1 + y2) / 2)
        v2 = det2.velocity
        
        # 计算相对位置和相对速度
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        dvx = v2[0] - v1[0]
        dvy = v2[1] - v1[1]
        
        # 检查是否在碰撞路径上
        # 如果相对速度为零或两者正在远离，则不会发生碰撞
        if dvx == 0 and dvy == 0:
            return None
            
        # 计算相对速度和距离的点积，判断是否在接近
        dot_product = dx * dvx + dy * dvy
        if dot_product >= 0:  # 如果点积为正，表示两者正在远离
            return None
        
        # 计算距离和相对速度大小
        distance = np.sqrt(dx**2 + dy**2)
        rel_speed = np.sqrt(dvx**2 + dvy**2)
        
        if rel_speed < 0.1:  # 避免除以接近零的值
            return None
            
        # 计算TTC (单位：帧)
        ttc_frames = distance / rel_speed
        
        # 转换为秒
        ttc_seconds = ttc_frames / self.fps
        
        return ttc_seconds
    
    def _calculate_pet(self, det1: DetectionResult, det2: DetectionResult) -> Optional[float]:
        """计算PET (Post Encroachment Time)
        
        PET是第一个道路使用者离开共同冲突区域和第二个道路使用者进入该区域之间的时间间隔。
        这里我们使用简化的方法，基于当前位置和速度估计PET。
        """
        # 如果没有定义冲突区域，则使用简化计算
        if not self.conflict_areas:
            # 获取中心点和速度
            x1, y1, x2, y2 = det1.bbox
            center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
            v1 = det1.velocity
            
            x1, y1, x2, y2 = det2.bbox
            center2 = ((x1 + x2) / 2, (y1 + y2) / 2)
            v2 = det2.velocity
            
            # 计算两个物体的大致尺寸
            size1 = max(x2 - x1, y2 - y1) / 2
            size2 = max(x2 - x1, y2 - y1) / 2
            
            # 计算两个物体之间的距离
            dx = center2[0] - center1[0]
            dy = center2[1] - center1[1]
            distance = np.sqrt(dx**2 + dy**2) - (size1 + size2)  # 减去物体尺寸
            
            # 计算相对速度
            dvx = v2[0] - v1[0]
            dvy = v2[1] - v1[1]
            rel_speed = np.sqrt(dvx**2 + dvy**2)
            
            if rel_speed < 0.1:  # 避免除以接近零的值
                return None
                
            # 估计PET (单位：帧)
            pet_frames = distance / rel_speed
            
            # 转换为秒
            pet_seconds = pet_frames / self.fps
            
            return max(0, pet_seconds)  # PET不应为负值
        
        # 如果定义了冲突区域，则进行更精确的计算
        # 此处需要更复杂的轨迹预测和冲突区域判断
        # 简化起见，此处省略
        return None
    
    def _calculate_deceleration_rate(self, det1: DetectionResult, det2: DetectionResult) -> Optional[float]:
        """计算减速率 (Deceleration Rate)
        
        减速率表示为避免碰撞所需的减速度。
        """
        # 简化计算，使用加速度的负值作为减速率估计
        if det1.acceleration and det2.acceleration:
            acc1_mag = np.sqrt(det1.acceleration[0]**2 + det1.acceleration[1]**2)
            acc2_mag = np.sqrt(det2.acceleration[0]**2 + det2.acceleration[1]**2)
            
            # 取较大的减速率
            return max(acc1_mag, acc2_mag)
        
        return None
    
    def _calculate_motion_adaptation(self, det1: DetectionResult, det2: DetectionResult) -> Optional[float]:
        """计算运动适应性 (Motion Adaptation)
        
        运动适应性衡量道路使用者为避免碰撞而改变运动状态的程度。
        这里我们使用速度变化的角度来估计。
        """
        # 如果没有足够的历史数据，无法计算
        if not (det1.velocity and det2.velocity):
            return None
            
        # 计算速度向量的角度变化
        v1_mag = np.sqrt(det1.velocity[0]**2 + det1.velocity[1]**2)
        v2_mag = np.sqrt(det2.velocity[0]**2 + det2.velocity[1]**2)
        
        if v1_mag < 0.1 or v2_mag < 0.1:  # 避免除以接近零的值
            return None
            
        # 简化的运动适应性度量，基于速度大小的变化
        # 实际应用中可以使用更复杂的方法
        ma = abs(v1_mag - v2_mag) / max(v1_mag, v2_mag)
        
        return ma
    
    def _calculate_severity_score(self, ttc: Optional[float], pet: Optional[float], 
                                 dr: Optional[float], ma: Optional[float]) -> float:
        """计算综合严重程度评分
        
        结合多个安全指标计算综合严重程度评分。
        分数越高表示风险越大。
        """
        score = 0.0
        count = 0
        
        # TTC评分（TTC越小风险越大）
        if ttc is not None:
            ttc_score = max(0, 1 - ttc / self.ttc_threshold) if ttc < self.ttc_threshold else 0
            score += ttc_score * 0.4  # TTC权重为0.4
            count += 1
        
        # PET评分（PET越小风险越大）
        if pet is not None:
            pet_score = max(0, 1 - pet / self.pet_threshold) if pet < self.pet_threshold else 0
            score += pet_score * 0.3  # PET权重为0.3
            count += 1
        
        # 减速率评分（减速率越大风险越大）
        if dr is not None:
            dr_norm = min(dr / 10.0, 1.0)  # 归一化，假设最大减速率为10
            score += dr_norm * 0.15  # 减速率权重为0.15
            count += 1
        
        # 运动适应性评分（适应性越大风险越大）
        if ma is not None:
            score += ma * 0.15  # 运动适应性权重为0.15
            count += 1
        
        # 如果没有有效指标，返回0
        if count == 0:
            return 0.0
            
        # 归一化分数
        return score / count
    
    def _analyze_collision_risk(self, detection: DetectionResult, 
                               all_detections: List[DetectionResult], 
                               center: Tuple,
                               safety_metrics: Dict) -> List[BehaviorAlert]:
        """分析碰撞风险（增强版）"""
        alerts = []
        
        for other_detection in all_detections:
            if (other_detection.track_id == detection.track_id or 
                other_detection.track_id is None):
                continue
                
            # 获取安全指标
            pair_id = (detection.track_id, other_detection.track_id)
            reverse_pair_id = (other_detection.track_id, detection.track_id)
            
            metrics = safety_metrics.get(pair_id) or safety_metrics.get(reverse_pair_id)
            
            if not metrics:
                continue
                
            other_x1, other_y1, other_x2, other_y2 = other_detection.bbox
            other_center = ((other_x1 + other_x2) // 2, (other_y1 + other_y2) // 2)
            
            # 计算距离
            distance = np.sqrt((center[0] - other_center[0])**2 + 
                             (center[1] - other_center[1])**2)
            
            # 基于安全指标的风险评估
            alert_level = AlertLevel.SAFE
            alert_type = ""
            description = ""
            
            # 根据TTC评估风险
            if metrics.ttc is not None and metrics.ttc < self.ttc_threshold:
                if metrics.ttc < self.ttc_threshold / 2:
                    alert_level = AlertLevel.DANGER
                    alert_type = "CriticalTTC"
                    description = f"Dangerous collision time(TTC): {metrics.ttc:.2f}s"
                else:
                    alert_level = AlertLevel.WARNING
                    alert_type = "LowTTC"
                    description = f"Low collision time(TTC): {metrics.ttc:.2f}s"
            
            # 根据PET评估风险
            elif metrics.pet is not None and metrics.pet < self.pet_threshold:
                if metrics.pet < self.pet_threshold / 2:
                    alert_level = AlertLevel.DANGER
                    alert_type = "CriticalPET"
                    description = f"Dangerous post encroachment time(PET): {metrics.pet:.2f}s"
                else:
                    alert_level = AlertLevel.WARNING
                    alert_type = "LowPET"
                    description = f"Low post encroachment time(PET): {metrics.pet:.2f}s"
            
            # 根据综合评分评估风险
            elif metrics.severity_score > 0.6:
                alert_level = AlertLevel.DANGER
                alert_type = "HighRiskInteraction"
                description = f"High risk interaction, score: {metrics.severity_score:.2f}"
            elif metrics.severity_score > 0.3:
                alert_level = AlertLevel.WARNING
                alert_type = "MediumRiskInteraction"
                description = f"Medium risk interaction, score: {metrics.severity_score:.2f}"
            
            # 如果检测到风险，创建预警
            if alert_level != AlertLevel.SAFE:
                alerts.append(BehaviorAlert(
                    alert_type=alert_type,
                    level=alert_level,
                    description=description,
                    position=center,
                    timestamp=time.time(),
                    involved_objects=[detection.track_id, other_detection.track_id],
                    safety_metrics=metrics
                ))
                
        return alerts
    
    def _analyze_abnormal_behavior(self, detection: DetectionResult, 
                                  history: List) -> Optional[BehaviorAlert]:
        """分析异常行为
        
        检测突然加速、急刹车、急转弯等异常行为
        """
        if len(history) < 5 or not detection.acceleration:
            return None
            
        # 计算加速度大小
        acc_magnitude = np.sqrt(detection.acceleration[0]**2 + detection.acceleration[1]**2)
        
        # 检测急刹车/急加速
        if acc_magnitude > 10:  # 阈值可调整
            return BehaviorAlert(
                alert_type="SuddenAcceleration" if detection.acceleration[0] > 0 else "SuddenBraking",
                level=AlertLevel.WARNING,
                description=f"{detection.class_name}{' sudden acceleration' if detection.acceleration[0] > 0 else ' sudden braking'}",
                position=history[-1],
                timestamp=time.time(),
                involved_objects=[detection.track_id]
            )
        
        # 检测急转弯
        if len(history) >= 7:
            # 计算路径曲率
            p1, p2, p3 = history[-7], history[-4], history[-1]
            try:
                # 计算三点确定的圆的曲率
                a = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                b = np.sqrt((p3[0]-p2[0])**2 + (p3[1]-p2[1])**2)
                c = np.sqrt((p1[0]-p3[0])**2 + (p1[1]-p3[1])**2)
                s = (a + b + c) / 2
                area = np.sqrt(s * (s-a) * (s-b) * (s-c))
                curvature = 4 * area / (a * b * c)
                
                if curvature > 0.05:  # 阈值可调整
                    return BehaviorAlert(
                        alert_type="SharpTurn",
                        level=AlertLevel.WARNING,
                        description=f"{detection.class_name} sharp turn",
                        position=history[-1],
                        timestamp=time.time(),
                        involved_objects=[detection.track_id]
                    )
            except:
                pass  # 忽略计算错误
        
        return None
        
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

class AlertSystem:
    """预警系统"""
    
    def __init__(self):
        self.alert_history = []
        self.alert_cooldown = {}  # 防止重复预警
        self.cooldown_time = 3.0  # 3秒冷却时间
        self.persistent_alerts = []  # 持续显示的预警
        self.alert_display_duration = 6  # 预警显示持续帧数
        self.risk_statistics = {  # 添加风险统计
            "ttc_violations": 0,
            "pet_violations": 0,
            "zone_violations": 0,
            "speed_violations": 0,
            "abnormal_behaviors": 0,
            "total_interactions": 0,
            "high_risk_interactions": 0,
            "medium_risk_interactions": 0,
            "risk_scores": []
        }

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
                
                # 更新风险统计
                self._update_risk_statistics(alert)
        
        # 更新持续显示的预警列表
        self._update_persistent_alerts()
        
        # 限制历史记录长度
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
            
        return self.get_current_display_alerts()
    
    def _update_risk_statistics(self, alert: BehaviorAlert):
        """更新风险统计信息"""
        self.risk_statistics["total_interactions"] += 1
        
        # 根据预警类型更新统计
        if "TTC" in alert.alert_type:
            self.risk_statistics["ttc_violations"] += 1
        elif "PET" in alert.alert_type:
            self.risk_statistics["pet_violations"] += 1
        elif "Violation" in alert.alert_type:
            self.risk_statistics["zone_violations"] += 1
        elif "speed" in alert.alert_type.lower() or "Overspeeding" in alert.alert_type:
            self.risk_statistics["speed_violations"] += 1
        elif "Sudden" in alert.alert_type or "Sharp" in alert.alert_type:
            self.risk_statistics["abnormal_behaviors"] += 1
            
        # 根据风险级别更新统计
        if alert.level == AlertLevel.DANGER:
            self.risk_statistics["high_risk_interactions"] += 1
        elif alert.level == AlertLevel.WARNING:
            self.risk_statistics["medium_risk_interactions"] += 1
            
        # 记录风险评分
        if alert.safety_metrics and alert.safety_metrics.severity_score is not None:
            self.risk_statistics["risk_scores"].append(alert.safety_metrics.severity_score)
    
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
            
        # 计算平均风险评分
        avg_risk_score = 0
        if self.risk_statistics["risk_scores"]:
            avg_risk_score = sum(self.risk_statistics["risk_scores"]) / len(self.risk_statistics["risk_scores"])
            
        return {
            "total_alerts": len(self.alert_history),
            "alert_types": alert_counts,
            "alert_levels": level_counts,
            "current_displaying": len(self.persistent_alerts),
            "risk_statistics": {
                **self.risk_statistics,
                "avg_risk_score": avg_risk_score,
                "risk_scores": []  # 不返回所有评分，避免数据过大
            }
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
            self.config.get('segmentation_data_config')
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
        self.target_fps =5  # 降低到15帧每秒，您可以根据需要调整
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
            
            # 绘制速度向量（如果有）
            if detection.velocity:
                vx, vy = detection.velocity
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                end_x = int(center_x + vx * 3)  # 放大向量以便可视化
                end_y = int(center_y + vy * 3)
                cv2.arrowedLine(result_frame, (center_x, center_y), (end_x, end_y), color, 2)
            
            # 绘制轨迹
            if detection.track_id in self.object_tracker.track_history:
                history = self.object_tracker.track_history[detection.track_id]
                for i in range(1, len(history)):
                    if i % 2 == 0:
                        cv2.line(result_frame, history[i-1], history[i], color, 1)
        
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
            
            # 如果有安全指标，显示详细信息
            if alert.safety_metrics:
                metrics = alert.safety_metrics
                metrics_text = []
                
                if metrics.ttc is not None:
                    metrics_text.append(f"TTC: {metrics.ttc:.2f}s")
                if metrics.pet is not None:
                    metrics_text.append(f"PET: {metrics.pet:.2f}s")
                if metrics.severity_score is not None:
                    metrics_text.append(f"Risk: {metrics.severity_score:.2f}")
                    
                if metrics_text:
                    metrics_str = ", ".join(metrics_text)
                    cv2.putText(result_frame, metrics_str, (10, alert_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 1)
                    alert_y += 20
        
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

        # 显示统计信息
        stats = self.alert_system.get_statistics()
        if stats:
            stats_text = f"Total alerts: {stats.get('total_alerts', 0)}"
            cv2.putText(result_frame, stats_text, (10, self.height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 显示风险统计
            risk_stats = stats.get('risk_statistics', {})
            if risk_stats:
                risk_text = f"High risk: {risk_stats.get('high_risk_interactions', 0)}, Medium risk: {risk_stats.get('medium_risk_interactions', 0)}"
                cv2.putText(result_frame, risk_text, (10, self.height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示帧率
        cv2.putText(result_frame, f"FPS: {int(self.fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return result_frame
    
    def run(self):
        """运行主循环"""
        print("Starting intelligent traffic analysis...")
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
            cv2.imshow("Intelligent Traffic Analysis System", result_frame)
            
            # 按键控制
            key = cv2.waitKey(50 if not paused else 0) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):  # 空格键暂停/恢复
                paused = not paused
                if paused:
                    print("Video paused, press space to continue...")
                else:
                    print("Video resumed...")
            elif key == ord('s'):  # 保存当前帧
                cv2.imwrite(f"frame_{self.frame_count}.jpg", result_frame)
                print(f"Frame {self.frame_count} saved")
            
            # 每100帧输出一次统计信息
            if self.frame_count % 100 == 0:
                stats = self.alert_system.get_statistics()
                print(f"Processed frames: {self.frame_count}, Alert statistics: {stats}")
        
        self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        
        # 保存最终统计报告
        final_stats = self.alert_system.get_statistics()
        with open('yolo_warning/traffic_analysis_report.json', 'w') as f:
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
        
        print("Analysis completed, report saved to traffic_analysis_report.json")

def main():
    """主函数"""
    # 配置参数
    config = {
        'detection_model_path': 'train_model/16b_1280_origin_yolov8/best.pt',
        'segmentation_model_path': 'RoadANDSidewalk_segmentation/RoadANDSidewalk_segmentation3/weights/best.pt',
        'segmentation_data_config': 'data_road.yaml',  # 添加分割模型配置
        'deepsort_config': 'deep_sort_pytorch/configs/deep_sort.yaml',
        'input_video': 'cross.mp4',
        'output_video': 'yolo_warning/intelligent_traffic_analysis_results.mp4'
    }
    
    # 创建并运行分析器
    analyzer = IntelligentTrafficAnalyzer(config)
    
    # 设置帧率和安全阈值
    analyzer.behavior_analyzer.fps = analyzer.original_fps
    analyzer.behavior_analyzer.ttc_threshold = 2.0  # 2秒TTC阈值
    analyzer.behavior_analyzer.pet_threshold = 2.0  # 2秒PET阈值
    
    # 如果需要设置十字路口区域，可以取消下面的注释并提供适当的值
    # crosswalk_zones = [
    #     [(100, 400), (300, 600)],  # 示例坐标，需要根据实际视频调整
    #     [(500, 300), (700, 500)]
    # ]
    # 
    # intersection_center = (400, 400)  # 示例坐标，需要根据实际视频调整
    # 
    # conflict_areas = [
    #     [(350, 350), (450, 450)]  # 示例坐标，需要根据实际视频调整
    # ]
    # 
    # analyzer.behavior_analyzer.set_intersection_zones(
    #     crosswalk_zones, intersection_center, conflict_areas
    # )
    
    analyzer.run()

if __name__ == "__main__":
    main()