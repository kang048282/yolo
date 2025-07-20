import multiprocessing
import os

import matplotlib.pyplot as plt

from ultralytics import YOLO


def evaluate_model():
    # 加载训练好的模型
    model = YOLO("train_model/64b_640_C3TR_yolov11n/best.pt")

    # 创建结果保存目录
    results_dir = "evaluation_results_64b_640_C3TR_yolov11n"
    os.makedirs(results_dir, exist_ok=True)

    # 使用自定义数据集进行评估
    results = model.val(
        data="cross4.yaml",
        project="validation_results",
        name="64b_640_C3TR_yolov11n",
        save_txt=True,
        save_json=True,
        plots=True,
    )

    # 提取性能指标
    metrics = results.box

    # 获取每个类别的指标
    class_names = model.names
    {i: name for i, name in class_names.items()}

    # 提取各类别的mAP, precision, recall, f1-score
    class_metrics = {}
    for i, class_name in class_names.items():
        # 检查索引是否有效
        valid_ap50_index = hasattr(metrics, "ap50") and i < len(metrics.ap50) if len(metrics.ap50) > 0 else False
        valid_ap_index = hasattr(metrics, "ap") and i < len(metrics.ap) if len(metrics.ap) > 0 else False

        class_metrics[class_name] = {
            "mAP50": metrics.ap50[i] if valid_ap50_index else 0,  # AP at IoU=0.5 for this class
            "mAP50-95": metrics.ap[i] if valid_ap_index else 0,  # AP50-95 for this class
            "precision": metrics.p[i] if i < len(metrics.p) else 0,  # precision for this class
            "recall": metrics.r[i] if i < len(metrics.r) else 0,  # recall for this class
            "f1": metrics.f1[i] if i < len(metrics.f1) else 0,  # F1-score for this class
        }

    # 打印总体性能指标
    print(f"Overall mAP@0.5: {metrics.map50:.4f}")
    print(f"Overall mAP@0.5-0.95: {metrics.map:.4f}")

    # 创建性能指标可视化图表
    classes = list(class_metrics.keys())
    colors = ["blue", "green"]

    # 1. 每个类别的mAP50柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, [class_metrics[cls]["mAP50"] for cls in classes], color=colors)
    plt.title("mAP@0.5 by Class")
    plt.ylabel("mAP@0.5")
    plt.ylim(0, 1.0)

    # 添加图例
    plt.legend(bars, classes, title="Classes")

    plt.savefig(os.path.join(results_dir, "map50_by_class.png"))

    # 2. 每个类别的mAP50-95柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, [class_metrics[cls]["mAP50-95"] for cls in classes], color=["purple", "orange"])
    plt.title("mAP@0.5-0.95 by Class")
    plt.ylabel("mAP@0.5-0.95")
    plt.ylim(0, 1.0)

    # 添加图例
    plt.legend(bars, classes, title="Classes")

    plt.savefig(os.path.join(results_dir, "map50_95_by_class.png"))

    # 3. 每个类别的Precision柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, [class_metrics[cls]["precision"] for cls in classes], color=["red", "cyan"])
    plt.title("Precision by Class")
    plt.ylabel("Precision")
    plt.ylim(0, 1.0)

    # 添加图例
    plt.legend(bars, classes, title="Classes")

    plt.savefig(os.path.join(results_dir, "precision_by_class.png"))

    # 4. 每个类别的Recall柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, [class_metrics[cls]["recall"] for cls in classes], color=["green", "lime"])
    plt.title("Recall by Class")
    plt.ylabel("Recall")
    plt.ylim(0, 1.0)

    # 添加图例
    plt.legend(bars, classes, title="Classes")

    plt.savefig(os.path.join(results_dir, "recall_by_class.png"))

    # 5. 每个类别的F1-Score柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, [class_metrics[cls]["f1"] for cls in classes], color=["blue", "skyblue"])
    plt.title("F1-Score by Class")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1.0)

    # 添加图例
    plt.legend(bars, classes, title="Classes")

    plt.savefig(os.path.join(results_dir, "f1_score_by_class.png"))

    # 6. 混淆矩阵已经由model.val()生成在runs/val/目录下

    print("\nDetailed metrics by class:")
    for cls, metrics in class_metrics.items():
        print(f"\n{cls}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    print(f"\nEvaluation complete. Results saved to {results_dir}")
    # plt.show()  # 如果在非GUI环境中运行，可以注释掉这一行


if __name__ == "__main__":
    # 在Windows上添加多进程支持
    multiprocessing.freeze_support()
    evaluate_model()
