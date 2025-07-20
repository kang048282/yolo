import json
import os
 
from sympy import print_glsl
 
# 类别列表和类别字典
all_classes = ['road', 'sidewalk']
class_dict = {'road': 0, 'sidewalk': 1}
 
# 根目录
rootdir = 'E:/yolov/gtFine/test'
 
# 输出目录
output_rootdir = 'E:/yolov/YOLOLabels_road_side/test'
 
 
def position(pos):
    x = [point[0] for point in pos]
    y = [point[1] for point in pos]
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)
    return float(x_min), float(x_max), float(y_min), float(y_max)
 
 
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh
 
 
def convert_annotation(json_id, city_name):
    json_file_path = os.path.join(rootdir, city_name, '%s.json' % json_id)
    out_file_path = os.path.join(output_rootdir, city_name, '%s.txt' % json_id)
 
    if not os.path.exists(os.path.dirname(out_file_path)):
        os.makedirs(os.path.dirname(out_file_path))
 
    with open(json_file_path, 'r') as load_f:
        load_dict = json.load(load_f)
 
    w = load_dict['imgWidth']
    h = load_dict['imgHeight']
    objects = load_dict['objects']
 
    with open(out_file_path, 'w') as out_file:
        for obj in objects:
            labels = obj['label']
            # 只保留 road 和 sidewalk
            if labels in class_dict:
                pos = obj['polygon']
                cls_id = class_dict[labels]
                # 归一化多边形点坐标为 x1 y1 x2 y2 ...
                coords = []
                for point in pos:
                    x_norm = point[0] / w
                    y_norm = point[1] / h
                    coords.extend([str(x_norm), str(y_norm)])
                out_file.write(str(cls_id) + " " + " ".join(coords) + '\n')
 
def jsons_id(rootdir):
    a = []
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.endswith('.json'):
                filename_without_ext = os.path.splitext(filename)[0]
                a.append(filename_without_ext)
    return a
 
 
# 获取所有子目录
subdirs = [d for d in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, d))]
# print(subdirs)
# ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf', 'erfurt', 'hamburg', 'hanover', 'jena',
# 'krefeld', 'monchengladbach', 'strasbourg', 'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']
 
# 为每个子目录生成YOLO格式的标注文件
for subdir in subdirs:
    names = jsons_id(os.path.join(rootdir, subdir))
    for json_id in names:
        convert_annotation(json_id, subdir)