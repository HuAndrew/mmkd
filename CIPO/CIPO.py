import os
import shutil
from tqdm import tqdm
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import json
from shapely.geometry import LineString, Polygon
import math
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import Polygon, LineString



def calculate_overlap_ratio(bbox, triangle):
    """
    判断检测框和三角形是否重叠，计算底边在三角形区域中的长度比例。

    参数:
        bbox: tuple - (x1, y1, x2, y2)，检测框的左上角和右下角坐标。
        triangle: list - [(x1, y1), (x2, y2), (x3, y3)]，三角形的三个顶点。

    返回:
        overlap_ratio: float - 检测框底边在三角形区域中的长度比例。如果没有重叠，返回 0。
    """
    # 检测框的四个顶点
    x1, y1, x2, y2 = bbox
    bbox_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    bbox_bottom_edge = LineString([(x1, y2), (x2, y2)])  # 检测框的底边

    # 三角形区域
    triangle_polygon = Polygon(triangle)

    # 判断是否重叠
    if not bbox_polygon.intersects(triangle_polygon):
        return 0.0  # 没有重叠

    # 计算底边与三角形的交集
    intersection = bbox_bottom_edge.intersection(triangle_polygon)

    # 如果交集是线段，计算长度
    if isinstance(intersection, LineString):
        overlap_length = intersection.length
    else:
        overlap_length = 0.0  # 没有交集

    # 计算比例
    bbox_bottom_length = bbox_bottom_edge.length
    overlap_ratio = overlap_length / bbox_bottom_length if bbox_bottom_length > 0 else 0.0

    return overlap_ratio


def curve_min_distance(a, b, c, d, x0, y0):
    def distance_squared(y):
        x = a * y**3 + b * y**2 + c * y + d
        return (x - x0)**2 + (y - y0)**2
    initial_guess = 0.0
    result = minimize(distance_squared, initial_guess)
    y_min = result.x[0]
    x_min = a * y_min**3 + b * y_min**2 + c * y_min + d
    min_distance = np.sqrt((x_min - x0)**2 + (y_min - y0)**2)
    return min_distance


########判断点是否在曲线左侧
def is_point_left_of_cubic_curve(point, a, b, c, d):
    value = a * point[1]**3 + b * point[1]**2 + c * point[1] + d
    if value > point[0]:
        return True
    else:
        return False


########计算iou，分母为两个框中较小框的面积
def compute_iou(rec1, rec2):
    
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    sum_area = S_rec1 + S_rec2
 
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        #######分母为车尾
        return intersect / S_rec1

#######车身和车头车尾进行匹配
def car_reg_match(detInfos):
    matched_detInfos = list()
    for i in range(len(detInfos)):
        if detInfos[i][-1] in ["car_reg", "car_big_reg", "car_front", "car_big_front"]:
            matched_detInfos.append(detInfos[i])
    unmatched_list = list()
    for i in range(len(detInfos)):
        if detInfos[i] in matched_detInfos:
            continue
        if detInfos[i][-1] not in ["car", "bus", "truck"]:
            unmatched_list.append(detInfos[i])
            continue
        flag = 0
        for j in range(len(matched_detInfos)):
            if len(matched_detInfos[j]) > 6:
                continue
#             if compute_iou(matched_detInfos[j], detInfos[i]) > 0.9 and 0.9 < (matched_detInfos[j][3]-matched_detInfos[j][1])/(detInfos[i][3]-detInfos[i][1]) < 1.1:
            if compute_iou(matched_detInfos[j], detInfos[i]) > 0.9:
                matched_detInfos[j].extend(detInfos[i])
                flag = 1
                break
        if flag == 0:
            unmatched_list.append(detInfos[i])
    
    matched_detInfos.extend(unmatched_list)
    
    return matched_detInfos

class XmlWriter(object):
    def __init__(self, filename, image_shape=None):
        self.doc = Document()
        self.annotation = self.doc.createElement('annotation')
        self.doc.appendChild(self.annotation)
        if image_shape is None:
            image_shape = [1080, 1920, 3]
        self.image_shape = image_shape
        self.filename = filename
        self._init_xml(filename)

    def _init_xml(self, image_name):

        folder = self.doc.createElement('folder')
        folder.appendChild(self.doc.createTextNode("VOC2017"))
        self.annotation.appendChild(folder)

        filename = self.doc.createElement('filename')
        filename.appendChild(self.doc.createTextNode(image_name))
        self.annotation.appendChild(filename)

        source = self.doc.createElement('source')
        database = self.doc.createElement('database')
        database.appendChild(self.doc.createTextNode('Unknown'))
        source.appendChild(database)
        self.annotation.appendChild(source)

        size = self.doc.createElement('size')
        width = self.doc.createElement('width')
        width.appendChild(self.doc.createTextNode(str(self.image_shape[1])))
        size.appendChild(width)
        height = self.doc.createElement('height')
        height.appendChild(self.doc.createTextNode(str(self.image_shape[0])))
        size.appendChild(height)
        depth = self.doc.createElement('depth')
        depth.appendChild(self.doc.createTextNode(str(self.image_shape[2])))
        size.appendChild(depth)
        self.annotation.appendChild(size)

        segmented = self.doc.createElement('segmented')
        segmented.appendChild(self.doc.createTextNode(str(0)))
        self.annotation.appendChild(segmented)

    def insert_object(self, datas):
        obj = self.doc.createElement('object')
        name = self.doc.createElement('name')
        name.appendChild(self.doc.createTextNode(datas[0]))
        obj.appendChild(name)
        fixratio = self.doc.createElement('fixratio')
        fixratio.appendChild(self.doc.createTextNode('0'))
        obj.appendChild(fixratio)
        pose = self.doc.createElement('pose')
        pose.appendChild(self.doc.createTextNode('Unspecified'))
        obj.appendChild(pose)
        truncated = self.doc.createElement('truncated')
        truncated.appendChild(self.doc.createTextNode(str(0)))
        obj.appendChild(truncated)
        difficult = self.doc.createElement('difficult')
        difficult.appendChild(self.doc.createTextNode(str(0)))
        obj.appendChild(difficult)
        bndbox = self.doc.createElement('bndbox')

        xmin = self.doc.createElement('xmin')
        xmin.appendChild(self.doc.createTextNode(str(datas[1])))
        bndbox.appendChild(xmin)
        ymin = self.doc.createElement('ymin')
        ymin.appendChild(self.doc.createTextNode(str(datas[2])))
        bndbox.appendChild(ymin)
        xmax = self.doc.createElement('xmax')
        xmax.appendChild(self.doc.createTextNode(str(datas[3])))
        bndbox.appendChild(xmax)
        ymax = self.doc.createElement('ymax')
        ymax.appendChild(self.doc.createTextNode(str(datas[4])))
        bndbox.appendChild(ymax)
        
        if len(datas) > 5:
            score = self.doc.createElement('score')
            score.appendChild(self.doc.createTextNode(str(datas[5])))
            bndbox.appendChild(score)

        
        obj.appendChild(bndbox)
        self.annotation.appendChild(obj)

    def write_xml(self, save_path=None):
        if save_path is None:
            save_path = os.path.splitext(self.filename)[0] + ".xml"
        with open(save_path, "wb") as f:
            f.write(self.doc.toprettyxml(indent='    ', encoding='utf-8'))

            
def write_xml(output_xml_path, image_path, boxes, image_shape=None):
    if "/" in image_path:
        image_path = os.path.split(image_path)[-1]
    xml_writer = XmlWriter(image_path, image_shape)
    for key, values in boxes.items():
        for box in values:
            xml_writer.insert_object([key] + box)
    xml_writer.write_xml(output_xml_path)
    
def standard_label(label):
    label = label.replace(" ", "_")
    if label == "person_o":
        label = "person"
    
    return label

det_info_root  = "xxx"
lane_info_root = "xxx"
cipv_info_root = "xxx"
vcppi_classes = ['car', 'bus','truck', 'car_reg', 'car_big_reg', 'car_front', 'car_big_front', 'person', 'bicyclist', 'motorcyclist']
outside_lane_classes = ['person', 'motorcyclist']
image_root = "xxx"
save_root_tmp  = "xxx"
det_info_names = os.listdir(det_info_root)

if not os.path.exists(cipv_info_root):
    os.makedirs(cipv_info_root)

def process_file(det_info_name):
    if det_info_name[0] == ".":
        return
#     if "HT_TRAIN_001011_SH_000.xml" != det_info_name:
#         return
    det_info_path  = os.path.join(det_info_root, det_info_name)
    lane_info_path = os.path.join(lane_info_root, det_info_name[:-4] + ".txt")
    if not os.path.exists(lane_info_path):
        return

    
    boxes_list = list()
    tree = ET.parse(det_info_path)
    root = tree.getroot()
    filename = root.find('filename').text
    for obj in root.iter('object'): 
        category = obj.find('name').text
        category = standard_label(category)
        if category not in vcppi_classes:
            continue
        xml_box = obj.find('bndbox')
        x_min = int(float(xml_box.find('xmin').text))
        x_max = int(float(xml_box.find('xmax').text))
        y_min = int(float(xml_box.find('ymin').text))
        y_max = int(float(xml_box.find('ymax').text))
#         score = float(xml_box.find('score').text)
        score = 1.0
        #########过滤掉过远的目标
#         if y_max <= vp + 10:
#             continue
        bbox = [x_min, y_min, x_max, y_max, score, category]         
        boxes_list.append(bbox)
    boxes_list = car_reg_match(boxes_list)

    #########解析车道线信息
    w, h = 1920, 1080
    with open(lane_info_path, "r") as f:
        lane_infos_raw = f.read().split(" ")
    lane_infos_raw = [elem.replace("\n", "") for elem in lane_infos_raw]
#     print(lane_infos_raw)
    vp = int(float(lane_infos_raw[-1]) * h)

    left_x, right_x = w / 3, 2 * w / 3
    left_y, right_y = h, h
    cross_x, cross_y = w / 2, vp
    important_targets = []
    targets = []
    if 'L' not in lane_infos_raw and 'R' not in lane_infos_raw:
        for i in range(len(boxes_list)):
##            根据底边中点判断
#             retangle_roi = [left_x, left_y, right_x, right_y, cross_x, cross_y]
#             center_x = (boxes_list[i][0] + boxes_list[i][2]) / 2
#             center_y = boxes_list[i][3]
#             point_in_retangle = is_point_in_triangle((center_x, center_y), (retangle_roi[0], retangle_roi[1]), (retangle_roi[2], retangle_roi[3]), (retangle_roi[4], retangle_roi[5]))
#             if point_in_retangle:
#                 targets.append(boxes_list[i])
#           根据底边在三角区域内的长度占底边总长度比例判断
            if len(boxes_list[i]) > 6:
                rectangle = (boxes_list[i][6], boxes_list[i][7], boxes_list[i][8], boxes_list[i][9])
            else:
                rectangle = (boxes_list[i][0], boxes_list[i][1], boxes_list[i][2], boxes_list[i][3])
            triangle = ((left_x, left_y), (right_x, right_y), (cross_x, cross_y))
            overlap_ratio = calculate_overlap_ratio(rectangle, triangle)
            if overlap_ratio > 0.2:
                targets.append(boxes_list[i])
        targets = sorted(targets, key=lambda x: x[3], reverse=True)
        if len(targets) != 0:
            important_targets.append(targets[0])

    elif 'L' not in lane_infos_raw and 'R' in lane_infos_raw:
        r_index = lane_infos_raw.index('R')
        third_coffe_r  = float(lane_infos_raw[r_index+1])
        second_coffe_r = float(lane_infos_raw[r_index+2])
        a_r = float(lane_infos_raw[r_index+3])
        m_r = float(lane_infos_raw[r_index+4])
        lane_infos = [third_coffe_r, second_coffe_r , a_r, m_r]

        for i in range(len(boxes_list)):
            middle_bottom = [(boxes_list[i][0]+boxes_list[i][2]) / 2, boxes_list[i][3]]
            if is_point_left_of_cubic_curve(middle_bottom, lane_infos[0], lane_infos[1], lane_infos[2], lane_infos[3]):
                targets.append(boxes_list[i])
        targets = sorted(targets, key=lambda x: x[3], reverse=True)
        if len(targets) != 0:
            important_targets.append(targets[0])      
            
    elif 'L' in lane_infos_raw and 'R' not in lane_infos_raw:
        l_index = lane_infos_raw.index('L')
        third_coffe_l  = float(lane_infos_raw[l_index+1])
        second_coffe_l = float(lane_infos_raw[l_index+2])
        a_l = float(lane_infos_raw[l_index+3])
        m_l = float(lane_infos_raw[l_index+4])
        lane_infos = [third_coffe_l, second_coffe_l , a_l, m_l]
        
        for i in range(len(boxes_list)):
            middle_bottom = [(boxes_list[i][0]+boxes_list[i][2]) / 2, boxes_list[i][3]]
            if not is_point_left_of_cubic_curve(middle_bottom, lane_infos[0], lane_infos[1], lane_infos[2], lane_infos[3]):
                targets.append(boxes_list[i])
        targets = sorted(targets, key=lambda x: x[3], reverse=True)
        if len(targets) != 0:
            important_targets.append(targets[0])      
            
    elif 'L' in lane_infos_raw and 'R' in lane_infos_raw:
        lane_infos = []
        r_index = lane_infos_raw.index('R')
        third_coffe_r  = float(lane_infos_raw[r_index+1])
        second_coffe_r = float(lane_infos_raw[r_index+2])
        a_r = float(lane_infos_raw[r_index+3])
        m_r = float(lane_infos_raw[r_index+4])
        lane_infos.append([third_coffe_r, second_coffe_r , a_r, m_r])
        l_index = lane_infos_raw.index('L')
        third_coffe_l  = float(lane_infos_raw[l_index+1])
        second_coffe_l = float(lane_infos_raw[l_index+2])
        a_l = float(lane_infos_raw[l_index+3])
        m_l = float(lane_infos_raw[l_index+4])
        lane_infos.append([third_coffe_l, second_coffe_l , a_l, m_l])
        
        for i in range(len(boxes_list)):
            middle_bottom = [(boxes_list[i][0]+boxes_list[i][2]) / 2, boxes_list[i][3]]
            if is_point_left_of_cubic_curve(middle_bottom, lane_infos[0][0], lane_infos[0][1], lane_infos[0][2], lane_infos[0][3]) and not is_point_left_of_cubic_curve(middle_bottom, lane_infos[1][0], lane_infos[1][1], lane_infos[1][2], lane_infos[1][3]) :
                targets.append(boxes_list[i])
        targets = sorted(targets, key=lambda x: x[3], reverse=True)
        if len(targets) != 0:
            important_targets.append(targets[0])  
        
        
    else:
        print("abnormal lane info！")

    
    cipv_target = dict()
    if len(important_targets) != 0:
        if len(important_targets[0]) > 6:
            for i in range(1,3):
                if important_targets[0][6*i-1] not in cipv_target:
                    cipv_target[important_targets[0][6*i-1]] = list()
                cipv_target[important_targets[0][6*i-1]].append(important_targets[0][6*i-6:6*i-1])
        else:
            if important_targets[0][5] not in cipv_target:
                    cipv_target[important_targets[0][5]] = list()
            cipv_target[important_targets[0][5]].append(important_targets[0][0:5]) 

    write_xml(os.path.join(cipv_info_root, det_info_name), filename, cipv_target)
    
if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_file, det_info_name) for det_info_name in tqdm(det_info_names) if det_info_name[0] != "."]
        for future in futures:
            future.result()
#    for det_info_name in tqdm(det_info_names):
#         if det_info_name[0] != ".":
#              process_file(det_info_name) 