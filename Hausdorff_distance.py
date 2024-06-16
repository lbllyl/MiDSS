import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import math

def load_image(image_path):
    # 读取图像并转换为灰度图
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def edge_detect(image):
    # 应用 Canny 边缘检测
    edges = cv2.Canny(image, 100, 200)
    return edges

def calculate_hausdorff_distance(points1, points2):
    # 计算两个点集之间的 Hausdorff 距离
    hd1 = directed_hausdorff(points1, points2)[0]
    hd2 = directed_hausdorff(points2, points1)[0]
    return max(hd1, hd2)

def convert_distance_to_similarity(distance, alpha=0.01):
    # 将距离转换为相似度
    return math.exp(-alpha * distance)

def calculate_dice_score(binary_image1, binary_image2):
    # 确保图像是以1和0的形式二值化
    binary_image1 = (binary_image1 == 255).astype(int)
    binary_image2 = (binary_image2 == 255).astype(int)

    # 计算交集和并集
    intersection = np.logical_and(binary_image1, binary_image2).sum()
    total = binary_image1.sum() + binary_image2.sum()

    # 避免除以0的情况
    if total == 0:
        return 1.0  # 如果两个图像都没有前景像素，可以认为它们完全相同
    
    # 计算 Dice 系数
    dice_score = 2. * intersection / total
    return dice_score

# 图像路径
image_path1 = 'C:\\Users\\99345\\Desktop\\MiDSS\\data\\Fundus\\Domain2\\train\\ROIs\\mask\\G-1-L.png'
image_path2 = 'C:\\Users\\99345\\Desktop\\MiDSS\\data\\Fundus\\Domain1\\train\\ROIs\\mask\\gdrishtiGS_002.png'

# 加载原图
image1 = load_image(image_path1)
image2 = load_image(image_path2)
breakpoint()
# 边缘检测
edges1 = edge_detect(image1)
edges2 = edge_detect(image2)

# 计算 Hausdorff 距离
points1 = np.column_stack(np.where(edges1 > 0))
points2 = np.column_stack(np.where(edges2 > 0))
distance = calculate_hausdorff_distance(points1, points2)
similarity = convert_distance_to_similarity(distance)

# 应用阈值
thresh1 = cv2.threshold(image1, 128, 255, cv2.THRESH_BINARY)[1]
thresh2 = cv2.threshold(image2, 128, 255, cv2.THRESH_BINARY)[1]

# 计算Dice系数
dice_score = calculate_dice_score(thresh1, thresh2)

print("Hausdorff Distance:", distance)
print("Similarity:", similarity)
print("Dice Score:", dice_score)

# 可选：显示原图和边缘图像
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.imshow(image1, cmap='gray')
plt.title('Original Image 1')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(image2, cmap='gray')
plt.title('Original Image 2')
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(edges1, cmap='gray')
plt.title('Edges of Image 1')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(edges2, cmap='gray')
plt.title('Edges of Image 2')
plt.axis('off')
plt.show()