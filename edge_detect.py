import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
image_path = 'C:\\Users\\99345\\Desktop\\MiDSS\\data\\Fundus\\Domain1\\test\\ROIs\\mask\\gdrishtiGS_001.png'  # 将这里的路径替换成你的图像文件的实际路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图

# 检查图像是否正确加载
if image is None:
    print("图像文件未找到，请检查路径。")
    exit()

# 应用Canny边缘检测
edges = cv2.Canny(image, threshold1=50, threshold2=150)

# 显示原始图像和边缘检测结果
plt.figure(figsize=(10, 5))
# plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.axis('off')
# plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.title('Edge Image'), plt.axis('off')
plt.imshow(edges, cmap='gray'), plt.title('Edge Image'), plt.axis('off')
# 保存到./edge文件夹下
plt.savefig('./edge/gdrishtiGS_001_edge.png')
plt.show()