import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path):
    """ 加载图像并转换为numpy数组 """
    image = Image.open(image_path)
    return np.array(image)

def svd_and_swap_channels(image1, image2):
    """ 对两个图像的每个通道执行SVD，交换它们的V矩阵，并返回新图像 """
    # 确保图像尺寸和通道数一致
    assert image1.shape == image2.shape
    
    # 初始化新图像数组
    new_image1 = np.zeros_like(image1)
    new_image2 = np.zeros_like(image2)
    
    # 分别处理每个颜色通道
    for i in range(3):  # 对于RGB，有三个通道
        # 对第一个图像的当前通道进行SVD
        U1, S1, V1T = np.linalg.svd(image1[:, :, i], full_matrices=False)
        # 对第二个图像的当前通道进行SVD
        U2, S2, V2T = np.linalg.svd(image2[:, :, i], full_matrices=False)
        
        # 创建新图像，通过交换V矩阵
        new_image1[:, :, i] = np.dot(np.dot(U2, np.diag(S1)), V2T)
        new_image2[:, :, i] = np.dot(np.dot(U1, np.diag(S2)), V1T)
    
    return new_image1.astype(np.uint8), new_image2.astype(np.uint8)

def display_images(images, titles):
    """ 显示图像列表 """
    plt.figure(figsize=(10, 5))
    for i, (image, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, len(images), i)
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
    plt.show()

# 图像路径
image1_path = 'C:\\Users\\99345\\Desktop\\MiDSS\\data\\Fundus\\Domain1\\train\\ROIs\\image\\gdrishtiGS_002.png'
image2_path = 'C:\\Users\\99345\\Desktop\\MiDSS\\data\\Fundus\\Domain2\\train\\ROIs\\image\\G-1-L.png'

# 加载图像
image1 = load_image(image1_path)
image2 = load_image(image2_path)

# 执行SVD并交换V矩阵
new_image1, new_image2 = svd_and_swap_channels(image1, image2)

# 显示原始图像和修改后的图像
display_images([image1, image2, new_image1, new_image2], ['Image 1', 'Image 2', 'New Image 1', 'New Image 2'])