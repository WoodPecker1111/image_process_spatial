
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def readpicture(path):
    # 读入图片
    im = Image.open(path).convert('L')
    # 如果图片为空，返回错误信息，并终止程序
    if im is None:
        print("图片打开失败！")
        exit()
    return im


def process(im):
    image = np.array(im)*1.0
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if 0 < image[i][j] <= 80:
                image[i][j] = image[i][j]/2
            if 80 < image[i][j] <= 150:
                image[i][j] = image[i][j]*1
            if 150 < image[i][j] <= 255:
                image[i][j] = image[i][j]*2
            if image[i][j] > 255:
                image[i][j] = 255
    return image


def drawcontrast(im1, im2):
    # 创建一个窗口
    plt.figure('对比图', figsize=(7, 5))
    # 显示原图
    plt.subplot(121)  # 子图1
    # 显示原图，设置标题和字体
    plt.imshow(im1, plt.cm.gray), plt.title('before')
    # 显示处理过的图像
    plt.subplot(122)  # 子图2
    # 显示处理后的图，设置标题和字体
    plt.imshow(im2, plt.cm.gray), plt.title('after')
    plt.show()


im1 = readpicture('role.jpg')
im2 = process(im1)
drawcontrast(im1, im2)