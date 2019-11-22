from PIL import Image
from pylab import *


def readpicture(path):
    # 读入图片
    im = array(Image.open(path).convert('L'))
    # 如果图片为空，返回错误信息，并终止程序
    if im is None:
        print("图片打开失败！")
        exit()
    return im


# 绘制原始直方图,先分格再画
def drawHistogram(image):
    # 把画布分成1*3的格子，把Y1放在第一格
    subplot(131)
    # flatten() 方法将任意数组按照行优先准则转换成一维数组
    hist(image.flatten(), 256)

    # 计算直方图
    # imhist表示某个区间中的数值，bins对应某个区间
    imhist, bins = histogram(image.flatten(), 256, range=(0,255))
    # 计算累积分布函数
    cdf = imhist.cumsum()
    # 累计函数归一化（变换至0~255）
    cdf = cdf*255/cdf[-1]

    # 绘制累积分布函数
    subplot(132)
    plot(cdf)

    # 线性插值函数
    im = interp(image.flatten(), bins[:256], cdf)
    #将压平的图像数组重新变成二维数组
    im = im.reshape(image.shape)
    subplot(133)
    hist(im.flatten(), 256)
    show()
    return image, im


def drawpicture(im1, im2):
    gray()
    subplot(121)
    imshow(im1)
    subplot(122)
    imshow(im2)
    show()


im1, im2 = drawHistogram(readpicture('role.jpg'))
drawpicture(im1, im2)
