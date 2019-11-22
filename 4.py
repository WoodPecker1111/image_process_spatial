from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

def readpicture(path):
    # 读入图片
    im = Image.open(path).convert('L')
    # 如果图片为空，返回错误信息，并终止程序
    if im is None:
        print("图片打开失败！")
        exit()
    return im

# 高斯噪声
def GaussianNoise(image, means, sigma, percetage):
    image = np.array(image)
    NoiseImg = image
    NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, image.shape[0] - 1)
        randY = random.randint(0, image.shape[1] - 1)
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg


# 椒盐噪声
def PepperandSalt(image, percetage):
    image = np.array(image)
    NoiseImg = image
    rows = NoiseImg.shape[0]
    cols = NoiseImg.shape[1]
    NoiseNum = int(percetage*rows*cols)
    for i in range(NoiseNum):
        randX = random.randint(0, rows - 1)
        randY = random.randint(0, cols - 1)

        if random.randint(0, 1) <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


def DrawPicture1(image, im1, im2):
    plt.subplot(131)
    plt.imshow(image, plt.cm.gray)
    plt.title('before')
    plt.subplot(132)
    plt.imshow(im1, plt.cm.gray)
    plt.title('GaussianNoise')
    plt.subplot(133)
    plt.imshow(im2, plt.cm.gray)
    plt.title('PepperandSalt')
    plt.show()


image = readpicture('im.jpg')
im1 = GaussianNoise(image, 0.01, 0.5, 1)
im2 = PepperandSalt(image, 0.1)
DrawPicture1(image, im1, im2)


#  均值去噪和中值去噪
def MeanFilter(Imge,dim):       #Image为待处理图像，dim为滤波器的大小dim*dim
    im=np.array(Imge)
    sigema=0
    for i in range(int(dim/2), im.shape[0] - int(dim/2)):
        for j in range(int(dim/2), im.shape[1] - int(dim/2)):
            for a in range(-int(dim/2), -int(dim/2)+dim):
                for b in range(-int(dim/2), -int(dim/2)+dim):
                    sigema = sigema + im[i + a, j + b]
            im[i, j] = sigema / (dim*dim)
            sigema = 0
    return im


def MedianFilter(Imge,dim):       #Image为待处理图像，dim为滤波器的大小dim*dim
    im=np.array(Imge)
    sigema=[]
    for i in range(int(dim/2), im.shape[0] - int(dim/2)):
        for j in range(int(dim/2), im.shape[1] - int(dim/2)):
            for a in range(-int(dim/2), -int(dim/2)+dim):
                for b in range(-int(dim/2), -int(dim/2)+dim):
                    sigema.append(im[i + a, j + b])
            sigema.sort()
            im[i, j] = sigema[int(dim*dim/2)]
            sigema = []
    return im


def DrawPicture2(image, im3, im5, im7, title1):
    plt.subplot(221)
    plt.imshow(image, plt.cm.gray)
    plt.title('before')
    plt.subplot(222)
    plt.imshow(im3, plt.cm.gray)
    plt.title(title1+'3')
    plt.subplot(223)
    plt.imshow(im5, plt.cm.gray)
    plt.title(title1+'5')
    plt.subplot(224)
    plt.imshow(im7, plt.cm.gray)
    plt.title(title1+'7')
    plt.show()


# 对椒盐噪声滤波
im = im2
im3 = MeanFilter(im, 3)
im5 = MeanFilter(im, 5)
im7 = MeanFilter(im, 7)
DrawPicture2(im, im3, im5, im7, 'MeanFilter')

im = im2
im3 = MedianFilter(im, 3)
im5 = MedianFilter(im, 5)
im7 = MedianFilter(im, 7)
DrawPicture2(im, im3, im5, im7, 'MedianFilter')


# 对高斯噪声滤波
im = im1
im3 = MeanFilter(im, 3)
im5 = MeanFilter(im, 5)
im7 = MeanFilter(im, 7)
DrawPicture2(im, im, im5, im7, 'MeanFilter')

im = im1
im3 = MedianFilter(im, 3)
im5 = MedianFilter(im, 5)
im7 = MedianFilter(im, 7)
DrawPicture2(im, im3, im5, im7, 'MedianFilter')