import cv2
import os
import numpy as np

def RandomCrop(dataset_path, save_path, size, num):
    ''' 将数据集中的每张图片随机裁剪为num个子图
    :param dataset_path: 数据集路径
    :param save_path: 保存路径
    :param size: 子图大小(a, b)分别对应子图的宽和高
    :param num: m每张图裁剪的子图的数量
    :return:
    '''
    filename = os.listdir(dataset_path)  # 文件名列表
    os.makedirs(save_path, exist_ok=True)
    for i in range(len(filename)):
        filepath = dataset_path + '\\' + filename[i]
        img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
        img = cv2.resize(img, (512, 512))
        x = np.random.randint(0, img.shape[0]-size[1], (num, 1))
        # 此处需要注意矩阵的行数与图片的宽和高的对应方式
        y = np.random.randint(0, img.shape[1]-size[0], (num, 1))
        start_point = np.concatenate([x, y], axis=1)
        if len(img.shape) == 2:         # 1 channels
            for j in range(num):
                sub_imgs = img[start_point[j, 0]:start_point[j, 0]+size[1],
                           start_point[j, 1]:start_point[j, 1] + size[0]]
                name = os.path.splitext(filename[i])[0] + "_" + str(j) + ".jpg"
                cv2.imencode('.jpg', sub_imgs)[1].tofile(save_path + '\\' + name)
        elif len(img.shape) == 3:         # 3 channels
            for j in range(num):
                sub_imgs = img[start_point[j, 0]:start_point[j, 0]+size[1],
                           start_point[j, 1]:start_point[j, 1] + size[0], :]
                name = os.path.splitext(filename[i])[0] + "_" + str(j) + ".jpg"
                cv2.imencode('.jpg', sub_imgs)[1].tofile(save_path + '\\' + name)
                # 将文件保存到中文路径

def NormalCrop(dataset_path, save_path, size, sub_size):
    ''' 将数据集中的每张图片随机裁剪为num个子图
    :param dataset_path: 数据集路径
    :param save_path: 保存路径
    :param size: 子图大小(a, b)分别对应子图的宽和高
    :param sub_size: 每张子图的宽和高
    :return:
    '''
    filename = os.listdir(dataset_path)  # 文件名列表
    os.makedirs(save_path, exist_ok=True)
    for i in range(len(filename)):
        filepath = dataset_path + '\\' + filename[i]
        img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
        img = cv2.resize(img, (size, size))
        if size % sub_size != 0:
            return
        num = int(size / sub_size)
        if len(img.shape) == 2:         # 1 channels
            for j in range(num):
                for k in range(num):
                    sub_imgs  = img[j*sub_size:(j+1)*sub_size,
                                k*sub_size:(k+1)*sub_size]
                    name = os.path.splitext(filename[i])[0] + "_" + str(j*num + k) + ".jpg"
                    cv2.imencode('.jpg', sub_imgs)[1].tofile(save_path + '\\' + name)
        elif len(img.shape) == 3:         # 3 channels
            for j in range(num):
                for k in range(num):
                    sub_imgs = img[j * sub_size:(j + 1) * sub_size,
                               k * sub_size:(k + 1) * sub_size, :]
                    name = os.path.splitext(filename[i])[0] + "_" + str(j*num + k) + ".jpg"
                    cv2.imencode('.jpg', sub_imgs)[1].tofile(save_path + '\\' + name)
                    # 将文件保存到中文路径

dataset_path = "../numclass9/Train/BMPImages/0"
save_path = "../numclass9/CutImages"

if __name__ == '__main__':
    #RandomCrop(dataset_path, save_path, (64, 64), 1500)
    NormalCrop(dataset_path, save_path, 640, 64)