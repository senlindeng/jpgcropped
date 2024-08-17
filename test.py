import cv2
import pandas as pd
import numpy as np

# 读取图片
img = cv2.imread('test.jpg')



# 将图片从BGR颜色空间转换为HSV颜色空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义红色在HSV颜色空间中的范围
lower_red1 = np.array([0,50,50])
upper_red1 = np.array([10,255,255])
lower_red2 = np.array([160,50,50])
upper_red2 = np.array([180,255,255])

# 创建红色区域的二值图像（mask）
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# 在mask上找到轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 对每个找到的轮廓计算最小的矩形边界框
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    # 截取边界框内的区域
    cropped = img[y:y+h, x:x+w]
    # 保存截取的区域为jpg文件
    cv2.imwrite(f'test_cropped_{i}.jpg', cropped)

