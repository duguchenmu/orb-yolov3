import os
import cv2
import numpy as np
from skimage.transform import AffineTransform, warp
# image_path = 'datasets/rgbd_dataset_freiburg1_desk/rgb'
image_path = 'datasets/data_gui'
out_path = './output'
out_path = os.path.join(out_path, image_path.split('/')[1])
os.makedirs(out_path, exist_ok=True)

images = os.listdir(image_path)

for image_name in images:
    image = cv2.imread(os.path.join(image_path, image_name))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    print(f'Finished processing Image: {image_name}, Keypoints_count: {len(keypoints)}')
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, image, color=(0, 255, 0), flags=2)
    cv2.imwrite(os.path.join(out_path, image_name), image_with_keypoints)


#有两个图像 img1 和 img2，将 img2 配准到 img1

# 特征提取与匹配
def feature_matching(img1, img2):
# 初始化ORB检测器
    orb = cv2.ORB_create()
# 找到关键点和描述符
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
# 创建BF匹配器并进行匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
# 根据距离排序匹配项
    matches = sorted(matches, key=lambda x:x.distance)
    return kp1, kp2, matches
# 粗略配准
def find_initial_transform(kp1, kp2, matches):
# 提取匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
# 使用RANSAC找到最佳变换矩阵
    transform_matrix, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return transform_matrix
# 互信息优化
def optimize_transform(img1, img2, initial_transform):
# 描述如何使用MI对初始变换进行优化
#优化互信息度量
#引用optimize_mi
    optimized_transform = optimize_mi(img1, img2, initial_transform)
    return optimized_transform
def optimize_mi(img1, img2, initial_transform):
    return img1
def apply_transform(img2, transform_matrix):
# 将AffineTransform应用于img2
    transformed_img2 = warp(img2, AffineTransform(matrix=transform_matrix))
    return transformed_img2