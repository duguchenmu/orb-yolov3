import os
import cv2
import numpy as np

imagepath = 'datasets/data_gui'
images = os.listdir(imagepath)
avg_error = []
for i, img_name in enumerate(images[:-1]):
    image1 = cv2.imread(os.path.join(imagepath, img_name), 0)
    print(os.path.join(imagepath, img_name))
    image2 = cv2.imread(os.path.join(imagepath, images[i+1]), 0)


    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)


    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)


    src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    src_points_transformed = cv2.perspectiveTransform(src_points, M)
    reprojection_errors = np.sqrt(np.sum((dst_points - src_points_transformed) ** 2, axis=2))
    mean_reprojection_error = np.mean(reprojection_errors)
    avg_error.append((100-mean_reprojection_error))
    print('准确率:', 100-mean_reprojection_error)
print("平均准确率:", sum(avg_error)/len(avg_error))