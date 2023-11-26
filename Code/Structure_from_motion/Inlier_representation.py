import numpy as np
import matplotlib.pyplot as plt
import cv2

def DrawInliers(i, j, inliers_a, inliers_b):
    img1 = cv2.imread('/home/venk/Downloads/Sfm/' + str(i) + '.png')
    img2 = cv2.imread('/home/venk/Downloads/Sfm/' + str(j) + '.png')

    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    out[:rows1, :cols1, :] = img1
    out[:rows2, cols1:cols1 + cols2, :] = img2
    radius = 4
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    thickness = 1
    # print(f"testing = {inliers_a}")
    assert len(inliers_a) == len(inliers_b), "inliers in images not equal"
    for m in range(0, len(inliers_a)):

        cv2.circle(out, (int(inliers_a[m,0]), int(inliers_a[m,1])), radius,
                   RED, -1)

        cv2.circle(out, (int(inliers_b[m][0]) + cols1, int(inliers_b[m][1])),
                   radius, BLUE, -1)
        
        cv2.line(out, (int(inliers_a[m][0]), int(inliers_a[m][1])),
                 (int(inliers_b[m][0]) + cols1, int(inliers_b[m][1])), GREEN,
                 thickness)
    plt.figure(figsize=(20, 15))
    plt.imshow(out)
    plt.show()