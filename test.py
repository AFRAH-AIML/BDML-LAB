import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.color import rgb2gray

# Read image
img = cv2.imread('image.jpg')
if img is None:
    print("Error: Image not found!")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- SIFT Feature Extraction ---
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    sift_img = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # --- HOG Feature Extraction ---
    hog_img = hog(rgb2gray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                  orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), visualize=True)[1]

    # --- Display Results ---
    titles = ['SIFT Features', 'HOG Features']
    imgs = [cv2.cvtColor(sift_img, cv2.COLOR_BGR2RGB), hog_img]

    plt.figure(figsize=(10,5))
    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.imshow(imgs[i], cmap='gray')
        plt.title(titles[i]); plt.axis('off')
    plt.show()
