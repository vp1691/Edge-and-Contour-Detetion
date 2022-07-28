import matplotlib.pyplot as plt
import numpy as np
import cv2

path = 'image/lena_color_256.tif'

img = cv2.imread(path)
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

_,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
edges = cv2.dilate(cv2.Canny(thresh,0,255),None)

cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, 
cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]

mask = np.zeros((256,256), np.uint8)
masked = cv2.drawContours(mask, [cnt],-1, 255, -1)

dst = cv2.bitwise_and(img, img, mask=mask)
segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

plt.subplot(2,2,1)
plt.imshow(img)
plt.title("Original Image")

plt.xticks([])
plt.yticks([])
plt.subplot(2,2,2)
plt.imshow(edges)
plt.title("Edges Image")

plt.xticks([])
plt.yticks([])
plt.subplot(2,2,3)
plt.imshow(masked)
plt.title("Masked Image")

plt.xticks([])
plt.yticks([])
plt.subplot(2,2,4)
plt.imshow(segmented)
plt.title("Segmented Image")

plt.xticks([])
plt.yticks([])
plt.show()