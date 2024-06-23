import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

image_path =Path("Type1")/"((Confidential))210509_090939_21L4C22DAY-S22L-VF12-C04_S02_REJECTED.jpg"
image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
print(image.shape)
# Thresholding to find blobs
# Thresholding to find blobs
_, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

thresh = thresh[1:-1, 1:-1, ...]
cv2.imwrite("t.jpg",thresh)

# Find connected components
num_labels, labels_im = cv2.connectedComponents(thresh)
# Display the result
plt.figure(dpi=600)
plt.imshow(labels_im, cmap='nipy_spectral')
plt.axis('off')
plt.show()
num_labels
