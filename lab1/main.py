import numpy as np
import math
import cv2

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Reading file
img = cv2.imread("C://Users//yaros//.spyder-py3//Bots//image.png")
viewImage(img, "Original")

# Detecting face
cascade = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade)
faces = faceCascade.detectMultiScale(img)
x, y, w, h = faces[0]
detectedFaceImage = img[y : y + h, x : x + w]
viewImage(detectedFaceImage, "FACE")

# Changing borders
img = detectedFaceImage[math.ceil(0.1 * h):math.ceil(0.9 * h), math.ceil(0.1 * w):math.ceil(0.9 * w), :]
viewImage(img, "Indentation from borders")

# Changing image to binary colors
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 100, 3);
viewImage(edges, "Binary images of edges")

# Removing small borders
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
newContours = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w >= 10 and h >= 10:
        newContours.append(c)
newMask = np.zeros_like(edges)
cv2.drawContours(newMask, newContours, -1, (255, 255, 255), cv2.FILLED)
edges = cv2.bitwise_and(edges, edges, mask = newMask)
viewImage(edges, "Removing small borders")

# Applying morphological augmentation operation
structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilatation = cv2.dilate(edges, structuringElement)
viewImage(dilatation, "Morphological augmentation operation")

# Using Gaussing blur
gauss = cv2.GaussianBlur(dilatation, (5, 5), 0)
viewImage(gauss, "Gauss")

# Normalize the image
dst = np.zeros_like(gauss)
normalizationImg = cv2.normalize(gauss, dst, 100, 100, cv2.NORM_INF)
viewImage(normalizationImg, "Normalization")

# Applying bilateral filter
bilateralFiltration = cv2.bilateralFilter(img, 20, 40, 10)
viewImage(bilateralFiltration, "Face with bilateral filtration")

# Increasing the contrast
sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
increasedContrast = cv2.filter2D(img, -1, sharp)
viewImage(increasedContrast, "Face with increased contrast")

# Applying final filtration
normalizationImg = np.expand_dims(normalizationImg, axis=2)
resultImg = normalizationImg * increasedContrast + (1 - normalizationImg) * bilateralFiltration
viewImage(resultImg, "Final filter")
