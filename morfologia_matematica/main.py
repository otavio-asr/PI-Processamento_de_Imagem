import cv2
import numpy as np

binaria = np.zeros((100, 100), dtype=np.uint8)
binaria[30:70, 30:70] = 255

elemento = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

img_erodida = cv2.erode(binaria, elemento)
img_dilatada = cv2.dilate(binaria, elemento)
contorno = cv2.subtract(binaria, img_erodida)

cv2.imwrite("resultado_erodida.png", img_erodida)
cv2.imwrite("resultado_dilatada.png", img_dilatada)
cv2.imwrite("resultado_contorno.png", contorno)