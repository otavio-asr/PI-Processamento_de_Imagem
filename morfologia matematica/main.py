import cv2
import numpy as np

img = cv2.imread('entrada.png', cv2.IMREAD_GRAYSCALE)
_, binaria = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
elemento_estruturante = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img_erodida = cv2.erode(binaria, elemento_estruturante, iterations=1)
img_dilatada = cv2.dilate(binaria, elemento_estruturante, iterations=1)
contorno = cv2.subtract(binaria, img_erodida)

cv2.imwrite('resultado_dilatada.png', img_dilatada)
cv2.imwrite('resultado_erodida.png', img_erodida)
cv2.imwrite('resultado_contorno.png', contorno)
print("Processamento concluído com sucesso!")