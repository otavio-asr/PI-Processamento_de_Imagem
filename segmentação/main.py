import cv2
import numpy as np

img = cv2.imread('entrada.png', cv2.IMREAD_GRAYSCALE)
hist, _ = np.histogram(img, bins=256, range=(0, 256))

hist_suave = np.convolve(hist, np.ones(5)/5, mode='same')

pico1 = np.argmax(hist_suave)

hist_temp = hist_suave.copy()
faixa = 30
hist_temp[max(0, pico1 - faixa):min(256, pico1 + faixa)] = 0
pico2 = np.argmax(hist_temp)

inicio, fim = min(pico1, pico2), max(pico1, pico2)
limiar = inicio + np.argmin(hist_suave[inicio:fim])

_, binaria = cv2.threshold(img, limiar, 255, cv2.THRESH_BINARY)

print(f"Picos detectados: {inicio} e {fim}")
print(f"Limiar encontrado pelo método do vale: {limiar}")

cv2.imwrite('segmentada_vale.png', binaria)