import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread("entrada.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Coloque o arquivo 'entrada.png' na mesma pasta do script.")
hist = np.bincount(img.flatten(), minlength=256)
cdf = hist.cumsum() / (img.shape[0] * img.shape[1])
lut = np.round(255 * cdf).astype(np.uint8)
img_equalizada = lut[img]
cv2.imwrite("imagem_equalizada.png", img_equalizada)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Histograma Original")
plt.hist(img.flatten(), bins=256, range=[0, 256], color='gray')
plt.subplot(1, 2, 2)
plt.title("Histograma Equalizado")
plt.hist(img_equalizada.flatten(), bins=256, range=[0, 256], color='black')
plt.tight_layout()
plt.savefig("grafico_histogramas.png")