import cv2
import numpy as np

def rotacionar(img, angulo_graus):
    rad = np.radians(angulo_graus)
    cos_t = np.cos(rad)
    sen_t = np.sin(rad)

    M, N = img.shape[:2]
    cx, cy = M / 2.0, N / 2.0

    out = np.zeros_like(img)

    for x_d in range(M):
        for y_d in range(N):
            x_shift = x_d - cx
            y_shift = y_d - cy

            x_o = int(round(cos_t * x_shift + sen_t * y_shift + cx))
            y_o = int(round(-sen_t * x_shift + cos_t * y_shift + cy))

            if 0 <= x_o < M and 0 <= y_o < N:
                out[x_d, y_d] = img[x_o, y_o]

    return out

caminho = input("Nome da imagem: ").strip()
img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Imagem nao encontrada.")
    exit()

ang = float(input("Angulo em graus: "))

resultado = rotacionar(img, ang)


cv2.imwrite("resultado_rotacao.png", resultado)