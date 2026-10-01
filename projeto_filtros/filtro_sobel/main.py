import numpy as np
import cv2

GX_MASK = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float64)

GY_MASK = np.array([[-1,  0,  1],
                    [-2,  0,  2],
                    [-1,  0,  1]], dtype=np.float64)

def aplicar_convolucao(imagem, mascara):
    h_img, w_img = imagem.shape
    h_k, w_k = mascara.shape
    pad_y, pad_x = h_k // 2, w_k // 2

    img_padded = np.pad(imagem, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)
    saida = np.zeros((h_img, w_img), dtype=np.float64)

    for i in range(h_img):
        for j in range(w_img):
            regiao = img_padded[i : i + h_k, j : j + w_k]
            saida[i, j] = np.sum(regiao * mascara)

    return saida

def tratar_escala(matriz):
    min_val, max_val = np.min(matriz), np.max(matriz)
    if max_val - min_val == 0:
        return np.zeros_like(matriz, dtype=np.uint8)
    return (255.0 * (matriz - min_val) / (max_val - min_val)).astype(np.uint8)

if __name__ == "__main__":
    img = cv2.imread("entrada.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((100, 100), dtype=np.uint8)
        img[25:75, 25:75] = 200

    img_f = img.astype(np.float64)
    gx = aplicar_convolucao(img_f, GX_MASK)
    gy = aplicar_convolucao(img_f, GY_MASK)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    mag_uint8 = np.clip(magnitude, 0, 255).astype(np.uint8)

    cv2.imwrite("sobel_magnitude.png", mag_uint8)
    cv2.imwrite("sobel_gx.png", tratar_escala(gx))
    cv2.imwrite("sobel_gy.png", tratar_escala(gy))