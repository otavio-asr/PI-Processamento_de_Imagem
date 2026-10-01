import numpy as np
import cv2

MASCARAS = {
    'A': np.array([[ 0,  1,  0],
                   [ 1, -4,  1],
                   [ 0,  1,  0]], dtype=np.float64),

    'B': np.array([[ 1,  1,  1],
                   [ 1, -8,  1],
                   [ 1,  1,  1]], dtype=np.float64),

    'C': np.array([[ 0, -1,  0],
                   [-1,  4, -1],
                   [ 0, -1,  0]], dtype=np.float64),

    'D': np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]], dtype=np.float64)
}

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

def tratar_negativos_zero(matriz):
    return np.clip(matriz, 0, 255).astype(np.uint8)

def tratar_negativos_escala(matriz):
    min_val, max_val = np.min(matriz), np.max(matriz)
    if max_val - min_val == 0:
        return np.zeros_like(matriz, dtype=np.uint8)
    escala = 255.0 * (matriz - min_val) / (max_val - min_val)
    return escala.astype(np.uint8)

def realce_laplaciano(imagem, tipo_mascara):
    mascara = MASCARAS[tipo_mascara]
    lap = aplicar_convolucao(imagem.astype(np.float64), mascara)
    c = -1.0 if mascara[1, 1] < 0 else 1.0
    realcada = imagem.astype(np.float64) + (c * lap)
    return np.clip(realcada, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    img = cv2.imread("entrada.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((100, 100), dtype=np.uint8)
        img[25:75, 25:75] = 200

    for chave in ['A', 'B', 'C', 'D']:
        lap = aplicar_convolucao(img.astype(np.float64), MASCARAS[chave])
        cv2.imwrite(f"laplaciano_{chave}_zero.png", tratar_negativos_zero(lap))
        cv2.imwrite(f"laplaciano_{chave}_escala.png", tratar_negativos_escala(lap))
        cv2.imwrite(f"realce_{chave}.png", realce_laplaciano(img, chave))