import numpy as np
import cv2

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

def filtro_media(imagem, tamanho=3):
    mascara = np.ones((tamanho, tamanho), dtype=np.float64) / (tamanho * tamanho)
    res = aplicar_convolucao(imagem.astype(np.float64), mascara)
    return np.clip(np.round(res), 0, 255).astype(np.uint8)

if __name__ == "__main__":
    img = cv2.imread("entrada.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((100, 100), dtype=np.uint8)
        img[25:75, 25:75] = 200

    cv2.imwrite("resultado_media_3x3.png", filtro_media(img, tamanho=3))
    cv2.imwrite("resultado_media_5x5.png", filtro_media(img, tamanho=5))