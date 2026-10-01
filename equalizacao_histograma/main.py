import numpy as np
import cv2

def equalizar_histograma(imagem, niveis=256):
    h, w = imagem.shape
    total_pixels = h * w
    
    hist = np.zeros(niveis, dtype=np.int32)
    for i in range(h):
        for j in range(w):
            hist[imagem[i, j]] += 1
            
    pr = hist / total_pixels
    
    cdf = np.zeros(niveis, dtype=np.float64)
    soma = 0.0
    for k in range(niveis):
        soma += pr[k]
        cdf[k] = soma
        
    lut = np.round((niveis - 1) * cdf).astype(np.uint8)
    
    saida = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            saida[i, j] = lut[imagem[i, j]]
            
    return saida, hist, lut

if __name__ == "__main__":
    img = cv2.imread("entrada.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        matriz_exemplo = np.array([
            [2, 2, 1, 1],
            [2, 2, 1, 1],
            [2, 1, 4, 3],
            [4, 4, 2, 0]
        ], dtype=np.uint8)
        img_eq, _, _ = equalizar_histograma(matriz_exemplo, niveis=8)
        print("Resultado da equalizacao da matriz do exercicio:")
        print(img_eq)
    else:
        resultado, _, _ = equalizar_histograma(img, niveis=256)
        cv2.imwrite("imagem_equalizada.png", resultado)