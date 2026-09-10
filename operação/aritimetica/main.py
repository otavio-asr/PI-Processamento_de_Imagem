import cv2
import numpy as np

def adicao_media(img1, img2):
    f1 = img1.astype(np.float32)
    f2 = img2.astype(np.float32)
    res = (f1 + f2) / 2.0
    return np.round(res).astype(np.uint8)

def subtracao(img1, img2):
    f1 = img1.astype(np.float32)
    f2 = img2.astype(np.float32)
    res = np.clip(f1 - f2, 0, 255)
    return np.round(res).astype(np.uint8)

caminho1 = input("Caminho da Imagem 1: ").strip()
caminho2 = input("Caminho da Imagem 2: ").strip()

img1 = cv2.imread(caminho1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(caminho2, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Erro ao carregar uma ou ambas as imagens.")
    exit()

if img1.shape != img2.shape:
    print("As imagens devem ter a mesma resolucao.")
    exit()

print("1 - Adicao (Media)\n2 - Subtracao (img1 - img2)")
op = input("Operacao: ").strip()



if op == '1':
    resultado = adicao_media(img1, img2)
elif op == '2':
    resultado = subtracao(img1, img2)
else:
    print("Opcao invalida.")
    exit()

cv2.imwrite("resultado_aritmetica.png", resultado)
