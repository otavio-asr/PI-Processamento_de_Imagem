import cv2
import numpy as np

def negativo(img):
    return 255 - img

caminho = input("Nome da imagem: ").strip()
img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Imagem nao encontrada.")
    exit()

resultado = negativo(img)

cv2.imwrite("resultado_negativo.png", resultado)