import cv2
import numpy as np

def vizinho_reducao(img):
    return img[::2, ::2]

def vizinho_ampliacao(img):
    M, N = img.shape[:2]
    out = np.zeros((M * 2, N * 2), dtype=img.dtype)
    out[0::2, 0::2] = img
    out[0::2, 1::2] = img
    out[1::2, 0::2] = img
    out[1::2, 1::2] = img
    return out

def bilinear_reducao(img):
    img_f = img.astype(np.float32)
    M_red, N_red = img.shape[0] // 2, img.shape[1] // 2
    bloco = img_f[:M_red * 2, :N_red * 2]
    
    p00 = bloco[0::2, 0::2]
    p01 = bloco[0::2, 1::2]
    p10 = bloco[1::2, 0::2]
    p11 = bloco[1::2, 1::2]
    
    return np.round((p00 + p01 + p10 + p11) / 4.0).astype(img.dtype)

def bilinear_ampliacao(img):
    img_f = img.astype(np.float32)
    M, N = img.shape[:2]
    out = np.zeros((2 * M - 1, 2 * N - 1), dtype=np.float32)

    out[0::2, 0::2] = img_f
    out[0::2, 1::2] = (img_f[:, :-1] + img_f[:, 1:]) / 2.0
    out[1::2, 0::2] = (img_f[:-1, :] + img_f[1:, :]) / 2.0

    p_top_l = img_f[:-1, :-1]
    p_top_r = img_f[:-1, 1:]
    p_bot_l = img_f[1:, :-1]
    p_bot_r = img_f[1:, 1:]
    out[1::2, 1::2] = (p_top_l + p_top_r + p_bot_l + p_bot_r) / 4.0

    return np.round(out).astype(img.dtype)

caminho = input("Nome da imagem: ")
img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Imagem nao encontrada.")
    exit()

print("1 - Reducao\n2 - Ampliacao")
op = input("Operacao: ")

print("1 - Vizinho Mais Proximo\n2 - Bilinear")
metodo = input("Metodo: ")

if op == '1' and metodo == '1':
    resultado = vizinho_reducao(img)
elif op == '1' and metodo == '2':
    resultado = bilinear_reducao(img)
elif op == '2' and metodo == '1':
    resultado = vizinho_ampliacao(img)
elif op == '2' and metodo == '2':
    resultado = bilinear_ampliacao(img)
else:
    print("Opcao invalida.")
    exit()

print(f"Resolucao anterior: {img.shape[0]}x{img.shape[1]}")
print(f"Resolucao atual: {resultado.shape[0]}x{resultado.shape[1]}")

cv2.imwrite("resultado.png", resultado)
print("Salvo em resultado.png")