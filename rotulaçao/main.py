import cv2

def rotular_imagem(imagem):
    linhas = len(imagem)
    colunas = len(imagem[0])
    rotulos = [[0] * colunas for _ in range(linhas)]
    
    equivalencias = {}
    proximo_rotulo = 1

    def encontrar_raiz(rotulo):
        while equivalencias[rotulo] != rotulo:
            rotulo = equivalencias[rotulo]
        return rotulo

    for i in range(linhas):
        for j in range(colunas):
            if imagem[i][j] != 0:
                r = rotulos[i][j-1] if j > 0 else 0
                t = rotulos[i-1][j] if i > 0 else 0

                if r == 0 and t == 0:
                    rotulos[i][j] = proximo_rotulo
                    equivalencias[proximo_rotulo] = proximo_rotulo
                    proximo_rotulo += 1
                elif r != 0 and t == 0:
                    rotulos[i][j] = r
                elif r == 0 and t != 0:
                    rotulos[i][j] = t
                elif r != 0 and t != 0:
                    if r == t:
                        rotulos[i][j] = r
                    else:
                        rotulos[i][j] = r
                        raiz_r = encontrar_raiz(r)
                        raiz_t = encontrar_raiz(t)
                        if raiz_r != raiz_t:
                            equivalencias[max(raiz_r, raiz_t)] = min(raiz_r, raiz_t)

    for i in range(linhas):
        for j in range(colunas):
            if rotulos[i][j] != 0:
                rotulos[i][j] = encontrar_raiz(rotulos[i][j])

    return rotulos

nome_arquivo = 'imagem.png' 

img_cinza = cv2.imread(nome_arquivo, cv2.IMREAD_GRAYSCALE)

if img_cinza is None:
    print(f"Erro: Não foi possível carregar a imagem '{nome_arquivo}'.")
else:
    _, img_binarizada = cv2.threshold(img_cinza, 127, 255, cv2.THRESH_BINARY)
    imagem_lista = img_binarizada.tolist()
    matriz_resultado = rotular_imagem(imagem_lista)

    rotulos_unicos = set()
    for linha in matriz_resultado:
        for pixel in linha:
            if pixel != 0:
                rotulos_unicos.add(pixel)

    print(f"Total de objetos encontrados: {len(rotulos_unicos)}")