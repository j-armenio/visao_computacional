import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import pi
import os

# Cria uma imagem de teste simples - um quadrado branco em fundo preto
imagem = np.zeros((200, 200), dtype=np.float32)
imagem[50:150, 50:150] = 1.0

# Definindo os parametros do filtro de Gabor
# thetas = [0, pi/4, pi/2, 3 * (pi / 4)]  # Orientacoes
thetas = [0]  # Orientacoes
psis = [0]  # Fases
lambdas = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # Comprimentos de onda
sigma = 5.0  # Desvio padrao da gaussiana
gamma = 0.5  # Relacao de aspecto espacial

# Funcao para aplicar o filtro de Gabor e retornar a resposta
def aplicar_gabor(img, theta, psi, lambd, sigma, gamma):
    # Cria o kernel do filtro de Gabor
    kernel = cv2.getGaborKernel(
        (31, 31),  # Tamanho do kernel
        sigma,     # Desvio padrao
        theta,     # Orientacao
        lambd,     # Comprimento de onda
        gamma,     # Relacao de aspecto
        psi,       # Fase
        ktype=cv2.CV_32F
    )
    
    # Normaliza o kernel para visualizacao
    kernel_normalizado = kernel / kernel.max()
    
    # Aplica o filtro na imagem
    imagem_filtrada = cv2.filter2D(img, cv2.CV_32F, kernel)
    
    return kernel_normalizado, imagem_filtrada

# Cria a pasta para salvar as imagens, se não existir
if not os.path.exists("imagens_filtro"):
    os.makedirs("imagens_filtro")

if not os.path.exists("kernels"):
    os.makedirs("kernels")

# Para cada combinacao de parametros, aplica o filtro e salva os resultados
for i, theta in enumerate(thetas):
    for j, psi in enumerate(psis):
        for k, lambd in enumerate(lambdas):
            # Aplica o filtro de Gabor
            kernel, filtrada = aplicar_gabor(imagem, theta, psi, lambd, sigma, gamma)
            
            # Salva o kernel do filtro
            plt.figure(figsize=(8, 8))
            plt.imshow(kernel, cmap='gray')
            plt.title(f'Kernel - theta={theta:.2f}, psi={psi:.2f}, lambda={lambd}')
            plt.axis('off')
            plt.savefig(f'kernels/kernel_theta_{theta:.2f}_psi_{psi:.2f}_lambda_{lambd}.png')
            plt.close()

# Versao alternativa usando uma imagem real
# Carrega uma imagem real em escala de cinza
imagem_real = cv2.imread('cropped_images/foto_19.jpg', cv2.IMREAD_GRAYSCALE)
if imagem_real is not None:
    # Redimensiona para um tamanho adequado
    imagem_real = cv2.resize(imagem_real, (200, 200))
    imagem_real = imagem_real.astype(np.float32) / 255.0  # Normaliza para [0,1]
    
    # Salva a imagem original real
    plt.figure(figsize=(8, 8))
    plt.imshow(imagem_real, cmap='gray')
    plt.title('Imagem Original Real')
    plt.axis('off')
    plt.savefig('imagens_filtro/imagem_real_original.png')
    plt.close()

    # Para cada combinacao de parametros, aplica o filtro e salva os resultados
    for i, theta in enumerate(thetas):
        for k, lambd in enumerate(lambdas):
            # Usa apenas psi = 0 para simplificar a visualizacao
            psi = 0
            
            # Aplica o filtro de Gabor
            kernel, filtrada = aplicar_gabor(imagem_real, theta, psi, lambd, sigma, gamma)
            
            # Salva a imagem filtrada
            plt.figure(figsize=(8, 8))
            plt.imshow(filtrada, cmap='gray')
            plt.title(f'Imagem Filtrada - theta={theta:.2f}, lambda={lambd}')
            plt.axis('off')
            plt.savefig(f'imagens_filtro/filtrada_real_theta_{theta:.2f}_lambda_{lambd}.png')
            plt.close()

else:
    print("Para executar o segundo exemplo, substitua 'cropped_images/foto_19.jpg' por um caminho válido.")
