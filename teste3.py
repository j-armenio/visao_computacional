import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import pi
import os

# Definindo os parametros do filtro de Gabor
thetas = [0, pi/4, pi/2, 3 * (pi / 4)]  # Orientacoes
# thetas = [0]  # Orientacoes
psi = pi/2  # Fases
lambd = 32  # Comprimentos de onda
sigmas = [1.0, 2.0, 3.0, 4.0]  # Desvio padrao da gaussiana
gamma = 0.3  # Relacao de aspecto espacial
kernel_size = 31

if not os.path.exists("kernels"):
    os.makedirs("kernels")

kernels = []

# Para cada combinacao de parametros, aplica o filtro e salva os resultados
for i, sig in enumerate(sigmas):
  for j, theta in enumerate(thetas):
    # Aplica o filtro de Gabor
    kernel = cv2.getGaborKernel(
        (kernel_size, kernel_size),  # Tamanho do kernel
        sig,     # Desvio padrao
        theta,     # Orientacao
        lambd,     # Comprimento de onda
        gamma,     # Relacao de aspecto
        psi,       # Fase
        ktype=cv2.CV_32F
    )
    
    # Normaliza o kernel para visualizacao
    kernel_normalizado = kernel / kernel.max()
    kernels.append(kernel_normalizado)

theta = 0  # Orientacoes
psis = [0, pi] # Fases
lambd = 32  # Comprimentos de onda
sig = 5.0  # Desvio padrao da gaussiana
gammas = [1, 1.3, 1.6]  # Relacao de aspecto espacial
for psi in psis:
  for gamma in gammas:
    kernel = cv2.getGaborKernel(
        (kernel_size, kernel_size),  # Tamanho do kernel
        sig*gamma,     # Desvio padrao
        theta,     # Orientacao
        lambd,     # Comprimento de onda
        gamma,     # Relacao de aspecto
        psi,       # Fase
        ktype=cv2.CV_32F
    )

    # Normaliza o kernel para visualizacao
    kernel_normalizado = kernel / kernel.max()
    kernels.append(kernel_normalizado)

for i, kernel in enumerate(kernels):
    # Salva o kernel do filtro
    plt.figure(figsize=(8, 8))
    plt.imshow(kernel, cmap='rainbow')
    plt.title(f'Kernel {i}')
    plt.axis('off')
    plt.savefig(f'kernels/k{i}.png')
    plt.close()

dxs = []
dys = []
ksize = []