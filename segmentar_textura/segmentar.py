import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import glob

# Variáveis globais
KSIZE = 31
TAM_JANELA = 5
NIVEIS_PIRAMIDE = 3
DEBUG = False

############ Funções Auxiliares ############

def visualizar_segmentacao(imagem_rgb, rotulos, titulo="Segmentação por Textura"):
    """
    Visualiza o resultado da segmentação.
    
    Parâmetros:
    - imagem_rgb: imagem original
    - rotulos: matriz com os rótulos de segmentação
    - titulo: título para a visualização
    """
    # Cria um mapa de cores aleatório para os rótulos
    n_clusters = len(np.unique(rotulos))
    cores = np.random.rand(n_clusters, 3)
    mapa_cores = ListedColormap(cores)
    
    # Cria uma figura com 2 subplots
    plt.figure(figsize=(12, 6))
    
    # Exibe a imagem original
    plt.subplot(1, 2, 1)
    plt.imshow(imagem_rgb)
    plt.title('Imagem Original')
    plt.axis('off')
    
    # Exibe a segmentação
    plt.subplot(1, 2, 2)
    plt.imshow(rotulos, cmap=mapa_cores)
    plt.title(titulo)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return plt

def processar_uma_imagem(caminho_imagem, n_clusters=5, tipo_segmentacao=1):
    """
    Processa uma única imagem.
    
    Parâmetros:
    - caminho_imagem: path para arquivo de imagem
    - n_clusters: numero de clusters para segmentação
    """
    # Processa a imagem
    imagem_rgb, tensor_caracteristicas = processar_imagem(caminho_imagem)

    if tipo_segmentacao == 1:
        rotulos3 = segmentar_textura_kmeans(tensor_caracteristicas, n_clusters=n_clusters)
    elif tipo_segmentacao == 2:
        rotulos3 = segmentar_textura_distancia_euclidiana(tensor_caracteristicas, n_clusters=n_clusters)

    visualizar_segmentacao(imagem_rgb, rotulos3, titulo=f"Segmentação de ({n_clusters} clusters)")

    rotulos_imagem = np.uint8(255 * rotulos3 / np.max(rotulos3))

    # Salvar a imagem segmentada com extensão
    caminho_saida = "imagem_segmentada.png"
    cv2.imwrite(caminho_saida, rotulos_imagem)
    print(f"Imagem segmentada salva como {caminho_saida}")

def exibir_resultados(resultados, n_cols=4):
    """
    Exibe os resultados da segmentação para um conjunto de imagens.
    
    Parâmetros:
    - resultados: lista de tuplas (nome_arquivo, imagem_rgb, rotulos)
    - n_cols: número de colunas para exibição
    """
    n_imagens = len(resultados)
    n_rows = (n_imagens + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, n_rows * 5))
    
    for i, (nome_arquivo, imagem_rgb, rotulos) in enumerate(resultados):
        # Cria um mapa de cores aleatório para os rótulos
        n_clusters = len(np.unique(rotulos))
        cores = np.random.rand(n_clusters, 3)
        mapa_cores = ListedColormap(cores)
        
        # Exibe a imagem original
        plt.subplot(n_rows, n_cols * 2, i * 2 + 1)
        plt.imshow(imagem_rgb)
        plt.title(f"Original: {nome_arquivo}")
        plt.axis('off')
        
        # Exibe a segmentação
        plt.subplot(n_rows, n_cols * 2, i * 2 + 2)
        plt.imshow(rotulos, cmap=mapa_cores)
        plt.title(f"Segmentação: {n_clusters} clusters")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def salvar_resultados(output_path, resultados):
    """
    Salva resultados de segmentação em arquivos.
    
    Parâmetros:
    - output_path: caminho que será salvo
    - resultados: objeto com os resultados da segmentação
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("Salvando resultados...")

    plt.figure(figsize=(15, 10))
    for nome_arquivo, imagem_rgb, rotulos in resultados:
        n_clusters = len(np.unique(rotulos))
        cores = np.random.RandomState(42).rand(n_clusters, 3)

        mapa_cores = ListedColormap(cores)

        # Exibe a imagem original
        plt.subplot(1, 2, 1)
        plt.imshow(imagem_rgb)
        plt.title(f"Original: {nome_arquivo}")
        plt.axis('off')
        
        # Exibe a segmentação
        plt.subplot(1, 2, 2)
        plt.imshow(rotulos, cmap=mapa_cores)
        plt.title(f"Segmentação: {n_clusters} clusters")
        plt.axis('off')

        plt.savefig(os.path.join(output_path, nome_arquivo))
    plt.close()

def processar_conjunto_imagens(arquivos_imagem, n_clusters=5, tipo_segmentacao=1):
    """
    Processa um conjunto de imagens realizando a segmentação por textura.
    
    Parâmetros:
    - pasta_imagens: caminho para a pasta com as imagens
    - n_clusters: número de clusters para segmentação
    """
    resultados = []

    # Processa cada imagem
    for caminho_imagem in arquivos_imagem:
        # Extrai o nome base do arquivo
        nome_arquivo = os.path.basename(caminho_imagem)
        
        print(f"Processando: {nome_arquivo}")
        
        # Processa a imagem
        imagem_rgb, tensor_caracteristicas = processar_imagem(caminho_imagem)
        
        # Segmenta a imagem
        if tipo_segmentacao == 1:
            rotulos = segmentar_textura_kmeans(tensor_caracteristicas, n_clusters=n_clusters)
        elif tipo_segmentacao == 2: 
            rotulos = segmentar_textura_distancia_euclidiana(tensor_caracteristicas, n_clusters=n_clusters)
        
        # Armazena os resultados
        resultados.append((nome_arquivo, imagem_rgb, rotulos))
    
    return resultados

############ Funções de Segmentação ############

def segmentar_textura_kmeans(tensor_caracteristicas, n_clusters=5, random_state=56):
    """
    Segmenta a imagem com base nas características de textura usando K-means.
    
    Parâmetros:
    - tensor_caracteristicas: tensor com as características de textura
    - n_clusters: número de clusters para o K-means
    - random_state: semente para reprodução dos resultados
    
    Retorna:
    - Matriz com os rótulos de segmentação
    """
    print("Segmentando usando K-Means...")

    # Obtem as dimensões do tensor
    altura, largura, n_caracteristicas = tensor_caracteristicas.shape

    # Reorganiza o tensor em uma matriz 2D: (n_pixels, n_caracteristicas)
    X = tensor_caracteristicas.reshape(-1, n_caracteristicas)

    # Normaliza as características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Reduz a dimensionalidade com PCA para melhorar a eficiência
    pca = PCA(n_components=min(10, n_caracteristicas))
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    # rotulos é um array de tamanho n_pixels que indica qual cluster cada pixel pertence
    rotulos = kmeans.fit_predict(X_pca)

    # Reorganiza os rótulos para o formato original da imagem
    rotulos_imagem = rotulos.reshape(altura, largura)
    return rotulos_imagem

def segmentar_textura_distancia_euclidiana(tensor_caracteristicas, n_clusters=5, max_iter=10):
    """
    Agrupa os pixels da imagem com base na distância euclidiana entre seus vetores de características.

    Parâmetros:
    - tensor_caracteristicas: tensor com as características de textura
    - n_clusters: número de clusters desejados
    - max_iter: número máximo de iterações

    Retorna:
    - Matriz com os rótulos de segmentação
    """
    print("Segmentando usando distância euclidiana...")

    # Obtém as dimensões do tensor
    altura, largura, n_caracteristicas = tensor_caracteristicas.shape
    
    # Reorganiza o tensor para uma matriz 2D: (altura*largura, n_características)
    X = tensor_caracteristicas.reshape(-1, n_caracteristicas)
    
    # Normaliza as características para o intervalo [0, 1]
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-10)
    
    # Inicializa os centros dos clusters aleatoriamente
    # Para maior determinismo, podemos escolher pontos espaçados no conjunto de dados
    n_pontos = X_norm.shape[0]
    indices_centros = np.linspace(0, n_pontos-1, n_clusters, dtype=int)
    centros = X_norm[indices_centros]
    
    # Inicializa rótulos
    rotulos = np.zeros(n_pontos, dtype=int)
    
    # Iterações do algoritmo
    for iteracao in range(max_iter):
        # Atribui cada ponto ao cluster mais próximo
        for i in range(n_pontos):
            # Calcula distâncias euclidianas aos centros
            distancias = np.sqrt(np.sum((X_norm[i] - centros)**2, axis=1))
            # Atribui ao cluster mais próximo
            rotulos[i] = np.argmin(distancias)
        
        # Guarda os centros antigos para verificar convergência
        centros_antigos = centros.copy()
        
        # Atualiza os centros
        for k in range(n_clusters):
            # Pontos do cluster k
            pontos_cluster = X_norm[rotulos == k]
            if len(pontos_cluster) > 0:
                centros[k] = np.mean(pontos_cluster, axis=0)
        
        # Verifica convergência
        if np.allclose(centros, centros_antigos):
            print(f"Convergência atingida na iteração {iteracao+1}")
            break
    
    # Reorganiza os rótulos para o formato da imagem
    rotulos_imagem = rotulos.reshape(altura, largura)
    
    return rotulos_imagem

############ Funções de Filtros ############

def extrair_caracteristicas_textura(resposta_filtro, kernel):
    """
    Extrai características de textura a partir das respostas dos filtros.
    
    Parâmetros:
    - resposta_filtro: resposta do filtro
    - kernel: janela para cálculo das estatísticas locais
    
    Retorna:
    - Imagem com as médias locais
    """
    media = cv2.filter2D(resposta_filtro, -1, kernel)
    return media

def aplicar_filtros(imagem, filtros):
    """
    Aplica uma lista de filtros à imagem.
    
    Parâmetros:
    - imagem: imagem em escala de cinza
    - filtros: lista de filtros
    
    Retorna:
    - Lista de respostas dos filtros
    """
    respostas = []

    for kernel in filtros:
        # Aplica filtro usando convolução
        resposta = cv2.filter2D(imagem, cv2.CV_32F, kernel)
        respostas.append(resposta)

    return respostas

def criar_filtro_gabor(ksize=KSIZE, sigma=4.0, theta=0, lambd=10.0, gamma=0.5, psi=0):
    """
    Cria um filtro de Gabor com os parâmetros especificados.
    
    Parâmetros:
    - ksize: tamanho do kernel (impar)
    - sigma: desvio padrão do envelope gaussiano
    - theta: orientação do filtro em radianos
    - lambd: comprimento de onda da função senoidal
    - gamma: razão de aspecto do envelope gaussiano
    - psi: deslocamento de fase
    
    Retorna:
    - Kernel do filtro de Gabor
    """
    kernel = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F
    )

    # Normaliza o kernel
    if kernel.sum() != 0:
        kernel /= 1.5 * kernel.sum()
    else:
        kernel /= kernel.max()

    return kernel

def criar_banco_filtros(ksize=KSIZE):
    """
    Cria um banco de filtros de Gabor (com diferentes orientações), Gausseanos e Laplacianos.
    
    Retorna:
    - Lista de filtros
    """
    filtros = []

    #### Filtros de Gabor ####
    # Detecta texturas nas direções indicadas (e circular)

    # Parâmetros para filtros de Gabor
    sigma = 4.0
    lambd = 10.0
    gamma = 0.5
    psi = [0, np.pi/2]
    theta = [0, np.pi/2, np.pi/4, 3*np.pi/4] # 0°, 90°, 45°, 135°

    # Filtros direcionais
    for p in psi:
        for t in theta:
            filtros.append(criar_filtro_gabor(ksize, sigma, theta=t, lambd=lambd, gamma=gamma, psi=p))

    # Filtro circular
    filtros.append(criar_filtro_gabor(ksize, sigma, theta=0, lambd=lambd, gamma=1.0, psi=0))

    #### Filtro Laplaciano ####
    # Detecta bordas independentemente da direção

    # getDerivKernels retorna uma tupla de dois arrays 1D
    kx = cv2.getDerivKernels(2, 0, ksize)  # Segunda derivada em x (ordem=2)
    ky = cv2.getDerivKernels(0, 2, ksize)  # Segunda derivada em y (ordem=2)

    # Precisamos calcular o produto externo para obter o kernel 2D
    kernel_dx2 = np.outer(kx[0], kx[1])
    kernel_dy2 = np.outer(ky[0], ky[1])

    # O Laplaciano é a soma das derivadas segundas
    filtros.append(kernel_dx2 + kernel_dy2)

    #### Filtro Gaussiano ####
    # Serve para suavizar os ruídos e detalhes finos

    # getGaussianKernel retorna um array 1D
    # Precisamos gerar o gaussiano apartir da multiplicação de arrays para ter uma matriz gausseana
    kernel_gauss = cv2.getGaussianKernel(ksize, sigma)
    filtros.append(kernel_gauss * kernel_gauss.T)

    return filtros

def criar_piramide_gaussiana(imagem, niveis=NIVEIS_PIRAMIDE):
    """
    Cria uma pirâmide gaussiana com o número especificado de níveis.
    
    Parâmetros:
    - imagem: imagem de entrada
    - niveis: número de níveis da pirâmide
    
    Retorna:
    - Lista de imagens em diferentes escalas
    """
    piramide = [imagem]

    for i in range(1, niveis):
        # Aplica um filtro gaussiano (borra) e dps reduz a imagem para metade do tamanho
        img_reduzida = cv2.pyrDown(piramide[i-1])
        piramide.append(img_reduzida)

    return piramide

def processar_imagem(caminho_imagem, visualizar=False, tamanho_janela=TAM_JANELA, niveis=NIVEIS_PIRAMIDE):
    """
    Processa uma imagem aplicando filtros de Gabor em múltiplas escalas.

    Parâmetros:
    - caminho_imagem: caminho para a imagem
    - visualizar: se True, exibe as respostas dos filtros

    Retorna:
    - Vetor de características da imagem
    """
    # Lê a imagem
    imagem_rgb = cv2.imread(caminho_imagem)
    imagem_rgb = cv2.cvtColor(imagem_rgb, cv2.COLOR_BGR2RGB)

    # Converte para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2GRAY)

    # Cria a pirâmide gaussiana
    piramide = criar_piramide_gaussiana(imagem_cinza, niveis)

    # Cria o banco de filtros de Gabor
    filtros = criar_banco_filtros()

    # Lista pra armazenar todas as características extraídas
    todas_caracteristicas = []

    # Cria o kernel para cálculo da média local para vetor de caracteristicas
    kernel = np.ones((tamanho_janela, tamanho_janela), np.float32) / (tamanho_janela * tamanho_janela)

    # Processa cada escala da piramida
    for idx_escala, imagem_escala in enumerate(piramide):
        # Aplica filtros
        respostas_filtros = aplicar_filtros(imagem_escala, filtros)

        # Extrai as características da textura
        caracteristicas_escala = []
        for resposta in respostas_filtros:
            # Extrai a média local como característica
            media = extrair_caracteristicas_textura(resposta, kernel)

            # Redimensiona a característica para o tamanho da imagem original
            if idx_escala > 0:
                media = cv2.resize(media, (imagem_cinza.shape[1], imagem_cinza.shape[0]), interpolation=cv2.INTER_LINEAR)

            caracteristicas_escala.append(media)
        
        todas_caracteristicas.extend(caracteristicas_escala)

    # Visualiza as respostas dos filtros se solicitado
    if visualizar:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(1, 2, 1)
        plt.imshow(imagem_rgb)
        plt.title('Imagem Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(todas_caracteristicas[9], cmap='jet')
        plt.title('Exemplo de Resposta (Laplace, Escala 1)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    # Prepara o vetor de características
    # Redimensiona as características para o formato (altura, largura, n_caracteristicas)
    altura, largura = imagem_cinza.shape
    n_caracteristicas = len(todas_caracteristicas)

    # Reorganiza as caracteristicas em um tensor 3D
    tensor_caracteristicas = np.zeros((altura, largura, n_caracteristicas), dtype=np.float32)

    for i, caracteristica in enumerate(todas_caracteristicas):
        tensor_caracteristicas[:, :, i] = caracteristica

    # Como interpretar o tensor de características
    # Se você quer saber o vetor de características do pixel (100, 500) acesse:
    # pixel = tensor_caracteristicas[100, 50, :]

    return imagem_rgb, tensor_caracteristicas

############ Função Principal ############

    visualizar_segmentação(imagem_rgb, rotulos, titulo=f"Segmentação por K-Means")

# Função principal que orquestra todo o processo
def main():
    # Seleciona o tipo de segmentação
    # 1 -> K-Means
    # 2 -> Distância euclidiana
    tipo_segmentacao = 1

    # Número de clusters para segmentação
    n_clusters = 5

    if DEBUG:
        imagem_alvo = "../other/more_imgs/foto_12.jpg"
        processar_uma_imagem(imagem_alvo, n_clusters=n_clusters, tipo_segmentacao=tipo_segmentacao)
        return

    pasta_imagens = "./imgs"
    pasta_segmentados = "./segmentados"

    # Número maximo de imagens a serem processadas
    maximo_imagens = 16

    # Seleciona arquivos de imagens
    extensoes = ['img*']
    arquivos_imagem = []
    salvar = True
    exibir = True

    for ext in extensoes:
        arquivos_imagem.extend(glob.glob(os.path.join(pasta_imagens, ext)))

    if len(arquivos_imagem) <= 0:
        print(f"Nenhuma imagem encontrada para ser processada em '{pasta_imagens}'.")

    # Chama função de processamento das imagens
    resultados = processar_conjunto_imagens(
        arquivos_imagem[:min(maximo_imagens, len(arquivos_imagem))], 
        n_clusters=n_clusters,
        tipo_segmentacao=tipo_segmentacao
    )

    # Salva os resultados na pasta_segmentados
    if salvar:
        salvar_resultados(pasta_segmentados, resultados)

    # Mostra os resultados para o usuário
    if exibir:
        exibir_resultados(resultados)

if __name__ == "__main__":
    main()