import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Carregar a imagem
image = cv2.imread('cropped_images/foto_19.jpg', cv2.IMREAD_GRAYSCALE)

# Função para criar filtros de Gabor
def create_gabor_filters():
    kernels = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kernel = cv2.getGaborKernel((21, 21), 5, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
        kernels.append(kernel)
    return kernels

# Função para aplicar filtros em uma imagem
def apply_filters(image, kernels):
    filtered_images = []
    for kernel in kernels:
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        filtered_images.append(filtered)
    return filtered_images

# Função para reduzir a imagem em 3 escalas
def reduce_image(image):
    scales = [image]
    for _ in range(2):  # Reduz a imagem em 2 níveis (3 escalas no total)
        image = cv2.GaussianBlur(image, (5, 5), 1)
        image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
        scales.append(image)
    return scales

# Função para calcular a média das janelas
def calculate_texture_features(image, window_size=5):
    height, width = image.shape
    features = []
    for y in range(0, height - window_size, window_size):
        for x in range(0, width - window_size, window_size):
            window = image[y:y+window_size, x:x+window_size]
            features.append(np.mean(window))
    return features

# Função para calcular a distância euclidiana entre dois vetores
def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

# Função para segmentar a imagem manualmente
def segment_image(features, n_clusters=3):
    # Inicializando os centros dos clusters com as primeiras 'n_clusters' características
    centers = [features[i] for i in range(n_clusters)]
    labels = np.zeros(len(features), dtype=int)
    
    # Iterações para atribuição das características aos clusters
    for _ in range(10):  # Número fixo de iterações
        # Atribuindo cada característica ao cluster mais próximo
        for i, feature in enumerate(features):
            distances = [euclidean_distance(feature, center) for center in centers]
            labels[i] = np.argmin(distances)
        
        # Atualizando os centros dos clusters
        for i in range(n_clusters):
            cluster_features = [features[j] for j in range(len(features)) if labels[j] == i]
            if cluster_features:
                centers[i] = np.mean(cluster_features, axis=0)
    
    return labels

# Criar filtros de Gabor
kernels = create_gabor_filters()

# Aplicar filtros em todas as escalas
scales = reduce_image(image)
filtered_images = []
for scale in scales:
    filtered_images.append(apply_filters(scale, kernels))

# Calculando características de textura para cada escala e filtro
all_features = []
for filtered_images_scale in filtered_images:
    for filtered_image in filtered_images_scale:
        features = calculate_texture_features(filtered_image)
        all_features.extend(features)

# Realizando a segmentação manual usando a distância euclidiana
n_clusters = 3  # Defina o número de clusters desejado
labels = segment_image(np.array(all_features).reshape(-1, 1), n_clusters)

# Mapeando os rótulos para a imagem original
height, width = image.shape
segmented_image = np.zeros_like(image)

feature_idx = 0
for y in range(0, height - 5, 5):  # Usando uma janela 5x5
    for x in range(0, width - 5, 5):
        window_label = labels[feature_idx]
        segmented_image[y:y+5, x:x+5] = window_label * 85  # Colorindo os clusters (código arbitrário)
        feature_idx += 1

# Exibindo os resultados
plt.figure(figsize=(10, 10))

# Imagem original
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Imagem Original")
plt.axis('off')

# Exibindo imagens filtradas em cada escala
for i, filtered_image in enumerate(filtered_images[0]):
    plt.subplot(2, 3, i+2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f"Escala 1 - Filtro {i+1}")
    plt.axis('off')

# Exibindo a imagem segmentada
plt.subplot(2, 3, 6)
plt.imshow(segmented_image, cmap='tab20b')
plt.title("Imagem Segmentada")
plt.axis('off')

plt.tight_layout()
plt.show()
