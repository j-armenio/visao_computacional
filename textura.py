import cv2
import os
import numpy as np
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json

# 1. Aplicar filtros de Gabor
def apply_gabor_filters(img, sigmas, thetas, lambdas, gammas, psis, kernel_size=31):
    gabor_features = []
    for sigma in sigmas:
      for theta in thetas:
        for lambd in lambdas:
          for gamma in gammas:
            for psi in psis:
              kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
              filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)
              gabor_features.append(filtered)
    return gabor_features

# 2. Aplicar Laplaciano do Gaussiano
def laplacian_of_gaussian(img, kernel_size=31):
    blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    log = cv2.Laplacian(blur, cv2.CV_64F)
    return np.uint8(np.absolute(log))

# 3. Classificar pixels manualmente usando distância mínima
def minimum_distance_classifier(features, training_samples):
    class_means = {}
    for class_id, pixels in training_samples.items():
        samples = np.array([features[y, x] for (y, x) in pixels])
        class_means[class_id] = np.mean(samples, axis=0)

    output = np.zeros(features.shape[:2], dtype=np.uint8)

    for y in range(features.shape[0]):
        for x in range(features.shape[1]):
            pixel_feature = features[y, x]
            distances = {class_id: np.linalg.norm(pixel_feature - mean) for class_id, mean in class_means.items()}
            output[y, x] = min(distances, key=distances.get)

    return output

# 4. Função principal de processamento
def process_image(image_path, output_path, training_samples):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    scale_image = img

    thetas = [0, np.pi/4, np. pi/2, 3 * (np.pi / 4)]  # Orientacoes
    psis = [np.pi/2]  # Fases
    lambdas = [32]  # Comprimentos de onda
    sigmas = [2.0]  # Desvio padrao da gaussiana
    gammas = [0.3]  # Relacao de aspecto espacial
    # Gabor filters
    combined_features = None
    num_of_scales = 3
    for scale in range(num_of_scales):
      gabor_features = apply_gabor_filters(img, sigmas, thetas, lambdas, gammas, psis)

      # Stack Gabor outputs
      gabor_stack = np.stack(gabor_features, axis=-1)

      # Edge detection (LoG)
      edge_img = laplacian_of_gaussian(scale_image, 5)

      # Adiciona a imagem de bordas ao stack de características
      new_combined_features = np.concatenate((gabor_stack, edge_img[..., np.newaxis]), axis=-1)
      if combined_features is None:
        combined_features = new_combined_features
      else:
        combined_features = np.concatenate((combined_features, new_combined_features), axis=-1)
      
      scale_image = cv2.GaussianBlur(scale_image, (5,5), 1)
      scale_image = cv2.resize(scale_image, (scale_image.shape[1]//2, scale_image.shape[0]//2))
      scale_image = cv2.resize(scale_image, (scale_image.shape[1]*2, scale_image.shape[0]*2))

    # Classify
    labels = minimum_distance_classifier(combined_features, training_samples)

    # Visualização
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('Imagem Original')
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    # Plotar pontos de treinamento
    colors = ['red', 'blue', 'green', 'yellow']
    for class_id, points in training_samples.items():
        ys, xs = zip(*points)
        plt.scatter(xs, ys, s=40, c=colors[class_id % len(colors)], label=f'Classe {class_id}', edgecolors='white')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.title('Bordas (LoG)')
    plt.imshow(edge_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Classificação')
    plt.imshow(labels, cmap='jet')
    plt.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

# Exemplo de amostras de treinamento (y, x):
# Para rodar de verdade, você precisa coletar alguns pontos manualmente da imagem
TRAINING_SAMPLES = {
    0: [(10, 10), (15, 15)],  # Classe 0
    1: [(50, 50), (55, 55)],  # Classe 1
    2: [(100, 100), (105, 105)],  # Classe 2
}


def load_training_samples(filename='training_samples.json'):
    # Carregar os pontos de treinamento do arquivo JSON
    with open(filename, 'r') as f:
        training_samples = json.load(f)
    
    # Garantir que os pontos de treinamento estejam no formato desejado
    formatted_samples = {int(k): [(int(y), int(x)) for y, x in v] for k, v in training_samples.items()}
    
    return formatted_samples

def process_images(input_dir, output_dir):
    # Criar diretório de saída se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Processar cada imagem no diretório
    # image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_extensions = ['foto_28.jpg']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(input_dir, ext)))
        all_images.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    for img_path in all_images:
        # Gerar nome do arquivo de saída
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        
        # Processar a imagem
        process_image(img_path, output_path, load_training_samples())

if __name__ == "__main__":
    INPUT_FOLDER = "cropped_images"
    OUTPUT_FOLDER = "output_images"
    
    process_images(INPUT_FOLDER, OUTPUT_FOLDER)
    print("Processamento concluído! Verifique as imagens em", OUTPUT_FOLDER)