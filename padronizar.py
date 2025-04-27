import cv2
import os
import numpy as np
import glob

TARGET_SIZE = 512

def resize_and_crop(img, target_size=512):
    """
    Redimensiona a imagem para preencher 512x512, cortando o excesso
    sem distorcer e sem bordas pretas
    """
    # Obtém as dimensões originais
    h, w = img.shape[:2]
    
    # Calcula a proporção necessária
    target_ratio = target_size / target_size  # 1.0 (quadrado)
    img_ratio = w / h

    # Determina como redimensionar
    if img_ratio >= target_ratio:
        # Imagem mais larga que alta (ou quadrada)
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        # Imagem mais alta que larga
        new_h = target_size
        new_w = int(w * (target_size / h))
    
    # Redimensiona mantendo proporções
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Faz o crop central para obter exatamente 512x512
    if resized.shape[0] >= target_size and resized.shape[1] >= target_size:
        start_x = (resized.shape[1] - target_size) // 2
        start_y = (resized.shape[0] - target_size) // 2
        cropped = resized[start_y:start_y+target_size, start_x:start_x+target_size]
    else:
        # Caso raro (imagem muito pequena), força 512x512 com resize direto
        cropped = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    return cropped

def process_images(input_dir, output_dir):
    # Criar diretório de saída se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Processar cada imagem no diretório
    # image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_extensions = ['agarras*.jpg', 'agarras*.png']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(input_dir, ext)))
        all_images.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    for img_path in all_images:
        # Gerar nome do arquivo de saída
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        
        # Processar a imagem
        img = cv2.imread(img_path)
        if img is None:
            print(f"Erro ao carregar: {filename}")
            continue
        
        # Passa pelas tranformações
        img = resize_and_crop(img, TARGET_SIZE)
        
        # Garante que o resultado é 512x512
        assert img.shape[0] == TARGET_SIZE and img.shape[1] == TARGET_SIZE, \
                f"Imagem {filename} não está 512x512! Tem {img.shape}"
        
        cv2.imwrite(output_path, img)
        print(f"Processado: {filename} -> {img.shape}")

if __name__ == "__main__":
    INPUT_FOLDER = "input_images"
    OUTPUT_FOLDER = "cropped_images"
    
    process_images(INPUT_FOLDER, OUTPUT_FOLDER)
    print("Processamento concluído! Verifique as imagens em", OUTPUT_FOLDER)