import cv2
import os
import numpy as np

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

def process_images(input_folder, output_folder, target_size=512):
    """
    Processa todas as imagens de uma pasta
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            img = cv2.imread(input_path)
            if img is None:
                print(f"Erro ao carregar: {filename}")
                continue
            
            processed_img = resize_and_crop(img, target_size)
            
            # Garante que o resultado é 512x512
            assert processed_img.shape[0] == target_size and processed_img.shape[1] == target_size, \
                   f"Imagem {filename} não está 512x512! Tem {processed_img.shape}"
            
            cv2.imwrite(output_path, processed_img)
            print(f"Processado: {filename} -> {processed_img.shape}")

if __name__ == "__main__":
    INPUT_FOLDER = "input_images"
    OUTPUT_FOLDER = "output_images"
    TARGET_SIZE = 512
    
    process_images(INPUT_FOLDER, OUTPUT_FOLDER, TARGET_SIZE)
    print("Processamento concluído! Verifique as imagens em", OUTPUT_FOLDER)