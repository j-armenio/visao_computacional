import cv2
import os
import numpy as np
import glob

def remove_black_borders(image_path, output_path):
    """
    Retira as linhas pretas (parte superior e inferior) que o IPhone colocar nas fotos
    """
    # Carregar a imagem
    img = cv2.imread(image_path)
    if img is None:
        print(f"Não foi possível carregar a imagem: {image_path}")
        return False
    
    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Encontrar a primeira linha não-preta a partir do topo
    h, w = gray.shape
    top = 0
    bottom = h - 1
    
    # Threshold para considerar como "preto" (ajuste conforme necessário)
    threshold = 10
    
    # Verificar do topo para baixo
    for i in range(h):
        if np.mean(gray[i, :]) > threshold:
            top = i
            break
    
    # Verificar do topo + 1 até próxima linha preta
    for i in range(top+1, h):
        if np.mean(gray[i, :]) <= threshold:
            bottom = i
            break
    
    # Recortar a imagem
    cropped_img = img[top:bottom+1, :]
    
    # Salvar a imagem recortada
    cv2.imwrite(output_path, cropped_img)
    
    return True

def process_images(input_dir, output_dir):
    # Criar diretório de saída se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Processar cada imagem no diretório
    # image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_extensions = ['foto_0.jpg']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(input_dir, ext)))
        all_images.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    for img_path in all_images:
        # Gerar nome do arquivo de saída
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        
        # Processar a imagem
        success = remove_black_borders(img_path, output_path)
        if success:
            print(f"Processado: {filename}")
        else:
            print(f"Falha ao processar: {filename}")

if __name__ == "__main__":
    INPUT_FOLDER = "input_images"
    OUTPUT_FOLDER = "output_images"
    
    process_images(INPUT_FOLDER, OUTPUT_FOLDER)
    print("Processamento concluído! Verifique as imagens em", OUTPUT_FOLDER)