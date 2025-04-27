import cv2
import numpy as np
import matplotlib.pyplot as plt

def remover_riscos_morfologia(imagem, mostrar_etapas=True):
    """
    Remove riscos pretos de uma imagem usando operações morfológicas.
    
    Args:
        imagem: Caminho da imagem ou array numpy da imagem
        mostrar_etapas: Se True, exibe as imagens intermediárias do processo
    
    Returns:
        Imagem com riscos removidos
    """
    # Carregar imagem se for um caminho
    if isinstance(imagem, str):
        img = cv2.imread(imagem)
        if img is None:
            raise FileNotFoundError(f"Não foi possível abrir a imagem: {imagem}")
    else:
        img = imagem.copy()
    
    # Converter para escala de cinza se for colorida
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Preservar a imagem original para comparação
    img_original = img.copy()
    
    # Etapa 1: Limiarização para detectar áreas escuras (riscos)
    # Usando limiarização adaptativa para lidar com diferentes iluminações
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Etapa 2: Usar operações morfológicas para isolar riscos
    # Kernel horizontal para riscos horizontais
    kernel_h = np.ones((1, 5), np.uint8)
    # Kernel vertical para riscos verticais
    kernel_v = np.ones((5, 1), np.uint8)
    
    # Aplicar abertura em ambas direções para remover ruídos pequenos
    opening_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    opening_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
    
    # Combinar resultados das duas direções
    combined_opening = cv2.bitwise_or(opening_h, opening_v)
    
    # Aplicar fechamento para preencher pequenos buracos nos riscos
    # Utilizando um kernel ligeiramente maior
    kernel_close = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(combined_opening, cv2.MORPH_CLOSE, kernel_close)
    
    # Criar máscara para os riscos
    # Dilatação para garantir que toda a extensão dos riscos seja coberta
    mask = cv2.dilate(closing, kernel_close, iterations=1)
    
    # Etapa 3: Aplicar reconstrução (inpainting) na área dos riscos
    # Converter máscara para o formato exigido pelo inpainting
    mask_uint8 = np.uint8(mask)
    
    # Aplicar inpainting para reconstruir a área dos riscos
    if len(img.shape) == 3:
        result = cv2.inpaint(img, mask_uint8, 3, cv2.INPAINT_TELEA)
    else:
        result = cv2.inpaint(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 
                            mask_uint8, 3, cv2.INPAINT_TELEA)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # Etapa 4: Aplicar filtro bilateral para preservar bordas
    if len(result.shape) == 3:
        result_bilateral = cv2.bilateralFilter(result, 9, 75, 75)
    else:
        result_bilateral = cv2.bilateralFilter(result, 9, 75, 75)
    
    # Exibir resultados intermediários se solicitado
    if mostrar_etapas:
        plt.figure(figsize=(15, 10))
        
        # Converter BGR para RGB se for imagem colorida
        if len(img_original.shape) == 3:
            img_original_plt = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            result_plt = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_bilateral_plt = cv2.cvtColor(result_bilateral, cv2.COLOR_BGR2RGB)
        else:
            img_original_plt = img_original
            result_plt = result
            result_bilateral_plt = result_bilateral
        
        # Original
        plt.subplot(2, 3, 1)
        plt.imshow(img_original_plt, cmap='gray' if len(img_original.shape) == 2 else None)
        plt.title('Imagem Original')
        plt.axis('off')
        
        # Limiarização
        plt.subplot(2, 3, 2)
        plt.imshow(thresh, cmap='gray')
        plt.title('Limiarização')
        plt.axis('off')
        
        # Operações morfológicas
        plt.subplot(2, 3, 3)
        plt.imshow(closing, cmap='gray')
        plt.title('Após Operações Morfológicas')
        plt.axis('off')
        
        # Máscara de riscos
        plt.subplot(2, 3, 4)
        plt.imshow(mask_uint8, cmap='gray')
        plt.title('Máscara de Riscos')
        plt.axis('off')
        
        # Após inpainting
        plt.subplot(2, 3, 5)
        plt.imshow(result_plt, cmap='gray' if len(result.shape) == 2 else None)
        plt.title('Após Inpainting')
        plt.axis('off')
        
        # Resultado final com filtro bilateral
        plt.subplot(2, 3, 6)
        plt.imshow(result_bilateral_plt, cmap='gray' if len(result_bilateral.shape) == 2 else None)
        plt.title('Resultado Final')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return result_bilateral

def aplicar_em_multiplas_imagens(lista_imagens, diretorio_saida=None):
    """
    Aplica a remoção de riscos em múltiplas imagens
    
    Args:
        lista_imagens: Lista com caminhos das imagens
        diretorio_saida: Diretório para salvar as imagens processadas (opcional)
    """
    import os
    
    resultados = []
    
    for i, img_path in enumerate(lista_imagens):
        print(f"Processando imagem {i+1}/{len(lista_imagens)}: {img_path}")
        
        try:
            resultado = remover_riscos_morfologia(img_path, mostrar_etapas=(i==0))
            resultados.append(resultado)
            
            # Salvar resultado se especificado
            if diretorio_saida:
                if not os.path.exists(diretorio_saida):
                    os.makedirs(diretorio_saida)
                    
                nome_arquivo = os.path.basename(img_path)
                nome_base, extensao = os.path.splitext(nome_arquivo)
                caminho_saida = os.path.join(diretorio_saida, f"{nome_base}_sem_riscos{extensao}")
                
                cv2.imwrite(caminho_saida, resultado)
                print(f"Imagem salva: {caminho_saida}")
                
        except Exception as e:
            print(f"Erro ao processar {img_path}: {e}")
    
    return resultados

# Exemplo de uso:
if __name__ == "__main__":
    # Caminho da imagem a ser processada
    imagem_exemplo = "cropped_images/foto_29.png"
    
    # Processar uma única imagem
    try:
        resultado = remover_riscos_morfologia(imagem_exemplo)
        cv2.imshow("Resultado", resultado)
        cv2.imwrite("output_images/foto_29.png", resultado)
    except FileNotFoundError:
        print("Por favor, atualize o caminho da imagem ou execute o código em um notebook")
        
    # Processar múltiplas imagens de um diretório
    # lista_imagens = glob.glob("diretorio_com_imagens/*.jpg")
    # aplicar_em_multiplas_imagens(lista_imagens, "diretorio_saida")