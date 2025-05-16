import cv2
import numpy as np

def main():
    # Lista para armazenar os pontos em que o mouse clica
    pts = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            # Adiciona o ponto à lista
            pts.append((x, y))
            # Desenha um círculo no ponto clicado
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(WINDOW_NAME, img)

    # Carrega a imagem
    img = cv2.imread('img.jpg')
    clone = img.copy()

    WINDOW_NAME = 'Selecione 4 pontos'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(WINDOW_NAME, img)

    # Registra o callback do mouse
    cv2.setMouseCallback(WINDOW_NAME, click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(pts) != 4:
        raise Exception("Você deve selecionar exatamente 4 pontos!")

    # Ordenar/definir vertices objetivo (retângulo)
    # Retangulo de largura W e altura H
    pts1 = np.array(pts, dtype='float32')

    # Calcula larguras e alturas
    widthA = np.linalg.norm(pts1[0] - pts1[1])
    widthB = np.linalg.norm(pts1[2] - pts1[3])
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(pts1[0] - pts1[3])
    heightB = np.linalg.norm(pts1[1] - pts1[2])
    maxH = int(max(heightA, heightB))

    # Calcula a matriz de transformação
    pts2 = np.array([
        [0,           0], # canto superior-esquerdo
        [maxW-1,      0], # canto superior-direito
        [maxW-1, maxH-1], # canto inferior-direito
        [0,      maxH-1]  # canto inferior-esquerdo
    ], dtype='float32')

    print("Ta rodando...")

    # Calcula a matriz de transformação
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # Aplica a transformação
    dst = cv2.warpPerspective(clone, M, (maxW, maxH))

    # Mostra a imagem transformada
    cv2.imwrite("img_perspectiva.jpg", dst)
    cv2.imshow("Imagem transformada", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()