import cv2
import numpy as np

WINDOW_NAME = 'Selecione 4 pontos'

# def order_points(pts):
#     s = pts.sum(axis=1)
#     diff = np.diff(pts, axis=1)

#     tl = pts[np.argmin(s)]    # superior esquerdo
#     br = pts[np.argmax(s)]    # inferior direito
#     tr = pts[np.argmin(diff)] # superior direito
#     bl = pts[np.argmax(diff)] # inferior esquerdo

#     return np.array([tl, tr, br, bl], dtype='float32')

def main():
    # guarda os pontos clicados
    pts = []

    # chamada quando algum esquerdo clique é feito
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x,y))

            # Desenha um circulo no ponto clicado
            cv2.circle(img, (x,y), 5, (0, 255, 0), -1)
            cv2.imshow(WINDOW_NAME, img)

    img_id = input("\nEscolha uma das imagens disponíveis e clique em 4 pontos:\nIMPORTANTE: clique em ordem horária! (canto superior-esquerdo, canto superior-direito, canto inferior-direito, canto inferior-esquerdo)\n1 - Museu\n2 - Ginásio de escalada\n")

    if (img_id == '1'):
        img = cv2.imread('museu.jpg')
    elif (img_id == '2'):
        img = cv2.imread('escalada.jpeg')

    clone = img.copy()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(WINDOW_NAME, img)

    cv2.setMouseCallback(WINDOW_NAME, click_event)

    # Espera até que o usuário clique 4 vezes
    while len(pts) < 4:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    pts = np.array(pts, dtype='float32')
    # pts = order_points(pts)

    widthA = np.linalg.norm(pts[0] - pts[1])
    widthB = np.linalg.norm(pts[2] - pts[3])
    maxW = int((widthA + widthB) / 2)
    heightA = np.linalg.norm(pts[0] - pts[3])
    heightB = np.linalg.norm(pts[1] - pts[2])
    maxH = int((heightA + heightB) / 2)

    pts2 = np.array([
        [0, 0],           # canto superior-esquerdo
        [maxW-1, 0],     # canto superior-direito
        [maxW-1, maxH-1],# canto inferior-direito
        [0, maxH-1]      # canto inferior-esquerdo
    ], dtype='float32')

    # Calcula a matriz de transformação
    M = cv2.getPerspectiveTransform(pts, pts2)
    # Aplica a transformação
    dst = cv2.warpPerspective(clone, M, (maxW, maxH), flags=cv2.INTER_LINEAR)

    # Aumenta o tamanho da imagem transformada
    if maxW < 300 or maxH < 300:
        resized_W = int(maxW * 2)
        resized_H = int(maxH * 2)
    else:
        resized_W = maxW
        resized_H = maxH
        
    dst = cv2.resize(dst, (resized_W, resized_H), interpolation=cv2.INTER_LINEAR)

    cv2.imshow('Resultado', dst)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty('Resultado', cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
