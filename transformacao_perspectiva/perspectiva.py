import cv2
import numpy

WINDOW_NAME = 'Selecione 4 pontos'

def main():
    # guarda os pontos clicados
    pts = []

    # chamada quando o clique é feito
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTON and len(pts) < 4:
            # Add ponto a lista
            pts.append((x,y))

            # Desenha um circulo no ponto clicado
            cv2.circle(img, (x,y), 5, (0, 255, 0), -1)
            cv2.imshow(WINDOW_NAME, img)

    # Carrega a imagem
    img = cv2.imread('img.jpg')

    # Cria a janela com a imagem original
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(WINDOW_NAME, img)

    # Registra funcao no clique do mouse
    cv2.setMouseCallback(WINDOW_NAME, click_event)

    while len(pts) < 4:
        pass

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
