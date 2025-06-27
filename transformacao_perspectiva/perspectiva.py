import cv2
import numpy as np
import os
import math

def transformacao_perspectiva():
    WINDOW_NAME = 'Selecione 4 pontos'

    # guarda os pontos clicados
    pts = []

    # chamada quando algum esquerdo clique é feito
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x,y))

            # Desenha um circulo no ponto clicado
            cv2.circle(img, (x,y), 5, (0, 255, 0), -1)
            cv2.imshow(WINDOW_NAME, img)

    print("\nEscolha uma das imagens disponíveis e clique em 4 pontos:")
    print("IMPORTANTE: clique em ordem horária! (canto superior-esquerdo, canto superior-direito, canto inferior-direito, canto inferior-esquerdo)")

    IMAGENS_DIR="imagens"

    files = os.listdir(IMAGENS_DIR)
    for i, file in enumerate(files):
        print(f"{i} - {file}")

    img_id = int(input())

    if img_id < 0 or img_id >= len(files):
        print("[ERRO]: Selecione um valor entre 0-" + str(len(files)))
        exit(1)

    img = cv2.imread(os.path.join(IMAGENS_DIR, files[img_id]))

    clone = img.copy()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(WINDOW_NAME, img)

    cv2.setMouseCallback(WINDOW_NAME, click_event)

    do_wrap = False

    # Espera até que o usuário clique 4 vezes
    while len(pts) < 4 or not do_wrap:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            do_wrap = True
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
        [maxW-1, 0],      # canto superior-direito
        [maxW-1, maxH-1], # canto inferior-direito
        [0, maxH-1]       # canto inferior-esquerdo
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

    print("Pressione \'s\' caso deseje salvar sua imagem gerada.")
    cv2.imshow('Resultado', dst)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            save_file = f"transformed_{files[img_id]}"
            print(f"saved to {save_file}")
            cv2.imwrite(save_file, dst)
        if cv2.getWindowProperty('Resultado', cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

def movimento_perspectiva():
    # Carrega a imagem
    img = cv2.imread('imagens/img8.png')
    if img is None:
        raise FileNotFoundError("Imagem não encontrada!")
    h, w = img.shape[:2]

    # Define o plano da imagem no espaço 3D (centralizado na origem)
    image_plane = np.array([
        [-w/2,  h/2, 0],  # top-left
        [ w/2,  h/2, 0],  # top-right
        [ w/2, -h/2, 0],  # bottom-right
        [-w/2, -h/2, 0],  # bottom-left
    ], dtype=np.float32)

    # Parâmetros da câmera
    camera_pos = np.array([0.0, 0.0, 1000.0], dtype=np.float32)
    yaw = 0.0   # Rotação em torno do eixo Y (esquerda/direita)
    pitch = 0.0 # Rotação em torno do eixo X (cima/baixo)
    focal_length = 1000

    def get_view_matrix(yaw, pitch):
        """Gera uma matriz de rotação a partir do yaw e pitch da câmera"""
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)

        # Rotação composta: primeiro pitch, depois yaw
        R = np.array([
            [cos_y, sin_y * sin_p, sin_y * cos_p],
            [0,     cos_p,         -sin_p],
            [-sin_y, cos_y * sin_p, cos_y * cos_p]
        ], dtype=np.float32)

        return R

    def project_point(P, C, R):
        """
        Projeta um ponto 3D P com base na posição da câmera C e rotação R
        """
        relative = P - C
        view = R @ relative  # aplica rotação
        if view[2] == 0:
            view[2] = 1e-5  # evita divisão por zero
        x = focal_length * view[0] / view[2]
        y = focal_length * view[1] / view[2]
        return np.array([x + w/2, y + h/2], dtype=np.float32)

    def project_all(pts3D, C, R):
        return np.array([project_point(p, C, R) for p in pts3D], dtype=np.float32)

    # Controles
    print("Movimento: W/S = cima/baixo | A/D = esquerda/direita | Q/E = aproxima/afasta")
    print("Rotação: I/K = olhar cima/baixo | J/L = olhar esquerda/direita | ESC para sair")

    while True:
        R = get_view_matrix(yaw, pitch)
        projected_pts = project_all(image_plane, camera_pos, R)

        # Verifica se os pontos são válidos
        if np.any(np.isnan(projected_pts)) or np.any(np.isinf(projected_pts)):
            print("Câmera em posição inválida. Ajuste a posição.")
            break

        dst_pts = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=np.float32)

        # Projeta a imagem no plano com perspectiva
        M = cv2.getPerspectiveTransform(dst_pts, projected_pts)
        scene = np.zeros((800, 1200, 3), dtype=np.uint8)
        cv2.warpPerspective(img, M, (scene.shape[1], scene.shape[0]), dst=scene, borderMode=cv2.BORDER_TRANSPARENT)

        # Mostra a cena renderizada
        cv2.imshow("Cena 3D com Rotacao", scene)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        elif key == ord('w'):
            camera_pos[1] += 20
        elif key == ord('s'):
            camera_pos[1] -= 20
        elif key == ord('d'):
            camera_pos[0] -= 20
        elif key == ord('a'):
            camera_pos[0] += 20
        elif key == ord('q'):
            camera_pos[2] += 20
        elif key == ord('e'):
            camera_pos[2] -= 20
        elif key == ord('j'):
            yaw -= 0.05
        elif key == ord('l'):
            yaw += 0.05
        elif key == ord('i'):
            pitch += 0.05
        elif key == ord('k'):
            pitch -= 0.05

    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Selecione uma opção:")
    print("1 - Transformação de perspectiva")
    print("2 - Movimentação em perspectiva")
    escolha = input("Digite 1 ou 2: ")

    if escolha == '1':
        transformacao_perspectiva()
    elif escolha == '2':
        movimento_perspectiva()
    else:
        print("Opção inválida.")

