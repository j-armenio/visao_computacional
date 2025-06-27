import cv2
import numpy as np
import math

# Carrega a imagem
img = cv2.imread('img8.png')
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
    elif key == ord('a'):
        camera_pos[0] -= 20
    elif key == ord('d'):
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

