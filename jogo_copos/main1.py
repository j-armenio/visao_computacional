from collections import defaultdict, deque
from ultralytics import YOLO
from cvzone.ColorModule import ColorFinder
import cvzone
import cv2
import numpy as np
import os

# ===== VARIÁVEIS DE CONTROLE DO MAPEAMENTO =====

FIXED_NUM_CUPS = 3      # Número de "slots" de copos que queremos rastrear
MAX_AGE = 50            # Nº de quadros para um slot ser considerado "perdido" e elegível para Re-ID
REID_DISTANCE_THRESHOLD = 100 # Distância máxima em pixels para re-identificar um copo
TRACK_HISTORY_LEN = 10  # Tamanho máximo do histórico de um track
CUP_YOLO_ID = 41

COLOR_BALL = (255, 0, 255) # roxo
COLOR_CUPS = {
    0: (255, 0, 0), # azul
    1: (0, 255, 0), # verde
    2: (0, 0, 255)  # vermelho
}

# ===== INICIALIZACAO =====

# Carrega o modelo YOLO
model = YOLO("yolo11n.pt")

# configuração do detector de cor para bola
myColorFinder = ColorFinder(False)
hsvVals = {
    'hmin': 10,
    'smin': 160,
    'vmin': 150,
    'hmax': 22,
    'smax': 255,
    'vmax': 255
}
INPUT_VIDEO = 0
OUTPUT_VIDEO = f"output_v6_camera.mp4"

opt = input("Selecione fonte do vídeo: \n1 - Câmera em tempo real \n2 - Vídeo gravado\n")

if opt == "1":
    INPUT_VIDEO = 0
elif opt == "2":
    INPUT_VIDEO = "video1.mp4"
    if not os.path.exists(INPUT_VIDEO):
        print("Erro: video nao encontrado")
        exit(1)
else:
    print("Opção inválida.")
    exit(1)

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"Erro: não consegui abrir o vídeo {INPUT_VIDEO}")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

# ===== VARIÁVEIS DE GERENCIAMENTO DE ESTADO =====

# Mapeamento do ID do YOLO para nosso ID de exibição estável
yolo_id_to_display_id = {}
display_id_to_yolo_id = {}

# Pool de IDs de exibição disponíveis (0, 1, 2...)
available_display_ids = set(range(FIXED_NUM_CUPS))

# Dicionário para "slots" perdidos, guardando a última posição e idade
lost_slots = {}  # Formato: {display_id: {'last_pos': (x,y), 'age': frames}}

# Histórico da trilha e cores, agora usando nosso display_id estável
track_history = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LEN))
track_colors = {}

# Variaveis de cntrole da bola
ball_color = COLOR_BALL
ball_track_history = deque(maxlen=TRACK_HISTORY_LEN)

# marca se o copo esta com a bola
cup_has_ball = {0: False, 1: False, 2: False}
last_ball_pos = None
last_ball_bbox = {}

# Loop através dos quadros do vídeo
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Roda o tracker interno do YOLO
    result = model.track(frame, persist=True, classes=[CUP_YOLO_ID], tracker="custom_tracker.yml", verbose=False)[0]

    current_yolo_ids = set()
    current_yolo_id_positions = {}
    current_yolo_id_boxes = {}

    if result.boxes and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu().numpy()
        yolo_ids = result.boxes.id.int().cpu().tolist()
        current_yolo_ids = set(yolo_ids)
        for box, yolo_id in zip(boxes, yolo_ids):
            current_yolo_id_positions[yolo_id] = (box[0], box[1])
            current_yolo_id_boxes[yolo_id] = (box[0], box[1], box[2], box[3])

    # ===== TRACKING DA BOLA =====

    # converte frame atual para HSV e encontra mascara de cor da bola
    imgColor, mask = myColorFinder.update(frame, hsvVals)
    imgContours, contours = cvzone.findContours(frame, mask, minArea=300)

    # se encontrou a bola, atualiza o histórico e desenha
    if contours:
        center_x, center_y = contours[0]['center']
        x, y, w, h = contours[0]['bbox']
        ball_pos = (center_x, center_y)
        last_ball_pos = ball_pos # guarda ultima posição conhecida da bola
        last_ball_bbox = (x, y, w, h)

        ball_track_history.append(ball_pos)

        # Se bola reapareceu: restaura todos copos para normal
        for k in cup_has_ball:
            cup_has_ball[k] = False

        # desenha bounding box e o texto
        cv2.rectangle(frame, (x, y), (x + w, y + h), ball_color, 2)
        cv2.putText(frame, "Bola", (center_x + 20, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ball_color, 2)

        # desenha a trilha
        if len(ball_track_history) > 1:
            points = np.array(ball_track_history, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=ball_color, thickness=2)
    else:
        # bola desaparecida
        if last_ball_bbox:
            bx, by, bw, bh = last_ball_bbox
            ball_rect = [bx, by, bx + bw, by + bh]
            best_display_id = None
            best_intersection = 0

            for display_id, yolo_id in display_id_to_yolo_id.items():
                if yolo_id in current_yolo_id_positions:
                    cx, cy, cw, ch = current_yolo_id_boxes[yolo_id]
                    cup_rect = [int(cx - cw / 2), int(cy - ch / 2), int(cx + cw / 2), int(cy + ch / 2)]

                    # Calcula interseção
                    x_left = max(ball_rect[0], cup_rect[0])
                    y_top = max(ball_rect[1], cup_rect[1])
                    x_right = min(ball_rect[2], cup_rect[2])
                    y_bottom = min(ball_rect[3], cup_rect[3])

                    if x_right < x_left or y_bottom < y_top:
                        continue

                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    if intersection_area > best_intersection:
                        best_intersection = intersection_area
                        best_display_id = display_id

            # Marca somente o melhor copo
            for k in cup_has_ball:
                cup_has_ball[k] = False
            if best_display_id is not None and best_intersection / (bw * bh) > 0.5:
                cup_has_ball[best_display_id] = True


    # ===== LÓGICA DE MAPEAMENTO E Re-ID =====

    # 1. Identificar tracks perdidos e novos
    active_yolo_ids = set(yolo_id_to_display_id.keys())
    lost_yolo_ids = active_yolo_ids - current_yolo_ids
    new_yolo_ids = current_yolo_ids - active_yolo_ids

    # 2. Mover slots dos tracks perdidos para a lista de "lost_slots"
    for yolo_id in lost_yolo_ids:
        display_id = yolo_id_to_display_id[yolo_id]
        # Precisamos da última posição, que deveríamos ter guardado no frame anterior.
        # Por simplicidade, vamos assumir que o último ponto no histórico é a última posição.
        last_pos = track_history[display_id][-1] if track_history[display_id] else (0,0)

        lost_slots[display_id] = {'last_pos': last_pos, 'age': 0}

        # Remove o mapeamento antigo
        del display_id_to_yolo_id[display_id]
        del yolo_id_to_display_id[yolo_id]

    # 3. Tentativa de Re-ID para tracks novos
    reidentified_new_ids = set()
    if lost_slots and new_yolo_ids:
        for yolo_id in new_yolo_ids:
            new_pos = current_yolo_id_positions[yolo_id]

            min_dist = float('inf')
            best_match_display_id = -1

            for display_id, data in lost_slots.items():
                dist = np.linalg.norm(np.array(new_pos) - np.array(data['last_pos']))
                if dist < min_dist:
                    min_dist = dist
                    best_match_display_id = display_id

            # Se a correspondência for boa o suficiente, re-identifica!
            if best_match_display_id != -1 and min_dist < REID_DISTANCE_THRESHOLD:
                display_id = best_match_display_id

                # Cria o novo mapeamento
                yolo_id_to_display_id[yolo_id] = display_id
                display_id_to_yolo_id[display_id] = yolo_id

                # Remove da lista de perdidos e marca o yolo_id como tratado
                del lost_slots[display_id]
                reidentified_new_ids.add(yolo_id)

    # 4. Atribuir IDs para tracks verdadeiramente novos
    remaining_new_ids = new_yolo_ids - reidentified_new_ids
    for yolo_id in remaining_new_ids:
        if available_display_ids:
            display_id = available_display_ids.pop()

            yolo_id_to_display_id[yolo_id] = display_id
            display_id_to_yolo_id[display_id] = yolo_id

    # 5. Gerenciar e limpar slots perdidos por muito tempo
    slots_to_purge = []
    for display_id, data in lost_slots.items():
        data['age'] += 1
        if data['age'] > MAX_AGE:
            slots_to_purge.append(display_id)
            available_display_ids.add(display_id) # Libera o ID para reutilização
    for display_id in slots_to_purge:
        del lost_slots[display_id]

    # 6. VISUALIZAÇÃO usando os IDs MAPEADOS
    if result.boxes and result.boxes.is_track:
        for box, yolo_id in zip(boxes, yolo_ids):
            # Se o ID do YOLO não tem um display_id, ele é ignorado (ou é novo demais)
            if yolo_id not in yolo_id_to_display_id:
                continue

            display_id = yolo_id_to_display_id[yolo_id]

            x, y, w, h = box
            track = track_history[display_id]
            track.append((float(x), float(y)))

            # Desenha o bounding box com o ID estável
            x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)

            if cup_has_ball.get(display_id, False):
                color = COLOR_BALL
                label = f"XCopoX {display_id}:{yolo_id}"
            else:
                color = COLOR_CUPS.get(display_id, (200, 200, 200))
                label = f"Copo {display_id}:{yolo_id}"

            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Desenha as trilhas
            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

    cv2.imshow("Jogo dos copos", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libera os recursos
print("Encerrando")
cap.release()
out.release()
cv2.destroyAllWindows()
