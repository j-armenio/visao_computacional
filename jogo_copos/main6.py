from collections import defaultdict, deque
import random
import cv2
import numpy as np
from ultralytics import YOLO

# --- PARÂMETROS DE CONTROLE DO MAPEAMENTO ---
FIXED_NUM_CUPS = 3      # Número de "slots" de copos que queremos rastrear
MAX_AGE = 50            # Nº de quadros para um slot ser considerado "perdido" e elegível para Re-ID
REID_DISTANCE_THRESHOLD = 100 # Distância máxima em pixels para re-identificar um copo
TRACK_HISTORY_LEN = 70 # Tamanho máximo do histórico de um track
CUP_YOLO_ID = 41

# --- SCRIPT PRINCIPAL ---

# Carrega o modelo YOLO
model = YOLO("yolo11n.pt")

# Abre o vídeo de entrada
# INPUT_VIDEO = "video1.mp4"
INPUT_VIDEO = 0
OUTPUT_VIDEO = f"output_v6_camera.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"Erro: não consegui abrir o vídeo {INPUT_VIDEO}")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

# --- VARIÁVEIS DE GERENCIAMENTO DE ESTADO ---
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
ball_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
ball_track_history = deque(maxlen=TRACK_HISTORY_LEN)

# Loop através dos quadros do vídeo
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Roda o tracker interno do YOLO
    result = model.track(frame, persist=True, classes=[CUP_YOLO_ID], tracker="custom_tracker.yml", verbose=False)[0]
    # result_ball = model.track(frame, persist=True, classes=[32], tracker="custom_tracker.yml", verbose=False)[0]

    current_yolo_ids = set()
    current_yolo_id_positions = {}

    if result.boxes and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu().numpy()
        yolo_ids = result.boxes.id.int().cpu().tolist()
        current_yolo_ids = set(yolo_ids)
        for box, yolo_id in zip(boxes, yolo_ids):
            current_yolo_id_positions[yolo_id] = (box[0], box[1])

    # --- LÓGICA DE MAPEAMENTO E Re-ID ---

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

            if display_id not in track_colors:
                track_colors[display_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            color = track_colors[display_id]

            # Desenha o bounding box com o ID estável
            x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Copo {display_id}:{yolo_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Desenha as trilhas
            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)


    # if result_ball.boxes and result_ball.boxes.is_track:
    #     boxes = result_ball.boxes.xywh.cpu().numpy()
    #     yolo_ids = result_ball.boxes.id.int().cpu().tolist()

    # if result_ball.boxes and result_ball.boxes.is_track:
    #     for box, yolo_id in zip(boxes, yolo_ids):
    #         x, y, w, h = box
    #         track = ball_track_history
    #         track.append((float(x), float(y)))

    #         # Desenha o bounding box com o ID estável
    #         x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), ball_color, 2)
    #         cv2.putText(frame, f"Bola {yolo_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ball_color, 2)

    #         # Desenha as trilhas
    #         points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
    #         cv2.polylines(frame, [points], isClosed=False, color=ball_color, thickness=2)

    cv2.imshow("Jogo dos copos", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libera os recursos
print("Encerrando")
cap.release()
out.release()
cv2.destroyAllWindows()
