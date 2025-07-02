from collections import defaultdict, deque
import random
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# --- PARÂMETROS DE CONTROLE DO MAPEAMENTO ---
FIXED_NUM_CUPS = None      # Número de "slots" de copos que queremos rastrear
MAX_AGE = 10            # Nº de quadros para um slot ser considerado "perdido" e elegível para Re-ID
REID_COST_THRESHOLD = 0.8    # Limiar de custo combinado para aceitar uma Re-ID
# Pesos para a função de custo da Re-ID
REID_DISTANCE_WEIGHT = 0.7
REID_MOTION_WEIGHT = 0.3

TRACK_HISTORY_LEN = 70 # Tamanho máximo do histórico de um track
CUP_YOLO_ID = 41
APPLE_YOLO_ID = 47

# --- FUNÇÕES AUXILIARES ---

def calculate_average_velocity(history, num_points=5):
    """Calcula o vetor de velocidade média dos últimos `num_points` da trajetória."""
    if len(history) < 2:
        return np.array([0, 0])

    # Pega os últimos pontos, mas no máximo `num_points`
    points_to_consider = min(len(history), num_points)
    recent_history = list(history)[-points_to_consider:]

    velocities = []
    for i in range(1, len(recent_history)):
        p1 = np.array(recent_history[i-1])
        p2 = np.array(recent_history[i])
        velocity = p2 - p1
        velocities.append(velocity)

    if not velocities:
        return np.array([0, 0])

    return np.mean(velocities, axis=0)

# --- SCRIPT PRINCIPAL ---

# Carrega o modelo YOLO
model = YOLO("yolo11n.pt")
model_apple = YOLO("yolo11n.pt")

# Abre o vídeo de entrada
INPUT_VIDEO = "video1.mp4"
# INPUT_VIDEO = 0
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
available_display_ids = None

# Dicionário para "slots" perdidos, guardando a última posição e idade
lost_slots = {}  # Formato: {display_id: {'last_pos': (x,y), 'age': frames}}

# Histórico da trilha e cores, agora usando nosso display_id estável
track_history = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LEN))
track_colors = {}

# Apple vars
apple_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
apple_cup = -1

def Track(frame):
    global apple_cup

    # Roda o tracker interno do YOLO
    result = model.track(frame, persist=True, classes=[CUP_YOLO_ID], tracker="custom_tracker.yml", verbose=False)[0]
    apple_result = model_apple(frame, classes=[APPLE_YOLO_ID], verbose=False)[0]
    apple_tensors = [r for r in apple_result.boxes if r.conf[0].item() > 0.2]

    current_yolo_ids = set()
    current_yolo_id_positions = {}
    if result.boxes and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu().numpy()
        yolo_ids = result.boxes.id.int().cpu().tolist()
        current_yolo_ids = set(yolo_ids)
        for box, yolo_id in zip(boxes, yolo_ids):
            current_yolo_id_positions[yolo_id] = (box[0], box[1])

    # --- LÓGICA DE MAPEAMENTO E Re-ID ---

    active_yolo_ids = set(yolo_id_to_display_id.keys())
    lost_yolo_ids = active_yolo_ids - current_yolo_ids
    new_yolo_ids = current_yolo_ids - active_yolo_ids

    # 2. Mover slots dos tracks perdidos para a lista de "lost_slots"
    for yolo_id in lost_yolo_ids:
        display_id = yolo_id_to_display_id[yolo_id]
        history = track_history[display_id]
        if history:
            last_pos = history[-1]
            avg_velocity = calculate_average_velocity(history)
            lost_slots[display_id] = {'last_pos': last_pos, 'velocity': avg_velocity, 'age': 0}
        del display_id_to_yolo_id[display_id]
        del yolo_id_to_display_id[yolo_id]

    # 3. Tentativa de Re-ID por Movimento e Proximidade (com Algoritmo Húngaro)
    reidentified_new_ids = set()
    if lost_slots and new_yolo_ids:
        lost_display_ids = list(lost_slots.keys())
        new_yolo_ids_list = list(new_yolo_ids)

        cost_matrix = np.ones((len(lost_display_ids), len(new_yolo_ids_list)))

        for i, display_id in enumerate(lost_display_ids):
            slot_data = lost_slots[display_id]
            predicted_pos = np.array(slot_data['last_pos']) + slot_data['velocity']
            hist_velocity = slot_data['velocity']

            for j, yolo_id in enumerate(new_yolo_ids_list):
                new_pos = np.array(current_yolo_id_positions[yolo_id])

                # Custo de Distância
                dist = np.linalg.norm(predicted_pos - new_pos)
                cost_dist = dist / (frame_width / 4) # Normaliza a distância

                # Custo de Direção
                reid_vector = new_pos - np.array(slot_data['last_pos'])
                # Similaridade de Cosseno entre vetores de velocidade
                norm_hist_v = np.linalg.norm(hist_velocity)
                norm_reid_v = np.linalg.norm(reid_vector)

                if norm_hist_v > 0 and norm_reid_v > 0:
                    cosine_sim = np.dot(hist_velocity, reid_vector) / (norm_hist_v * norm_reid_v)
                    cost_dir = (1 - cosine_sim) # Varia de 0 (mesma direção) a 2 (oposta)
                else:
                    cost_dir = 1.0 # Custo neutro se não houver movimento

                # Custo Combinado
                cost_matrix[i, j] = (REID_DISTANCE_WEIGHT * cost_dist) + (REID_MOTION_WEIGHT * cost_dir)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < REID_COST_THRESHOLD:
                display_id = lost_display_ids[r]
                yolo_id = new_yolo_ids_list[c]

                yolo_id_to_display_id[yolo_id] = display_id
                display_id_to_yolo_id[display_id] = yolo_id

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
            available_display_ids.add(display_id)
    for display_id in slots_to_purge:
        del lost_slots[display_id]

    # 6. VISUALIZAÇÃO
    if result.boxes and result.boxes.is_track:
        # ... (código de visualização idêntico ao anterior, ele já usa o display_id) ...
        for box, yolo_id in zip(boxes, yolo_ids):
            if yolo_id not in yolo_id_to_display_id: continue
            display_id = yolo_id_to_display_id[yolo_id]
            x, y, w, h = box
            track = track_history[display_id]
            track.append((float(x), float(y)))
            if display_id not in track_colors: track_colors[display_id] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            color = track_colors[display_id]
            x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Copo {display_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

            # Desenhar onde esta a apple
            if apple_cup == display_id:
                cv2.arrowedLine(frame, (x1, y1-20), (x1, y1-2), color=(0, 255, 0), thickness=3)


    # 7. Assciar maça com copo mais perto
    if apple_tensors:
        # Primeira e unica maça
        x, y, w, h = map(int, apple_tensors[0].xywh[0])
        apple_center = np.array((x, y))
        better_dist = float('inf')
        for box, yolo_id in zip(boxes, yolo_ids):
            if yolo_id not in yolo_id_to_display_id:
                continue
            display_id = yolo_id_to_display_id[yolo_id]
            x, y, w, h = box
            dist = np.linalg.norm(np.array((x, y)) - apple_center)
            if dist < better_dist:
                better_dist = dist
                apple_cup = display_id

    # 8. VISUALIZAÇÃO da "Bola/Apple"
    for tensor in apple_tensors:
        x1, y1, x2, y2 = map(int, tensor.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), apple_color, 2)
        cv2.putText(frame, f"Apple {tensor.conf[0].item()}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, apple_color, 2)

    return frame

def GetNumCups(frame):
    # 1. Pegar os copos em cena
    result = model(frame, classes=[CUP_YOLO_ID], verbose=False)[0]
    tensors = [r for r in result.boxes if r.conf[0].item() > 0.2]
    num_cups = len(tensors)

    for tensor in tensors:
        x1, y1, x2, y2 = map(int, tensor.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"Cup {tensor.conf[0].item():.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, apple_color, 2)

    cv2.putText(frame, f"Num cups: {num_cups}", (0, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return (frame, num_cups)


# Variaveis de controle do trabalho
start = False
start_counter = 0
# Loop através dos quadros do víde
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    if start and start_counter <= 10:
        frame = Track(frame)
    else:
        frame, FIXED_NUM_CUPS = GetNumCups(frame)
        available_display_ids = set(range(FIXED_NUM_CUPS))
        if start:
            start_counter += 1

    cv2.imshow("Jogo dos copos", frame)
    out.write(frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("s"):
        start = True

# Libera os recursos
print("Encerrando")
cap.release()
out.release()
cv2.destroyAllWindows()