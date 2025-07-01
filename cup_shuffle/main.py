from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import os
import random

# ==== CONFIGURACAO ====
MODEL = "yolov8n.pt" # yolov8n.pt , yolo11n.pt
VIDEO_PATH = "media/cup_shuffle3.mp4"

CUP_CLASS_ID = 41
MAX_CUPS = 3
MAX_BALLS = 1
TOTAL_IDS = MAX_BALLS + MAX_CUPS

FIXED_IDS = list(range(TOTAL_IDS))   # id 0, 1, 2, 3
object_id_assignments = {}           # track_id do deepsort -> fixed_id
fixed_id_to_class = {}               # fixed_id -> 'cup'/'ball'
fixed_id_colors = {i: (random.randint(0,255), random.randint(0,255)) for i in FIXED_IDS}
fixed_id_positions = {}              # fixed_id -> ultima posicao (x,y)

opt = input("Selecione fonte do vídeo: \n1 - Câmera em tempo real \n2 - Vídeo gravado\n")

if opt == "1":
    VIDEO_PATH = 0
elif opt == "2":
    VIDEO_PATH = "media/cup_shuffle3.mp4"
    if not os.path.exists(VIDEO_PATH):
        print("Erro: video nao encontrado")
        exit(1)
else:
    print("Opção inválida.")
    exit(1)

video = cv2.VideoCapture(VIDEO_PATH)
model = YOLO(MODEL)
tracker = DeepSort(max_age=30)

if not video.isOpened():
    print("Erro ao abrir o video.")
    exit(1)

# ==== LOOP PRINCIPAL ====
while True:
    ret, frame = video.read()
    if not ret:
        break

    # aplica YOLO e pega deteccoes
    results = model(frame, verbose=False)[0]

    detections = []
    cup_count = 0
    ball_count = 0

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        # filtra apenas copos
        if cls == CUP_CLASS_ID and cup_count < MAX_CUPS:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "cup"))
            cup_count += 1

    # atualiza tracker com deteccoes do frame
    tracks = tracker.update_tracks(detections, frame=frame)

    used_fixed_ids = set(object_id_assignments.values())

    # desenha resultados
    for track in tracks:
        if not track.is_confirmed():
            continue

        ds_id = track.track_id
        cls_name = track.get_det_class()
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # atribuicao fixa
        if ds_id in object_id_assignments:
            fixed_id = object_id_assignments[ds_id]
        else:
            # Reidentificação manual baseada em distância
            best_match_id = None
            min_dist = float('inf')

            for fid in FIXED_IDS:
                if fid in fixed_id_positions and fid not in used_fixed_ids:
                    dist = np.linalg.norm(np.array([cx, cy]) - np.array(fixed_id_positions[fid]))
                    if dist < min_dist and dist < 100:  # limiar de reidentificação
                        min_dist = dist
                        best_match_id = fid

            if best_match_id is not None:
                fixed_id = best_match_id
            elif len(used_fixed_ids) < TOTAL_IDS:
                fixed_id = min(set(FIXED_IDS) - used_fixed_ids)
            else:
                continue  # Ignora se não há IDs livres

            object_id_assignments[ds_id] = fixed_id
            fixed_id_to_class[fixed_id] = cls_name

                # Atualiza posição
        fixed_id_positions[fixed_id] = (cx, cy)
        used_fixed_ids.add(fixed_id)

        # Desenho
        color = fixed_id_colors[fixed_id]
        label = f"{cls_name} {fixed_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # exibe o frame atual e sai ao pressionar 'q'
    cv2.imshow("Deteccao YOLO", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()