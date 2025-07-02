import cv2
import random
from collections import defaultdict
from ultralytics import YOLO  # ou from ultralytics import YOLO

# ===== CONFIGURAÇÕES =====
MODEL_PATH = "yolo11n.pt"
INPUT_VIDEO = "video3.mp4"
OUTPUT_VIDEO = "output_tracking_with_trails.mp4"
CUP_CLASS_ID = 41  # Classe do copo no COCO

# ===== 1) Carregar modelo =====
model = YOLO(MODEL_PATH)

# ===== 2) Vídeo =====
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"Erro: não consegui abrir o vídeo {INPUT_VIDEO}")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

# ===== 3) Histórico de trilhas e cores =====
tracks_history = defaultdict(list)
track_colors = {}  # ID -> cor

print("[INFO] Tracking com trilhas coloridas em andamento...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO track embutido
    # results = model.track(frame, classes=[CUP_CLASS_ID], persist=True, verbose=False)
    results = model.track(frame, classes=[CUP_CLASS_ID], persist=True, verbose=False)

    if results.boxes and results.boxes.is_track:
        frame = results.plot()

    for result in results:
        for box in result.boxes:
            track_id = int(box.id[0]) if box.id is not None else -1
            if track_id == -1:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Atualiza histórico
            tracks_history[track_id].append((cx, cy))

            # Gera cor única se ainda não existir
            if track_id not in track_colors:
                track_colors[track_id] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )

            color = track_colors[track_id]

            # # Desenha bbox
            # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # label = f"Copo ID: {track_id}"
            # cv2.putText(frame, label, (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Desenha trilhas
    for track_id, points in tracks_history.items():
        color = track_colors[track_id]
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], color, 2)
        if points:
            cv2.circle(frame, points[-1], 4, color, -1)

    cv2.imshow("Tracking Copos com Trilhas Coloridas", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== 4) Encerrar =====
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[INFO] Tracking finalizado. Vídeo salvo em {OUTPUT_VIDEO} 🚀🍹")
