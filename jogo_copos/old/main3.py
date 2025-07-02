import cv2
from ultralytics import YOLO

# --- Parâmetros ---
input_video_path = 'video1.mp4'   # Caminho do vídeo de entrada
output_video_path = 'output.mp4' # Caminho do vídeo de saída
model_path = 'yolo11n.pt'        # Você pode usar o modelo padrão YOLOv8 Nano

# --- Carregar modelo YOLO ---
model = YOLO(model_path)

# --- Abrir vídeo ---
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Erro ao abrir o vídeo")
    exit()

# --- Definir parâmetros do vídeo de saída ---
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

points = []

# --- Loop de leitura ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rodar a detecção
    results = model.predict(frame)  # Ajuste o threshold se quiser

    # Pegar resultados e desenhar bounding boxes
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = box.conf[0].item()
            label = model.names[cls_id]
            if label != "cup":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # centro da bbox

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            points.append((cx, cy))
            if len(points) > 50:
                points.pop(0)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for cx, cy in points:
        cv2.circle(frame, (cx, cy), 1, (0, 255, 0), 2)

    # Mostrar em tempo real
    cv2.imshow('YOLO', frame)

    # Salvar no arquivo de saída
    out.write(frame)

    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Liberação ---
cap.release()
out.release()
cv2.destroyAllWindows()

