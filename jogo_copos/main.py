import cv2
import numpy as np
import os
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from collections import deque

# --- PARÂMETROS DE CONFIGURAÇÃO PARA Re-ID ---
VIDEO_SOURCE = "media/video1-2.mp4"
WEBCAM_SOURCE = 0
# VIDEO_OUTPUT = f"output_{VIDEO_SOURCE}"
VIDEO_OUTPUT = f"output_v5_output.mp4"
MODEL_PATH = 'yolo11n.pt'
CONFIDENCE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.3
HISTORY_LENGTH = 50
FIXED_NUM_CUPS = 3

# Parâmetros de ciclo de vida e Re-ID
MAX_AGE = 5                # Quadros para um tracker ativo ser considerado "perdido"
MAX_LOST_AGE = 60           # Quadros para um tracker perdido ser permanentemente excluído
REID_COLOR_THRESHOLD = 0.7  # Limiar de similaridade de histograma (1.0 = idêntico)
REID_DISTANCE_THRESHOLD = 150 # Distância máxima em pixels para considerar uma re-identificação

# Parâmetros para reconhecer a bola
LOWER_ORANGE = np.array([10, 160, 150])
UPPER_ORANGE = np.array([22, 255, 255])

def detect_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 10:  # evita ruído
            cx = x + w // 2
            cy = y + h // 2
            return (x, y, x + w, y + h), (cx, cy)
    return None, None

def iou_batch(bboxes1, bboxes2):
    """Calcula a IoU entre dois conjuntos de bounding boxes."""
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    inter_x1 = np.maximum(bboxes1[:, 0][:, np.newaxis], bboxes2[:, 0])
    inter_y1 = np.maximum(bboxes1[:, 1][:, np.newaxis], bboxes2[:, 1])
    inter_x2 = np.minimum(bboxes1[:, 2][:, np.newaxis], bboxes2[:, 2])
    inter_y2 = np.minimum(bboxes1[:, 3][:, np.newaxis], bboxes2[:, 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = area1[:, np.newaxis] + area2 - inter_area
    iou = inter_area / (union_area + 1e-6)
    return iou

def get_color_histogram(frame, bbox):
    """Calcula o histograma de matiz (Hue) normalizado para a região do bbox."""
    x1, y1, x2, y2 = map(int, bbox)
    # Garante que as coordenadas não saiam dos limites do frame
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x1 >= x2 or y1 >= y2:
        return np.zeros(16) # Retorna histograma vazio se o bbox for inválido

    roi = frame[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Calcula o histograma para o canal Hue (matiz)
    hist = cv2.calcHist([hsv_roi], [0], None, [16], [0, 180])
    # Normaliza o histograma para que a comparação seja consistente
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()

class KalmanTracker:
    def __init__(self, bbox, track_id, initial_hist):
        self.id = track_id
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # ... (configuração do Kalman igual à anterior) ...
        self.kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.kf.R *= 10.
        self.kf.P[2:,2:] *= 1000.
        self.kf.Q[2:,2:] *= 0.01

        # Matriz de Transição de Estado (F)
        # Descreve como o estado evolui de t-1 para t sem controle externo
        # x_t = x_{t-1} + vx_{t-1} * dt
        # y_t = y_{t-1} + vy_{t-1} * dt
        # vx_t = vx_{t-1}
        # vy_t = vy_{t-1}
        # (assumindo dt=1 frame)
        self.kf.F = np.array([[1, 0, 1, 0],
                               [0, 1, 0, 1],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

        # Matriz de Medição (H)
        # Converte o estado para o espaço de medição
        # Medimos apenas a posição (x, y)
        self.kf.H = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]])

        # Covariância do Ruído da Medição (R)
        # Incerteza da nossa detecção (YOLO). Ajustável.
        self.kf.R[2:, 2:] *= 10.
        self.kf.R = np.eye(2) * 10

        # Covariância da Incerteza do Estado (P)
        # Incerteza inicial sobre o estado. Começamos com alta incerteza.
        self.kf.P[2:, 2:] *= 1000.
        self.kf.P *= 10.

        # Covariância do Ruído do Processo (Q)
        # Incerteza no modelo de movimento (ex: aceleração não modelada).
        self.kf.Q[2:, 2:] *= 0.01
        self.kf.Q = np.eye(4) * 1

        # Estado inicial (x)
        # Extrai o centroide do bbox inicial
        self.kf.x[:2] = self.bbox_to_centroid(bbox).reshape((2, 1))

        self.bbox = bbox
        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        self.history = deque(maxlen=HISTORY_LENGTH)
        self.histogram = initial_hist # ### NOVO ###: Armazena a assinatura de cor
        self.lost_age = 0 # ### NOVO ###: Contador para quando o tracker está na lista de perdidos

    # ... (métodos bbox_to_centroid e predict iguais) ...
    @staticmethod
    def bbox_to_centroid(bbox): return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
    def predict(self):
        self.kf.predict(); self.age += 1; self.time_since_update += 1
        predicted_centroid = self.kf.x[:2].flatten()
        w = self.bbox[2] - self.bbox[0]; h = self.bbox[3] - self.bbox[1]
        return np.array([predicted_centroid[0] - w/2, predicted_centroid[1] - h/2, predicted_centroid[0] + w/2, predicted_centroid[1] + h/2])

    def update(self, bbox, hist):
        self.bbox = bbox
        self.histogram = hist # Atualiza a assinatura de cor
        self.time_since_update = 0
        self.hits += 1
        measurement = self.bbox_to_centroid(bbox)
        self.kf.update(measurement)
        self.history.append(tuple(measurement.astype(int)))

def main():
    model = YOLO(MODEL_PATH)

    opt = input("Selecione fonte do vídeo: \n1 - Câmera em tempo real \n2 - Vídeo gravado\n")
    if opt == "1":
        INPUT_VIDEO = WEBCAM_SOURCE
    elif opt == "2":
        INPUT_VIDEO = VIDEO_SOURCE
        if not os.path.exists(INPUT_VIDEO):
            print("Erro: video nao encontrado")
            exit(1)
    else:
        print("Opção inválida.")
        exit(1)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo")
        exit()

    # --- Definir parâmetros do vídeo de saída ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

    active_trackers = []
    lost_trackers = [] # ### NOVO ###: Lista para trackers perdidos
    available_ids = set(range(FIXED_NUM_CUPS)) # Pool de IDs
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(100)]

    ball_track = deque(maxlen=10)
    ball_missing_counter = 0
    MAX_MISSING_FRAMES = 8
    last_ball_bbox = None
    cup_with_ball_id = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        cv2.waitKey(15)

        # SEGMENTAÇÃO DA BOLA
        ball_bbox, ball_center = detect_ball(frame)
        if ball_bbox:
            x1, y1, x2, y2 = ball_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(frame, "Bola", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

        if ball_center:
            ball_track.append(ball_center)
            # Desenhar a trilha da bola
            for i in range(1, len(ball_track)):
                if ball_track[i - 1] is None or ball_track[i] is None:
                    continue
                cv2.line(frame, ball_track[i - 1], ball_track[i], (0, 140, 255), 2)

        # 1. DETECÇÃO E EXTRAÇÃO DE FEATURES
        results = model(frame, verbose=False)[0]
        detections_data = [r for r in results.boxes.data.tolist() if r[4] > CONFIDENCE_THRESHOLD and int(r[5]) == 41]

        detections_bboxes = np.array([d[:4] for d in detections_data]) if detections_data else np.empty((0, 4))
        detections_hists = np.array([get_color_histogram(frame, d[:4]) for d in detections_data]) if detections_data else np.empty((0, 16))

        # 2. ASSOCIAÇÃO PRIMÁRIA (IoU com trackers ativos)
        predicted_bboxes = np.array([t.predict() for t in active_trackers]) if active_trackers else np.empty((0, 4))
        iou_matrix = iou_batch(predicted_bboxes, detections_bboxes) if len(predicted_bboxes) > 0 and len(detections_bboxes) > 0 else np.empty((0,0))

        if iou_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(1 - iou_matrix)
            # Associações e atualizações
            matched_trackers_idx = set()
            matched_detections_idx = set()
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= IOU_THRESHOLD:
                    active_trackers[r].update(detections_bboxes[c], detections_hists[c])
                    matched_trackers_idx.add(r)
                    matched_detections_idx.add(c)
        else:
            matched_trackers_idx = set()
            matched_detections_idx = set()

        # 3. GERENCIAR TRACKERS ATIVOS NÃO ASSOCIADOS -> Mover para "perdidos"
        unmatched_trackers = set(range(len(active_trackers))) - matched_trackers_idx
        for idx in sorted(list(unmatched_trackers), reverse=True):
            if active_trackers[idx].time_since_update > MAX_AGE:
                lost_trackers.append(active_trackers.pop(idx))

        # 4. ASSOCIAÇÃO SECUNDÁRIA (Re-ID com trackers perdidos)
        unmatched_detections_idx = set(range(len(detections_bboxes))) - matched_detections_idx
        reid_detections_bboxes = detections_bboxes[list(unmatched_detections_idx)]
        reid_detections_hists = detections_hists[list(unmatched_detections_idx)]

        if len(lost_trackers) > 0 and len(reid_detections_bboxes) > 0:
            # Construir matriz de custo para Re-ID
            reid_cost_matrix = np.ones((len(lost_trackers), len(reid_detections_bboxes)))
            for i, tracker in enumerate(lost_trackers):
                for j, hist in enumerate(reid_detections_hists):
                    hist_sim = cv2.compareHist(tracker.histogram, hist, cv2.HISTCMP_CORREL)
                    dist = np.linalg.norm(tracker.bbox_to_centroid(tracker.bbox) - tracker.bbox_to_centroid(reid_detections_bboxes[j]))

                    # Se a detecção estiver muito longe OU a cor for muito diferente, o custo é alto (ignora)
                    if dist > REID_DISTANCE_THRESHOLD or hist_sim < REID_COLOR_THRESHOLD:
                        reid_cost_matrix[i,j] = 1.0
                    else:
                        # Custo é uma combinação de (1 - similaridade_cor) e (distancia_normalizada)
                        reid_cost_matrix[i,j] = (1 - hist_sim) + (dist / REID_DISTANCE_THRESHOLD)

            row_ind, col_ind = linear_sum_assignment(reid_cost_matrix)

            # Reativar trackers
            detections_to_remove_from_reid = set()
            matches_to_process = []

            for r, c in zip(row_ind, col_ind):
                if reid_cost_matrix[r, c] < 0.9: # Um limiar de custo combinado
                    matches_to_process.append((r, c))

            for r, c in sorted(matches_to_process, key=lambda x: -x[0]):
                    lost_tracker = lost_trackers.pop(r)
                    original_detection_idx = list(unmatched_detections_idx)[c]

                    lost_tracker.update(detections_bboxes[original_detection_idx], detections_hists[original_detection_idx])
                    active_trackers.append(lost_tracker)
                    detections_to_remove_from_reid.add(original_detection_idx)

            unmatched_detections_idx -= detections_to_remove_from_reid

        # 5. CRIAR TRACKERS VERDADEIRAMENTE NOVOS
        for idx in unmatched_detections_idx:
            if available_ids:
                new_id = available_ids.pop()
                new_tracker = KalmanTracker(detections_bboxes[idx], new_id, detections_hists[idx])
                active_trackers.append(new_tracker)

        # 6. EXCLUIR PERMANENTEMENTE TRACKERS PERDIDOS HÁ MUITO TEMPO
        lost_to_keep = []
        for tracker in lost_trackers:
            tracker.lost_age += 1
            if tracker.lost_age < MAX_LOST_AGE:
                lost_to_keep.append(tracker)
            else:
                available_ids.add(tracker.id) # Devolve o ID ao pool
        lost_trackers = lost_to_keep

        if ball_bbox:
            # Bola visível
            last_ball_bbox = ball_bbox
            ball_missing_counter = 0
            cup_with_ball_id = None  # Reset se a bola reapareceu
        else:
            # Bola sumida
            ball_missing_counter += 1

            if last_ball_bbox and ball_missing_counter == MAX_MISSING_FRAMES:
                # Só tenta atribuir depois de N frames sumida
                bx, by, bw, bh = last_ball_bbox[0], last_ball_bbox[1], last_ball_bbox[2] - last_ball_bbox[0], last_ball_bbox[3] - last_ball_bbox[1]
                ball_rect = [bx, by, bx + bw, by + bh]
                best_tracker_id = None
                best_intersection = 0

                for tracker in active_trackers:
                    cx1, cy1, cx2, cy2 = tracker.bbox
                    cup_rect = [int(cx1), int(cy1), int(cx2), int(cy2)]

                    x_left = max(ball_rect[0], cup_rect[0])
                    y_top = max(ball_rect[1], cup_rect[1])
                    x_right = min(ball_rect[2], cup_rect[2])
                    y_bottom = min(ball_rect[3], cup_rect[3])

                    if x_right < x_left or y_bottom < y_top:
                        continue

                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    if intersection_area > best_intersection:
                        best_intersection = intersection_area
                        best_tracker_id = tracker.id

                if best_tracker_id is not None and best_intersection / (bw * bh) > 0.5:
                    cup_with_ball_id = best_tracker_id


        # 7. VISUALIZAÇÃO
        for tracker in active_trackers:
            # ... (código de visualização igual ao anterior) ...
            color = colors[tracker.id % len(colors)]
            if tracker.time_since_update == 0:
                x1, y1, x2, y2 = map(int, tracker.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"Copo {tracker.id}"
                if tracker.id == cup_with_ball_id:
                    cv2.arrowedLine(frame, (x1, y1 - 20), (x1, y1 - 2), color=(0, 0, 255), thickness=3)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                
            for i in range(1, len(tracker.history)):
                cv2.line(frame, tracker.history[i-1], tracker.history[i], color, 2)

        cv2.imshow("Jogo dos copos", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()