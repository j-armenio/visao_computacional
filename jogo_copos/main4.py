import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from collections import deque ### NOVO ###

# --- PARÂMETROS DE CONFIGURAÇÃO ---
VIDEO_SOURCE = "video1.mp4"  # 0 para webcam, ou "caminho/para/seu/video.mp4"
VIDEO_OUTPUT = f"output_{VIDEO_SOURCE}"  # Caminho para salvar video
MODEL_PATH = 'yolo11n.pt'  # Modelo YOLOv8 pré-treinado (n-nano é o mais rápido)
FIXED_NUM_CUPS = 3  # Quantidade fixa de copos a serem rastreados
CONFIDENCE_THRESHOLD = 0.1  # Confiança mínima para uma detecção ser válida
DISTANCE_THRESHOLD = 100 # Distância máxima (em pixels) para associar uma detecção a um tracker
HISTORY_LENGTH = 50 # ### NOVO ###: Quantos pontos do histórico manter para cada tracker

class KalmanTracker:
    """
    Esta classe representa um objeto rastreado.
    Agora com histórico de trajetória.
    """
    count = 0

    def __init__(self, bbox):
        """
        bbox: Bounding box inicial (x1, y1, x2, y2) da detecção.
        """
        self.id = KalmanTracker.count
        KalmanTracker.count += 1

        # O Filtro de Kalman
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kf.R = np.eye(2) * 10
        self.kf.P *= 10.
        self.kf.Q = np.eye(4) * 1
        self.kf.x[:2] = self.get_centroid(bbox).reshape((2, 1))

        self.bbox = bbox
        self.time_since_update = 0
        self.hits = 1
        self.age = 0

        ### NOVO ###
        # Histórico de centroides para desenhar a trajetória
        self.history = deque(maxlen=HISTORY_LENGTH)

    @staticmethod
    def get_centroid(bbox):
        """Calcula o centroide de um bounding box."""
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def predict(self):
        """
        Prevê o próximo estado do objeto.
        Retorna o centroide previsto (x, y).
        """
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        predicted_centroid = self.kf.x[:2].flatten()
        return predicted_centroid

    def update(self, bbox):
        """
        Atualiza o estado do filtro com uma nova medição.
        bbox: O novo bounding box detectado.
        """
        self.bbox = bbox
        measurement = self.get_centroid(bbox)
        self.kf.update(measurement)
        self.time_since_update = 0
        self.hits += 1

        ### NOVO ###
        # Adiciona o centroide atualizado ao histórico
        corrected_centroid = self.kf.x[:2].flatten().astype(int)
        self.history.append(tuple(corrected_centroid))


def main():
    # Carregar o modelo YOLOv8
    model = YOLO(MODEL_PATH)

    # Iniciar captura de vídeo
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo")
        exit()

    # --- Definir parâmetros do vídeo de saída ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

    trackers = []

    # Cores aleatórias para os trackers
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(FIXED_NUM_CUPS * 2)]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. DETECÇÃO
        results = model(frame, verbose=False)[0]
        detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if score > CONFIDENCE_THRESHOLD and int(class_id) == 41: # ID da classe 'cup'
                detections.append([int(x1), int(y1), int(x2), int(y2)])

        if not trackers and detections:
            for i in range(min(len(detections), FIXED_NUM_CUPS)):
                trackers.append(KalmanTracker(detections[i])) ### MODIFICADO ### (não muda o código, mas o objeto agora tem histórico)

        # 2. PREVISÃO
        predicted_centroids = []
        for t in trackers:
            predicted_centroids.append(t.predict())

        # 3. ASSOCIAÇÃO
        if predicted_centroids and detections:
            detected_centroids = [KalmanTracker.get_centroid(d) for d in detections]
            cost_matrix = np.zeros((len(predicted_centroids), len(detections)))
            for i, pred_c in enumerate(predicted_centroids):
                for j, det_c in enumerate(detected_centroids):
                    cost_matrix[i, j] = np.linalg.norm(pred_c - det_c)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # 4. ATUALIZAÇÃO
            matched_indices = []
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < DISTANCE_THRESHOLD:
                    trackers[r].update(detections[c])
                    matched_indices.append(c)

            unmatched_detections = [d for i, d in enumerate(detections) if i not in matched_indices]
            if len(trackers) < FIXED_NUM_CUPS:
                for d in unmatched_detections:
                    if len(trackers) < FIXED_NUM_CUPS:
                        trackers.append(KalmanTracker(d)) ### MODIFICADO ###

        # 5. VISUALIZAÇÃO
        for tracker in trackers:
            color = colors[tracker.id % len(colors)]

            # Desenha o bounding box apenas se o tracker foi atualizado recentemente
            if tracker.time_since_update == 0:
                x1, y1, x2, y2 = map(int, tracker.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Copo {tracker.id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Desenha o centroide previsto pelo Kalman
            pred_x, pred_y = map(int, tracker.kf.x[:2])
            cv2.circle(frame, (pred_x, pred_y), 4, color, -1)

            ### NOVO: Desenha a trajetória ###
            # Itera sobre o histórico de pontos e desenha linhas entre eles
            for i in range(1, len(tracker.history)):
                if tracker.history[i - 1] is None or tracker.history[i] is None:
                    continue
                # Desenha uma linha do ponto anterior para o ponto atual
                cv2.line(frame, tracker.history[i-1], tracker.history[i], color, 2)


        # Exibe o quadro
        cv2.imshow("Jogo dos Copos", frame)

        # Salva o quadro
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()