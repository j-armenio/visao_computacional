from collections import defaultdict
import random
import cv2
import numpy as np

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Open the video file
INPUT_VIDEO = "video1.mp4"
OUTPUT_VIDEO = "output_tracking_with_trails_v1.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"Erro: não consegui abrir o vídeo {INPUT_VIDEO}")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

# Store the track history
track_history = defaultdict(lambda: [])
track_colors = {}  # ID -> cor

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        result = model.track(frame, persist=True, classes=[41], tracker="custom_tracker.yml", verbose=False)[0]

        # Get the boxes and track IDs
        if result.boxes and result.boxes.is_track:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()

            # Visualize the result on the frame
            frame = result.plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point

                if track_id not in track_colors:
                    track_colors[track_id] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )

                color = track_colors[track_id]
                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", frame)
        out.write(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
print("Encerrando")
cap.release()
out.release()
cv2.destroyAllWindows()