import numpy as np
import cv2
from time import time
from collections import deque
from ultralytics import YOLO
from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator
import torch

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device:", self.device)
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.5)

        self.track_memory = {}
        self.travel_time_memory = {}

        self.detection_history = deque()
        self.history_timestamps = deque()
        self.speed_history = deque()
        self.travel_history = deque()


        self.segment_length_m = 50
        self.segment_km = self.segment_length_m / 1000.0
        self.vmax_kmph = 60.0

        self.spi_time_window = 15 
        self.stuck_time_threshold = 10

    def load_model(self):
        model = YOLO("yolov12m.pt")
        model.fuse()
        return model

    def predict(self, frame):
        return self.model.track(frame, persist=True, tracker="bytetrack.yaml")

    def plot_bboxes(self, results, frame):
        current_detections = 0
        result = results[0]
        now = time()

        boxes = result.boxes
        if not boxes or boxes.id is None:
            return frame, current_detections

        conf = boxes.conf.to(self.device).cpu().numpy()
        xyxy = boxes.xyxy.to(self.device).cpu().numpy()
        class_id = boxes.cls.to(self.device).cpu().numpy().astype(int)
        track_ids = boxes.id.to(self.device).cpu().numpy().astype(int)

        valid_mask = (conf >= 0.3) & np.isin(class_id, [0, 1, 2, 3, 5, 7])

        if np.any(valid_mask):
            filtered_conf = conf[valid_mask]
            filtered_xyxy = xyxy[valid_mask]
            filtered_class_id = class_id[valid_mask]
            filtered_ids = track_ids[valid_mask]

            detections = Detections(
                xyxy=filtered_xyxy,
                confidence=filtered_conf,
                class_id=filtered_class_id,
                tracker_id=filtered_ids,
            )

            labels = [
                # f"{self.CLASS_NAMES_DICT[c]} ID:{tid} {conf:.2f}"
                # for c, conf, tid in zip(filtered_class_id, filtered_conf, filtered_ids)
                f"{self.CLASS_NAMES_DICT[c]} {conf:.2f}"
                for c, conf in zip(filtered_class_id, filtered_conf)
            ]
            frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=labels)

            for box, tid in zip(filtered_xyxy, filtered_ids):
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2

                if tid in self.track_memory:
                    prev_x, prev_y, prev_time = self.track_memory[tid]
                    dist_px = np.linalg.norm([cx - prev_x, cy - prev_y])
                    delta_t = now - prev_time
                    if delta_t > 0:
                        speed_px_per_sec = dist_px / delta_t
                        speed_mps = speed_px_per_sec * 0.2
                        self.speed_history.append((now, speed_mps))

                self.track_memory[tid] = (cx, cy, now)

                # # Clean up old track IDs
                # expired_ids = [tid for tid, (_, _, t) in self.track_memory.items() if now - t > 10]
                # for tid in expired_ids:
                #     del self.track_memory[tid]

                # # Clean up travel time memory
                # expired_travel_ids = [tid for tid, (_, _, t) in self.travel_time_memory.items() if now - t > 15]
                # for tid in expired_travel_ids:
                #     del self.travel_time_memory[tid]

                if tid not in self.travel_time_memory:
                    self.travel_time_memory[tid] = (cx, cy, now)
                else:
                    ox, oy, start_time = self.travel_time_memory[tid]
                    dist_px = np.linalg.norm([cx - ox, cy - oy])
                    if dist_px > 160:
                        travel_time = now - start_time
                        trr = (travel_time / 60.0) / self.segment_km
                        self.travel_history.append((now, trr))
                        del self.travel_time_memory[tid]
                    elif now - start_time > self.stuck_time_threshold:
                        trr = "Stuck"  
                        self.travel_history.append((now, trr))

            current_detections = len(filtered_class_id)

        return frame, current_detections

    def compute_spi_per_minute(self):
        now = time()
        while self.speed_history and now - self.speed_history[0][0] > self.spi_time_window:
            self.speed_history.popleft()

        if self.speed_history:
            speeds = [s for t, s in self.speed_history]
            vavg = np.mean(speeds)
        else:
            vavg = 0.0
        vmax_mps = self.vmax_kmph / 3.6
        spi = (vavg / vmax_mps) * 100 if vmax_mps > 0 else 0.0
        return min(spi, 100.0)

    def compute_trr_per_minute(self):
        now = time()
        while self.travel_history and now - self.travel_history[0][0] > self.spi_time_window:
            self.travel_history.popleft()

        valid_trrs = [r for r in self.travel_history if isinstance(r[1], (int, float))]

        if valid_trrs:
            trrs = [r[1] for r in valid_trrs]
            return np.mean(trrs)
        return "Stuck"

    def get_spi_status_and_color(self, spi):
        if spi < 25:
            return f"Very Low ({spi:.2f})", (0, 0, 255)
        elif spi <= 50:
            return f"Low ({spi:.2f})", (0, 165, 255)
        elif spi <= 75:
            return f"Moderate ({spi:.2f})", (0, 255, 255)
        else:
            return f"High ({spi:.2f})", (0, 255, 0)

    def is_window_closed(self, window_name):
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        window_name = 'YOLOv12 Detection'
        cv2.namedWindow(window_name)

        while True:
            start_time = time()
            ret, frame = cap.read()
            assert ret

            results = self.predict(frame)
            frame, current_detections = self.plot_bboxes(results, frame)

            now = time()
            self.detection_history.append(current_detections)
            self.history_timestamps.append(now)

            while self.history_timestamps and now - self.history_timestamps[0] > self.spi_time_window:
                self.detection_history.popleft()
                self.history_timestamps.popleft()

            avg_per_minute = (
                sum(self.detection_history) / len(self.detection_history)
                if self.detection_history else 0
            )

            spi = self.compute_spi_per_minute()
            spi_status, spi_color = self.get_spi_status_and_color(spi)
            trr = self.compute_trr_per_minute()

            trr_text = f"Trr: {trr}" if isinstance(trr, str) else f"Trr: {trr:.2f} min/km"

            fps = 1 / max(time() - start_time, 1e-6)

            cv2.putText(frame, f'FPS: {int(fps)} | Active: {current_detections} | Avg/min: {avg_per_minute:.1f}',
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'SPI: {spi_status}', (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, spi_color, 2)
            cv2.putText(frame, trr_text, (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(5) & 0xFF == 27 or self.is_window_closed(window_name):
                break

        cap.release()
        cv2.destroyAllWindows()

detector = ObjectDetection(capture_index=2)
detector()