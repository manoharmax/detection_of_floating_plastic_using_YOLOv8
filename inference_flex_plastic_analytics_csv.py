import os
import cv2
import csv
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tkinter import Tk, filedialog
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# --- CONFIG ---
DISPLAY_RESULTS = True
CONFIDENCE_THRESHOLD = 0.01
IOU_THRESHOLD = 0.5
MAX_OVERLAP_THRESHOLD = 0.5
CONFIDENCE_MEAN_THRESHOLD = 0.2

# --- Setup ---
Tk().withdraw()
model_path = "C:/Users/manoh/plastic_detection_yolov8/yolov8_dataset_cleaner/runs/detect/train_yolov8x_augmented_50ep/weights/best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found at: {model_path}")
model = YOLO(model_path)

# --- Input Selection ---
paths = filedialog.askopenfilenames(title="Select image(s) or folder", filetypes=[("Images", "*.jpg *.jpeg *.png")])
image_paths = []
for path in paths:
    if os.path.isdir(path):
        image_paths.extend([os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    else:
        image_paths.append(path)

if not image_paths:
    print("❌ No valid images selected. - inference_flex_plastic_analytics_csv.py:34")
    exit()

# --- Output Directory ---
base_dir = "output/inference_results_plastic"
version = 1
output_dir = f"{base_dir}_{version}"
while os.path.exists(output_dir):
    version += 1
    output_dir = f"{base_dir}_{version}"
os.makedirs(output_dir)

# --- Analytics Data ---
image_stats = []
csv_rows = []

# --- Inference Loop ---
for img_path in image_paths:
    image = cv2.imread(img_path)
    filename = os.path.basename(img_path)
    results = model.predict(source=img_path, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]

    confidences = [float(b.conf) for b in results.boxes]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    count = len(boxes)
    avg_conf = np.mean(confidences) if confidences else 0

    # --- Apply NMS ---
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=confidences,
        score_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=MAX_OVERLAP_THRESHOLD
    )
    if len(indices) > 0:
        indices = indices.flatten()
        boxes = boxes[indices]
        confidences = [confidences[i] for i in indices]
        count = len(boxes)
        avg_conf = np.mean(confidences)

    # --- Metrics ---
    y_true = [1] * count  # All boxes are "plastic"
    y_pred = [1 if c >= CONFIDENCE_THRESHOLD else 0 for c in confidences]
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # --- Draw Detections ---
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        conf = confidences[i]
        csv_rows.append([filename, x1, y1, x2, y2, f"{conf:.4f}"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"plastic {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- Save Output ---
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, image)

    image_stats.append([filename, count, avg_conf, precision, recall, f1])

    if DISPLAY_RESULTS:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Detections in {filename}")
        plt.axis("off")
        plt.show()

# --- Save CSVs ---
csv_path = os.path.join(output_dir, "detection_log.csv")
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "x1", "y1", "x2", "y2", "confidence"])
    writer.writerows(csv_rows)

summary_path = os.path.join(output_dir, "image_summary_metrics.csv")
with open(summary_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "detections", "avg_confidence", "precision", "recall", "f1_score"])
    writer.writerows(image_stats)

print(f"✅ Inference complete. Results saved to: {output_dir} - inference_flex_plastic_analytics_csv.py:114")
print(f"📄 Detection CSV: {csv_path} - inference_flex_plastic_analytics_csv.py:115")
print(f"📄 Summary Metrics CSV: {summary_path} - inference_flex_plastic_analytics_csv.py:116")
