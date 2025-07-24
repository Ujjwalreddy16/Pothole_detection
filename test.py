import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import cv2
import shutil

# --- Configuration ---
# 1. This name matches the 'name' from your CPU training script.
RUNS_DIR = r'D:\Projects\runs\detect\pothole_yolov8_cpu_run'

# 2. IMPORTANT: Update this to the name of your downloaded dataset folder.
DATASET_FOLDER = r'D:\Projects\Python\DS\Dataset\Pothole Detection.v10-v5-rotation-only.yolov8'

# --- The rest of the script is fully automated ---

MODEL_PATH = os.path.join(RUNS_DIR, 'weights/best.pt')
OUTPUT_DIR = 'publication_figures'
SAMPLE_IMAGES_DIR = os.path.join(DATASET_FOLDER, 'valid/images')

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

try:
    results_df = pd.read_csv(os.path.join(RUNS_DIR, 'results.csv'))
    results_df.columns = results_df.columns.str.strip()
except FileNotFoundError:
    print(f"❌ Error: Could not find 'results.csv' in '{RUNS_DIR}'.")
    print("Please ensure training has completed and the 'RUNS_DIR' path is correct.")
    exit()

print("📈 Generating training metrics plot...")
fig, ax = plt.subplots(1, 2, figsize=(20, 7))

ax[0].plot(results_df['epoch'], results_df['train/box_loss'], label='Training Box Loss', color='b')
ax[0].plot(results_df['epoch'], results_df['val/box_loss'], label='Validation Box Loss', color='r', linestyle='--')
ax[0].set_title('Box Loss vs. Epochs')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(results_df['epoch'], results_df['metrics/mAP50-95(B)'], label='Validation mAP@50-95', color='g')
ax[1].set_title('Validation mAP@50-95 vs. Epochs')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('mAP@50-95')
ax[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=300)
plt.close()

print("📑 Copying key visualizations...")
for file_name in ['confusion_matrix.png', 'PR_curve.png']:
    source_path = os.path.join(RUNS_DIR, file_name)
    if os.path.exists(source_path):
        shutil.copy(source_path, os.path.join(OUTPUT_DIR, file_name))

print("🖼️ Generating qualitative results on sample images...")
model = YOLO(MODEL_PATH)
if os.path.exists(SAMPLE_IMAGES_DIR):
    val_images = os.listdir(SAMPLE_IMAGES_DIR)
    for i, image_name in enumerate(val_images[:5]):
        image_path = os.path.join(SAMPLE_IMAGES_DIR, image_name)
        results = model(image_path)
        annotated_image = results[0].plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        plt.imsave(os.path.join(OUTPUT_DIR, f'qualitative_result_{i}.png'), annotated_image_rgb)
else:
    print(f"⚠️ Warning: Validation images not found at '{SAMPLE_IMAGES_DIR}'. Skipping qualitative results.")

print("\n--- 📊 Quantitative Results for Your Paper ---")
final_metrics = results_df.iloc[-1]
print("\n| Metric             | Value   |")
print("|--------------------|---------|")
print(f"| Precision          | {final_metrics.get('metrics/precision(B)', 0):.4f}   |")
print(f"| Recall             | {final_metrics.get('metrics/recall(B)', 0):.4f}   |")
print(f"| mAP@50             | {final_metrics.get('metrics/mAP50(B)', 0):.4f}   |")
print(f"| mAP@50-95          | {final_metrics.get('metrics/mAP50-95(B)', 0):.4f}   |")

print("\n✅ All publication assets generated successfully!")