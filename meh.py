from ultralytics import YOLO

# --- Configuration ---

# 1. Set the path to the data.yaml file from your downloaded dataset.
DATASET_YAML_PATH = r'D:\Projects\Python\DS\Dataset\Pothole Detection.v10-v5-rotation-only.yolov8\data.yaml'

# 2. Choose the YOLOv8 model size. 'yolov8n.pt' is recommended for CPU training.
MODEL = 'yolov8n.pt'

# --- Training on CPU ---

# Load a pre-trained model
model = YOLO(MODEL)

print("\nStarting training on CPU...")
print(f"Dataset: {DATASET_YAML_PATH}")
print(f"Model: {MODEL}")

# Train the model with early stopping, using the CPU
results = model.train(
    data=DATASET_YAML_PATH,
    
    # Image Size
    imgsz=224,
    
    # Early Stopping to prevent overfitting
    patience=20,
    
    # Force the use of CPU
    device='cpu',
    
    # Standard parameters
    optimizer='AdamW',
    epochs=100,
    batch=8,  # A smaller batch size is better for CPU training
    name='pothole_yolov8_cpu_run'
)

print("\n🎉 Training complete!")
print("The best model is saved in 'runs/detect/pothole_yolov8_cpu_run'")