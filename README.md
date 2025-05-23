# Pothole Detection and Smart Navigation System

This project implements a comprehensive solution for detecting potholes in road images and dynamically navigating a vehicle around them using intelligent path planning. The system integrates deep learning for pothole classification, bounding box localization, and algorithmic route planning to ensure safe and efficient traversal.

## 🚀 Features

- ✅ CNN-based pothole classification
- ✅ Bounding box-based pothole localization
- ✅ CSV annotation of pothole coordinates and dimensions
- ✅ Path planning using A* algorithm to avoid potholes
- ✅ Real-time visualization of path and obstacles
- ✅ Speed and distance analytics for optimized travela


## 📂 Project Structure

Pothole_Detector/
│
├── dataset/ # Annotated pothole images and CSV labels
├── models/ # Trained CNN models
├── path_planning/ # A* algorithm and visualization code
├── utils/ # Helper scripts (e.g., bounding box rendering)
├── main.py # Main entry point to run the project
├── requirements.txt # Python dependencies
└── README.md # Project overview



## 🖼️ Sample Outputs

### Pothole Annotation Output

| image_id  | num_potholes | x     | y     | w  | h  |
|-----------|--------------|-------|-------|----|----|
| G0010033  | 6            | 1990  | 1406  | 66 | 14 |
| G0010033  | 6            | 1464  | 1442  | 92 | 16 |
| ...       | ...          | ...   | ...   | .. | .. |



📈 Evaluation Metrics

Detection Accuracy

IoU for bounding boxes

Path deviation from shortest route

Speed impact due to detours


📌 Future Work

Integrate into autonomous vehicle simulators like CARLA

Extend to real-time video feeds

Include road quality prediction module


🧑‍💻 Author
Krish @sam-2707


📃 License
This project is licensed under the MIT License.
