# ⚽ Football Analysis System (Computer Vision Capstone)

## 📌 Overview

This project is a **computer vision-based football analysis system** that detects, tracks, and analyzes players, teams, ball movement,
and **team formations** from match footage.
Using deep learning models such as YOLOv8 and computer vision techniques, the system extracts both **physical** and **tactical insights**, 
including player positioning, ball possession, movement patterns, and formation structures.

## 🚀 Features

* 🎯 **Player Detection & Tracking**
  Detects players and assigns consistent IDs across frames.

* 🧤 **Goalkeeper Handling**
  Goalkeepers are detected separately and excluded from team classification.

* 👕 **Team Classification**
  Players are grouped into teams based on shirt color.

* ⚽ **Ball Detection & Possession Assignment**
  Assigns ball possession to the nearest player.

* 📏 **Speed & Distance Estimation**
  Computes player speed and total distance covered.

* 🎥 **Camera Movement Compensation**
  Adjusts for camera motion to improve positional accuracy.

* 🧩 **Team Formation Detection**
  Infers team formations (e.g., 4-3-3, 4-4-2) using spatial positioning of players.

* 📊 **Annotated Video Output**
  Outputs a video with:

  * Player IDs
  * Team classification
  * Ball possession
  * Movement tracking
  * Formation structure


## 🧠 How It Works

1. **Frame Processing**
   The input video is processed frame by frame.
2. **Object Detection (YOLOv8)**
   Players, ball, and goalkeepers are detected.
3. **Tracking**
   Players are assigned persistent IDs across frames.
4. **Team Assignment**
   Players are clustered into teams using jersey color.
5. **Ball Assignment**
   Ball is assigned to the closest player.
6. **Camera Motion Adjustment**
   Camera movement is compensated for stable tracking.
7. **Formation Detection**
   Player positions are analyzed to infer team formations.
8. **Motion Analysis**
   Speed and distance are computed from trajectories.
9. **Rendering Output**
   All data is visualized and exported as an annotated video.


## 🧰 Tech Stack

* Python
* OpenCV
* NumPy
* YOLOv8 (Ultralytics)

---

## 📂 Project Structure

```plaintext
Capstone/
├── camera_movement/        # Handles camera motion estimation & compensation
├── dev_analysis/           # Experimental / development scripts and analysis
├── formation_detector/     # Logic for detecting team formations
├── Formations.csv          # Formation templates / reference data
├── input_videos/           # Input match videos
├── output_videos/          # Generated annotated videos (created by user)
├── player_ball_assigner/   # Assigns ball possession to players
├── speed_distance/         # Computes player speed and distance covered
├── stubs/                  # Cached tracking data (.pkl files)
├── team_assigner/          # Assigns players to teams based on jersey color
├── team_structure/         # Handles team shape and positional structure
├── trackers/               # Object tracking logic (players, ball)
├── utils/                  # Helper functions and utilities
├── view_transformer/       # Perspective transformation (top-down view, etc.)
├── visualization/          # Drawing overlays and rendering output
├── main.py                 # Main pipeline script
├── yolo_inference.py       # YOLO model inference logic
├── yolov8l.pt              # Model weights (NOT included in repo)
└── README.md
```

---

## ⚙️ Setup & Usage Notes

### 📁 Required Setup

1. **Create an output folder**

```bash
mkdir output_videos
```

---

2. **Add model weights**

Download the YOLOv8 model (`yolov8l.pt`) and place it in the project root directory.

Example:

```
Capstone/
├── yolov8l.pt
├── output_videos/
├── main.py
```

---

### ▶️ Run the Project

```bash
python main.py
```
---

### 📤 Output

After running:

* The processed video will be saved in the `output_videos/` folder

---

## ⚠️ Important Note

If you **change the input video**, you must remove cached tracking files:

```bash
rm stubs/*.pkl
```

These files store intermediate tracking results and can cause incorrect outputs if reused.

---

## ⚠️ Limitations

* Formation detection depends on tracking accuracy
* Jersey color classification may fail in similar colors
* Ball detection may be inconsistent in crowded scenes
* Not optimized for real-time performance
* Requires manual model setup

---

## 🧩 Future Improvements

* Improve formation detection robustness
* Add real-time processing
* Build a UI/dashboard for visualization
* Deploy as a web application
* Integrate advanced tracking (e.g., DeepSORT)

---

## 💡 Notes

* Large files (models, videos) are not included due to GitHub size limits
* Recommended to use Google Colab with GPU for heavy workloads

---

## ⭐ Acknowledgments

* Ultralytics YOLOv8
* OpenCV community
* Computer vision research community
