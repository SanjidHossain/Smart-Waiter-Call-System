# 🍽️ Smart Waiter Call System

![Banner](banner.jpg)

A real-time AI-driven system that detects raised hands in a video and identifies the corresponding desk. This system can be used in restaurants, conference halls, or similar environments to streamline service requests.

---

## 🚀 Features

- 🖐️ **Detects raised hands** using **MediaPipe Pose**.
- 🔍 **Identifies desks** based on OCR-processed desk labels.
- 🎥 **Processes video input** to track hand-raising events.
- ⚡ **Real-time analysis** for accurate and fast responses.
- 📌 **Displays the detected desk name** on the video output.

---

## 🔧 Installation

### 📋 Prerequisites

Ensure you have the following installed:

- Python 3.8+
- Pip
- Virtual environment (optional but recommended)

### 📥 Clone the Repository

```bash
git clone https://github.com/yourusername/smart-waiter-call.git
cd smart-waiter-call
```

### 🏗️ Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Run the Application

```bash
python App.py
```

---

## 🛠️ Usage

1. Place an image (`IMG.png`) with desk labels in the root directory.
2. Provide a video file (`desk_video.mp4`) for hand-raising detection.
3. Run the script and observe desk name detection in real-time.
4. Press `q` to exit the application.

---

## 📚 Libraries Used

| Library         | Purpose                                      |
|----------------|----------------------------------------------|
| `cv2` (OpenCV) | Video processing and visualization          |
| `mediapipe`    | Pose detection for hand-raising recognition |
| `numpy`        | Numerical computations                      |
| `easyocr`      | Optical Character Recognition (OCR)         |
| `time`         | Managing real-time event display            |

---

## 📂 Project Structure

```
├── App.py
├── IMG.png  # Image with desk names
├── desk_video.mp4  # Video file for analysis
├── banner.jpg  # Repository banner image
├── requirements.txt
└── README.md
```

---

## 🚀 Future Improvements

- 🏆 Enhance accuracy with better OCR filtering.
- 📊 Implement a real-time webcam version.
- 💻 Develop a user-friendly GUI for easier monitoring.

---

## 🤝 Contributing

Pull requests are welcome! Feel free to contribute by improving the project, fixing bugs, or adding new features.

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

✨ Developed with ❤️ by MD. Sanjid Hossain ✨

