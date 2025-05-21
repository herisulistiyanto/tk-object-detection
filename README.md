# tk-object-detection

## 🖼️ Project Overview
This project is a Python-based object detection application featuring a modern GUI built with `ttkbootstrap`. It leverages deep learning libraries such as PyTorch, torchvision, and Ultralytics for robust object detection tasks. The application uses Ultralytics YOLO12 as its main object detection engine, providing state-of-the-art accuracy and performance. The application is designed to be user-friendly and efficient for various computer vision needs.

## ✨ Features
- Object detection using pre-trained models
- Modern and intuitive GUI
- Easy image selection and detection

## ⚙️ Requirements
- Python 3.13.0
- All dependencies listed in `requirements.txt`

## 🚀 Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone git@github.com:herisulistiyanto/tk-object-detection.git
   cd tk-object-detection
   ```
2. **Create a virtual environment:**
   ```bash
   python3.13 -m venv venv
   ```
3. **Activate the virtual environment:**
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```cmd
     venv\Scripts\activate
     ```
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage
1. **Activate the virtual environment** (if not already active):
   ```bash
   source venv/bin/activate
   ```
2. **Run the application:**
   ```bash
   python main.py
   ```
3. **Follow the on-screen instructions to perform object detection.**

## 📝 Notes
- Ensure you are using Python 3.13.0 for compatibility.
- For best results, use the virtual environment as described above.
- If you are using a Mac and iPhone, it is recommended to disable Camera Continuity to avoid camera conflicts.
- If you encounter issues with dependencies, try upgrading `pip` first:
   ```bash
   pip install --upgrade pip
   ```

## 📁 Project Structure
- `main.py` — Main application file
- `requirements.txt` — List of required Python packages

## 🪪 License
This project is licensed under the MIT License.
