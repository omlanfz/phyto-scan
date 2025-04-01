# PhytoScan: Plant Skin Disease Detection and Treatment Recommendation

**PhytoScan** is a machine learning project aimed at detecting plant skin diseases and recommending appropriate treatments. This project uses deep learning models to classify plant diseases from images and provide suggested treatments for the identified disease.

---

## Setup and Installation

### 1. Install Python 3.10.11
- Download and install Python 3.10.11 from the official [Python website](https://www.python.org/downloads/release/python-31011/).
- During installation, ensure you check the box to **Add Python to PATH**.

### 2. Check Installation
After installation, verify the Python version:
`python --version`
You should see Python 3.10.11

### 3. Create a Virtual Environment
To avoid conflicts with other projects, it's best to use a virtual environment:
`py -3.10 -m venv tf_env`

### 4. Activate the Virtual Environment
For Windows:
`tf_env\Scripts\activate`

### 5. Install Dependencies
Upgrade pip and install required libraries:
`pip install --upgrade pip
pip install flask-cors
pip install flask tensorflow numpy pillow`

### 6. Verify TensorFlow Installation
Check if TensorFlow is installed correctly:
`python -c "import tensorflow as tf; print(tf.__version__)"`
This should print the installed TensorFlow version.

### 7. Run the Flask App
Navigate to the project directory and run the Flask app:
`cd /d "file_directory"
python app.py
`

---
## How It Works
- The app accepts an image of a plant and detects the disease based on the trained ML model.
- Once the disease is identified, it provides a treatment recommendation.
