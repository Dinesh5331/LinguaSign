# Sign Conversion

This project captures hand landmarks from a webcam, prepares labeled sign samples, trains a graph neural network (GNN) in a notebook, and runs real-time sign prediction using the trained model.

## Project Flow

1. `HandTrackingModule.py`
   Provides the `HandDetector` class built with MediaPipe Hands and OpenCV.

2. `collect_data.py`
   Opens the webcam, detects 21 hand landmarks, normalizes landmark positions relative to the wrist, and appends labeled samples to `sign_data.csv`.

3. `Sign_conversion.ipynb`
   Loads the collected CSV data, balances the classes, encodes labels, converts hand landmarks into graph data, trains a GCN model with PyTorch Geometric, and exports:
   - `gnn_model.pth`
   - `label_encoder.pkl`

4. `detector.py`
   Loads the saved model and label encoder, runs webcam inference, smooths recent predictions, and displays the predicted sign with confidence.

## Main Files

- `HandTrackingModule.py`: reusable hand detection and landmark extraction module.
- `collect_data.py`: sample collection script for creating sign training data.
- `Sign_conversion.ipynb`: notebook for preprocessing, training, evaluation, and model export.
- `detector.py`: real-time sign recognition using the trained GNN.
- `sign_data.csv`: collected landmark dataset.
- `gnn_model.pth`: trained model weights.
- `label_encoder.pkl`: saved label mapping for inference.

## Requirements

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

## How To Use

### 1. Collect data

Update the `label` variable inside `collect_data.py` to the sign you want to record, then run:

```bash
python collect_data.py
```

Notes:
- Press `P` to pause or resume saving samples.
- Press `Ctrl + C` in the terminal to stop.
- Samples are appended to `sign_data.csv`.

### 2. Train the model

Open and run `Sign_conversion.ipynb`.

The notebook:
- reads the collected CSV data
- balances samples across labels
- builds graph samples from the 21 hand landmarks
- trains a GCN model
- evaluates the trained model
- saves `gnn_model.pth` and `label_encoder.pkl`

Note:
- The notebook includes `google.colab` download cells for Colab usage.
- If you run it locally, you may need to adjust the CSV path used in the notebook.

### 3. Run live detection

After generating `gnn_model.pth` and `label_encoder.pkl`, run:

```bash
python detector.py
```

Controls:
- Press `Esc` to close the detector window.

## Model Details

The current inference pipeline:
- uses 21 MediaPipe hand landmarks
- normalizes each landmark using the wrist position and hand width/height
- creates a hand graph based on finger connections
- applies a two-layer GCN with batch normalization and dropout
- smooths predictions using a short queue before display

## Notes

- `hand_detection.py` and `hand_tracking.py` are simple experimentation scripts for webcam and landmark testing.
- Large generated assets such as videos, datasets, model files, and virtual environment files are ignored through `.gitignore`.
