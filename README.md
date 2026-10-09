# Crime Activity Detection Using EfficientNetB0 and LSTM

## Project Overview

This project explores video-based crime activity classification using deep learning. It uses a pretrained **EfficientNetB0** model to extract visual features from video frames and an **LSTM (Long Short-Term Memory)** layer to learn temporal patterns across a sequence of frames.

The notebook demonstrates dataset loading, frame extraction and resizing, model construction, training with validation, saving/loading the trained model, and predicting a class for a test video.

> **Note:** This is an academic/computer-vision prototype. Predictions are not proof that a crime occurred and should not be used as the sole basis for security, legal, or emergency decisions.

## Classes

The notebook configures these four classes:

- `Explosion`
- `fighting`
- `Shooting`
- `z_Normal_Videos_event` (normal videos)

The notebook's class names must match the corresponding folder names in the dataset exactly. `Robbery` is commented out in the current class list.

## Model Architecture

1. **Input video:** A video is sampled into a sequence of 20 frames.
2. **Preprocessing:** Each frame is converted from BGR to RGB and resized to `224 × 224`.
3. **Feature extraction:** A pretrained EfficientNetB0 model (`ImageNet` weights, classification head removed, global average pooling enabled) extracts features from each frame. Its weights are frozen.
4. **Temporal learning:** `TimeDistributed` applies the CNN to each frame, and an LSTM with 128 units learns sequence-level patterns.
5. **Classification head:** Dense layers with dropout feed into a softmax output layer for the four classes.

The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as a metric. Training uses early stopping and a model checkpoint based on validation loss.

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV (`cv2`)
- NumPy
- scikit-learn
- Jupyter Notebook
- UCF-Crime video dataset (local dataset required)

## Project Structure

A suggested GitHub repository layout:

```text
crime-activity-detection/
├── crime-activity-detection.ipynb
├── README.md
├── requirements.txt          # optional: add your tested package versions
├── .gitignore
└── best_crime_model.keras    # optional; large model file, often kept outside Git
```

The notebook currently contains a local Windows dataset path. Keep the dataset outside the repository unless you have confirmed its license and redistribution permissions.

## Dataset Setup

1. Obtain the UCF-Crime dataset from its official/source distribution and follow its terms of use.
2. Arrange the video files in class folders, for example:

```text
Videos/
├── Explosion/
├── fighting/
├── Shooting/
└── z_Normal_Videos_event/
```

3. In the notebook, update `dataset_dir` to the path of your dataset folder. For example:

```python
dataset_dir = r"C:\path\to\UCF_Crimes\Videos"
```

Make sure the folder names in `classes` match the actual dataset folders. The dataset itself is not included in this repository.

## Installation

Create and activate a virtual environment, then install the libraries used by the notebook.

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install tensorflow opencv-python numpy scikit-learn jupyter
```

TensorFlow installation compatibility depends on your operating system, Python version, and hardware. If installation fails, use a Python version supported by the TensorFlow release you choose.

## Running the Notebook

1. Clone or download this repository.
2. Install the dependencies.
3. Download and arrange the dataset separately.
4. Update `dataset_dir` in the notebook.
5. Open Jupyter Notebook:

```bash
jupyter notebook
```

6. Open `crime-activity-detection.ipynb` and run the cells in order.

The notebook extracts frames from the videos, builds the model, trains it, saves the best checkpoint as `best_crime_model.keras`, reloads the checkpoint, and runs predictions on example videos. Training time depends on dataset size and available hardware.

## Model Files

The notebook saves the best checkpoint as:

```text
best_crime_model.keras
```

It also saves a copy to a local Windows path. Change that path or use a repository-relative path if you want to save the model elsewhere.

Trained model files can be large. Consider using Git LFS or a release/artifact link rather than committing a large `.keras` file directly to GitHub.

## Important Implementation Notes

- The notebook uses a fixed sequence length of **20 frames** and frame size of **224 × 224**.
- The model expects input shaped like `(batch_size, 20, 224, 224, 3)`.
- Ensure the frame extraction function returns exactly 20 valid frames for each video.
- The current notebook uses the test generator as `validation_data` during training. For a more reliable evaluation, split data into separate training, validation, and test sets, and use the test set only for final evaluation.
- Check that all directory entries are valid video files before adding them to `video_paths`.
- The notebook's final section opens multiple videos for visual playback; this is a demonstration, separate from model training.
- Report measured evaluation results only after running the notebook on your own environment. Results can vary with the dataset split, preprocessing, and software/hardware setup.

## Future Improvements

- Add a dedicated validation set and report precision, recall, F1-score, and a confusion matrix.
- Add configurable dataset paths instead of hard-coded local paths.
- Handle unreadable or corrupted videos more explicitly.
- Add a simple interface for uploading a video and displaying the predicted class and confidence.
- Evaluate class imbalance and test the model on videos not used during training.

## Disclaimer

This project is intended for learning and research in video classification. It may misclassify videos and should not be treated as a reliable crime-detection or public-safety system.
