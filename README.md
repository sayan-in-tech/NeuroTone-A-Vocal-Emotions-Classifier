# NeuroTone: Multimodal Emotion Recognition

NeuroTone is a Streamlit application that analyzes an uploaded video and estimates emotion from three modalities:

* Facial expressions
* Vocal characteristics
* The spoken transcript

The application combines the predictions into one weighted result and presents probability charts, timelines, transcription, and video statistics.

## Current Application

The production entry point is `app.py`.

### Analysis Flow

1. Upload an `mp4`, `avi`, `mov`, `mkv`, or `webm` video.
2. Save the upload to a temporary file.
3. Extract the video's audio as a temporary WAV file with `pydub`.
4. Sample video frames once per second.
5. Detect faces with OpenCV's Haar cascade and classify the first detected face with MobileNetV2.
6. Extract 40 MFCC features from the audio and classify vocal emotion with the Keras model.
7. Transcribe the audio with Whisper.
8. Classify the transcript with the serialized scikit-learn text pipeline.
9. Map modality labels to a shared vocabulary and combine active predictions.
10. Display the result and detailed modality breakdowns in Streamlit.

### Fusion Weights

| Modality | Weight |
| --- | ---: |
| Face | 35% |
| Voice | 55% |
| Text | 20% |

Only modalities that produce a prediction participate in the final normalized score.

### Supported Emotion Labels

The shared output vocabulary is `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, and `neutral`.

The text model originally uses `anger`, `joy`, and `sadness`; the application maps those labels to `angry`, `happy`, and `sad`. `shame` is mapped to `neutral`.

## Repository Structure

```text
.
├── app.py                         Main multimodal Streamlit application
├── audio_formatter.py             Standalone audio-to-WAV helper
├── video_processor.py              Video sampling and face inference
├── requirements.txt                Root dependency list
├── models/
│   ├── audio/
│   │   ├── emotion_recognition_model.keras
│   │   ├── best_model.keras        Alternative audio model, not used by app.py
│   │   ├── scaler_params.npy
│   │   └── label_encoder.pkl
│   ├── cv/
│   │   ├── face_model.pth
│   │   └── face_model.py
│   └── text/
│       └── emotion_classifier_pipe_lr.pkl
├── cv_addition/                   Face-model experiments and training code
├── Emotion-Detection-in-Text/     Standalone text emotion application
├── Jupyter Notebook/               Audio-classifier notebook
├── labels/                         Audio features and labels
└── whisper-audio-to-text/          Standalone Whisper script
```

## Requirements

The main application uses Streamlit, TensorFlow/Keras, PyTorch, torchvision, OpenCV, librosa, soundfile, pydub, OpenAI Whisper, scikit-learn, joblib, NumPy, pandas, Pillow, and Plotly.

Python 3.8 or newer is recommended. A virtual environment is strongly recommended.

Audio and video decoding through `pydub` generally requires **FFmpeg** to be installed and available on `PATH`.

The checked-in root `requirements.txt` currently appears to have a malformed or non-standard text encoding. Review or rewrite it as normal UTF-8 before installing it with `pip`.

## Installation

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Whisper downloads its selected model the first time the application starts. The default model is `small.en`. Set `WHISPER_MODEL_NAME` to choose another model.

## Run the Main Application

```bash
streamlit run app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`, and upload a video containing a visible face and speech for the fullest analysis.

The application can return partial results when a modality is unavailable. For example, a video without a detectable face can still be analyzed for voice and transcript emotion.

## Model Artifacts

The main app expects these files:

| File | Purpose |
| --- | --- |
| `models/audio/emotion_recognition_model.keras` | Audio emotion classifier |
| `models/audio/scaler_params.npy` | Mean and scale values for audio features |
| `models/audio/label_encoder.pkl` | Maps audio class indexes to labels |
| `models/cv/face_model.pth` | MobileNetV2 face-emotion weights |
| `models/text/emotion_classifier_pipe_lr.pkl` | CountVectorizer plus LogisticRegression pipeline |

The application loads the audio and text artifacts with Streamlit resource caching and loads the face model on CPU or CUDA depending on availability.

## Training and Experiments

### Audio

`Jupyter Notebook/Vocal_Emotional_Classifier.ipynb` contains the audio-model workflow. The runtime feature path uses 22,050 Hz audio, 40 MFCC coefficients, delta and delta-delta features, padding/truncation, and saved normalization parameters.

### Face

`cv_addition/train_face_model.ipynb` is the more complete face-training workflow. It documents a MobileNetV2 transfer-learning model with seven classes, image augmentation, an 80/20 train-validation split, validation metrics, a confusion matrix, and model export.

`cv_addition/train_face.py` is a simpler older training script. It expects images under `data/images/<emotion>/` and writes a checkpoint under `checkpoints/`.

### Text

`Emotion-Detection-in-Text/notebooks/Emotion Detection in Text.ipynb` trains the text model from an eight-class emotion dataset using text cleaning, `CountVectorizer`, and LogisticRegression. The resulting pipeline is saved as `emotion_classifier_pipe_lr.pkl`.

## Standalone Text Application

The text project can also be run independently:

```bash
cd Emotion-Detection-in-Text
streamlit run app.py
```

It provides text prediction, probability charts, an About page, and a Monitor page backed by SQLite. Runtime tracking data is created under `Emotion-Detection-in-Text/data/data.db`.

## Important Implementation Notes

* The main application accepts videos, not direct audio uploads.
* Face analysis samples frames and analyzes only the first face found in each sampled frame.
* The application uses OpenCV Haar-cascade detection before MobileNetV2 classification.
* Whisper transcription is restricted to English in the main app.
* The audio timeline analyzes separate three-second segments.
* Temporary video and WAV files are removed after processing when cleanup is reached.
* Model predictions are estimates and should not be treated as clinical or psychological diagnoses.

## Known Limitations

* Video files without an audio track cannot produce voice or transcript predictions.
* Face detection depends on lighting, pose, resolution, and the Haar cascade's limitations.
* The default Whisper model can require substantial RAM, disk space, and startup time.
* CPU inference can be slow for long videos.
* Some model paths are relative to the working directory, so launch the app from the repository root.
* The checked-in training data is not included; training notebooks require their documented datasets and folder structures.
* There is no automated test suite currently included in the repository.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
