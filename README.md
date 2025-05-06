# Real-Time Sign Language Translation Using GRU and MLP

A deep learning-based system that translates sign language gestures into text and speech in real time using MediaPipe landmark detection and neural network models.

## Project Overview

This project implements a real-time sign language translation system that captures hand gestures through a webcam and converts them into meaningful text or speech output. By leveraging MediaPipe Holistic for landmark detection and employing dual model architecture (GRU and MLP), the system delivers accurate and responsive sign language interpretation.

## Repository Contents

- **data_collection.ipynb**: Jupyter notebook for capturing and preprocessing sign language gesture data
- **model_training.ipynb**: Implementation of GRU and MLP models with training procedures
- **realtime_gru_inference.ipynb**: Real-time inference script using the GRU model
- **realtime_mlp_inference.ipynb**: Real-time inference script using the MLP model
- **verify_data.ipynb**: Utilities for validating the collected landmark data

## Features

- **Real-time translation**: Processes webcam input to recognize sign language gestures with minimal latency
- **MediaPipe integration**: Extracts 3D keypoints from hands, face, and pose for comprehensive gesture representation
- **Dual model support**:
  - **GRU model**: 96.8% accuracy with superior temporal sequence modeling
  - **MLP model**: 93.2% accuracy with faster inference for resource-constrained environments
- **10 common sign vocabulary**: hello, goodbye, thank you, please, yes, no, stop, come, go, and sorry
- **Text-to-speech capability**: Converts recognized signs to spoken output
- **Visual feedback**: Displays landmarks and predictions on the video feed

## Technical Approach

The system follows a modular pipeline architecture:

1. **Data Collection**: Captures 30-frame sequences of sign gestures via webcam
2. **Landmark Extraction**: Uses MediaPipe Holistic to detect 3D keypoints (face, hands, body)
3. **Preprocessing**: Normalizes and flattens landmark vectors into structured sequences
4. **Model Training**: Implements both GRU (temporal) and MLP (non-temporal) models
5. **Real-Time Inference**: Processes live webcam input through trained models
6. **User Interface**: Provides immediate visual and optional auditory feedback

## Requirements

### Hardware
- Standard webcam
- Processor: Intel Core i5 (8th Gen+) or equivalent
- RAM: 4GB minimum (8GB recommended)
- GPU: Integrated graphics sufficient; dedicated GPU optional

### Software
- Python 3.11
- TensorFlow/Keras
- MediaPipe
- OpenCV
- NumPy
- Pandas
- Matplotlib
- pyttsx3 (for text-to-speech)

## Getting Started

1. Clone this repository:
```
git clone https://github.com/Teja220/Real-Time-Sign-Translation.git
cd Real-Time-Sign-Translation
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run data collection (optional if using pre-trained models):
```
jupyter notebook data_collection.ipynb
```

4. Train models (optional if using pre-trained models):
```
jupyter notebook model_training.ipynb
```

5. Start real-time translation:
```
jupyter notebook realtime_gru_inference.ipynb  # For GRU model
# OR
jupyter notebook realtime_mlp_inference.ipynb  # For MLP model
```

## Model Performance

### GRU Model
- **Validation accuracy**: 92.4%
- **Inference time**: ~1.1 seconds per gesture
- **Advantages**: Better temporal sequence modeling, higher accuracy

### MLP Model
- **Validation accuracy**: 84.7%
- **Inference time**: <1 second per gesture
- **Advantages**: Faster inference, lower computational requirements

## Future Enhancements

- Integration of Explainable AI (XAI) techniques
- Support for continuous sign language sentences
- Emotion recognition from facial expressions
- Expanded vocabulary and multiple sign language support
- Mobile deployment
- Web interface using Flask or Streamlit

## Acknowledgments

This project was developed as part of a Bachelor of Technology capstone project at VIT University under the supervision of Dr. Rajkumar R.

## License

This project is available under the MIT License.

## Citation

If you use this project in your research, please cite:
```
Bhimavarapu Saiteja. (2025). Real Time Sign Language Translation Using GRU and MLP. 
Vellore Institute of Technology, Bachelor of Technology Capstone Project.
```
