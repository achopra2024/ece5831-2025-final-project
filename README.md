# Animal Sound Recognition 🐶🐱🐮🐷

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning project that recognizes and classifies 10 different animal sounds using Convolutional Neural Networks (CNNs) with mel-spectrogram features extracted from the ESC-50 environmental sound dataset.

## 🎯 Project Overview

This project implements an end-to-end audio classification system capable of identifying animal sounds including:
- **Domestic Animals:** Dog, Cat, Cow, Pig, Sheep
- **Birds:** Rooster, Hen, Crow
- **Others:** Frog, Insects

The system achieves **70-80% accuracy** on test data using attention-based CNN architecture trained on mel-spectrogram representations of audio signals.

### Key Features

✅ **Audio Feature Extraction** - Mel-spectrogram representations at 44.1 kHz  
✅ **Deep Learning Models** - Attention-based CNN with batch normalization  
✅ **Model Persistence** - Save/load trained models for deployment  
✅ **Video Support** - Extract audio from videos for prediction  
✅ **Interactive Demo** - Test on custom audio files  
✅ **Comprehensive Evaluation** - Confusion matrices, classification reports, visualizations  

## 📊 Project Resources

| Resource | Link |
|----------|------|
| 🎥 **Presentation Video** | [Watch on YouTube](https://youtu.be/kaKb7_GryvI) |
| 📽️ **Presentation Slides** | [View Slides](https://drive.google.com/file/d/1FxaPLpfRZv63Xk2IzNLaa8ElJhKu5Nl0/view?usp=drive_link) |
| 📄 **Project Report** | [Read Report (PDF)](https://drive.google.com/file/d/1inip4OykwmhV6Su-BJJApyRvp904jz_w/view?usp=drive_link) |
| 📁 **Dataset (ESC-50)** | [Download from Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) |
| 🎬 **Demo Video** | [Watch Demo](https://youtu.be/sWr0Basz-k0) |

## 🚀 Quick Start

### Demo Usage

Test the model on your own audio files:

```bash
# For audio files
python scripts/test_audio_demo.py --audio path/to/your/audio.wav

# For video files (extracts audio automatically)
python scripts/test_video_demo.py --video path/to/your/video.mp4
```

### Example Output

```
======================================================================
🔊 AUDIO DEMO - ANIMAL SOUND PREDICTION
======================================================================

Audio: dog_bark.wav

                    PREDICTION                    
----------------------------------------------------------------------
  Animal: DOG
  Confidence: 94.32%

======================================================================
```

## 🏗️ Architecture

### Model Architecture
- **Input:** Mel-spectrogram (128 x 128)
- **Convolutional Blocks:** 3 blocks (32→64→128 filters)
- **Attention Mechanism:** Channel-wise attention
- **Global Pooling:** Adaptive average pooling
- **Output:** 10 classes (softmax)
- **Total Parameters:** ~2.5M

### Feature Extraction
- **Sample Rate:** 44.1 kHz
- **Duration:** 5 seconds per clip
- **Feature Type:** Mel-spectrogram
- **Mel Bands:** 128
- **FFT Size:** 2048
- **Hop Length:** 512

## 📁 Project Structure

```
animal-sound-recognition/
├── data/                      # Data directory
│   ├── raw/                   # Original, immutable ESC-50 data
│   ├── processed/             # Preprocessed audio files and features
│   ├── interim/               # Intermediate data transformations
│   └── external/              # External datasets or additional resources
│
├── src/                       # Source code
│   ├── data/                  # Data loading and preprocessing
│   ├── features/              # Feature extraction (MFCCs, spectrograms, etc.)
│   ├── models/                # Model architectures
│   ├── training/              # Training loops and pipelines
│   ├── evaluation/            # Model evaluation and metrics
│   ├── utils/                 # Utility functions
│   └── visualization/         # Visualization tools
│
├── notebooks/                 # Jupyter notebooks
│   ├── exploratory/           # EDA and data exploration
│   └── experiments/           # Model experiments and prototyping
│
├── models/                    # Trained models
│   ├── saved_models/          # Final trained models
│   └── checkpoints/           # Training checkpoints
│
├── configs/                   # Configuration files
│   ├── config.yaml            # Main configuration
│   └── model_config.yaml      # Model-specific configurations
│
├── scripts/                   # Standalone scripts
│   ├── download_data.sh       # Download ESC-50 dataset
│   ├── preprocess.py          # Data preprocessing script
│   └── train.py               # Training script
│
├── tests/                     # Unit tests
│
├── logs/                      # Training logs and TensorBoard files
│
├── outputs/                   # Generated outputs (predictions, reports)
│
├── docs/                      # Documentation
│
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation script
├── .gitignore                 # Git ignore file
└── README.md                  # Project documentation
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager
- FFmpeg (for video processing)

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/animal-sound-recognition.git
cd animal-sound-recognition
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download ESC-50 Dataset
```bash
# Option 1: Using Kaggle API
kaggle datasets download -d mmoreaux/environmental-sound-classification-50

# Option 2: Manual download from Kaggle
# Download from: https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50
# Extract to: data/raw/
```

### Step 4: Install FFmpeg (for video demo)
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows: Download from https://ffmpeg.org/download.html
```

## 📖 Usage

### Training a Model

Train from scratch:
```bash
python scripts/train.py --config configs/config.yaml
```

With custom parameters:
```bash
python scripts/train.py \
    --config configs/config.yaml \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001
```

### Testing and Inference

#### Test on Audio Files
```bash
python scripts/test_audio_demo.py --audio data/raw/audio/1-100032-A-0.wav
```

#### Test on Video Files
```bash
python scripts/test_video_demo.py --video my_video.mp4 --keep-audio
```

#### Batch Testing
```bash
# Test multiple files
for file in data/raw/audio/*.wav; do
    python scripts/test_audio_demo.py --audio "$file"
done
```

### Jupyter Notebook

For interactive exploration and experimentation:
```bash
jupyter notebook notebooks/final-project.ipynb
```

## 📊 Dataset

### ESC-50 Dataset
- **Total Clips:** 2000 audio recordings (5 seconds each)
- **Categories:** 50 environmental sound classes
- **Animal Subset:** 400 clips across 10 animal categories
- **Format:** WAV files (44.1 kHz, mono)
- **Folds:** 5-fold cross-validation structure
- **Source:** [Kaggle - ESC-50](https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50)

### Animal Categories (10 Classes)
1. Dog - Barking sounds
2. Cat - Meowing sounds
3. Cow - Mooing sounds
4. Pig - Oinking sounds
5. Sheep - Bleating sounds
6. Frog - Croaking sounds
7. Hen - Clucking sounds
8. Rooster - Crowing sounds
9. Crow - Cawing sounds
10. Insects - Flying insects (bees, mosquitoes)

## 🎯 Results

### Model Performance
- **Test Accuracy:** 72-80%
- **Training Time:** ~15-20 minutes (CPU), ~5-10 minutes (GPU)
- **Model Size:** ~8-10 MB
- **Inference Time:** <1 second per audio file

### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Dog | 0.89 | 0.85 | 0.87 |
| Cat | 0.78 | 0.73 | 0.75 |
| Cow | 0.82 | 0.88 | 0.85 |
| Rooster | 0.91 | 0.87 | 0.89 |
| Average | 0.80 | 0.78 | 0.79 |

*Note: Results may vary based on training run and hyperparameters*

## 🔬 Technical Details

### Technologies Used
- **Deep Learning:** TensorFlow/Keras, PyTorch
- **Audio Processing:** Librosa, NumPy
- **Data Processing:** Pandas, scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Video Processing:** FFmpeg

### Model Improvements Implemented
- Attention mechanism for better feature focus
- Batch normalization for training stability
- Dropout and L2 regularization to prevent overfitting
- Early stopping and learning rate scheduling
- Data augmentation (noise, pitch shift, time stretch)

## 📚 Documentation

- [Demo Scripts Guide](docs/DEMO_GUIDE.md) - How to use the demo scripts
- [Animal Categories](docs/animal_categories.md) - Details about each animal class
- [API Documentation](docs/README.md) - API reference

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ESC-50 Dataset:** [Karol J. Piczak](https://github.com/karolpiczak/ESC-50)
- **Librosa:** Audio processing library
- **TensorFlow/Keras:** Deep learning framework
- Inspiration from various audio classification research papers

---

**⭐ If you find this project helpful, please consider giving it a star!**
