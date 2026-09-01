# Multimodal Deepfake Detection System

A deepfake detection system designed to identify manipulated **images, videos, and audio** using deep learning and multimodal analysis.

The project focuses on building separate detection pipelines for visual and audio content and combining their predictions to improve the reliability and generalization of deepfake detection.

---

## 1. Problem Statement

Deepfake generation techniques are rapidly improving and can manipulate faces, videos, and voices with high realism. Detecting such manipulated content using only one type of information can be difficult.

For example:

* An image-based detector only analyzes visual information.
* A video detector can analyze both facial appearance and temporal information.
* An audio detector can identify manipulated or synthesized speech.
* A multimodal system can use information from multiple modalities.

Therefore, this project aims to develop a system that can analyze:

* **Images**
* **Videos**
* **Audio**

and classify the input as:

```text
REAL
or
FAKE
```

The long-term objective is to combine the predictions from different modalities into a unified multimodal deepfake detection system.

---

# 2. Project Objectives

The main objectives are:

1. Build an image-based deepfake detection pipeline.
2. Build a video-based deepfake detection pipeline.
3. Build an audio-based deepfake detection pipeline.
4. Use appropriate preprocessing for each modality.
5. Train and evaluate deep learning models for each modality.
6. Prevent data leakage between training, validation, and test sets.
7. Evaluate models using appropriate classification metrics.
8. Compare individual modality performance.
9. Combine modality predictions using multimodal fusion.
10. Provide an inference interface for detecting deepfake content.

---

# 3. Overall System

The proposed system follows this architecture:

```text
                    USER INPUT
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
        IMAGE         VIDEO         AUDIO
          │             │             │
          ▼             ▼             ▼
     Face Detection   Frame         Audio
     / Preprocessing  Extraction    Preprocessing
          │             │             │
          ▼             ▼             ▼
       Image Model   Video Model   Audio Model
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                 Multimodal Fusion
                        │
                        ▼
                Final REAL / FAKE
                    Prediction
```

Each modality is processed independently first. The predictions are then combined during the multimodal fusion stage.

---

# 4. Modalities

## 4.1 Image Detection

The image pipeline analyzes facial images and classifies them as REAL or FAKE.

Pipeline:

```text
Input Image
     │
     ▼
Face Detection
     │
     ▼
Face Crop
     │
     ▼
Resize / Normalize
     │
     ▼
Image Classification Model
     │
     ▼
REAL / FAKE
```

The current image baseline uses **EfficientNet-B0 pretrained on ImageNet**.

The image pipeline is intended to provide a strong visual baseline before moving to more advanced multimodal experiments.

---

# 5. Video Detection

The video pipeline analyzes facial content across multiple frames.

```text
Input Video
     │
     ▼
Video Reading
     │
     ▼
Frame Sampling
     │
     ▼
Face Detection
     │
     ▼
Face Cropping
     │
     ▼
Frame Preprocessing
     │
     ▼
Deepfake Detection Model
     │
     ▼
Frame-level Predictions
     │
     ▼
Temporal / Video-level Aggregation
     │
     ▼
REAL / FAKE
```

For the video detection research experiment, **GenConViT** is being used with its pretrained model weights.

The GenConViT implementation contains:

* GenConViTED
* GenConViTVAE
* GenConViT combined inference

The implementation uses the provided pretrained GenConViT checkpoints rather than building the architecture completely from scratch.

---

# 6. GenConViT Video Model

The GenConViT architecture combines an encoder-decoder based visual representation with a transformer/convolutional backbone.

The project repository contains the GenConViT implementation and pretrained checkpoints.

The two major model components are:

```text
                 Input Frames
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
     GenConViTED             GenConViTVAE
          │                       │
          ▼                       ▼
      Prediction              Prediction
          │                       │
          └───────────┬───────────┘
                      ▼
               Combined Output
                      │
                      ▼
                 REAL / FAKE
```

The current implementation was verified using the available GenConViT source code and pretrained weights.

---

# 7. Audio Detection

The audio branch is designed to analyze speech/audio extracted from videos or provided as an independent audio file.

Pipeline:

```text
Input Audio / Video
        │
        ▼
Audio Extraction
        │
        ▼
Audio Preprocessing
        │
        ▼
Feature Extraction
        │
        ▼
Audio Deepfake Model
        │
        ▼
Audio Prediction
        │
        ▼
REAL / FAKE
```

The audio branch will be developed independently and later integrated with the visual branches.

The audio component should not be considered complete until its preprocessing, model training, validation, testing, and evaluation have been completed.

---

# 8. Datasets

The current research work uses multiple publicly available deepfake datasets.

The datasets currently identified and processed for the GenConViT experiment are:

### Celeb-DF-v2

Used for real and manipulated facial videos.

Current verified count:

```text
Real : 890
Fake : 5,639
```

### FaceForensics++ C23

Current verified count:

```text
Real : 1,000
Fake : 6,000
```

Manipulation categories identified:

```text
DeepFakeDetection : 1,000
Deepfakes         : 1,000
Face2Face         : 1,000
FaceShifter       : 1,000
FaceSwap          : 1,000
NeuralTextures    : 1,000
```

### DFDC-10

Current verified count:

```text
Real : 2,512
Fake : 17,397
```

### DeepfakeTIMIT

DeepfakeTIMIT contains:

```text
Higher quality : 320
Lower quality  : 320
```

The current metadata construction uses the 640 usable DeepfakeTIMIT videos.

### DeeperForensics-1.0

The currently available Kaggle copy was inspected.

The expected source/manipulated video directories were not present in the available dataset copy. Only the repository/support files and a sample perturbation video were found.

Therefore, DeeperForensics-1.0 is **not currently included in the final video metadata/split**.

---

# 9. Video Dataset Metadata

The verified video metadata currently contains:

```text
Total videos : 34,078

Real videos  : 4,402
Fake videos  : 29,676
```

The metadata is stored at:

```text
/kaggle/working/genconvit_project/metadata_all.csv
```

The metadata contains fields including:

```text
path
label
dataset
quality
manipulation
```

where:

```text
label = 0 → REAL
label = 1 → FAKE
```

---

# 10. Dataset Splitting

The video dataset is divided into:

```text
TRAIN
VALIDATION
TEST
```

Current split:

```text
TRAIN
Videos : 7,043
Real   : 3,521
Fake   : 3,522
```

```text
VALIDATION
Videos : 880
Real   : 440
Fake   : 440
```

```text
TEST
Videos : 1,521
Real   : 441
Fake   : 1,080
```

The test set also contains DeepfakeTIMIT:

```text
DeepfakeTIMIT : 640
```

This provides an important cross-dataset evaluation component.

---

# 11. Dataset Split Distribution

### Training

```text
DFDC               : 3,989
FaceForensics++    : 1,628
Celeb-DF-v2        : 1,426
```

### Validation

```text
DFDC               : 505
FaceForensics++    : 211
Celeb-DF-v2        : 164
```

### Test

```text
DeepfakeTIMIT      : 640
DFDC               : 530
Celeb-DF-v2        : 190
FaceForensics++    : 161
```

The final split files are:

```text
/kaggle/working/genconvit_project/splits/train.csv
/kaggle/working/genconvit_project/splits/val.csv
/kaggle/working/genconvit_project/splits/test.csv
```

---

# 12. Leakage Prevention

Deepfake datasets commonly contain multiple frames or related videos originating from the same source.

If related samples are placed in both training and testing sets, the reported performance may become misleading.

Therefore, the project aims to maintain a leakage-safe split.

The split should be performed at the appropriate **video/source level**, rather than randomly splitting individual frames from the same video between train and test.

This is especially important for video deepfake detection.

---

# 13. GenConViT Environment

The GenConViT experiment was prepared in a Kaggle GPU environment.

Verified environment:

```text
Python      : 3.12.13
PyTorch     : 2.10.0+cu128
CUDA        : 12.8
GPU         : Tesla T4
GPU count   : 2
```

The GenConViT project is located at:

```text
/kaggle/working/GenConViT-main
```

---

# 14. GenConViT Configuration

The current configuration includes:

```text
Backbone       : ConvNeXt-Tiny
Embedder       : Swin-Tiny
Image size     : 224 × 224
Latent dims    : 12544
Number classes : 2
Batch size     : 32
Learning rate  : 0.0001
Weight decay   : 0.0001
```

The model is configured for binary classification:

```text
0 → REAL
1 → FAKE
```

---

# 15. Pretrained Model Weights

The GenConViT experiment contains two pretrained checkpoints:

```text
weight/genconvit_ed_inference.pth
weight/genconvit_vae_inference.pth
```

The checkpoints were successfully loaded and inspected.

The checkpoint objects were verified as:

```text
collections.OrderedDict
```

The model state dictionaries contain the expected encoder and model parameters.

---

# 16. GenConViT Implementation Verification

The following components were verified from the project source:

```text
GenConViT
GenConViTED
GenConViTVAE
```

The combined GenConViT model supports:

```text
ED
VAE
Combined
```

For combined inference, the ED and VAE predictions are generated separately and combined according to the implementation.

---

# 17. Video Frame Processing

A video cannot be directly processed as a single ordinary image.

The video pipeline therefore performs:

```text
Video
  ↓
Frame Extraction
  ↓
Selected Frames
  ↓
Face Detection / Preprocessing
  ↓
224 × 224 Input
  ↓
GenConViT
```

Frame-level predictions must then be aggregated to obtain a final video-level prediction.

Possible aggregation methods include:

* Mean probability
* Mean logits
* Majority voting
* Temporal aggregation

The final method should be selected and evaluated experimentally.

---

# 18. Training

The training stage consists of:

```text
Training CSV
      ↓
Video Dataset Loader
      ↓
Frame Extraction
      ↓
Preprocessing
      ↓
GenConViT
      ↓
Loss Calculation
      ↓
Backpropagation
      ↓
Optimizer
      ↓
Validation
      ↓
Best Checkpoint
```

The training configuration uses:

```text
Loss       : CrossEntropyLoss
Optimizer  : AdamW
LR         : 1e-4
Weight decay : 1e-4
Scheduler  : ReduceLROnPlateau
```

The scheduler reduces the learning rate when validation performance stops improving.

---

# 19. Image Model Baseline

The image branch uses:

```text
EfficientNet-B0
```

with ImageNet pretrained weights.

The baseline experiment produced the following recorded results:

```text
FF++ Validation AUC : 0.8323
FF++ Test AUC       : 0.8355
FF++ Test Accuracy  : 74.8%
```

These values belong specifically to the EfficientNet-B0 image baseline and should not be presented as GenConViT or multimodal results.

---

# 20. Evaluation Metrics

The project will evaluate the models using:

### Accuracy

Measures the percentage of correctly classified samples.

### Precision

Measures how many samples predicted as FAKE are actually FAKE.

### Recall

Measures how many actual FAKE samples are detected.

### F1-Score

Provides a balance between precision and recall.

### ROC-AUC

Measures the ability of the model to distinguish REAL from FAKE across different thresholds.

### Confusion Matrix

Shows:

```text
                 Predicted
              REAL      FAKE
Actual REAL     TN        FP
Actual FAKE     FN        TP
```

For video detection, both frame-level and video-level metrics should be considered where appropriate.

---

# 21. Cross-Dataset Evaluation

A major objective of this project is to evaluate generalization.

A model should not only perform well on samples similar to its training data.

The evaluation will therefore include:

```text
Training Dataset
       ↓
Model
       ↓
Same-dataset Test
       +
Cross-dataset Test
       ↓
Generalization Analysis
```

DeepfakeTIMIT is particularly useful for examining whether the model can detect manipulation from a dataset that differs from the main training distribution.

---

# 22. Multimodal Fusion

After developing the individual branches, the predictions will be combined.

The three modality outputs can be represented as:

```text
Image Model  → P(image is fake)
Video Model  → P(video is fake)
Audio Model  → P(audio is fake)
```

These predictions can then be combined.

A simple initial fusion method is weighted probability fusion:

```text
P_final =
    w_image  × P_image
  + w_video  × P_video
  + w_audio  × P_audio
```

where:

```text
w_image + w_video + w_audio = 1
```

The weights should be selected using validation data rather than arbitrarily.

More advanced fusion can later be investigated, such as feature-level fusion or a learned fusion network.

---

# 23. Final Multimodal Pipeline

The intended final system is:

```text
                         INPUT
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
           IMAGE         VIDEO         AUDIO
             │             │             │
             ▼             ▼             ▼
        Face/Image      Frame + Face    Audio
        Processing      Processing      Processing
             │             │             │
             ▼             ▼             ▼
       EfficientNet      GenConViT     Audio Model
             │             │             │
             ▼             ▼             ▼
        P(image)        P(video)      P(audio)
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    Fusion Module
                           │
                           ▼
                  Final Prediction
                           │
                    ┌──────┴──────┐
                    ▼             ▼
                  REAL           FAKE
```

---

# 24. Deployment

The final system can be exposed through an API.

The planned deployment architecture is:

```text
                Client
                  │
                  ▼
                API
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
     Image      Video      Audio
     Model      Model      Model
       │          │          │
       └──────────┼──────────┘
                  ▼
              Fusion
                  │
                  ▼
           Final Prediction
```

The API can accept:

```text
Image
Video
Audio
```

and return a structured response such as:

```text
Prediction : FAKE
Confidence : 0.94
```

For multimodal input, the API can additionally return the individual modality predictions.

---

# 25. Current Implementation Status

### Completed / Verified

```text
✓ Dataset discovery
✓ Dataset verification
✓ Celeb-DF-v2 verification
✓ FaceForensics++ C23 verification
✓ DFDC-10 verification
✓ DeepfakeTIMIT verification
✓ DeeperForensics dataset inspection
✓ Combined video metadata creation
✓ Train/validation/test CSV creation
✓ GenConViT source inspection
✓ GenConViTED verification
✓ GenConViTVAE verification
✓ Pretrained ED checkpoint loading
✓ Pretrained VAE checkpoint loading
✓ GenConViT configuration verification
✓ Video dataset preparation
✓ GenConViT training pipeline setup
✓ EfficientNet-B0 image baseline
✓ Image baseline evaluation
```

---

# 26. Work Remaining

The following work needs to be completed to obtain the final multimodal system.

## Video Branch

```text
□ Complete efficient GenConViT training
□ Select the best checkpoint using validation performance
□ Perform complete video-level evaluation
□ Calculate accuracy, precision, recall and F1
□ Calculate ROC-AUC
□ Generate confusion matrix
□ Perform cross-dataset evaluation
□ Analyze DeepfakeTIMIT generalization
□ Optimize frame sampling and inference speed
```

## Audio Branch

```text
□ Finalize audio dataset
□ Define REAL/FAKE audio labels
□ Extract audio from videos where required
□ Preprocess audio
□ Extract suitable audio representations/features
□ Select and implement audio deepfake model
□ Train audio model
□ Validate audio model
□ Test audio model
□ Perform cross-dataset evaluation
```

## Image Branch

```text
□ Finalize image preprocessing
□ Finalize EfficientNet-B0 baseline
□ Evaluate on the final test set
□ Perform external-image testing
□ Compare with stronger image models if required
```

## Multimodal Fusion

```text
□ Obtain reliable image predictions
□ Obtain reliable video predictions
□ Obtain reliable audio predictions
□ Design fusion strategy
□ Tune fusion weights using validation data
□ Evaluate fused predictions
□ Compare individual vs multimodal performance
```

## Final Application

```text
□ Create unified inference pipeline
□ Add image inference
□ Add video inference
□ Add audio inference
□ Add multimodal inference
□ Integrate final model checkpoints
□ Build API
□ Test API with real inputs
□ Measure inference time
□ Prepare final demonstration
```

---

# 27. Expected Final Comparison

The final project should compare:

```text
                    Accuracy   Precision   Recall   F1   AUC
----------------------------------------------------------------
Image Model            ✓          ✓          ✓      ✓     ✓
Video Model            ✓          ✓          ✓      ✓     ✓
Audio Model            ✓          ✓          ✓      ✓     ✓
Multimodal Model       ✓          ✓          ✓      ✓     ✓
```

The purpose of this comparison is to determine whether combining multiple modalities improves deepfake detection compared with using only one modality.

---

# 28. Important Research Goal

The primary research objective is **not simply to obtain a high accuracy on one dataset**.

The important goal is to determine whether the model can generalize to:

* Different datasets
* Different manipulation techniques
* Different video qualities
* Different compression levels
* Different identities
* Unseen deepfake samples

Therefore, cross-dataset testing and leakage-safe evaluation are important parts of the project.

---

# 29. Project Structure

A suggested final project structure is:

```text
multimodal-deepfake-detection/
│
├── image_detection/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   └── evaluation/
│
├── video_detection/
│   ├── GenConViT/
│   ├── preprocessing/
│   ├── datasets/
│   ├── training/
│   └── evaluation/
│
├── audio_detection/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   ├── training/
│   └── evaluation/
│
├── multimodal/
│   ├── fusion.py
│   └── inference.py
│
├── api/
│   └── app.py
│
├── metadata/
│   ├── metadata_all.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── checkpoints/
│
├── requirements.txt
│
└── README.md
```

---

# 30. Technologies Used

### Programming

* Python

### Deep Learning

* PyTorch
* timm
* EfficientNet-B0
* GenConViT

### Computer Vision

* OpenCV
* MTCNN

### Data Processing

* NumPy
* Pandas

### Deployment

* FastAPI

### Environment

* Kaggle GPU environment
* CUDA

---

# 31. Conclusion

This project aims to develop a **multimodal deepfake detection system** capable of analyzing image, video, and audio content.

The current work has established the dataset preparation and video-model foundation, including verification of multiple deepfake datasets, creation of video metadata and train/validation/test splits, and verification of the GenConViT architecture and pretrained checkpoints. An EfficientNet-B0 image detection baseline has also been evaluated.

The remaining work is primarily focused on completing robust video training/evaluation, developing the audio detection branch, improving image evaluation, combining the three modalities through a validated fusion strategy, and integrating the complete system into a unified inference API.

The final objective is to compare **single-modality detection against multimodal detection** and determine whether combining visual and audio information provides better and more generalizable deepfake detection.
THE END
