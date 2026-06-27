# Hybrid ViT-LSTM Framework for Physical and Mental Fatigue Detection

This repository contains the implementation of a dual-branch Vision Transformer (ViT) and Long Short-Term Memory (LSTM) framework designed to detect and differentiate between physical and mental fatigue

## Abstract
Fatigue significantly reduces decision-making, focus, and reaction time. However, distinguishing between mental cognitive overload and physical muscular strain is challenging due to overlapping outward expressions. This project proposes a hybrid deep learning model that leverages big data analytics to process facial landmarks and classify fatigue types. By integrating multimodal datasets, the framework achieves an overall accuracy of 99% on combined test data, establishing a highly reliable foundation for fatigue prediction.

## Datasets
The model integrates three publicly available datasets to capture diverse, multimodal fatigue contexts:
* Driver Drowsiness Dataset (DDD): Provides physical fatigue indicators like prolonged eye closure and head drooping.
* YAWDD Dataset: Captures the temporal dynamics of physical fatigue, such as yawning and blink rates.
* FER2013 Dataset: Fine-tuned to identify mental fatigue through emotional dullness, blank affect, and reduced facial expressivity.

## Architecture
The architecture is built on a dual-channel design to process distinct data modalities independently before feature fusion:
* Channel 1 (Physical Fatigue): Processes sequences of 3D landmarks extracted from DDD and YAWDD video frames. It utilizes a Multi-Head Self-Attention (MHSA) block followed by a two-layer LSTM to capture long-term temporal dependencies like head nods and blink cycles.
* Channel 2 (Mental Fatigue): Processes normalized static frames from the FER2013 dataset. A Vision Transformer extracts attention-weighted spatial relationships among facial regions without requiring temporal recurrence.
* Feature Fusion & Classification: Outputs from both channels are concatenated and passed through fully connected layers with Dropout (p=0.3). A Softmax layer is then used for the final binary classification.

## Preprocessing & Mathematical Foundations
* Landmark Extraction: MediaPipe Face Mesh extracts 468 3D facial landmarks from each frame, flattening them into a dense feature vector of length 1,404.
* Normalization: Coordinates are normalized across all datasets using Min-Max scaling to ensure numerical stability and account for variations in facial size and camera angle.
* Sequence Padding: Variable-length video sequences are dynamically padded to a maximum length of 200 frames to enable efficient GPU batch processing.
* Optimization & Loss: To handle dataset class imbalances, a weighted cross-entropy loss function is optimized using AdamW.

## Installation & Dependencies
Ensure you have a CUDA-enabled GPU for optimal training performance. 

Install the required Python libraries using:
pip install torch torchvision timm mediapipe opencv-python numpy scipy scikit-learn pillow tqdm seaborn matplotlib kaggle gdown

## Usage
1.  Dataset Preparation: Use the provided Kaggle and gdown scripts in the notebook to download the DDD, YAWDD, and FER2013 datasets directly into your working directory.
2.  Feature Extraction: Run the MediaPipe extraction blocks to convert raw .mp4 and .jpg files into .npy landmark arrays.
3.  Model Training: Train the DualBranchFatigueModel using the custom UnifiedFatigueDataset dataloader.
4.  Evaluation: The notebook automatically generates a classification report, an ROC curve, and a Seaborn confusion matrix for performance analysis.

## Results
The model was evaluated on an unseen test set of 9,515 samples.
* Accuracy: The hybrid architecture achieved an outstanding 99% overall accuracy.
* Precision & Recall: Scored 0.99 for both mental and physical fatigue, indicating an absence of class bias.
* Misclassification: Maintained a total misclassification rate below 1.2%, proving its robust capability to separate distinct physiological and cognitive states.

## Authors
* Vidhi
