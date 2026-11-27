# Cassava AI System 🌿

An AI-powered cassava leaf disease classification system developed using Deep Learning and Computer Vision. The system uses a ResNet101 convolutional neural network to identify cassava leaf diseases from images and classify them into five categories.

## Project Overview

Cassava is an important food and economic crop in many African countries, including Tanzania. However, cassava production is affected by several diseases that can significantly reduce crop yields.

This project applies deep learning to automate the identification of cassava leaf diseases, helping users recognize potential infections through image-based classification.

## Objectives

- Develop a deep learning model for cassava leaf disease classification.
- Apply transfer learning using the ResNet101 architecture.
- Classify cassava leaf images into five disease categories.
- Develop an application for image-based disease prediction.
- Evaluate model performance using classification metrics.

## Disease Categories

The system recognizes five categories:

| Class | Disease |
|---|---|
| CMD | Cassava Mosaic Disease |
| Healthy | Healthy Cassava Leaf |
| CGM | Cassava Green Mottle |
| CBSD | Cassava Brown Streak Disease |
| CBB | Cassava Bacterial Blight |

## Dataset

The project uses a cassava leaf image dataset containing 21,397 images.

| Dataset | Images |
|---|---:|
| Training | 17,117 |
| Validation | 4,280 |
| Total | 21,397 |

### Class Distribution

| Category | Images |
|---|---:|
| CMD | 13,158 |
| Healthy | 2,577 |
| CGM | 2,386 |
| CBSD | 2,189 |
| CBB | 1,087 |

## Deep Learning Model

**Architecture:** ResNet101

ResNet101 is a deep convolutional neural network that uses residual connections to facilitate the training of deeper networks.

The project uses transfer learning to leverage pretrained image-recognition capabilities and adapt the network to cassava leaf disease classification.

### Reported Model Performance

The ResNet101 training experiment achieved the following validation results:

| Metric | Result |
|---|---:|
| Accuracy | 83.46% |
| Weighted F1-score | 0.82 |
| Training epoch | 10 |

These results represent the reported evaluation of the trained model.

## Trained Model Download

The trained model checkpoint is hosted externally on Google Drive because of GitHub's individual file size restrictions.

**Download the trained ResNet101 checkpoint:**

[Download best_resnet101.pth](https://drive.google.com/file/d/1XCOmyVIuzzNvoAzd5XvraUWq-AtPGHb3/view?usp=drive_link)

After downloading, place the file in the root directory of the project.

The expected filename is:

`best_resnet101.pth`

## Project Structure

```text
Cassava_AI_System/
│
├── app.py
├── model.py
├── predictor.py
├── imagenet_classes.txt
├── test_model.py
├── test_images/
│   ├── 4.jpg
│   ├── sample.jpg
│   └── ...
│
├── best_resnet101.pth  (download separately)
├── .gitignore
└── README.md
```

## Technologies Used

- Python
- PyTorch
- Torchvision
- Deep Learning
- Convolutional Neural Networks (CNN)
- ResNet101
- Computer Vision
- Image Classification

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/josephmasasi/cassava-ai-sytsem.git
```

### 2. Navigate to the project directory

```bash
cd cassava-ai-sytsem
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install torch torchvision streamlit pillow numpy
```

### 5. Download the trained model

Download `best_resnet101.pth` from the Google Drive link provided above and place it in the project root directory.

### 6. Run the application

If using Streamlit as the application framework:

```bash
streamlit run app.py
```

## Testing

The project includes `test_model.py` for testing model functionality and prediction-related components.

## Repository

GitHub: https://github.com/josephmasasi/cassava-ai-sytsem

## Author

**Joseph Cletus Masasi**

Computer Engineering | Artificial Intelligence | Machine Learning | Cybersecurity

GitHub: https://github.com/josephmasasi

---

*This project is developed for educational and research purposes, demonstrating the application of deep learning in agricultural disease identification.*
