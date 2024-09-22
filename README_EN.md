# A Four-Class Inappropriate Image Model Based on Over 16,000 Scraped Photos  
## Updates  
- Added a second model (v2), which improves the distinction between "politic" and "other" compared to the first model (v1).  
- Pass the `version` parameter (`v1` / `v2`) when constructing the Detect class.  

### Download
- Download the `./release/dist/SensitiveImgDetect-0.1.5-py3-none-any.whl` file.  
```shell
pip install ${path_to_whl}
```  

### v1
- Original trained model  
```shell
Class: cartoon
  Precision: 0.95
  Recall: 0.95
Class: other
  Precision: 0.92
  Recall: 0.82
Class: politic
  Precision: 0.79
  Recall: 0.93
Class: sex
  Precision: 0.95
  Recall: 0.88
Overall Accuracy: 0.89
```

### v2  
- Enhanced v1 with additional data from the "other" dataset.
```shell
Class: cartoon
  Precision: 0.92
  Recall: 0.88
Class: other
  Precision: 0.93
  Recall: 0.93
Class: politic
  Precision: 0.88
  Recall: 0.88
Class: sex
  Precision: 0.86
  Recall: 0.90
Overall Accuracy: 0.89
```

## Introduction
> There are four categories: Cartoon, Pornography, Politics, and Other.
- Each category has approximately 4,000 images.
> Due to the sensitivity of the dataset, it will not be made public.

## During training, `ResNet34` and `ResNet50` classification models were used for transfer learning.
### Dataset Description
- **Pornography**: Approximately 4,000 images filtered from nearly 20,000 scraped images.
- **Cartoon**: Over 4,000 images selected from nearly 10,000 scraped images.
- **Politics**: Utilized some portrait images from the `wider_face` dataset along with self-scraped sensitive images.
- **Other**: Includes 2,000 scraped images and over 5,000 compliant images from Kaggle’s pet and scenery datasets, which may appear on blogs or websites.

### Model Introduction
- Applied transfer learning to `ResNet34` and `ResNet50`, adjusting the output layer to fit the four-class classification needs.

### Output Explanation
- The model's output is processed through the softmax function to obtain the probability for each category.
- The four categories are:
  - **Cartoon**: Non-pornographic cartoon images.
  - **Pornography**: Pornographic images.
  - **Politics**: Politically sensitive and violent content.
  - **Other**: Generally compliant images that may appear on various blogs and websites.

### Model Performance
```shell
Class: Cartoon  
  Precision: 0.95  
  Recall: 0.95  
Class: Other  
  Precision: 0.92  
  Recall: 0.82  
Class: Politics  
  Precision: 0.79  
  Recall: 0.93  
Class: Pornography  
  Precision: 0.95  
  Recall: 0.88  
Overall Accuracy: 0.89  
```

### Usage Instructions
- `./release`
- Or install directly using pip:
```shell
pip install SensitiveImgDetect
```
> If the package cannot be found, please switch to the official repository.

### Development
- `./dev`

# SensitiveImgDetect

This package provides a way to load pre-trained PyTorch models.

## Installation

You can install this package using pip:

```bash
pip install SensitiveImgDetect
```

## Overview

The `Detect` class is designed for image classification tasks, allowing users to predict the categories of single or multiple images using a pre-trained model. This document provides an overview of how to initialize and use this class, including example code snippets.

## Installation

Before using the `Detect` class, ensure that the necessary libraries are installed:

```bash
pip install torch pillow torchvision
```

## Initialization

To use the `Detect` class, you need to initialize it. The class constructor allows you to specify the computing device (CPU or GPU).

```python
from SensitiveImgDetect import Detect 

# Initialize the Detect class
detector = Detect(device='cuda')  # Use 'cpu' if no CUDA is available
```

## Methods

### 1. Detecting the Category of a Single Image

#### Method: `detect_single_type`

**Description**: Predicts the category label of a single image.

**Parameters**:
- `img`: An image object (PIL Image).

**Returns**: A string representing the predicted category label.

#### Example Usage:

```python
from PIL import Image

# Load an image
img = Image.open("path_to_your_image.jpg")

# Predict the category label
predicted_label = detector.detect_single_type(img)
print(f"The predicted category of this image is: {predicted_label}")
```

### 2. Detecting the Probability of a Single Image's Category

#### Method: `detect_single_prob`

**Description**: Predicts the probability of a single image's category.

**Parameters**:
- `img`: An image object (PIL Image).

**Returns**: A dictionary containing category labels and their respective probabilities.

#### Example Usage:

```python
# Predict category probabilities
predicted_probs = detector.detect_single_prob(img)
print("The category probabilities for this image are:")
for class_label, probability in predicted_probs.items():
    print(f"{class_label}: {probability}")
```

### 3. Detecting Categories for a List of Images

#### Method: `detect_list_type`

**Description**: Predicts the category labels for a list of images.

**Parameters**:
- `img_list`: A list of image objects (PIL Images).

**Returns**: A list of predicted category labels.

#### Example Usage:

```python
# Load multiple images
images = [Image.open("image_1.jpg"), Image.open("image_2.jpg")]

# Predict the category labels for the list of images
predicted_labels = detector.detect_list_type(images)
print("Predicted categories for the list of images:")
print(predicted_labels)
```

### 4. Detecting Probabilities for a List of Images

#### Method: `detect_list_prob`

**Description**: Predicts the category probabilities for a list of images.

**Parameters**:
- `img_list`: A list of image objects (PIL Images).

**Returns**: A list of dictionaries containing category labels and their respective probabilities.

#### Example Usage:

```python
# Predict category probabilities for the list of images
predicted_probs_list = detector.detect_list_prob(images)
for index, probs in enumerate(predicted_probs_list):
    print(f"Probabilities for image {index + 1}:")
    for class_label, probability in probs.items():
        print(f"{class_label}: {probability}")
```