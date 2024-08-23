# Facial Gender Classification

## Project Overview

This project aims to develop a machine learning model for classifying gender based on facial images. The dataset used for this project is the UTKFace dataset, which contains a large collection of images labeled with gender, age, and ethnicity. The primary goal of this project is to build a model that can accurately distinguish between male and female faces.

## Dataset

The dataset is sourced from Kaggle and includes over 20,000 images of faces with the following attributes:
- **Gender:** Male or Female
- **Age:** From 0 to 116 years
- **Ethnicity:** White, Black, Asian, Indian, and Others

The dataset can be found [here](https://www.kaggle.com/datasets/jangedoo/utkface-new/data).

## Model Architecture

The model is built using convolutional neural networks (CNNs) to capture the spatial hierarchies in the facial images. The architecture includes:
- **Input Layer:** Accepts 48x48 pixel grayscale images.
- **Convolutional Layers:** Extracts features through a series of convolutional operations.
- **Pooling Layers:** Reduces the spatial dimensions of the feature maps.
- **Fully Connected Layers:** Combines the extracted features to predict the gender.

## Implementation

The implementation is done in Python using the following libraries:
- TensorFlow/Keras for building and training the neural network.
- OpenCV for image preprocessing.
- NumPy and Pandas for data manipulation.

## Training and Evaluation

The model is trained on 80% of the dataset, with the remaining 20% used for validation and testing. Various data augmentation techniques, such as rotation, flipping, and scaling, are applied to increase the robustness of the model.

### Key Metrics:
- **Accuracy:** The primary metric for evaluating the model's performance.
- **Precision, Recall, and F1-Score:** Additional metrics for assessing the model's classification abilities.

## Results

The model achieves an accuracy of over 90% on the validation set, demonstrating its effectiveness in gender classification. Further improvements could involve fine-tuning the model architecture or experimenting with more complex neural networks.

## Usage

To use the model, follow these steps:
1. Clone the repository.
2. Install the necessary dependencies listed in `requirements.txt`.
3. Run the script `train.py` to train the model or `predict.py` to make predictions on new images.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
