# Email Classification Project

## Overview

This project aims to classify emails as either "SPAM" or "NOT SPAM" using different machine learning and deep learning techniques. Various models were trained and evaluated on a dataset containing email texts and their corresponding labels.

## Dataset

The dataset used in this project is `spam_ham_dataset.csv`, which contains email texts and their corresponding labels (spam or ham). It was preprocessed to handle missing values and duplicates.

## Models Explored

1. **Naive Bayes with TF-IDF**: This model uses TF-IDF vectorization and a Multinomial Naive Bayes classifier.

2. **LSTM RNN**: A custom LSTM recurrent neural network model was built using TensorFlow's Keras API.

3. **Pretrained Universal Sentence Encoder (USE)**: This model utilizes a pretrained feature extractor from TensorFlow Hub, specifically the Universal Sentence Encoder.

## Model Comparison

Several metrics were used to evaluate model performance, including accuracy, precision, recall, and F1-score. The confusion matrix was also analyzed to understand the model's behavior.

The best-performing model was determined to be the Pretrained Universal Sentence Encoder model, which achieved an accuracy of 97% on the test set.

## Usage

To use this project, follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have the required dependencies installed (TensorFlow, scikit-learn, pandas, etc.).
3. Run the provided Jupyter Notebook or Python scripts to train and evaluate the models.
4. Use the trained models to classify new email messages by following the provided examples.

## Future Work

Potential areas for future improvement and exploration include:

- Fine-tuning the pretrained Universal Sentence Encoder for domain-specific tasks.
- Experimenting with different neural network architectures and hyperparameters.
- Incorporating ensemble methods for model combination and boosting performance.

## Conclusion

In conclusion, this project demonstrates the effectiveness of deep learning techniques, particularly pretrained feature extractors like the Universal Sentence Encoder, for email classification tasks. By leveraging advanced natural language processing models, we can achieve robust and accurate spam detection systems.
