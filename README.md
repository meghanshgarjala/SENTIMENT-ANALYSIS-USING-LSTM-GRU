# Sentiment Analysis of IMDB Dataset Using GRU and LSTM

## Overview
This project demonstrates a **sentiment analysis** application using **Recurrent Neural Networks (RNNs)**, specifically **GRU (Gated Recurrent Unit)** and **LSTM (Long Short-Term Memory)** layers, to classify movie reviews from the IMDB dataset as either positive or negative. By leveraging sequential data processing capabilities of these networks, we achieve robust performance in understanding sentiment from textual data.

## Project Structure
- `app.py`: Main Flask application for deploying the model as a REST API.
- `sentiment_model.h5`: Trained model file.
- `tokenizer.pickle`: Tokenizer for processing input text.
- `static/` and `templates/`: HTML, CSS, and JavaScript files for a simple web interface.
- `README.md`: Project documentation.

## Dataset
The **IMDB movie review dataset** contains 50,000 highly polarized reviews, split equally into 25,000 reviews for training and 25,000 for testing. Each review is labeled as positive (1) or negative (0), making it ideal for binary sentiment analysis.

## Model Architecture
The model architecture is based on sequential neural networks with GRU and LSTM layers:
1. **Embedding Layer**: Converts words into dense vectors of fixed size.
2. **GRU Layer**: Captures dependencies in sequential data.
3. **LSTM Layer**: Handles long-term dependencies better, crucial for sentiment analysis.
4. **Dense Layer with Sigmoid Activation**: Outputs a probability score for binary classification.

## Training
The model was trained using the IMDB dataset, where:
- **Input sequences** are tokenized and padded to a maximum length for uniform input size.
- **Binary Cross-Entropy Loss** and **Adam Optimizer** were used to optimize the model for accuracy.

## Evaluation
The model achieved an accuracy of ~89% on the test set, showing strong performance in identifying positive and negative sentiments from reviews.

## Deployment
A Flask API was created to deploy the model:
- `/predict`: Accepts a POST request with a review as input and returns the predicted sentiment.

## How to Run

### Prerequisites
- Python 3.7+
- TensorFlow
- Flask
