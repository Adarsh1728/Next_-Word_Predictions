## 🧠 Next Word Prediction Using LSTM

## 📌 Project Description

This project focuses on building a Next Word Prediction System using Deep Learning and Natural Language Processing (NLP). It uses a Long Short-Term Memory (LSTM) neural network to understand sequential patterns in text and predict the most probable next word based on the given input sentence.

The model is trained on a text dataset using tokenization, sequence generation, and padding techniques. The trained model is then integrated with a Streamlit web application to provide an interactive user experience for next-word prediction and text generation.

The main objective is to train a language model that learns patterns from text and predicts what word is most likely to come next.

## Deep Learning Project
## NLP

## 1. Project objective

The project aims to build a next-word prediction system.

For example:

- Input: `What are you`
- Predicted next word: `doing` (illustrative)
- Input: `I am going`
- Predicted next word: `to` (illustrative)

The model learns word relationships and sentence patterns from a training dataset.

## 2. Technologies used

| Technology         | Purpose                                          |
| ------------------ | ------------------------------------------------ |
| Python             | Main programming language                        |
| TensorFlow / Keras | Building and loading the deep learning model     |
| RNN                | Learning sequential text patterns                |
| LSTM               | Remembering previous words and context           |
| NumPy              | Numerical operations                             |
| Pickle             | Saving and loading tokenizer and sequence length |
| Streamlit          | Creating the interactive web application         |

The repository includes an LSTM model, tokenizer, maximum sequence length, quote dataset, Python application, and Jupyter notebooks.&#x20;

## 3. How the project works

1\. Text Dataset

Quote dataset (qoute_dataset.csv)

2\. Text Preprocessing

Tokenization and sequence preparation

3\. LSTM Model Training

Learns word patterns and context

4\. Next Word Prediction

Predicts the most likely next word

5\. Streamlit Application

Displays prediction to the user

## 4. Main project components

A. Dataset: The project contains `qoute_dataset.csv`, which is used as the text dataset.

B. Tokenization: Converts words into numerical indexes so the neural network can process text.

C. LSTM model: Learns sequential relationships between words and predicts the next word based on previous input.

D. Model saving: The trained model is saved as `lstm_model (1).h5`, with `tokenizer.pkl` and `max_len.pkl` storing the supporting information.

E. Streamlit: Provides a user interface where users enter text and receive the predicted next word.&#x20;
