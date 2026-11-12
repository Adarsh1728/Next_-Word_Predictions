
import streamlit as st
import os
import pickle
import numpy as np
import glob

import tensorflow as tf
from tensorflow.keras.utils import pad_sequences

# --------------------------------
# 1. Page Configuration
# --------------------------------
st.set_page_config(
    page_title="LSTM Next Word Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 LSTM Next Word Prediction")
st.write("Predict the next word and generate text using LSTM.")

# --------------------------------
# 2. Project Directory
# --------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------
# 3. Find Model Automatically
# --------------------------------
def find_model():

    model_files = (
        glob.glob(os.path.join(BASE_DIR, "*.h5")) +
        glob.glob(os.path.join(BASE_DIR, "*.keras"))
    )

    if not model_files:
        st.error("No LSTM model file found!")
        st.write("Available files:", os.listdir(BASE_DIR))
        st.info("Place your trained .h5 or .keras model in the project folder.")
        st.stop()

    return model_files[0]


# --------------------------------
# 4. Find Tokenizer
# --------------------------------
def find_tokenizer():

    tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")

    if not os.path.exists(tokenizer_path):
        st.error("tokenizer.pkl not found!")
        st.write("Available files:", os.listdir(BASE_DIR))
        st.stop()

    return tokenizer_path


# --------------------------------
# 5. Load Resources
# --------------------------------
@st.cache_resource
def load_resources(model_path, tokenizer_path):

    model = tf.keras.models.load_model(model_path)

    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer


try:

    model_path = find_model()
    tokenizer_path = find_tokenizer()

    model, tokenizer = load_resources(
        model_path,
        tokenizer_path
    )

    max_len = model.input_shape[1]

    index_to_word = {
        index: word
        for word, index in tokenizer.word_index.items()
    }

    st.success("Model and Tokenizer loaded successfully!")

except Exception as e:

    st.error(f"Error loading resources: {e}")
    st.stop()


# --------------------------------
# 6. Next Word Prediction
# --------------------------------
def predictor(model, tokenizer, text, max_len):

    text = text.lower()

    sequence = tokenizer.texts_to_sequences([text])[0]

    if not sequence:
        return ""

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len,
        padding="pre"
    )

    predictions = model.predict(sequence, verbose=0)

    predicted_index = int(np.argmax(predictions[0]))

    predicted_word = index_to_word.get(
        predicted_index, ""
    )

    return predicted_word


# --------------------------------
# 7. Generate Text
# --------------------------------
def generate_text(model, tokenizer, seed_text, max_len, n_words):

    generated_text = seed_text

    for _ in range(n_words):

        next_word = predictor(
            model,
            tokenizer,
            generated_text,
            max_len
        )

        if not next_word:
            break

        generated_text += " " + next_word

    return generated_text


# --------------------------------
# 8. Next Word Prediction UI
# --------------------------------
st.markdown("---")

st.subheader("✍️ Next Word Prediction")

user_input = st.text_input(
    "Enter your sentence:",
    placeholder="What are you..."
)

if st.button("Predict Next Word"):

    if user_input.strip():

        next_word = predictor(
            model,
            tokenizer,
            user_input,
            max_len
        )

        if next_word:
            st.success(
                f"Predicted Next Word: **{next_word}**"
            )
        else:
            st.warning(
                "No prediction found. Try another sentence."
            )

    else:
        st.warning("Please enter some text.")


# --------------------------------
# 9. Text Generation UI
# --------------------------------
st.markdown("---")

st.subheader("✨ Text Generation")

seed_text = st.text_input(
    "Enter starting sentence:",
    placeholder="Are you a..."
)

n_words = st.slider(
    "Number of words to generate:",
    min_value=1,
    max_value=50,
    value=10
)

if st.button("Generate Text"):

    if seed_text.strip():

        generated_text = generate_text(
            model,
            tokenizer,
            seed_text,
            max_len,
            n_words
        )

        st.success("Generated Text:")
        st.write(generated_text)

    else:
        st.warning("Please enter starting text.")


# --------------------------------
# 10. Footer
# --------------------------------
st.markdown("---")

st.caption(
    "LSTM Next Word Prediction | Python | TensorFlow | Streamlit"
)
