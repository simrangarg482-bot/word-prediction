import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------------------------------
# FIX: Custom Layers to handle version mismatch
# These filter out the 'quantization_config' argument for all layer types
# -------------------------------------------------------------------------
class FixedEmbedding(Embedding):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

class FixedDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

class FixedLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

# -------------------------------------------------------------------------
# Load Resources
# -------------------------------------------------------------------------

# Load model with all custom object fixes
try:
    custom_layers = {
        'Embedding': FixedEmbedding,
        'Dense': FixedDense,
        'LSTM': FixedLSTM
    }
    model = load_model("lstm_model.h5", custom_objects=custom_layers)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load tokenizer
try:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except FileNotFoundError:
    st.error("Error: tokenizer.pkl not found. Make sure the file exists.")
    st.stop()

# Create reverse mapping
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

max_len = 477   # use your actual max_len

# -------------------------------------------------------------------------
# Core Logic
# -------------------------------------------------------------------------

def sample_with_temperature(preds, temperature=1.0, top_k=10):
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    top_indices = np.argsort(preds)[-top_k:]
    top_probs = preds[top_indices]
    top_probs = top_probs / np.sum(top_probs)

    return np.random.choice(top_indices, p=top_probs)

def generate_text(seed_text, next_words=20, temperature=1.0):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = sample_with_temperature(predicted_probs[0], temperature)
        predicted_word = index_to_word.get(predicted_index, "")

        if predicted_word == "":
            continue
        if predicted_word == seed_text.split()[-1]:
            continue

        seed_text += " " + predicted_word
    return seed_text

# -------------------------------------------------------------------------
# UI
# -------------------------------------------------------------------------

st.title("🎬 Movie Dialogue Generator (LSTM)")

seed_text = st.text_input("Enter starting text:")
temperature = st.slider("Creativity (Temperature)", 0.1, 1.5, 0.8)

if st.button("Generate"):
    if seed_text:
        with st.spinner("Generating dialogue..."):
            output = generate_text(seed_text, 20, temperature)
            st.write("### Generated Text:")
            st.write(output)
    else:
        st.warning("Please enter some starting text.")