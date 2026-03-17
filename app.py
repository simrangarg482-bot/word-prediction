import streamlit as st
import numpy as np
import pickle
import os
import json
import shutil
import h5py
import tempfile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------------------------------
# FIX: Patch model config JSON to handle Keras version mismatch
# InputLayer in newer Keras saves 'batch_shape' and 'optional' which
# older versions don't understand. We fix the h5 file config directly.
# -------------------------------------------------------------------------

def patch_config(config):
    """Recursively patch model config dict to fix version incompatibilities."""
    if isinstance(config, dict):
        # We want to modify this dict if it's an InputLayer or has InputLayer-like keys
        if config.get('class_name') == 'InputLayer' or 'batch_shape' in config:
            # If it's a layer-style dict: {'class_name': '...', 'config': {...}}
            if 'config' in config and isinstance(config['config'], dict):
                inner_cfg = config['config']
                if 'batch_shape' in inner_cfg:
                    inner_cfg['batch_input_shape'] = inner_cfg.pop('batch_shape')
                inner_cfg.pop('optional', None)
                inner_cfg.pop('quantization_config', None)
            
            # If it's the config dict itself: {'batch_shape': ..., 'optional': ...}
            if 'batch_shape' in config:
                config['batch_input_shape'] = config.pop('batch_shape')
            config.pop('optional', None)
            config.pop('quantization_config', None)
            
        return {k: patch_config(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [patch_config(item) for item in config]
    return config

# -------------------------------------------------------------------------
# Load Resources
# -------------------------------------------------------------------------

@st.cache_resource
def load_resources():
    try:
        model_path = "lstm_model.h5"
        # Use system temp directory for the patched file
        temp_dir = tempfile.gettempdir()
        patched_path = os.path.join(temp_dir, "temp_patched_model.h5")

        # Copy and patch the model config inside the h5 file
        if os.path.exists(patched_path):
            try:
                os.remove(patched_path)
            except:
                pass
        
        shutil.copy(model_path, patched_path)
        
        with h5py.File(patched_path, "r+") as f:
            if "model_config" in f.attrs:
                raw_config = f.attrs["model_config"]
                if isinstance(raw_config, bytes):
                    raw_config = raw_config.decode("utf-8")
                
                config = json.loads(raw_config)
                patched = patch_config(config)
                f.attrs["model_config"] = json.dumps(patched)

        # Load the patched model
        model = load_model(patched_path, compile=False)
        
        # Load tokenizer
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
            
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading resources: {str(e)}")
        st.stop()

model, tokenizer = load_resources()
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