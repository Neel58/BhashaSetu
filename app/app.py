"""
English -> Gujarati Transformer NMT — Streamlit demo.

Loads the trained model (transformer_model.keras) and the saved vocabularies
(vectorizer_vocab.pkl) produced by the training notebook, then translates
input sentences using either greedy decoding or beam search.

Run:
    streamlit run app.py

Expects, in the same folder (or set via the sidebar):
    - transformer_model.keras
    - vectorizer_vocab.pkl
"""

import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "transformer_model.keras"
DEFAULT_VOCAB_PATH = "vectorizer_vocab.pkl"

st.set_page_config(page_title="English → Gujarati", page_icon="🔤", layout="centered")


# --------------------------------------------------------------------------
# Custom layer required to load the model
# (must match the definition used in the training notebook exactly)
# --------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_length, embed_size, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert embed_size % 2 == 0, "embed_size must be even"
        self.max_length = max_length
        self.embed_size = embed_size
        p, i = np.meshgrid(np.arange(max_length), 2 * np.arange(embed_size // 2))
        pos_emb = np.empty((1, max_length, embed_size))
        pos_emb[0, :, ::2] = np.sin(p / 10_000 ** (i / embed_size)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10_000 ** (i / embed_size)).T
        self.pos_encodings = tf.constant(pos_emb.astype(self.dtype))
        self.supports_masking = True

    def call(self, inputs):
        batch_max_length = tf.shape(inputs)[1]
        return inputs + self.pos_encodings[:, :batch_max_length]

    def get_config(self):
        config = super().get_config()
        config.update({"max_length": self.max_length, "embed_size": self.embed_size})
        return config


# --------------------------------------------------------------------------
# Loading (cached so the ~260MB model only loads once per session)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path):
    # compile=False: we only need this model for inference, so skip
    # reconstructing the training-time optimizer/LR-schedule/loss entirely.
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"PositionalEncoding": PositionalEncoding},
        compile=False,
    )


@st.cache_resource(show_spinner="Loading vocabularies...")
def load_vectorizers(vocab_path):
    with open(vocab_path, "rb") as f:
        data = pickle.load(f)

    max_length = data["max_length"]

    vec_en = tf.keras.layers.TextVectorization(
        max_tokens=data["vocab_size_en"], output_sequence_length=max_length
    )
    vec_en.set_vocabulary(data["en_vocab"])

    vec_gu = tf.keras.layers.TextVectorization(
        max_tokens=data["vocab_size_gu"], output_sequence_length=max_length
    )
    vec_gu.set_vocabulary(data["gu_vocab"])

    return vec_en, vec_gu, max_length


# --------------------------------------------------------------------------
# Inference — mirrors the notebook's translate() / beam_search() exactly
# --------------------------------------------------------------------------
def translate_greedy(sentence_en, model, vec_en, vec_gu, max_length):
    translation = ""
    for word_idx in range(max_length):
        X = vec_en(tf.constant([sentence_en]))
        X_dec = vec_gu(tf.constant(["startofseq " + translation]))
        y_logits = model((X, X_dec), training=False).numpy()[0, word_idx]
        predicted_word_id = np.argmax(y_logits)
        predicted_word = vec_gu.get_vocabulary()[predicted_word_id]
        if predicted_word == "endofseq":
            break
        translation += " " + predicted_word
    return translation.strip()


def beam_search(sentence_en, model, vec_en, vec_gu, max_length, beam_width=3):
    vocab = vec_gu.get_vocabulary()
    eos_id = vocab.index("endofseq")

    X = vec_en(tf.constant([sentence_en]))
    X_dec = vec_gu(tf.constant(["startofseq"]))
    y_logits = model((X, X_dec), training=False).numpy()[0, 0]
    y_logp = tf.nn.log_softmax(y_logits).numpy()
    top_k = np.argsort(-y_logp)[:beam_width]
    candidates = [(y_logp[idx], [idx], idx == eos_id) for idx in top_k]

    for step in range(1, max_length):
        all_candidates = []
        for log_prob, token_ids, finished in candidates:
            if finished:
                all_candidates.append((log_prob, token_ids, True))
                continue
            translation = " ".join(vocab[t] for t in token_ids if t not in (0, eos_id))
            X_dec = vec_gu(tf.constant(["startofseq " + translation]))
            y_logits = model((X, X_dec), training=False).numpy()[0, step]
            y_logp = tf.nn.log_softmax(y_logits).numpy()
            top_k = np.argsort(-y_logp)[:beam_width]
            for idx in top_k:
                new_log_prob = log_prob + y_logp[idx]
                all_candidates.append((new_log_prob, token_ids + [idx], idx == eos_id))

        def score(c):
            log_prob, token_ids, _ = c
            return log_prob / len(token_ids)

        candidates = sorted(all_candidates, key=score, reverse=True)[:beam_width]
        if all(c[2] for c in candidates):
            break

    def score(c):
        log_prob, token_ids, _ = c
        return log_prob / len(token_ids)

    ranked = sorted(candidates, key=score, reverse=True)
    results = []
    for log_prob, token_ids, _ in ranked:
        words = [vocab[t] for t in token_ids if t not in (0, eos_id)]
        results.append((" ".join(words).strip(), score((log_prob, token_ids, None))))
    return results  # list of (translation, avg_log_prob), best first


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🔤 English → Gujarati")
st.caption("From-scratch Transformer NMT, decoded with greedy or beam search.")

with st.sidebar:
    st.header("Model files")
    model_path = st.text_input("Model path (.keras)", value=DEFAULT_MODEL_PATH)
    vocab_path = st.text_input("Vocab path (.pkl)", value=DEFAULT_VOCAB_PATH)
    st.divider()
    st.header("Decoding")
    beam_width = st.slider("Beam width", min_value=1, max_value=8, value=3)
    show_all_beams = st.checkbox("Show all beam candidates", value=False)
    st.caption("Beam width 1 is equivalent to greedy decoding.")

if not Path(model_path).exists() or not Path(vocab_path).exists():
    st.warning(
        f"Couldn't find `{model_path}` and/or `{vocab_path}` next to this app. "
        "Download them from your Kaggle notebook's Output tab and place them "
        "in this folder (or point the sidebar paths at them)."
    )
    st.stop()

model = load_model(model_path)
vec_en, vec_gu, max_length = load_vectorizers(vocab_path)

sentence = st.text_area("English sentence", placeholder="I like soccer and also going to the beach")

col1, col2 = st.columns(2)
run_greedy = col1.button("Translate (greedy)", use_container_width=True)
run_beam = col2.button("Translate (beam search)", use_container_width=True, type="primary")

if (run_greedy or run_beam) and not sentence.strip():
    st.error("Enter a sentence first.")
elif run_greedy and sentence.strip():
    with st.spinner("Translating..."):
        result = translate_greedy(sentence.strip(), model, vec_en, vec_gu, max_length)
    st.subheader("Gujarati translation")
    st.write(result if result else "*(empty — model produced endofseq immediately)*")
elif run_beam and sentence.strip():
    with st.spinner("Translating..."):
        results = beam_search(sentence.strip(), model, vec_en, vec_gu, max_length, beam_width)
    st.subheader("Gujarati translation")
    if results:
        st.write(results[0][0] if results[0][0] else "*(empty)*")
        if show_all_beams:
            st.caption("All beam candidates (avg. log-probability):")
            for text, sc in results:
                st.text(f"{sc:.3f}  {text if text else '(empty)'}")
    else:
        st.write("*(no candidates)*")

st.divider()
st.caption(
    "Note: inference re-runs the full decoder from scratch for every output "
    "token (no KV-caching), so translation of longer sentences can take a "
    "few seconds, especially on CPU."
)
