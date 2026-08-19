# English → Gujarati NMT — Transformer

A from-scratch Transformer encoder–decoder for English→Gujarati translation, built in
TensorFlow/Keras and tuned to train inside a 10-hour session on a single 16GB GPU
(P100/T4-class).

Follows the encoder-decoder → attention → transformer progression from
*Hands-On Machine Learning* (Géron), Ch. 16, applied to a real ~3M-pair
English-Gujarati corpus.

## Dataset

[English-to-Gujarati Machine Translation Dataset](https://www.kaggle.com/datasets/parvmodi/english-to-gujarati-machine-translation-dataset)
(Kaggle, ~3M sentence pairs).

```bash
kaggle datasets download -d parvmodi/english-to-gujarati-machine-translation-dataset -p ./data --unzip
```

## Approach

- Vocabulary built directly from the corpus via `TextVectorization.adapt()` rather
  than an external vocab file (an earlier bundled-vocab approach had a ~36-38% OOV
  rate because the shipped vocab didn't match the actual training corpus). Capped
  at 32k tokens/side.
- Corpus pre-vectorized once into a `tf.data` pipeline instead of vectorizing raw
  strings inside the model on every training step.
- Transformer encoder-decoder (`embed_size=128`, `N=2` blocks, `num_heads=8`) with
  int32 token-id inputs, mixed precision, and logits-based loss.
- Training budget trimmed to ~1.5M pairs (from the full ~2.6-3M) to trade raw data
  volume for more epochs within the 10-hour GPU quota.

## Notebook

[`eng_guj_transformer.ipynb`](./eng_guj_transformer.ipynb) — full pipeline: setup,
data loading/cleaning, vocabulary, `tf.data` pipeline, model, training, and
evaluation.

## Demo app

[`app/`](./app) has a small Streamlit UI (greedy + beam search decoding) that
loads the trained model checkpoint and serves translations. See
[`app/README.md`](./app/README.md) for setup — you'll need to download
`transformer_model.keras` and `vectorizer_vocab.pkl` from the Kaggle
notebook's Output tab and drop them in `app/`.

## Status

Training/eval notebook, developed and run on Kaggle.
