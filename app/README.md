# English → Gujarati — Streamlit demo

A small Streamlit UI around the trained Transformer NMT model, with greedy
and beam-search decoding.

## Setup

1. From your Kaggle notebook's **Output** tab, download:
   - `transformer_model.keras` (~260 MB)
   - `vectorizer_vocab.pkl` (~4 MB)

   Place both files in this folder (next to `app.py`). If you'd rather keep
   them elsewhere, the sidebar lets you point at custom paths.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run:
   ```bash
   streamlit run app.py
   ```

## What it does

- Loads the saved `.keras` model (registering the custom `PositionalEncoding`
  layer so it deserializes correctly) and rebuilds the two `TextVectorization`
  layers from the pickled vocabularies — this mirrors exactly what the
  notebook's `translate()` / `beam_search()` functions do, just wrapped in a UI.
- **Greedy** button: argmax decoding, one word at a time.
- **Beam search** button: keeps the top `beam_width` partial translations at
  each step (length-normalized by average log-probability), same algorithm as
  the notebook's `beam_search()`. Optionally shows all beam candidates, not
  just the best one.

## Notes / known limitations (carried over from the notebook)

- No KV-caching — each output token re-runs the full decoder stack, so
  translation isn't instant, especially on CPU. Fine for a demo; not meant
  for production-scale serving.
- Quality is bounded by whatever the underlying model achieved (see the
  notebook's OOV-rate / training notes) — this app doesn't change the model,
  just serves it.
