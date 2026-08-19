# English -> Gujarati Neural Machine Translation

An end-to-end English-to-Gujarati translation system built with a custom Transformer encoder-decoder in TensorFlow/Keras and deployed as an interactive Streamlit application.

**Live demo:** [Open the deployed translator](https://english-gujarati-nmt-fl9rkfkmgimvgdqngbkepp.streamlit.app/)

The project covers the complete machine-learning workflow: dataset preparation, text vectorization, Transformer design, GPU training, checkpoint export, custom model deserialization, and cloud deployment.

## Highlights

- Custom Transformer encoder-decoder implemented with TensorFlow/Keras
- English and Gujarati vocabularies learned directly from the training corpus
- Pre-vectorized `tf.data` input pipeline with integer token IDs
- Registered sinusoidal positional encoding Keras layer
- Greedy decoding and configurable beam search
- Streamlit interface for interactive translation
- Approximately 260 MB model checkpoint managed with Git LFS
- Public deployment on Streamlit Community Cloud

## Model Architecture

```text
English sentence -> TextVectorization -> Transformer encoder
                                             |
Gujarati tokens <- autoregressive Transformer decoder
```

The model uses 128-dimensional embeddings, two Transformer blocks, eight attention heads, mixed-precision training, and logits-based loss computation. The training notebook uses approximately 1.5 million sentence pairs to fit a single-GPU development budget while retaining a substantial corpus.

## Demo Features

- English sentence input
- Greedy decoding for a direct prediction
- Beam search with adjustable beam width
- Optional display of all beam candidates and average log-probabilities
- Cached model and vocabulary loading for repeated inference

Try an input such as:

```text
I like soccer and also going to the beach
```

## Dataset

The model was trained using the [English-to-Gujarati Machine Translation Dataset](https://www.kaggle.com/datasets/parvmodi/english-to-gujarati-machine-translation-dataset) on Kaggle. The source corpus contains approximately three million sentence pairs.

```bash
kaggle datasets download \
  -d parvmodi/english-to-gujarati-machine-translation-dataset \
  -p ./data --unzip
```

## Repository Structure

```text
.
├── eng_guj_transformer.ipynb   # Training, evaluation, and export pipeline
├── app/
│   ├── app.py                  # Streamlit inference application
│   ├── requirements.txt        # Deployment dependencies
│   ├── transformer_model.keras # Trained model, stored with Git LFS
│   ├── vectorizer_vocab.pkl    # Vocabularies and sequence settings
│   └── README.md               # App-specific setup notes
├── .gitattributes              # Git LFS configuration
└── runtime.txt                 # Streamlit Cloud Python runtime
```

## Run Locally

```bash
git clone https://github.com/Neel58/english-gujarati-nmt.git
cd english-gujarati-nmt
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

The app expects `transformer_model.keras` and `vectorizer_vocab.pkl` in `app/`. Install Git LFS before cloning or run this after cloning:

```bash
git lfs install
git lfs pull
```

## Deployment

The live app is deployed from the `main` branch on Streamlit Community Cloud.

| Setting | Value |
| --- | --- |
| Repository | `Neel58/english-gujarati-nmt` |
| Branch | `main` |
| Main file | `app/app.py` |
| Python | `3.11` |

Dependencies are read from `app/requirements.txt`, and the model artifact is downloaded through Git LFS.

## Engineering Notes

- `st.cache_resource` loads the model and vocabularies once per Streamlit process.
- The custom `PositionalEncoding` layer is registered during Keras deserialization.
- Beam search ranks candidates by length-normalized average log-probability.
- Inference currently recomputes decoder predictions for each output token and does not use KV caching, so longer translations can take several seconds on CPU.

## Future Improvements

- Add BLEU, chrF, and Gujarati-specific evaluation reports
- Add attention and translation-quality visualizations
- Introduce decoder KV caching for faster inference
- Experiment with subword tokenization and multilingual baselines
- Add automated serialization and decoding tests

## License

This is a portfolio and learning project. Review the dataset terms and the licenses of third-party dependencies before redistributing trained artifacts or derived datasets.
