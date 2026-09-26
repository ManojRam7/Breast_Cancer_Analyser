# Breast Cancer Predictor

Classifies breast tumour samples as malignant or benign from 30 measurements of cell nuclei (radius,
texture, concavity and so on) in the Wisconsin Diagnostic Breast Cancer dataset. A scaled logistic
regression is trained by a reproducible script, and a Streamlit app lets you adjust each measurement
and see the predicted class and probabilities.

**Live app:** https://breastcancerpredictor-app.streamlit.app

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B?logo=streamlit&logoColor=white)

## Results

Stratified 80/20 split (455 training, 114 test samples), `random_state=42`:

| Metric | Test set |
|---|---|
| Accuracy | **0.982** |
| Precision | 0.986 |
| Recall | 0.986 |
| F1 | 0.986 |
| ROC AUC | **0.996** |

Precision, recall and F1 are for the benign class (label 1 in scikit-learn's version of the data).
`scripts/train_model.py` writes these figures to `artifacts/metrics.json` on every run.

## Approach

- **Data:** 569 samples, 30 numeric features, 212 malignant and 357 benign, loaded with
  `sklearn.datasets.load_breast_cancer`. `EDA.IPYNB` checks feature ranges and class balance.
- **Model:** `StandardScaler` + `LogisticRegression` (liblinear) in one pipeline, so scaling is
  learned only from the training split. A linear model was chosen because it is accurate on this
  data, fast, and its coefficients are easy to explain.
- **Evaluation:** stratified hold-out split to keep the class balance, with accuracy, precision,
  recall, F1 and ROC AUC.
- **App:** `Streamlit_app.py` starts every slider at the dataset median, shows the model's test
  metrics, and trains the model on first launch if no saved model is present.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_model.py      # trains and saves artifacts/model.joblib and metrics.json
streamlit run Streamlit_app.py
pytest -q
```

## Project structure

```text
EDA.IPYNB                          dataset checks
src/breast_cancer_predictor/
    config.py                      paths and training settings
    data.py                        dataset loading
    modeling.py                    scikit-learn pipeline
    train.py                       train, evaluate, save
    predict.py                     load the saved model; default inputs
scripts/train_model.py             training entry point
Streamlit_app.py                   web app
tests/test_training_pipeline.py    end-to-end training test
docs/                              methodology and runbook
```

## Disclaimer

A machine learning exercise on a public benchmark dataset. It is not a medical device and must not
be used for diagnosis.
