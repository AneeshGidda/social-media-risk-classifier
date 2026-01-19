# Social Media Risk Classifier

Machine learning project that fine-tunes a DistilBERT model to classify tweets for indicators of suicidal ideation. The goal of the project is to build a clear end-to-end NLP pipeline, covering data preprocessing, feature caching, model training, inference on new text inputs, and serving predictions through a lightweight API.

## Project Structure

- data/ – raw dataset files
- models/ – cached features and trained model artifacts
- src/ – preprocessing, training, and inference code
- api/ – API for serving model predictions
- frontend/ – demo frontend

## How to Run

### 1. Install dependencies

    pip install -r requirements.txt

### 2. Download the dataset

The dataset is hosted on Kaggle:  
https://www.kaggle.com/datasets/aunanya875/suicidal-tweet-detection-dataset

Download the dataset and place the raw CSV file inside the `data/` directory.

Update the dataset path in `src/config.py` if the filename differs from the default.

### 3. Train the model

    python3 -m src.train

### 4. Run inference on new text

    python3 -m src.infer

    ⚠️ **Note:** This feature is next to be implemented and is not yet available.

## Status

Work in progress.
