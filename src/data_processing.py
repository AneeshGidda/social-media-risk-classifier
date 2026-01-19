from pathlib import Path
import tomli
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer

# Load configuration from config.toml
ROOT_DIR = Path(__file__).resolve().parent.parent
with open(ROOT_DIR / "config.toml", "rb") as f:
    config = tomli.load(f)

paths = config["paths"]
RAW_DATA_PATH = ROOT_DIR / paths["raw_data"]
FEATURES_DIR = ROOT_DIR / paths["features_dir"]


def main():
    # -----------------------------------
    # Load raw dataset
    # -----------------------------------
    data = pd.read_csv(RAW_DATA_PATH)

    # -----------------------------------
    # Clean and encode labels
    # -----------------------------------
    # Strip whitespace and map string labels to integer classes
    data["Suicide"] = data["Suicide"].astype(str).str.strip()
    label_map = {"Not Suicide post": 0, "Potential Suicide post": 1}
    data["Suicide"] = data["Suicide"].map(label_map)

    # Fail loudly if any labels could not be mapped
    if data["Suicide"].isna().any():
        bad = data.loc[data["Suicide"].isna(), "Suicide"].unique()
        raise ValueError(f"Unmapped labels found after cleaning: {bad}")

    # -----------------------------------
    # Extract text and labels
    # -----------------------------------
    x_data = data["Tweet"].values.astype(str)
    y_data = data["Suicide"].astype(int).values

    # -----------------------------------
    # Train / test split
    # -----------------------------------
    # Stratification preserves label distribution
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.2,
        random_state=42,
        stratify=y_data,
    )

    # -----------------------------------
    # Tokenizer initialization
    # -----------------------------------
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # -----------------------------------
    # Tokenize training data (PyTorch tensors)
    # -----------------------------------
    x_train_tokenized = tokenizer.batch_encode_plus(
        x_train,
        add_special_tokens=True,
        return_attention_mask=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    x_train_input_ids = x_train_tokenized["input_ids"]
    x_train_attention_mask = x_train_tokenized["attention_mask"]

    # -----------------------------------
    # Tokenize test data (PyTorch tensors)
    # -----------------------------------
    x_test_tokenized = tokenizer.batch_encode_plus(
        x_test,
        add_special_tokens=True,
        return_attention_mask=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    x_test_input_ids = x_test_tokenized["input_ids"]
    x_test_attention_mask = x_test_tokenized["attention_mask"]

    # -----------------------------------
    # Convert labels to PyTorch tensors
    # -----------------------------------
    # Use class indices (not one-hot) for CrossEntropyLoss
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # -----------------------------------
    # Persist preprocessed feature artifacts
    # -----------------------------------
    # These artifacts are reused across training runs to
    # avoid repeated tokenization of the raw dataset
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(x_train_input_ids, FEATURES_DIR / "x_train_input_ids.pt")
    torch.save(x_train_attention_mask, FEATURES_DIR / "x_train_attention_mask.pt")
    torch.save(x_test_input_ids, FEATURES_DIR / "x_test_input_ids.pt")
    torch.save(x_test_attention_mask, FEATURES_DIR / "x_test_attention_mask.pt")
    torch.save(y_train, FEATURES_DIR / "y_train.pt")
    torch.save(y_test, FEATURES_DIR / "y_test.pt")

    print(f"Saved preprocessed feature artifacts to: {FEATURES_DIR.resolve()}")

if __name__ == "__main__":
    main()
