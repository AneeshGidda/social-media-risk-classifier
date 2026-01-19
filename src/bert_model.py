import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import warnings
warnings.filterwarnings("ignore", message=".*np.object.*", category=FutureWarning)

from pathlib import Path
import tomli
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Load configuration from config.toml
ROOT_DIR = Path(__file__).resolve().parent.parent
with open(ROOT_DIR / "config.toml", "rb") as f:
    config = tomli.load(f)

paths = config["paths"]
FEATURES_DIR = ROOT_DIR / paths["features_dir"]
MODEL_DIR = ROOT_DIR / paths["model_dir"]

# Import the data preprocessing entrypoint
# This generates and saves preprocessed feature artifacts to disk
from src.data_processing import main as run_data_processing


# -----------------------------------
# Preprocessed feature artifact checks
# -----------------------------------

def preprocessed_features_exist():
    """
    Check whether preprocessed feature artifacts already exist on disk.

    These files are derived from the raw dataset via tokenization.
    If they exist, we reuse them to avoid repeating expensive preprocessing.
    """
    required_files = [
        "x_train_input_ids.pt",
        "x_train_attention_mask.pt",
        "x_test_input_ids.pt",
        "x_test_attention_mask.pt",
        "y_train.pt",
        "y_test.pt",
    ]
    return all((FEATURES_DIR / f).exists() for f in required_files)


def main():
    # -----------------------------------
    # Preprocessing control flow
    # -----------------------------------
    if not preprocessed_features_exist():
        print("Preprocessed feature artifacts not found. Running data processing...")
        run_data_processing()
    else:
        print("Using existing preprocessed feature artifacts.")

    # -----------------------------------
    # Device selection (cross-platform)
    # -----------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # -----------------------------------
    # Load preprocessed feature artifacts
    # -----------------------------------
    x_train_input_ids = torch.load(FEATURES_DIR / "x_train_input_ids.pt")
    x_train_attention_mask = torch.load(FEATURES_DIR / "x_train_attention_mask.pt")
    x_test_input_ids = torch.load(FEATURES_DIR / "x_test_input_ids.pt")
    x_test_attention_mask = torch.load(FEATURES_DIR / "x_test_attention_mask.pt")

    y_train = torch.load(FEATURES_DIR / "y_train.pt")
    y_test = torch.load(FEATURES_DIR / "y_test.pt")

    # -----------------------------------
    # Build PyTorch datasets and loaders
    # -----------------------------------
    train_dataset = TensorDataset(
        x_train_input_ids,
        x_train_attention_mask,
        y_train
    )
    test_dataset = TensorDataset(
        x_test_input_ids,
        x_test_attention_mask,
        y_test
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # -----------------------------------
    # Model initialization
    # -----------------------------------
    bert_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    bert_model.to(device)

    # -----------------------------------
    # Optimizer and loss
    # -----------------------------------
    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # -----------------------------------
    # Training loop
    # -----------------------------------
    epochs = 3
    for epoch in range(epochs):
        bert_model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # -----------------------------------
        # Evaluation loop
        # -----------------------------------
        bert_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for input_ids, attention_mask, labels in test_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        test_acc = correct / total

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- train loss: {train_loss:.4f} "
            f"- train acc: {train_acc:.4f} "
            f"- test acc: {test_acc:.4f}"
        )

    # -----------------------------------
    # Save trained model and tokenizer
    # -----------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    bert_model.save_pretrained(MODEL_DIR)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.save_pretrained(MODEL_DIR)

    print(f"Model saved to: {MODEL_DIR.resolve()}")


if __name__ == "__main__":
    main()
