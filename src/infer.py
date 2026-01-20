import warnings
warnings.filterwarnings("ignore", message=".*np.object.*", category=FutureWarning)

from pathlib import Path
import tomli
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Load configuration from config.toml
ROOT_DIR = Path(__file__).resolve().parent.parent
with open(ROOT_DIR / "config.toml", "rb") as f:
    config = tomli.load(f)

paths = config["paths"]
MODEL_DIR = ROOT_DIR / paths["model_dir"]

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
# Load trained model and tokenizer
# -----------------------------------
# These were saved after training and must be reused
# to ensure identical tokenization and model behavior
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()


# -----------------------------------
# Label mapping (matches training)
# -----------------------------------
label_map = {
    0: "Not Suicide post",
    1: "Potential Suicide post"
}


def predict(text):
    """
    Run inference on a single input string.

    Returns:
        - predicted label (string)
        - confidence score (softmax probability)
    """
    # -----------------------------------
    # Tokenize input text
    # -----------------------------------
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # -----------------------------------
    # Forward pass (inference only)
    # -----------------------------------
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return label_map[pred], probs[0][pred].item()


if __name__ == "__main__":
    # -----------------------------------
    # Example inference
    # -----------------------------------
    text = "I feel completely hopeless and exhausted."
    label, confidence = predict(text)

    print(f"Prediction: {label} (confidence: {confidence:.4f})")
