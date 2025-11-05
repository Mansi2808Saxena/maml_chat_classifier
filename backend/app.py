# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from model import load_meta_model
from utils import get_embedder, texts_to_embeddings, adapt_model_on_support, predict_with_model

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cpu")

# Load embedder and meta-model once at startup
print("Loading embedder...")
embedder = get_embedder()
print("Loading meta model...")
# Ensure the filename matches where you saved your pth
meta_model = load_meta_model(path="maml_meta_model.pth", device=DEVICE, input_dim=384, hidden_dim=128, output_dim=2)
print("Ready.")

@app.route("/")
def index():
    return "Few-shot MAML backend running."

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON:
    {
      "support": {
        "Complaint": ["text1", "text2"],
        "Query": ["text1", "text2"]
      },
      "query": "Some new conversation text",
      "inner_steps": 10,     # optional
      "lr": 0.001            # optional
    }
    """
    data = request.get_json(force=True)
    support = data.get("support", {})
    query = data.get("query", "")
    inner_steps = int(data.get("inner_steps", 10))
    lr = float(data.get("lr", 1e-3))

    if not support or not query:
        return jsonify({"error": "support (dict) and query (string) required"}), 400

    # Build class -> index mapping
    class_names = list(support.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # Build support dataset
    support_texts = []
    support_labels = []
    for cls, texts in support.items():
        for t in texts:
            support_texts.append(t)
            support_labels.append(class_to_idx[cls])

    if len(support_texts) < 1:
        return jsonify({"error": "Need at least one support example"}), 400

    # Convert to embeddings
    support_embeddings = texts_to_embeddings(embedder, support_texts)
    support_labels = torch.LongTensor(support_labels)

    # Adapt the meta-model on the support set
    adapted_model = adapt_model_on_support(meta_model, support_embeddings, support_labels,
                                          lr=lr, inner_steps=inner_steps, device=DEVICE)

    # Encode query
    query_emb = texts_to_embeddings(embedder, [query])

    # Predict
    probs, preds = predict_with_model(adapted_model, query_emb)
    pred_idx = int(preds[0].item())
    pred_prob = float(probs[0, pred_idx].item())

    result = {
    "category": class_names[pred_idx],
    "confidence": round(pred_prob * 100, 2)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
