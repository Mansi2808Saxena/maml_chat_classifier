# backend/utils.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sentence_transformers import SentenceTransformer
from copy import deepcopy

# Load the embedder once (shared)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

def get_embedder():
    # re-usable embedder instance
    return SentenceTransformer(EMBED_MODEL_NAME)

def texts_to_embeddings(embedder, texts):
    """
    texts: list[str]
    returns: torch.FloatTensor of shape (len(texts), 384)
    """
    vecs = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return torch.from_numpy(np.array(vecs)).float()

def adapt_model_on_support(meta_model, support_embeddings, support_labels,
                           lr=1e-3, inner_steps=10, device=torch.device("cpu")):
    """
    Create a copy of meta_model, fine-tune on the support set for inner_steps and return adapted model.
    support_embeddings: torch.Tensor (N, 384)
    support_labels: torch.LongTensor (N,)
    """
    # Deep copy so we don't change the loaded meta model
    model = deepcopy(meta_model).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    support_embeddings = support_embeddings.to(device)
    support_labels = support_labels.to(device)

    for _ in range(inner_steps):
        preds = model(support_embeddings)
        loss = loss_fn(preds, support_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model

def predict_with_model(model, query_embeddings):
    """
    query_embeddings: torch.Tensor (M, 384)
    returns (probs, predicted_indices)
    """
    with torch.no_grad():
        logits = model(query_embeddings)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return probs.cpu(), preds.cpu()
