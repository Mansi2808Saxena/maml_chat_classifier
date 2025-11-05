# 🧠 Few-Shot Conversational Text Classifier (MAML + NLP)

This project demonstrates a **few-shot learning-based conversational intent classifier** built using **Model-Agnostic Meta-Learning (MAML)**.  
It adapts quickly to new conversation domains (e.g., complaints, queries, feedback) with very few examples.

---

## 🚀 Features

- Few-shot training with dynamic support examples
- Sentence-transformer embeddings (`all-MiniLM-L6-v2`)
- Flask backend with PyTorch integration
- Interactive frontend built with HTML/CSS/JS
- Adaptable for new intents or customer service domains

---

## 📦 Project Structure

backend/ → Flask + MAML model
frontend/ → Web interface
notebooks/ → Colab training code

---

## ⚙️ Setup Instructions

1. Clone the repo:
   git clone https://github.com/Mansi2808Saxena/maml_chat_classifier.git
   cd maml-chat-classifier/backend

2. Create a virtual environment:
   python -m venv venv
   venv\Scripts\activate # On Windows
   pip install -r requirements.txt

3. Run the Flask server:
   python app.py

Open frontend/index.html in your browser to test the UI.
