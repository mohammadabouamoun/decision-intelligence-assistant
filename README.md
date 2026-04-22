# 📞 Decision Intelligence Assistant

An AI‑powered customer support assistant that answers user questions using **Retrieval‑Augmented Generation (RAG)** and predicts ticket priority with both a **trained ML classifier** and a **zero‑shot LLM**. Built with FastAPI, React, ChromaDB, Groq, and Docker.

**Live demo** (local): `docker compose up --build` → open `http://localhost`

## 📓 Colab Notebook

EDA, weak supervision labeling, feature engineering, and model comparison are documented in the Colab notebook:  
[Decision Intelligence Assistant – Colab Notebook](https://colab.research.google.com/drive/1C6C32Ml_pbrEabSXdHR2ipjQ4OegxvCg?usp=sharing)  

---

## 🚀 Features

- 🔍 **RAG answer** – retrieves similar past tickets from a vector database and generates a grounded answer.
- 💬 **Non‑RAG answer** – LLM answers without context (flexible but may hallucinate).
- 🤖 **ML priority predictor** – Random Forest classifier trained on engineered features (fast, zero cost).
- 🧠 **LLM zero‑shot priority predictor** – Groq LLM predicts urgency without training (slower, small cost).
- 📊 **Comparison table** – side‑by‑side view of both priority predictors with confidence, latency, and cost.
- 📄 **Source panel** – shows retrieved ticket snippets with similarity scores and metadata.
- 🐳 **Docker Compose** – one‑command start: backend, frontend, and vector DB with volume persistence.

---

## 🧠 Architecture
User query → React UI (nginx) → FastAPI backend → ChromaDB (retrieval) → Groq LLM (RAG & zero‑shot) → Random Forest (ML) → response

text

- **Frontend**: React + Vite, served by nginx (port 80). API calls are proxied to the backend via nginx reverse proxy.
- **Backend**: FastAPI (port 8000) with endpoints `/query` (main logic) and `/health`.
- **Vector DB**: ChromaDB (persistent, mounted volume) stores embeddings of tweet chunks.
- **LLM**: Groq `llama-3.3-70b-versatile` (free tier) for RAG, non‑RAG, and zero‑shot priority.
- **ML Model**: Random Forest trained on engineered features (length, word count, keyword presence, punctuation, caps ratio).

---

## 📊 Dataset

- **Source**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (2.8M tweets)
- **Sampling**: 50,000 inbound (customer) tweets (reservoir sampling, memory‑efficient)
- **Preprocessing**: lowercasing, remove URLs and mentions, drop nulls
- **Chunking**: each tweet treated as one chunk (`chunk_size=500`, no overlap)

---

## 🏷️ Weak Supervision (Priority Labeling)

A heuristic rule labels tweets as **urgent** (1) if any of these conditions hold:
- Contains urgency keywords (`refund`, `broken`, `cancel`, `help`, `urgent`, `problem`, etc.)
- At least two exclamation marks (`!!`)
- More than 50% of characters are uppercase (all‑caps)

This is **weak supervision** – the model learns the rule, not true urgency.  
**Consequence**: The ML model achieves near‑perfect accuracy (≥0.999) because the engineered features directly encode the rule.  
This is documented and accepted.

---

## 📈 Feature Engineering & ML Baseline

Engineered features for each tweet:

| Feature | Description |
|---------|-------------|
| `length` | Number of characters |
| `word_count` | Number of words |
| `exclamation_count` | Count of `!` |
| `question_count` | Count of `?` |
| `caps_ratio` | Proportion of uppercase letters |
| `urgent_keyword_count` | Number of urgency keywords |
| `has_urgent_keyword` | Binary indicator |

**Models compared** (5‑fold CV + test set):
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

**Winner**: Random Forest (Test F1 = 0.9991, Test Accuracy = 0.9993) – saved as `backend/models/ml_model.pkl`.

---

## 🔁 Retrieval (RAG)

- **Embedding model**: `all-MiniLM-L6-v2` (sentence‑transformers)
- **Vector store**: ChromaDB (persistent, collection `support_tickets_chunks`)
- **Retrieval**: Top‑k similar chunks (default 5) based on cosine similarity.

---

## 💬 LLM Generation (Groq)

- **RAG prompt**: includes retrieved chunks as context.
- **Non‑RAG prompt**: answers without context.
- **Zero‑shot priority prompt**: asks the LLM to classify urgency and output JSON with label & confidence.
- **Metrics**: latency (ms) and estimated cost ($) are returned for every LLM call.

---

## 🖥️ Frontend (React)

- Input box for user query.
- Displays:
  - RAG and non‑RAG answers with latency & cost.
  - Comparison table for ML vs LLM priority (label, confidence, latency, cost).
  - Source panel with retrieved chunks (text, similarity, metadata).
- Styling: simple CSS, responsive.

---

## 🐳 Docker Compose

The entire stack is containerised and orchestrated with Docker Compose.

| Service | Dockerfile | Port | Description |
|---------|------------|------|-------------|
| `backend` | `Dockerfile.backend` | 8000 | FastAPI app |
| `frontend` | `Dockerfile.frontend` | 80 | React app served by nginx (reverse proxy) |

**Volumes**:
- `./data/chroma_db:/app/data/chroma_db` – persist vector database.
- `./logs:/app/logs` – persist query logs.
- `~/.cache/huggingface:/root/.cache/huggingface` – cache sentence‑transformers model.

**Network**: Internal Docker network allows `frontend` to reach `backend` via service name `backend:8000`.

---

## 📦 How to Run

### Prerequisites
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Git

### 1. Clone the repository
```bash
git clone https://github.com/mohammadabouamoun/decision-intelligence-assistant.git
cd decision-intelligence-assistant
2. Set up environment variables
Create a .env file in the project root with your Groq API key:

text
GROQ_API_KEY=your_groq_api_key
3. Build and start the stack
bash
docker compose up --build
Wait for the build to complete (first time may take 10‑15 minutes).
The backend will be ready when you see Application startup complete.
The frontend will be served at http://localhost.

4. Use the application
Open http://localhost in your browser.

Type a query (e.g., “My phone is broken and I need a refund”).

Click Ask.

View the four answers, comparison table, and retrieved tickets.

5. Stop the stack
bash
docker compose down
📓 Colab Notebook
EDA, weak supervision labeling, feature engineering, and model comparison are documented in the Colab notebook:
Decision Intelligence Assistant – Colab Notebook
(Set sharing to “Anyone with the link → Viewer”)

📈 Results & Trade‑off Analysis
Priority Predictors Comparison (per query, from logs)
Predictor	Accuracy (test set)	Latency (ms)	Cost (USD)
ML Classifier (Random Forest)	0.9993	~10	0.000000
LLM Zero‑shot	~0.80 (estimated)	~200‑500	~0.0001
Recommendation for 10,000 tickets/hour:

Deploy the ML classifier for real‑time priority classification – it is 20‑50× faster and has zero incremental cost.

Use the LLM zero‑shot as a fallback for complex edge cases or when an explanation is required.

RAG vs Non‑RAG
RAG provides grounded answers (when relevant tickets exist) but may say “cannot be found” if no match.

Non‑RAG is more flexible and always produces an answer, but may be less accurate or hallucinate.

📁 Project Structure (Selected)
text
decision-intelligence-assistant/
├── backend/
│   ├── app/
│   │   └── main.py
│   ├── models/
│   │   ├── ml_model.pkl
│   │   └── feature_columns.pkl
│   ├── routers/
│   │   └── query.py
│   ├── utils/
│   │   ├── retrieval.py
│   │   ├── llm_client.py
│   │   ├── ml_predictor.py
│   │   ├── logger.py
│   │   └── helpers.py
│   ├── data_preparation.py
│   └── train_models.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── App.css
│   ├── nginx.conf
│   └── ...
├── data/
│   ├── chroma_db/
│   └── features.csv
├── logs/
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── requirements.txt
└── README.md
🙏 Acknowledgements
Dataset: Customer Support on Twitter

LLM: Groq (free tier)

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Vector DB: Chroma

Built with FastAPI, React, Vite, nginx, Docker Compose

