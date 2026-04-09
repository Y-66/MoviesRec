# MoviesRec - Conversational Movie Recommendation AI System

🥂 UI repository: [https://github.com/Y-66/movies-webui](https://github.com/Y-66/movies-webui)

MoviesRec is an intelligent, dialogue-driven movie recommendation system powered by LLMs and Collaborative Filtering algorithms. It combines the reasoning and conversational capabilities of Large Language Models (orchestrated via LangGraph) with the robust personalization of traditional recommendation algorithms (Singular Value Decomposition from scikit-surprise). 

The system understands user intentions (e.g., "Find me a sci-fi movie like Inception"), retrieves and filters metadata via SQL, ranks candidates using SVD-based collaborative filtering, enhances results through diversity filtering, and finally summarizes the recommendations into a natural, conversational response.

## 🎯 Key Features

- **Natural Language Understanding**: Uses LLMs to parse complex user requests and intents.
- **Dynamic Workflow Orchestration**: Powered by **LangGraph** to route requests through various analytical nodes (Intent Analysis, SQL Filtering, CF, Diversity, and Summarization).
- **Hybrid Recommendation Engine**:
  - **Collaborative Filtering**: SVD models trained on user-item ratings interactions (`scikit-surprise`).
  - **Content-Based & Diversity**: Feature extraction and clustering techniques to ensure varied and comprehensive recommendations.
- **RESTful API**: Fast and scalable web services built using **FastAPI** & **Uvicorn**.
- **Chat History & Session Management**: Tracks conversations locally using JSON-based persistent states.

---

## 🏗 Technical Architecture

The core of MoviesRec is built as a state-driven computational graph (`src/movies/graph.py`):

1. **Intent Analyzer Node**: Classifies user prompts and extracts key criteria (genres, actors, semantic searches).
2. **SQL Filter Node**: Dynamically translates criteria into SQL queries (or database filters) against the movie metadata (SQLite/CSV datasets).
3. **Collaborative Filtering (CF) Node**: Ranks the filtered movie candidates for specific users using pre-trained SVD models.
4. **Diversity Node**: Re-ranks the CF output to maximize the variety of genres or features.
5. **Summarize Node**: Synthesizes the final recommendations back into human-readable text.

### Tech Stack

- **Backend Framework:** FastAPI, Uvicorn, Python 3.10+
- **LLM / AI Orchestration:** LangChain, LangGraph, DeepAgents, LangChain-OpenAI
- **Machine Learning & Data Science:** Scikit-Surprise (SVD), Scikit-Learn (Clustering), Pandas, Numpy

---

## 📁 Project Structure

```text
MoviesRec/
├── API_DOCS.md                     # Documentation for FastAPI routes
├── main.py                         # Application entry point (FastAPI Server)
├── pyproject.toml                  # Project metadata and dependencies
├── README.md                       # This file
├── chat_history/                   # JSON blobs storing conversational states per session
├── datasets/                       # Raw/Aggregated data (movies.csv, ratings.csv, tags.csv, etc.)
├── db/                             # SQLite database files
├── models/                         # Serialized ML models (e.g., svd_model.pkl)
├── scripts/                        # Utility scripts (e.g., data_processing.py script)
├── src/
│   ├── algos/                      # 🧠 CORE RECOMMENDATION ALGORITHMS (See details below)
│   └── movies/                     # 🌐 WEB & LANGGRAPH APPLICATION
│       ├── agents/                 # Specialized LLM agents (Intent, Summarizer)
│       ├── api/                    # FastAPI application, routers (chat.py, system.py), and schemas
│       ├── nodes/                  # LangGraph operational nodes (cf_node, diversity_node, etc.)
│       ├── states/                 # State management definitions for LangGraph
│       └── utils/                  # Shared utilities (file storage, config loaders)
└── tests/                          # Automated tests (PyTest)
```

### 🧠 Deep Dive into `src/algos/`

The `algos` directory handles all offline model training, evaluation, and non-LLM machine learning processes. It acts as the algorithmic backbone for the Collaborative Filtering component inside the LangGraph nodes.

- **`svd_model_trainer.py`**: Handles offline dataset preparation and model training. It uses `surprise` to construct full training sets from user ratings and serializes (`.pkl`) the SVD model to the `models/` folder.
- **`svd_model_predictor.py`**: The runtime component that loads the pre-trained SVD model to predict ratings and generate top-N personalized movie lists.
- **`SVD_evaluate.py` & `SVD_error_analysis.py`**: Model evaluation scripts. They compute RMSE/MAE metrics on test sets, analyze error distributions (e.g., why a model heavily missed a prediction), and cross-validate hyper-parameters. 
- **`feature_extraction.py`**: Extracts content-based features from unstructured movie metadata (like genres, titles, and tags). Uses NLP techniques like TF-IDF or embeddings.
- **`feature_extraction_cluster_analysis.py`**: Performs clustering (e.g., K-Means or DBSCAN) on the extracted movie features. This helps the Diversity Node logically group similar movies to inject variety into the final recommendations.
- **`feature_extraction_eval.py`**: Evaluates the quality of extracted features and clusters (e.g., Silhouette scores) to tune the diversity engine.

---

## 🚀 Getting Started

### 1. Installation

Requires Python >= 3.10. It is recommended to use a virtual environment or conda:

```bash
# Clone the repository and navigate into it
cd MoviesRec

# Install dependencies (development mode)
pip install -e .
```

### 2. Configure Environment

Create a `.env` file in the root directory and supply the necessary API keys.
```env
OPENAI_API_KEY=your_openai_api_key_here
# Add TMDB keys or other keys if applicable
```

### 3. Prepare Data & Train Models

Before running the application, make sure the recommendation models are trained:

```bash
# Train the SVD model locally (ensure datasets are in the /datasets folder)
python -m src.algos.svd_model_trainer
```

### 4. Run the API Server

Start the interactive FastAPI backend:

```bash
python main.py
```
The server will start on `http://0.0.0.0:8000`. You can test endpoints via the Swagger UI at `http://localhost:8000/docs`.

---

## Testing

Run unit tests and verify core LangGraph workflow logic and API routes:

```bash
pytest tests/
```
