## Strikingly Support Center Chatbot (LangChain + FAISS)

### Project Goal
Build a retrieval‑augmented QA (RAG) chatbot over Strikingly Support Center (Zendesk) articles:
- Index support content with OpenAI Embeddings into a FAISS vector store
- Retrieve relevant chunks per question and generate grounded answers
- Return answer with source URLs for traceability

### Why Two Environments (Two Tasks)
- Task 1 uses the legacy `VectorDBQAWithSourcesChain`, which is stable with older langchain/openai versions.
- Task 2 uses the modern `ConversationalRetrievalChain`, recommended for better UX and includes prompt‑length safeguards.
- Each task has its own dependency set and its own FAISS pickle format. Mixing environments and stores can cause deserialization failures.

Mapping:
- Task 1 → virtual env: `.venv-task1`, store: `faiss_store_task1.pkl`
- Task 2 → virtual env: `.venv-task2`, store: `faiss_store.pkl`

### Prerequisites
- Python 3.10+
- A `.env` file at project root:
  ```
  OPENAI_API_KEY=your_api_key
  ```

### Quickstart (Copy & Paste)
```bash
# Task 1
python3 -m venv .venv-task1
source .venv-task1/bin/activate
pip install --upgrade pip
pip install langchain==0.0.117 openai==0.27.0 tiktoken==0.3.0 python-dotenv==1.0.1 faiss-cpu==1.8.0 bs4==0.0.1 requests==2.28.2 xmltodict==0.13.0
python create_embeddings.py --mode zendesk --zendesk "https://support.strikingly.com/api/v2/help_center/en-us/articles.json" --store faiss_store_task1.pkl
python ask_question.py --mode task1 --store faiss_store_task1.pkl "Your question"

# Task 2
deactivate 2>/dev/null || true
python3 -m venv .venv-task2
source .venv-task2/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python create_embeddings.py --mode zendesk --zendesk "https://support.strikingly.com/api/v2/help_center/en-us/articles.json" --store faiss_store.pkl
python ask_question.py --mode task2 --store faiss_store.pkl "Your question"
```

### Testing
- Single‑turn QA (recommended: Task 2)
  ```bash
  source .venv-task2/bin/activate
  python ask_question.py --mode task2 --store faiss_store.pkl "How do I ...?"
  ```

- Legacy compatibility check (Task 1)
  ```bash
  source .venv-task1/bin/activate
  python ask_question.py --mode task1 --store faiss_store_task1.pkl "How do I ...?"
  ```

- Optional multi‑turn chat demo
  ```bash
  python start_chat_app.py
  ```

### Notes
- Always keep environment and store files matched (Task 1 ↔ `faiss_store_task1.pkl`, Task 2 ↔ `faiss_store.pkl`).
- Task 2 includes safeguards against prompt overflow: MMR retrieval (k=2, fetch_k=8) and `max_tokens_limit=3000`.
- If you prefer sitemap ingestion instead of Zendesk, use:
  ```bash
  python create_embeddings.py --mode sitemap \
    --sitemap https://your.site/sitemap.xml \
    --filter https://your.site/blog/posts \
    --store faiss_store.pkl
  ```