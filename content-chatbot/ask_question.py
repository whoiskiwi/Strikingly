import argparse
import os
import pickle

from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


load_dotenv()

parser = argparse.ArgumentParser(description='Strikingly Support Q&A')
parser.add_argument('question', type=str, help='Your question')
parser.add_argument('--mode', type=str, choices=['task1', 'task2'], default='task2', help='Choose task implementation to run')
parser.add_argument('--store', type=str, default='faiss_store.pkl', help='Path to FAISS store pickle file')
args = parser.parse_args()

with open(args.store, "rb") as f:
    store = pickle.load(f)

# Ensure the vector store uses a compatible embedding function at runtime
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
try:
    store.embedding_function = embeddings.embed_query
except Exception:
    pass

if args.mode == 'task1':
    # Task1: use Chat model + stuff chain to avoid chat/completions mismatch in old stack
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-3.5-turbo",
    )
    combine_chain = load_qa_with_sources_chain(llm, chain_type="stuff")
    chain = VectorDBQAWithSourcesChain(
        combine_documents_chain=combine_chain,
        vectorstore=store,
        return_source_documents=True,
        k=4,
    )
    result = chain({"question": args.question})
else:
    # Task2: import chat-only deps lazily to keep Task1 compatible with older langchain in a separate venv
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.prompts.prompt import PromptTemplate

    retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 8})
    llm = ChatOpenAI(temperature=0.3, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-3.5-turbo")
    friendly_template = (
        "You are a helpful, empathetic assistant for the Strikingly Support Center.\n"
        "Use ONLY the provided context to answer the user's question.\n"
        "- Be concise and friendly.\n"
        "- Prefer numbered steps or short bullet points.\n"
        "- If the answer is not in the context, say \"Hmm, I'm not sure.\" and suggest keywords to search.\n\n"
        "Question: {question}\n"
        "=========\n"
        "{context}\n"
        "=========\n"
        "Answer in Markdown:"
    )
    QA_PROMPT = PromptTemplate(template=friendly_template, input_variables=["question", "context"])
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        # Hard cap on tokens passed to the LLM from concatenated docs
        max_tokens_limit=3000,
    )
    result = chain({"question": args.question, "chat_history": []})

print(f"Answer: {result['answer']}")

# Collect sources (works for both modes)
sources = []
if isinstance(result, dict) and result.get("source_documents"):
    for d in result["source_documents"]:
        src = getattr(d, "metadata", {}).get("source") if hasattr(d, "metadata") else None
        if src:
            sources.append(src)

# Task1 chain may return a newline-joined string under key 'sources'
if isinstance(result, dict):
    src_str = result.get("sources")
    if isinstance(src_str, str):
        for s in src_str.split("\n"):
            s = s.strip()
            if s:
                sources.append(s)

# Fallback: direct similarity search to surface URLs if chain didn't attach docs
if not sources:
    try:
        retrieved = store.similarity_search(args.question, k=3)
        for d in retrieved:
            src = getattr(d, "metadata", {}).get("source") if hasattr(d, "metadata") else None
            if src:
                sources.append(src)
    except Exception:
        pass

if sources:
    print("Source: (Internal-only)")
    for s in dict.fromkeys(sources):
        print(s)
