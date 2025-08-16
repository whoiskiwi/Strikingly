import os
import pickle
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ChatVectorDBChain
from dotenv import load_dotenv

_template = """Given the following conversation and a follow up question,
rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are a helpful support assistant for the Strikingly Support Center.
You are given the following extracted parts of a long support article and a question.
Provide a concise, accurate, and friendly answer grounded strictly in the provided context.
If the answer is not in the context, say "Hmm, I'm not sure." and suggest keywords to search.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
If the user asks for topics unrelated to Strikingly or its help center, politely decline and redirect.
But Use the following pieces of context to answer the question at the end. 

Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    load_dotenv()
    llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-3.5-turbo")
    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain


if __name__ == "__main__":
    with open("faiss_store.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    qa_chain = get_chain(vectorstore)
    chat_history = []
    print("Chat with the Strikingly Support bot:")
    while True:
        print("Your question:")
        question = input()
        result = qa_chain({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print(f"AI: {result['answer']}")
        if "source_documents" in result and result["source_documents"]:
            unique = []
            for d in result["source_documents"]:
                src = getattr(d, "metadata", {}).get("source") if hasattr(d, "metadata") else None
                if src and src not in unique:
                    unique.append(src)
            if unique:
                print("Sources:")
                for s in unique:
                    print(s)
