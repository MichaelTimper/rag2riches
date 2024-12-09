"""
FastAPI for sample RAG endpoint

"""
import os

from fastapi import FastAPI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, SeleniumURLLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIARerank
from langchain_nvidia_ai_endpoints.embeddings import NVIDIAEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from openai import OpenAI
from pydantic import BaseModel
from typing import List


NVIDIA_CLIENT = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = os.environ['NVIDIA_API_KEY'],
)

MODEL = ChatNVIDIA(model="nvidia/llama-3.1-nemotron-70b-instruct")
RERANKER = NVIDIARerank(model="nvidia/nv-rerankqa-mistral-4b-v3")
EMBEDDING = NVIDIAEmbeddings()

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Answer as a friendly support person based on the following context:\n<Documents>\n{context}\n</Documents>."
            ),
        ),
        (
            "user",
            "{question}"
        ),
    ]
)
VECTOR_STORE = None
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def add_documents_to_store(doc_urls):
    """
    Add web page content as documents to the VECTOR_STORE
    if the VECTOR_STORE does not exist create one
    """
    global VECTOR_STORE
    loader = WebBaseLoader(web_paths=doc_urls)
    splits = loader.load_and_split(TEXT_SPLITTER)
    # print(splits)
    if VECTOR_STORE is not None:
        VECTOR_STORE.add_documents(documents=splits)
    else:
        VECTOR_STORE = FAISS.from_documents(documents=splits, embedding=EMBEDDING)

"""
Load an intial document
"""
doc_urls = ["https://getrocketbook.com/pages/faqs"]
add_documents_to_store(doc_urls)


# Initialize app
app = FastAPI()


# User question
class Question(BaseModel):
    text: str


def generate_response(question: Question) -> str:
    """
    Generate a response to user question using documents
    from VECTORE_STORE
    """
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=RERANKER, base_retriever=VECTOR_STORE.as_retriever()
    )
    chain = (
        {"context": compression_retriever, "question": RunnablePassthrough()}
        | PROMPT
        | MODEL
        | StrOutputParser()
    )
    return chain.invoke(question.text)


@app.post("/question")
async def answer_question(question: Question) -> dict:
    """
    generate a response to an input question
    """
    response = generate_response(question)
    return {"answer": response}


@app.post("/documents")
async def add_documents(documents: List[str]) -> dict:
    """
    add other documents to VECTOR_STORE for RAG
    """
    add_documents_to_store(documents)
    return {"status": f"{len(documents)} added to store"}
