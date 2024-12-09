"""
Simple chatbot for testing RAG using content from web pages
"""
import requests
from requests.exceptions import HTTPError
import streamlit as st

# interact with FastAPI endpoint
BE_HOST = "http://127.0.0.1:8000"
RAG_URLS = set(["https://getrocketbook.com/pages/faqs"])


def get_answer(question):
    """
    call backend to get generated response to question
    """
    try:
        resp_json = make_json_request(f"{BE_HOST}/question", {"text": question})
        return resp_json["answer"]
    except HTTPError as e:
        return f"Error: {e}"


def add_documents(doc_urls):
    """
    call backend to add documents to RAG vector store
    """
    resp_json = make_json_request(f"{BE_HOST}/documents", doc_urls)
    return


def make_json_request(endpoint, json_data):
    resp = requests.post(endpoint, json=json_data, timeout=8000)
    resp.raise_for_status()
    resp_json = resp.json()
    return resp_json
    

with st.sidebar:
    input_urls = st.text_area("URLs", key="urls", value=",".join(RAG_URLS))
    "[Add URLs to use for RAG (comma separated)]"

st.title("RAG 2 Riches")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question := st.chat_input():
    if input_urls:
        urls = input_urls.split(",")
        urls_to_add = list(set(urls) - RAG_URLS)
        if urls_to_add:
            print(f"Adding URL(s) {urls_to_add}")
            add_documents(urls_to_add)
            RAG_URLS.update(set(urls_to_add))

    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    response = get_answer(question)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
