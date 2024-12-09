# rag2riches
Sample RAG enhanced chatbot using Nvidia NIM for LLM.

There are 2 separate compenents
1. backend - (a FastAPI web service, using LangChain and NIM)
2. webapp - (a streamlit chatbot webapp)

## backend
### Requirements
Nvidia API Key
LangChain API Key

### How to run
~~~
export NVIDIA_API_KEY="<your Nvidaia API key>"
export LANGCHAIN_API_KEY="<your LangChain API key>"
cd backend
python -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
fastapi dev main.py
~~~

## webapp
### How to run
~~~
cd webapp
python -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run bot.py
~~~

