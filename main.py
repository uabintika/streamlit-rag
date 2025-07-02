import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain import hub
import bs4

# Load .env
load_dotenv()
token = os.getenv("SECRET")
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

# Function to clean text
def clean_text(text: str) -> str:
    text = re.sub(r'\[\d+\]', '', text)  # Remove citation brackets like [1], [2]
    text = re.sub(r'(?<=[.?!])\s+(?=[A-Z])', '\n\n', text)  # Add paragraph breaks
    return text.strip()

# Load local file
loader1 = TextLoader("data/alytus_wiki.txt")
docs1 = loader1.load()

# Scrape Wikipedia Alytus page
loader2 = WebBaseLoader(
    web_paths=("https://en.wikipedia.org/wiki/Alytus",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(["p"]))
)
docs2 = loader2.load()

# Load another page (example: Alytus travel)
loader3 = WebBaseLoader(
    web_paths=("https://www.lonelyplanet.com/lithuania/alytus",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(["p"]))
)
docs3 = loader3.load()

# Combine and clean all documents
all_docs = docs1 + docs2 + docs3
for doc in all_docs:
    doc.page_content = clean_text(doc.page_content)

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(all_docs)

# Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://models.inference.ai.azure.com",
    api_key=token,
)

# Vector store
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# App UI
st.title("Alytus RAG Chatbot 🇱🇹")

def generate_response(input_text):
    llm = ChatOpenAI(base_url=endpoint, temperature=0.7, api_key=token, model=model)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke(input_text)
    st.info(result)

    st.subheader("📚 Sources")
    results = retriever.get_relevant_documents(input_text)
    for i, doc in enumerate(results, 1):
        with st.expander(f"Source {i}"):
            st.write(doc.page_content)

with st.form("my_form"):
    user_input = st.text_area("Ask about Alytus:", "What is Alytus known for?")
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(user_input)
