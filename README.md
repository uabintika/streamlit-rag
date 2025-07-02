# 🇱🇹 Alytus RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **LangChain**, and **FAISS**, designed to answer questions about the city of **Alytus, Lithuania** using multiple sources like Wikipedia, Lonely Planet, and a local `.txt` file.

## Features

- Scrapes and cleans web content (Wikipedia and Lonely Planet)
- Loads additional content from a local file
- Splits documents into manageable chunks
- Embeds documents using OpenAI's `text-embedding-3-small` model
- Stores vectors in a FAISS vector store
- Uses a ChatOpenAI model (`gpt-4.1-nano`) for answering questions
- Cleans up sources (removes [1], [2], etc. and formats into readable paragraphs)
- Displays sources used to answer each question
- UI powered by **Streamlit**

---

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/alytus-rag-chatbot.git
cd alytus-rag-chatbot
