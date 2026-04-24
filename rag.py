from multiprocessing import process
import importlib
from pathlib import Path
import uuid
import params
from dotenv import  load_dotenv
from groq.types import embedding
from langchain_community import embeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders.base_o365 import CHUNK_SIZE
from langchain_huggingface import HuggingFaceEmbeddings
# Naya import
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

from sqlalchemy.cyextension.collections import Collection
from sqlalchemy.dialects.oracle import vector

load_dotenv()
#write the main function to store the url as array
CHUNK_SIZE = 1000
VECTORSTORE_DIR = Path(__file__).parent /"resources"/ "vectorstore"
COLLECTION_NAME = "real_estate"
llm = None
vector_store = None
def intialize_components():
    global llm,vector_store
    if(llm is  None):
        llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.9,
        max_tokens=500,

        # other params...
            )

    ef = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs = {'trust_remote_code': True})
    if(vector_store is None ):
        vector_store  = Chroma(collection_name=COLLECTION_NAME, embedding_function=ef)
def main():
    urls = ['https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html','https://www.cnbc.com/2026/03/18/fed-interest-rate-decision-march-2026.html']
    print(urls)
    return urls


def process_urls(urls):
    global vector_store
    intialize_components()

    # Pehle wale documents delete karne ka sahi tarika bina collection udaye
    existing_ids = vector_store.get()['ids']
    if len(existing_ids) > 0:
        vector_store.delete(ids=existing_ids)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }

    loader = UnstructuredURLLoader(urls=urls, headers=headers)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", "."], chunk_size=CHUNK_SIZE,
                                                   chunk_overlap=0)
    docs = text_splitter.split_documents(data)

    uuids = [str(uuid.uuid4()) for _ in range(len(docs))]

    # Ab ye bina error ke chalega kyunki collection abhi bhi exist karta hai
    vector_store.add_documents(docs, ids=uuids)
    print("Documents added successfully!")


if __name__ == "__main__":
    urls = main()
    process_urls(urls)
    results =  vector_store.similarity_search(
        "30 years mortgage rates",
        k=3,

    )
    print(results)
