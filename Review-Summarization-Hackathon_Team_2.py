import os
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
# from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
import openai
import streamlit as st


# Setup environment variables for API keys and endpoint
os.environ["AZURE_OPENAI_API_KEY"] = "251bab4a00c6407c9edb695c5f450a5d"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://surface-llm-poc.openai.azure.com/"

# Function to read text from file
def get_text_from_file(txt_file):
    with open(txt_file, 'r',encoding='latin') as file:
        text = file.read()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and store embeddings
def get_vector_store(text_chunks):
    embeddings = AzureOpenAIEmbeddings(azure_deployment="Large_Embeddings")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index_new")
    return vector_store

# Function to setup the vector store (to be run once or upon text update)
def setup(txt_file_path):
    raw_text = get_text_from_file(txt_file_path)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    print("Setup completed. Vector store is ready for queries.")

# Function to get conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context,if question asked about future based on the provided data give answers by prediction, make sure to provide all the details, if the answer is not in
    provided context just say,"answer is not available in the context", don't provide the wrong answer. And if query is about comparision between 2 or 3 devices, compare all the aspects of the device and provide summary. And if the query is regarding any specification like processors, RAM, Storage, screensize without mentioning any device name, Summarize all the reviews for that mentioned specification\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = AzureChatOpenAI(
    azure_deployment="SurfaceLLM",
    api_version='2023-12-01-preview')
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user queries using the existing vector store
def query(user_question, vector_store_path="faiss_index_new"):
    embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    chain = get_conversational_chain()
    docs = vector_store.similarity_search(user_question)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    logo_url = "https://ipsgwalior.org/images/MuSigma.jpg"
    st.image(logo_url, width=150)
    st.title("Retail Consumer Reviews Summarization tool") 
    # Predefined text file path
    # txt_file_path = "Reviews with Specs.txt"
 
    # Automatically call setup with the predefined file on startup
    # if not os.path.exists("faiss_index_new"):
        # setup(txt_file_path)
 
    # User input for query
    user_question = st.text_input("Ask your question")
 
    if st.button("Submit"):
        if os.path.exists("faiss_index_new"):
            response = query(user_question)
            st.write("Response:", response)
        else:
            st.error("The vector store setup has failed. Please check the file path and try again.")
 
if __name__ == "__main__":
    main()


# In[ ]:




