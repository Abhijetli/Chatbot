import os
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load the sample document
loader = TextLoader("sample_data.txt")
documents = loader.load()

# Extract text content from Document objects
texts = [doc.page_content for doc in documents]

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
split_documents = text_splitter.split_documents(documents)  # Use split_documents for splitting

# Create TF-IDF embeddings
vectorizer = TfidfVectorizer()
split_texts = [chunk.page_content for chunk in split_documents]
document_embeddings = vectorizer.fit_transform(split_texts)

# Setup memory for the conversation
memory_store = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Function to find the most similar document
def find_most_similar_document(query):
    query_embedding = vectorizer.transform([query])
    similarities = cosine_similarity(query_embedding, document_embeddings)
    most_similar_index = np.argmax(similarities)
    return split_texts[most_similar_index]

# Streamlit interface
st.title("Chatbot")

st.sidebar.header('Navigation')
st.sidebar.text('Interact with the chatbot.')
st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTDrfJ0eik7uydN_Z-bNW8S-2jMTDEP4_m29A&s', caption='AI Chatbot', width=300)


st.write("Ask questions")
query = st.text_input("Enter Your Query:", "")
if st.button("Ask") and query:
    try:
        response = find_most_similar_document(query)
        st.write("**Answer:**", response)
    except Exception as e:
        st.write(f"An error occurred: {e}")
st.markdown("""
    <footer>
        <p style="text-align: center;">Built with Streamlit | For PreScienceDS</p>
    </footer>
    """, unsafe_allow_html=True)