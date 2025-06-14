#REVISED RAG_CFL FLASK CODE FOR HEROKU

from flask import Flask, request, render_template, redirect, url_for, session
from groq import Groq
import os
from pinecone import Pinecone, Index, ServerlessSpec
from sentence_transformers import SentenceTransformer
from torch import Tensor # Still might be needed depending on internal SentenceTransformer workings
import pandas as pd # Assuming product data is in a CSV
import uuid
import time
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# --- Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # Corrected variable name
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "gemini-rag") # Default to "gemini-rag" if not set
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MASTER_PASSWORD = os.getenv("MASTER_PASSWORD") # Get master password from environment

# --- Initialize Components ---
# 1. Initialize Pinecone
try:
    pinecone = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    # Handle error appropriately, e.g., exit or disable Pinecone features

# 2. Initialize Sentence Transformer for Embeddings
# It's good practice to ensure the model is loaded only once
try:
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
except Exception as e:
    print(f"Error loading SentenceTransformer: {e}")
    # Handle error appropriately

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# 3. Initialize Groq Chat Model
try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    # Handle error appropriately

chat_model_name = "llama-3.3-70b-versatile"

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

index_name = PINECONE_INDEX_NAME
try:
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768, # all-mpnet-base-v2 has an embedding dimension of 768, not 1536
            metric="cosine",
            spec=spec
        )
    index = pc.Index(index_name)
except Exception as e:
    print(f"Error initializing or creating Pinecone index: {e}")
    # Handle error appropriately, e.g., exit or inform user

app = Flask(__name__)
app.secret_key = os.urandom(24) # Keep this for session management

# --- Functions ---
def chunk_text(text, chunk_size=100):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def store_data(data):
    if not embedding_model:
        return "Embedding model not loaded. Cannot store data."
    if not index:
        return "Pinecone index not initialized. Cannot store data."

    chunks = chunk_text(data)
    embeddings = embedding_model.encode(chunks)
    vectors = [
        (str(uuid.uuid4()), embedding.tolist(), {'text': chunk})
        for chunk, embedding in zip(chunks, embeddings)
    ]
    try:
        index.upsert(vectors=vectors)
        return f"{len(chunks)} chunks of data stored."
    except Exception as e:
        return f"Error storing data to Pinecone: {e}"

def retrieve_relevant_chunks(query, top_k=3):
    if not embedding_model:
        return []
    if not index:
        return []

    query_embedding = embedding_model.encode(query)
    try:
        results = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)
        relevant_chunks = [match.metadata['text'] for match in results.matches]
        return relevant_chunks
    except Exception as e:
        print(f"Error retrieving chunks from Pinecone: {e}")
        return []

def generate_answer(question, context_chunks):
    if not groq_client:
        return "Groq client not initialized. Cannot generate answer."

    context = "\n".join(context_chunks)
    prompt = f"Answer the following question based on the provided context:\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    try:
        response = groq_client.chat.completions.create(
            model=chat_model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer with Groq: {e}"

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'new_data' in request.form and request.form['new_data']:
            data_to_store = request.form['new_data']
            result = store_data(data_to_store)
            return render_template('ragWeb.html', storage_result=result)
        elif 'query' in request.form and request.form['query']:
            user_query = request.form['query']
            relevant_chunks = retrieve_relevant_chunks(user_query)
            answer = generate_answer(user_query, relevant_chunks)
            return render_template('ragWeb.html', query=user_query, answer=answer)
    return render_template('ragWeb.html')

@app.route('/input_data_login', methods=['GET', 'POST'])
def input_data_login():
    if request.method == 'POST':
        password = request.form['password']
        if password == MASTER_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('input_new_data'))
        else:
            error = "Invalid password. Please try again."
            return render_template('login.html', error=error)
    return render_template('login.html')

@app.route('/input_new_data', methods=['GET', 'POST'])
def input_new_data():
    if not session.get('logged_in'):
        return redirect(url_for('input_data_login'))
    storage_result = None
    if request.method == 'POST':
        new_data = request.form['new_data']
        # The store_data function already handles storing to Pinecone
        storage_result = store_data(new_data)
        return render_template('input_new_data.html', storage_result=storage_result)
    return render_template('input_new_data.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) # Heroku assigns a port via PORT environment variable
    app.run(debug=True, host='0.0.0.0', port=port)