import os
import json
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
from google.cloud import aiplatform
import vertexai

# Load credentials from Streamlit secrets
if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in st.secrets:
    service_account_info = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
else:
    st.error("GCP credentials not found. Please set them in Streamlit Cloud Secrets.")
    st.stop()
    
# --- Configuration ---
PROJECT_ID = "lunar-outlet-447200-h6"
PROJECT_NUMBER = "852735963073"  # Add project number for consistency
LOCATION = "us-central1"
BQ_DATASET_ID = "contract_data"
BQ_TABLE_ID = "documents"

# --- Vector Search Configuration ---
# Based on your create_embeddings.py, update these with your actual deployed values
INDEX_ENDPOINT_RESOURCE_NAME = "projects/852735963073/locations/us-central1/indexEndpoints/6479573755088601088"
DEPLOYED_INDEX_ID = "vs_quickstart_deployed_08221803"  # This should match your deployed index

# --- Embedding Configuration ---
# This should match exactly what you used in create_embeddings.py
TEXT_CHUNK_SIZE = 512
EMBEDDING_MODEL_NAME = "text-embedding-004"

# --- Caching and Initialization ---
@st.cache_resource
def init_clients():
    """Initialize all the necessary clients for Google Cloud services."""
    print("Initializing clients...")
    
    # Initialize Vertex AI
    aiplatform.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
    
    # Initialize BigQuery client
    bq_client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

    # Initialize embedding model - use same model as create_embeddings.py
    embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    
    # Initialize generative model using Vertex AI SDK (not genai)
    generative_model = GenerativeModel('gemini-2.0-flash')
    
    # Initialize Vector Search Index Endpoint
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=INDEX_ENDPOINT_RESOURCE_NAME
    )
    
    print("Clients initialized.")
    return bq_client, embedding_model, generative_model, index_endpoint

# Load the clients
try:
    bq_client, embedding_model, generative_model, index_endpoint = init_clients()
except Exception as e:
    st.error(f"💥 Failed to initialize Google Cloud clients. Please check your configuration and authentication.")
    st.error(f"Error details: {e}")
    st.stop()

# --- Backend Functions ---
def get_text_from_bigquery(match_id: str) -> str:
    """
    Fetches the original text chunk from BigQuery based on the ID returned by Vector Search.
    
    Args:
        match_id: The ID of the embedding (e.g., "invoice.pdf_0").

    Returns:
        The text content of the chunk, or an empty string if not found.
    """
    try:
        # Parse the match_id to get file_name and chunk_index
        file_name, chunk_index_str = match_id.rsplit('_', 1)
        chunk_index = int(chunk_index_str)

        # Query BigQuery for the document
        query = f"""
        SELECT extracted_data
        FROM `{PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}`
        WHERE file_name = @file_name
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("file_name", "STRING", file_name),
            ]
        )
        
        query_job = bq_client.query(query, job_config=job_config)
        rows = list(query_job.result())


        # print("retrieved rows are ", rows)

        if not rows:
            print(f"No document found for file_name: {file_name}")
            return ""

        # Parse the JSON data
        document_data = json.loads(rows[0]['extracted_data'])

        print(document_data)

        # Recreate the searchable text and chunks EXACTLY as in create_embeddings.py
        searchable_text = create_searchable_text(document_data)
        text_chunks = chunk_text(searchable_text, TEXT_CHUNK_SIZE)  # Use same chunk size
        
        # Return the specific chunk
        if 0 <= chunk_index < len(text_chunks):
            return text_chunks[chunk_index]
        else:
            print(f"Chunk index {chunk_index} out of range for {file_name}")
            return ""
            
    except Exception as e:
        print(f"Error fetching text for ID {match_id}: {e}")
        return ""

def create_searchable_text(document_data):
    """Convert document JSON to searchable text - MUST match create_embeddings.py exactly."""
    searchable_parts = []
    
    # Add basic document info - exactly as in create_embeddings.py
    for key, value in document_data.items():
        if isinstance(value, (str, int, float)) and value:
            searchable_parts.append(f"{key}: {value}")
        elif isinstance(value, list):
            # Handle arrays like line items
            for item in value:
                if isinstance(item, dict):
                    for sub_key, sub_value in item.items():
                        if sub_value:
                            searchable_parts.append(f"{sub_key}: {sub_value}")
                elif item:
                    searchable_parts.append(str(item))
    
    return " | ".join(searchable_parts)

def chunk_text(text, max_length=512):
    """Split text into chunks - MUST match create_embeddings.py exactly."""
    if len(text) <= max_length:
        return [text]
    
    # Split by sentences/phrases first - exactly as in create_embeddings.py
    sentences = text.replace(" | ", " |SPLIT| ").split("|SPLIT|")
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(current_chunk + sentence) <= max_length:
            current_chunk += sentence + " | "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " | "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def get_context_from_vector_search(query_text: str, num_neighbors: int = 3) -> list[str]:
    """
    Performs a search on Vertex AI Vector Search and retrieves the original text.
    Uses the same embedding model and process as create_embeddings.py.
    
    Args:
        query_text: The user's question.
        num_neighbors: Number of similar chunks to retrieve.

    Returns:
        A list of relevant text chunks from the document.
    """
    try:
        st.write("➡️ Generating embedding for your question...")
        
        # Generate embedding for the query using SAME model as create_embeddings.py
        query_embeddings = embedding_model.get_embeddings([query_text])
        query_embedding = query_embeddings[0].values
        
        print(f"Generated query embedding with {len(query_embedding)} dimensions")
        
        st.write(f"➡️ Searching Vector Store for relevant information...")
        
        # Search the vector index
        search_response = index_endpoint.find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[query_embedding],
            num_neighbors=num_neighbors
        )
        
        print(f"Search response: {len(search_response) if search_response else 0} results")

        print("the search response is ", search_response)
        retrieved_contexts = []
        neighbor_details = []
        
        if search_response and len(search_response) > 0 and search_response[0]:
            for neighbor in search_response[0]:
                neighbor_details.append({
                    'id': neighbor.id,
                    'distance': neighbor.distance
                })
                print(f"Found neighbor: {neighbor.id} with distance: {neighbor.distance:.4f}")
                
                # Get the actual text content
                context_text = get_text_from_bigquery(neighbor.id)
                if context_text:
                    retrieved_contexts.append(context_text)
                    print(f"Retrieved context for {neighbor.id}: {len(context_text)} characters")
                else:
                    print(f"Warning: No text found for {neighbor.id}")
        
        st.write(f"✅ Found {len(retrieved_contexts)} relevant context sections.")
        print(f"Final retrieved contexts: {len(retrieved_contexts)}")
        
        # Log similarity scores for debugging
        if neighbor_details:
            st.write("🎯 Similarity scores:")
            for detail in neighbor_details[:3]:
                st.write(f"  • {detail['id']}: {detail['distance']:.4f}")
        
        return retrieved_contexts
        
    except Exception as e:
        print(f"Error in vector search: {e}")
        st.error(f"Error during vector search: {e}")
        return []

def generate_answer(query: str, context: list[str]) -> dict:
    """
    Generates an answer using the Gemini model based on the provided context.
    
    Args:
        query: The user's original question.
        context: A list of relevant text chunks from the document.

    Returns:
        A dictionary containing the generated answer.
    """
    context_str = "\n\n---\n\n".join(context)
    
    prompt = f"""
    You are an expert assistant for answering questions about invoices and contract documents.
    Answer the following question based *only* on the context provided below.
    If the context does not contain the information needed to answer the question, state that clearly.
    Do not use any information outside of the provided context.
    Be specific and cite relevant details from the context when possible.

    **Context:**
    ---
    {context_str}
    ---

    **Question:**
    {query}

    **Answer:**
    """

    st.write("🤖 Asking Gemini to generate the final answer...")
    try:
        response = generative_model.generate_content(prompt)
        answer = response.text
        return {"answer": answer}
    except Exception as e:
        print(f"Error generating content: {e}")
        return {"error": f"Failed to generate an answer from the model: {str(e)}"}

# --- Streamlit UI ---
st.set_page_config(page_title="Invoice Q&A", page_icon="📄", layout="wide")

st.title("📄 Invoice Inquiry Assistant")
st.markdown("Ask any question about the invoice document, and I'll find the answer for you using RAG (Retrieval-Augmented Generation).")

# Add some helpful information
with st.expander("ℹ️ How this works"):
    st.markdown("""
    1. **Your Question**: You ask a question about the invoice/contract data
    2. **Vector Search**: The system converts your question to an embedding and searches for similar content
    3. **Context Retrieval**: Relevant text chunks are retrieved from BigQuery
    4. **Answer Generation**: Gemini generates an answer based only on the retrieved context
    """)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

print("\n\n\n -- new interaction ---")
# React to user input
if prompt := st.chat_input("What would you like to know about the document? (e.g., 'What is the invoice number?')"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            # 1. Retrieve context from Vector Search
            context = get_context_from_vector_search(prompt)
            
            if not context:
                st.warning("Could not find any relevant information in the document to answer your question.")
                response_text = "I'm sorry, I couldn't find any relevant information in the document to answer that question. Please try rephrasing your question or asking about different aspects of the document."
            else:
                # 2. Generate the answer
                response = generate_answer(prompt, context)
                
                if "answer" in response:
                    response_text = response['answer']
                    # Display the retrieved context in an expander for transparency
                    with st.expander("📚 View Retrieved Context"):
                        for i, ctx in enumerate(context, 1):
                            st.markdown(f"**Context {i}:**")
                            st.text(ctx)
                            if i < len(context):
                                st.markdown("---")
                else:
                    response_text = response['error']
                    st.error(response_text)
            
            st.markdown(response_text)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_text})

# Add a sidebar with configuration info
with st.sidebar:
    st.header("🔧 Configuration")
    st.text(f"Project: {PROJECT_ID}")
    st.text(f"Project #: {PROJECT_NUMBER}")
    st.text(f"Location: {LOCATION}")
    st.text(f"Dataset: {BQ_DATASET_ID}")
    st.text(f"Table: {BQ_TABLE_ID}")
    
    st.header("🔍 Vector Search")
    st.text(f"Endpoint: ...{INDEX_ENDPOINT_RESOURCE_NAME.split('/')[-1]}")
    st.text(f"Index: {DEPLOYED_INDEX_ID}")
    st.text(f"Model: {EMBEDDING_MODEL_NAME}")
    st.text(f"Chunk Size: {TEXT_CHUNK_SIZE}")
    
    st.header("📊 Debug Info")
    if st.button("Test Embedding"):
        with st.spinner("Testing..."):
            try:
                test_embeddings = embedding_model.get_embeddings(["test query"])
                st.success(f"✅ Embedding: {len(test_embeddings[0].values)}D")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()