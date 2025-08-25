# create_embeddings.py
import json
import time
from google.cloud import aiplatform
from google.cloud import bigquery
from google.cloud import storage
from vertexai.language_models import TextEmbeddingModel

def generate_embeddings_from_bigquery(project_id, dataset_id, table_id, chunk_size=512):
    """Generate embeddings from BigQuery data with proper text chunking."""
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location="us-central1")

    # Initialize embedding model
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    
    # Initialize BigQuery client
    bq_client = bigquery.Client(project=project_id)
    
    # Fetch data from BigQuery
    query = f"""
    SELECT file_name, extracted_data, processed_timestamp 
    FROM `{project_id}.{dataset_id}.{table_id}`
    ORDER BY processed_timestamp DESC
    """
    
    print("🔍 Fetching documents from BigQuery...")
    query_job = bq_client.query(query)
    rows = list(query_job.result())
    print(f"📄 Found {len(rows)} documents to process")
    
    embeddings_data = []
    
    for row_idx, row in enumerate(rows):
        file_name = row['file_name']
        extracted_data = row['extracted_data']
        
        try:
            # Parse the JSON data
            document_data = json.loads(extracted_data)
            
            # Create a searchable text representation
            searchable_text = create_searchable_text(document_data)
            
            # Split into chunks if text is too long
            text_chunks = chunk_text(searchable_text, chunk_size)
            
            print(f"🔤 Processing {file_name}: {len(text_chunks)} chunks")
            
            for chunk_idx, chunk in enumerate(text_chunks):
                try:
                    # Generate embedding for this chunk
                    embeddings = embedding_model.get_embeddings([chunk])
                    embedding_vector = embeddings[0].values
                    
                    # Create embedding entry
                    embedding_entry = {
                        "id": f"{file_name}_{chunk_idx}",
                        "embedding": embedding_vector,
                        "restricts": [
                            {"namespace": "file_name", "allow": [file_name]},
                            {"namespace": "chunk_index", "allow": [str(chunk_idx)]},
                        ],
                        "crowding_tag": file_name  # Group chunks from same document
                    }
                    
                    embeddings_data.append(embedding_entry)
                    
                except Exception as e:
                    print(f"⚠️ Error generating embedding for {file_name} chunk {chunk_idx}: {e}")
                    continue
                    
        except json.JSONDecodeError as e:
            print(f"⚠️ Error parsing JSON for {file_name}: {e}")
            continue
        except Exception as e:
            print(f"⚠️ Error processing {file_name}: {e}")
            continue
    
    print(f"✅ Generated {len(embeddings_data)} embeddings")
    return embeddings_data

def create_searchable_text(document_data):
    """Convert document JSON to searchable text."""
    
    searchable_parts = []
    
    # Add basic document info
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
    """Split text into chunks that fit within embedding model limits."""
    
    if len(text) <= max_length:
        return [text]
    
    # Split by sentences/phrases first
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

def upload_embeddings_to_gcs(embeddings_data, project_id, bucket_name, folder_path="embeddings/"):
    """Upload embeddings to GCS in JSONL format for Vector Search."""
    
    # Initialize storage client
    storage_client = storage.Client(project=project_id)
    
    # Check if bucket exists, create if not
    try:
        bucket = storage_client.bucket(bucket_name)
        bucket.reload()  # This will raise an exception if bucket doesn't exist
        print(f"✅ Using existing bucket: gs://{bucket_name}")
    except Exception:
        print(f"📦 Creating new bucket: gs://{bucket_name}")
        try:
            bucket = storage_client.create_bucket(bucket_name, location="us-central1")
            print(f"✅ Created bucket: gs://{bucket_name}")
        except Exception as e:
            print(f"❌ Failed to create bucket: {e}")
            print("💡 Try creating it manually:")
            print(f"   gsutil mb gs://{bucket_name}")
            raise
    
    # Convert to JSONL format
    jsonl_content = ""
    for embedding in embeddings_data:
        jsonl_content += json.dumps(embedding) + "\n"
    
    # Create filename with timestamp
    timestamp = int(time.time())
    filename = f"document_embeddings_{timestamp}.json"
    gcs_path = f"{folder_path}{filename}"
    
    # Upload to GCS
    blob = bucket.blob(gcs_path)
    
    print(f"📤 Uploading embeddings to gs://{bucket_name}/{gcs_path}")
    blob.upload_from_string(jsonl_content, content_type='application/json')
    
    print(f"✅ Uploaded {len(embeddings_data)} embeddings to GCS")
    return f"gs://{bucket_name}/{gcs_path}"

def update_existing_vector_index(project_number, location, index_id, gcs_contents_uri): # <-- Change project_id to project_number
    """Update an existing Vector Search index with new embeddings."""
    
    try:
        from google.cloud import aiplatform_v1
        
        # THIS LINE IS CRITICAL - It forces the client to use the correct region
        client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
        client = aiplatform_v1.IndexServiceClient(client_options=client_options)

        
        # Get the full index name
        index_name = f"projects/{project_number}/locations/{location}/indexes/{index_id}"
        
        print(f"🔄 Updating Vector Search index: {index_name}")
        
        # Create update request
        update_request = {
            "index": {
                "name": index_name,
                "metadata": {
                    "contents_delta_uri": gcs_contents_uri,
                    "config": {
                        "dimensions": 768,
                        "approximate_neighbors_count": 10,
                        "distance_measure_type": "DOT_PRODUCT_DISTANCE",
                        "algorithm_config": {
                            "tree_ah_config": {
                                "leaf_node_embedding_count": 500,
                                "leaf_nodes_to_search_percent": 7,
                            }
                        },
                    }
                }
            },
            "update_mask": {"paths": ["metadata.contents_delta_uri"]}
        }
        
        # Update the index
        operation = client.update_index(**update_request)
        
        print(f"✅ Index update initiated: {operation.operation.name}")
        print("⏰ Index update will take 30-60 minutes to complete...")
        
        return operation.operation.name
        
    except Exception as e:
        print(f"❌ Error updating vector search index: {e}")
        print("\n💡 Manual update steps:")
        print("   1. Go to Vertex AI > Vector Search in Google Cloud Console")
        print(f"   2. Find your index (ID: {index_id})")
        print("   3. Click 'Update' and set new data source:")
        print(f"      Data source: {gcs_contents_uri}")
        print("   4. Click 'Update Index'")
        
        return None

def get_index_status(project_number, location, index_id): # <-- Change project_id to project_number
    """Check the status of a Vector Search index."""
    
    try:
        from google.cloud import aiplatform_v1

        # THIS LINE IS CRITICAL - It forces the client to use the correct region
        client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
        client = aiplatform_v1.IndexServiceClient(client_options=client_options)
        
        index_name = f"projects/{project_number}/locations/{location}/indexes/{index_id}" # <-- Use project_number here
        
        index = client.get_index(name=index_name)
        
        print(f"📊 Index Status: {index.index_stats.vectors_count} vectors")
        print(f"🔄 State: {index.display_name}")
        
        return index
        
    except Exception as e:
        print(f"⚠️ Error getting index status: {e}")
        return None

if __name__ == "__main__":
    # Configuration
    PROJECT_ID = "lunar-outlet-447200-h6"
    PROJECT_NUMBER = "852735963073"
    LOCATION = "us-central1"
    BQ_DATASET_ID = "contract_data"
    BQ_TABLE_ID = "documents"
    GCS_EMBEDDINGS_BUCKET = "invoice-bucket-temp"  # Use your existing bucket
    TEXT_CHUNK_SIZE = 512  # Adjust based on your needs
    
    # ========== UPDATE THESE VALUES FOR YOUR EXISTING INDEX ==========
    # Set this to your existing Vector Search index ID
    EXISTING_INDEX_ID = "3808305857360297984"  # Replace with your actual index ID
    
    # Choose what to do:
    UPDATE_EXISTING_INDEX = True  # Set to True to update existing index
    SKIP_VECTOR_SEARCH_CREATION = True  # Set to False when updating existing index
    # ================================================================

    print("🚀 Starting embedding generation pipeline...")
    
    # Step 1: Generate embeddings from BigQuery data
    print("\n📊 Step 1: Generating embeddings from BigQuery...")
    embeddings_data = generate_embeddings_from_bigquery(
        PROJECT_ID, BQ_DATASET_ID, BQ_TABLE_ID, TEXT_CHUNK_SIZE
    )
    
    if not embeddings_data:
        print("❌ No embeddings generated. Exiting...")
        exit(1)
    
    # Step 2: Upload embeddings to GCS
    print("\n☁️ Step 2: Uploading embeddings to GCS...")
    gcs_embeddings_uri = upload_embeddings_to_gcs(
        embeddings_data, PROJECT_ID, GCS_EMBEDDINGS_BUCKET
    )
    
    # Step 3: Handle existing Vector Search index
    if UPDATE_EXISTING_INDEX :
        print(f"\n🔄 Step 3: Updating existing Vector Search index...")
        
        # Check current index status
        print("📊 Checking current index status...")
        #get_index_status(PROJECT_ID, LOCATION, EXISTING_INDEX_ID)
        get_index_status(PROJECT_NUMBER, LOCATION, EXISTING_INDEX_ID)
        
        # Update the index with new embeddings
        operation_name = update_existing_vector_index(
        PROJECT_NUMBER, LOCATION, EXISTING_INDEX_ID, gcs_embeddings_uri
    )
        
        if operation_name:
            print(f"\n✅ Index update initiated successfully!")
            print(f"📍 Operation: {operation_name}")
            print("⏰ Monitor the operation in Google Cloud Console")
        
    elif SKIP_VECTOR_SEARCH_CREATION:
        print("\n✅ Skipping Vector Search operations (as configured)")
        
    else:
        print("\n⚠️ Please set EXISTING_INDEX_ID to update your existing index")
        print("   Or set SKIP_VECTOR_SEARCH_CREATION = True to skip this step")
    
    # Final summary
    print(f"\n📁 Embeddings uploaded to: {gcs_embeddings_uri}")
    print(f"📊 Generated {len(embeddings_data)} embeddings from your documents")
    
    print("\n🔄 NEXT STEPS:")
    print("=" * 50)
    if UPDATE_EXISTING_INDEX and EXISTING_INDEX_ID != "2100779692586958848":
        print("1. ✅ Your existing Vector Search index is being updated")
        print("2. ⏰ Wait 30-60 minutes for the update to complete")
        print("3. 🧪 Test your updated index with new semantic searches")
        print("4. 🔍 Your invoices now include the latest data!")
    else:
        print("1. 📝 Set EXISTING_INDEX_ID in the configuration section")
        print("2. 🔄 Re-run this script to update your index")
        print("3. 💡 Or manually update via Google Cloud Console")
    
    print(f"\n🎉 Successfully processed {len(set([emb['id'].split('_')[0] for emb in embeddings_data]))} unique documents")
    print("🔍 Your invoices are ready for semantic search!")