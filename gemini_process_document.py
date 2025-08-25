# process_documents.py
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
from google.cloud import bigquery
from google import genai

# Initialize Gemini client
genai_client = genai.Client(
    vertexai=True,
    project="lunar-outlet-447200-h6",
    location="us-central1"
)

def extract_kv_with_gemini(project_id: str, location: str, text: str) -> dict:
    """Extract key-value pairs from document text using Gemini with proper JSON formatting."""
    
    # Flexible prompt that adapts to any document structure
    prompt = f"""
Analyze this document text and extract ALL relevant information into a well-structured JSON format. Return ONLY the JSON object with no additional text, explanations, markdown formatting, or code blocks.

Guidelines for extraction:
1. Extract ALL identifiable fields and values from the document
2. Use descriptive, snake_case field names (e.g., "invoice_number", "billing_address", "line_items")
3. Group related information logically (sender info, recipient info, amounts, dates, etc.)
4. For line items/products, create an "items" or "line_items" array with all available details
5. Use appropriate data types: strings for text, numbers for quantities/amounts, arrays for lists
6. Use null for truly missing values, not empty strings
7. Ensure the JSON is syntactically correct and complete
8. Include currency symbols and units where relevant
9. Preserve important formatting like phone numbers, dates, addresses

Important rules:
- Return ONLY valid JSON, no surrounding text or markdown
- Do not invent or assume data that isn't clearly present
- Include ALL fields you can identify, even if some seem less important
- Make field names descriptive and consistent

Document text to analyze:
{text}
"""

    try:
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash-exp",  # Updated to newer model
            contents=prompt
        )
        
        response_text = response.text.strip()
        print(f"Raw Gemini response length: {len(response_text)} characters")
        
        # Clean up common formatting issues
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Parse JSON with better error handling
        try:
            kv = json.loads(response_text)
            print(f"✅ Successfully parsed JSON with {len(kv)} top-level fields")
            
            # Validate and clean any array fields (items, line_items, etc.)
            array_fields = [k for k, v in kv.items() if isinstance(v, list)]
            for field_name in array_fields:
                if field_name in ['items', 'line_items', 'products', 'services']:
                    # Filter out incomplete or malformed items
                    valid_items = []
                    for item in kv[field_name]:
                        if isinstance(item, dict) and len(item) > 0:
                            # Keep items that have at least some content
                            # Don't enforce specific field names since they vary
                            valid_items.append(item)
                    
                    kv[field_name] = valid_items
                    print(f"✅ Cleaned {field_name} array: {len(valid_items)} valid items")
            
            return kv
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Response text preview: {response_text[:500]}...")
            
            # Try to fix common JSON issues
            try:
                # Remove trailing commas and fix incomplete objects
                cleaned = response_text.rstrip(',}] \n') + '}'
                # Try parsing again
                kv = json.loads(cleaned)
                print("✅ Fixed JSON parsing after cleanup")
                return kv
            except:
                print("⚠️ Could not parse Gemini output even after cleanup. Returning empty dict.")
                return {}
                
    except Exception as e:
        print(f"❌ Error calling Gemini: {e}")
        return {}

def process_document_and_store_in_bq(
    project_id, location, processor_id,
    gcs_bucket_name, gcs_file_name,
    bq_dataset_id, bq_table_id,
    save_json_locally=True,
    update_if_exists=True
):
    """Process document with Document AI and store results in BigQuery."""
    
    storage_client = storage.Client(project=project_id)
    bq_client = bigquery.Client(project=project_id)
    blob = storage_client.bucket(gcs_bucket_name).blob(gcs_file_name)

    # Download file to temporary location
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / os.path.basename(gcs_file_name)
        print(f"📥 Downloading {gcs_file_name} to {tmp_path}")
        blob.download_to_filename(str(tmp_path))

        with open(tmp_path, "rb") as f:
            content = f.read()

    # Process with Document AI
    docai_client = documentai.DocumentProcessorServiceClient()
    processor_path = docai_client.processor_path(project_id, location, processor_id)
    request = documentai.ProcessRequest(
        name=processor_path,
        raw_document=documentai.RawDocument(content=content, mime_type="application/pdf")
    )
    
    print("🔍 Processing document with Document AI...")
    result = docai_client.process_document(request=request)
    text = result.document.text
    print(f"📄 Extracted {len(text)} characters of text")

    # Extract structured data with Gemini
    print("🤖 Extracting key-value pairs with Gemini...")
    kv = extract_kv_with_gemini(project_id, location, text)

    if not kv:
        print("⚠️ No data extracted, skipping BigQuery insertion")
        return

    # Save JSON locally if enabled
    if save_json_locally:
        output_dir = Path("extracted_json")
        output_dir.mkdir(exist_ok=True)
        
        json_filename = Path(gcs_file_name).stem + "_extracted.json"
        json_path = output_dir / json_filename
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(kv, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved JSON to: {json_path}")
        except Exception as e:
            print(f"⚠️ Failed to save JSON locally: {e}")

    # Check if file already exists in BigQuery
    if update_if_exists:
        existing_query = f"""
        SELECT COUNT(*) as count 
        FROM `{project_id}.{bq_dataset_id}.{bq_table_id}` 
        WHERE file_name = '{gcs_file_name}'
        """
        
        try:
            query_job = bq_client.query(existing_query)
            result = list(query_job.result())
            file_exists = result[0].count > 0
            
            if file_exists:
                print(f"🔄 File {gcs_file_name} already exists in BigQuery. Updating...")
                
                # Delete existing record
                delete_query = f"""
                DELETE FROM `{project_id}.{bq_dataset_id}.{bq_table_id}` 
                WHERE file_name = '{gcs_file_name}'
                """
                bq_client.query(delete_query).result()
                print(f"🗑️ Deleted existing record for {gcs_file_name}")
            else:
                print(f"➕ New file {gcs_file_name}, inserting fresh record...")
                
        except Exception as e:
            print(f"⚠️ Could not check for existing records: {e}")
            print("   Proceeding with insert (may create duplicates)")

    # Prepare data for BigQuery
    rows = [{
        "file_name": gcs_file_name, 
        "extracted_data": json.dumps(kv, ensure_ascii=False),
        "processed_timestamp": datetime.utcnow().isoformat()
    }]
    
    table_ref = bq_client.dataset(bq_dataset_id).table(bq_table_id)
    
    print("💾 Inserting data into BigQuery...")
    errors = bq_client.insert_rows_json(table_ref, rows)
    
    if not errors:
        print(f"✅ Successfully inserted {gcs_file_name} with {len(kv)} fields")
        # Count items in any array field that might contain line items
        item_counts = []
        for field_name, field_value in kv.items():
            if isinstance(field_value, list) and len(field_value) > 0:
                item_counts.append(f"{len(field_value)} {field_name}")
        if item_counts:
            print(f"   📋 Including: {', '.join(item_counts)}")
    else:
        print(f"❌ BigQuery insertion error: {errors}")

def setup_bigquery(project_id, dataset_id, table_id, recreate_table=False):
    """Set up BigQuery dataset and table with correct schema."""
    
    client = bigquery.Client(project=project_id)
    dataset_ref = client.dataset(dataset_id)
    
    # Create dataset if it doesn't exist
    try:
        client.get_dataset(dataset_ref)
        print(f"✅ Dataset {dataset_id} already exists")
    except Exception:
        client.create_dataset(dataset_ref)
        print(f"📊 Created dataset {dataset_id}")
    
    # Handle table creation/recreation
    table_ref = dataset_ref.table(table_id)
    
    if recreate_table:
        try:
            client.delete_table(table_ref)
            print(f"🗑️ Deleted existing table {table_id}")
        except Exception:
            print(f"ℹ️ Table {table_id} didn't exist, creating new one")
    
    try:
        existing_table = client.get_table(table_ref)
        print(f"✅ Table {table_id} already exists")
        
        # Check if schema is correct
        field_names = [field.name for field in existing_table.schema]
        if 'extracted_data' not in field_names:
            print(f"⚠️ WARNING: Table schema may be incorrect. Fields: {field_names}")
            print(f"   Expected field 'extracted_data' not found. Consider setting recreate_table=True")
        
    except Exception:
        # Create table with correct schema
        schema = [
            bigquery.SchemaField("file_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("extracted_data", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("processed_timestamp", "TIMESTAMP", mode="NULLABLE"),
        ]
        client.create_table(bigquery.Table(table_ref, schema=schema))
        print(f"📋 Created table {table_id} with correct schema")

if __name__ == "__main__":
    # Configuration
    PROJECT_ID = "lunar-outlet-447200-h6"
    LOCATION = "us"
    PROCESSOR_ID = "ac69c238331a1c43"
    GCS_BUCKET_NAME = "invoice-bucket-temp"
    BQ_DATASET_ID = "contract_data"
    BQ_TABLE_ID = "documents"
    
    # Toggles
    SAVE_JSON_LOCALLY = True    # Set to False to disable local JSON saving
    RECREATE_BQ_TABLE = True    # Set to True to recreate table with correct schema
    UPDATE_IF_EXISTS = False     # Set to True to update existing records, False to skip duplicates

    print("🚀 Starting document processing pipeline...")
    print(f"📁 Local JSON saving: {'ENABLED' if SAVE_JSON_LOCALLY else 'DISABLED'}")
    print(f"🔄 BigQuery table recreation: {'ENABLED' if RECREATE_BQ_TABLE else 'DISABLED'}")
    print(f"🔄 Update existing records: {'ENABLED' if UPDATE_IF_EXISTS else 'DISABLED'}")
    
    # Setup BigQuery resources
    setup_bigquery(PROJECT_ID, BQ_DATASET_ID, BQ_TABLE_ID, RECREATE_BQ_TABLE)

    # Process all PDFs in the bucket
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    
    pdf_count = 0
    for blob in bucket.list_blobs():
        if blob.name.lower().endswith(".pdf"):
            pdf_count += 1
            print(f"\n📄 Processing PDF {pdf_count}: {blob.name}")
            process_document_and_store_in_bq(
                PROJECT_ID, LOCATION, PROCESSOR_ID,
                GCS_BUCKET_NAME, blob.name,
                BQ_DATASET_ID, BQ_TABLE_ID,
                SAVE_JSON_LOCALLY, UPDATE_IF_EXISTS
            )
    
    print(f"\n🎉 Completed processing {pdf_count} PDF files")
    if SAVE_JSON_LOCALLY:
        print("📁 Check the 'extracted_json' folder for saved JSON files")