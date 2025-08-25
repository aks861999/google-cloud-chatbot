# upload_to_gcs.py
import os
from google.cloud import storage

from google.cloud import storage



def upload_files_to_gcs(bucket_name, source_directory):
    """Uploads all files from a directory to the specified GCS bucket."""
    storage_client = storage.Client(project="lunar-outlet-447200-h6")
    bucket = storage_client.bucket(bucket_name)

    for filename in os.listdir(source_directory):
        if filename.endswith('.pdf'):
            local_path = os.path.join(source_directory, filename)
            blob = bucket.blob(filename)
            
            print(f"Uploading {filename} to gs://{bucket_name}/{filename}...")
            blob.upload_from_filename(local_path)
            print(f"File {filename} uploaded.")

if __name__ == '__main__':
    # --- Configuration ---
    GCS_BUCKET_NAME = "invoice-bucket-temp"  #  Bucket for storing raw PDFs
    SOURCE_PDF_DIR = "./docs"          # A local folder containing your sample PDFs

    # Create the directory if it doesn't exist and add a placeholder
    if not os.path.exists(SOURCE_PDF_DIR):
        os.makedirs(SOURCE_PDF_DIR)
        print(f"Created directory '{SOURCE_PDF_DIR}'. Please add your sample PDF contracts there and run again.")
    else:
        upload_files_to_gcs(GCS_BUCKET_NAME, SOURCE_PDF_DIR)