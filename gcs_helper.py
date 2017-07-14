from google.cloud import storage

BUCKET_NAME = "glove-tf-model"
MODEL_FNAME = "glove_model.txt"

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client("sunway-14050926")
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

def download_blob(bucket_name, source_blob_name, destination_file_name):
    import pathlib
    """Downloads a blob from the bucket."""
    if not pathlib.Path(destination_file_name).exists():
        storage_client = storage.Client("sunway-14050926")
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print('Blob {} downloaded to {}.'.format(
            source_blob_name,
            destination_file_name))

    return destination_file_name



