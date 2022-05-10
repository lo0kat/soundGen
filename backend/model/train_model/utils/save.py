import shutil
import logging
import boto3
from botocore.exceptions import ClientError
import os


def zip(archive_name:str,input_dir:str):
    shutil.make_archive(archive_name,'zip',input_dir)


# Retrieve the list of existing buckets

def upload_file(file_name:str, bucket:str, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')

    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

    
def zip_and_upload(zip_name:str, input_dir ="model",bucket_name = "dl-model-bucket-cytech64"):
    """
    Zip trained model directory and upload it to an S3 bucket    
    """
    full_file_name = "{}.zip".format(zip_name)

    print('Zipping model now ...')
    zip(zip_name,input_dir)
    print('Zipping step has succeeded !')

    print('Sending zipped model to S3 bucket')
    upload_res = upload_file(full_file_name,bucket_name)
    
    if upload_res : 
       print('S3 upload step completed')
    else :
        print ('Something wrong happened during the S3 upload')

