import shutil

def zip(archive_name,input_dir):
    shutil.make_archive(archive_name,'zip',input_dir)
