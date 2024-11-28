from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

# Authenticate and initialize the PyDrive client
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Creates a local webserver and automatically handles authentication
drive = GoogleDrive(gauth)

def upload_file_to_drive(file_path):
    """
    Uploads a file to Google Drive.
    
    Args:
    - file_path (str): Path to the file you want to upload.
    """
    # Get the filename from the path
    filename = os.path.basename(file_path)
    
    # Create a GoogleDriveFile instance with the specified filename
    file_drive = drive.CreateFile({'title': filename})
    file_drive.SetContentFile(file_path)  # Set the content to the file
    file_drive.Upload()  # Upload the file
    print(f"Uploaded '{filename}' to Google Drive successfully!")

# Example usage
checkpoint_path = "./attribution_img/checkpoint_best_regular.pt"  # Replace with the path to your checkpoint file
upload_file_to_drive(checkpoint_path)
