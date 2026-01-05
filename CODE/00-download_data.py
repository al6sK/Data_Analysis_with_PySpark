import urllib.request
import os
import zipfile

url = 'https://archive.ics.uci.edu/static/public/487/gas+sensor+array+temperature+modulation.zip'

folder_path = 'DATA'
os.makedirs(folder_path, exist_ok=True)

local_filename = os.path.join(folder_path, 'data.zip')
local_filename2 = os.path.join(folder_path, "gas-sensor-array-temperature-modulation.zip")

try:
    print(f"Downloading data...")
    urllib.request.urlretrieve(url, local_filename)
    
    
    print("Unzipping files...")
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall(folder_path)

    with zipfile.ZipFile(local_filename2, 'r') as zip_ref:
        zip_ref.extractall(folder_path)

    print("Deleting zip files...")
    os.remove(local_filename)
    os.remove(local_filename2)
except Exception as e:
    print(f"ERROR: {e}")