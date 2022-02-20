# from google.cloud import storage
import glob
from turtle import up
import requests
from requests.auth import HTTPBasicAuth
import json
import os 
import uuid 

url = "https://api.scale.com/v1/files/upload"

# auth = HTTPBasicAuth('{{live_63775b487bd540cdb5dbaa3d5c6d5be8}}', '')

headers = {
    "Accept": "application/json",
    "Authorization": "Basic bGl2ZV82Mzc3NWI0ODdiZDU0MGNkYjVkYmFhM2Q1YzZkNWJlODo="
}

uploaded_files_status = {}
uploaded_files = set()

f = open("uploaded_files2.txt", "r")
for uploaded_file in f.readlines():
  uploaded_files.add(uploaded_file.strip())


files = glob.glob("D:\TreeHacks\\almacam\*.jpg")
files = list(files)

import random
random.shuffle(files)

for file in files:  
  print(file)
  if file in uploaded_files:
    print(f"Skipping {file} - already uploaded")
    continue
  with open(file, "rb") as f:
    fnam = file.split('\\')[-1].split(".")[0]
    fname = f"{fnam}"

    metadata_json = json.dumps({
      "fname": fname,
      "fpath": file
    })
    print(fnam)
    response = requests.request(
        "POST",
        url,
        data={
            "project_name": "Almacam",
            "metadata": metadata_json,
            "name": file
        },
        files={"file": f},
        headers=headers

    )
    print(response.text)
    uploaded_files_status[file] = response.text
    if response.status_code == 200:
      f = open("uploaded_files2.txt", "a")
      f.write(file)
      f.write("\n")
      f.close()



# curl -X POST https://api.scale.com/v1/files/upload -u "live_63775b487bd540cdb5dbaa3d5c6d5be8:" -F "file=@D:\TreeHacks\\almacam\weustis2_1645304809_0.jpg" -F "metadata={\"id\": \"testing\"}" -F "project_name=UIUC AlmaCam"