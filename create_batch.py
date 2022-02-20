import requests

url = "https://api.scale.com/v1/batches"

payload = {
    "project": "UIUC AlmaCam",
    "name": "almacam_batch_2",
    "calibration_batch": False,
    "self_label_batch": False
}
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Basic bGl2ZV82Mzc3NWI0ODdiZDU0MGNkYjVkYmFhM2Q1YzZkNWJlODo="
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)