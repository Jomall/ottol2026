import requests
import pandas as pd
import os

# Test script to upload a file, then add a draw and check if cleaned_lotto_copy2.csv is updated

# First, load the file before adding
file_path = 'cleaned_lotto_copy2.csv'
if os.path.exists(file_path):
    df_before = pd.read_csv(file_path)
    print(f"Rows before: {len(df_before)}")
else:
    print("File does not exist")
    exit()

# Simulate uploading a file via POST to /
upload_url = 'http://localhost:3000/'
files = {'file': open('sample.csv', 'rb')}
try:
    upload_response = requests.post(upload_url, files=files)
    print(f"Upload response status: {upload_response.status_code}")
    print(f"Upload response text: {upload_response.text[:200]}...")  # Print first 200 chars
except Exception as e:
    print(f"Upload error: {e}")
    exit()

# Now simulate adding a draw via POST to /add_nla
add_url = 'http://localhost:3000/add_nla'
data = {
    'draw': '9999',
    'nla': '1 2 3 4 5',
    'bb': '6',
    'letter': 'Z'
}

try:
    response = requests.post(add_url, data=data)
    print(f"Add response status: {response.status_code}")
    print(f"Add response text: {response.text}")
except Exception as e:
    print(f"Add error: {e}")

# Load the file after adding
if os.path.exists(file_path):
    df_after = pd.read_csv(file_path)
    print(f"Rows after: {len(df_after)}")
    if len(df_after) > len(df_before):
        print("SUCCESS: File has been updated with new draw.")
    else:
        print("FAILURE: File was not updated.")
else:
    print("File does not exist after")
