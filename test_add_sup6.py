import requests
import pandas as pd
import os

# Test script for critical-path testing of /add_sup6 route

# First, upload a sample CSV to initialize data
files = {'file': open('sample.csv', 'rb')}
response = requests.post('http://127.0.0.1:3000/', files=files)
print("Upload response status:", response.status_code)

# Now, test adding SUP6
data = {
    'draw_num': '2548',
    'sup6': '13 16 20 21 22 28 J'
}
response = requests.post('http://127.0.0.1:3000/add_sup6', data=data)
print("Add SUP6 response status:", response.status_code)
print("Response text:", response.text[:500])  # First 500 chars

# Check if reconstructed_lotto.csv was updated
if os.path.exists('reconstructed_lotto.csv'):
    df = pd.read_csv('reconstructed_lotto.csv')
    print("reconstructed_lotto.csv exists, rows:", len(df))
    if len(df) > 0:
        print("Last row SUP6:", df.iloc[-1]['SUP6'])
else:
    print("reconstructed_lotto.csv not found")
