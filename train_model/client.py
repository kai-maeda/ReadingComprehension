import requests

url = "http://127.0.0.1:5000/word_count"
# data = {"url": "https://www.nationalgeographic.com/animals/birds"}
data = {"url": "https://pbskids.org"}

response = requests.post(url, json=data)
print(response.json())