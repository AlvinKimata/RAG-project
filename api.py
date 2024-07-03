import requests

API_URL = ""
headers = {
	"Accept" : "application/json",
	"Authorization": "Bearer hf_XX",
	"Content-Type": "application/json" 
}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
