import requests

API_URL = "https://zamjmbhxi82i4h39.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
	"Accept" : "application/json",
	"Authorization": "Bearer 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",
	"Content-Type": "application/json" 
}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

output = query({
	"inputs": "Can you please let us know more details about your expertise",
	"parameters": {}
})

print(output)