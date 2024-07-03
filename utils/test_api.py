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
	"inputs": "Can you please let us know more details about your expertise?",
	"parameters": {
		"top_k": 10,
		"top_p": 0.95,
		"temperature": 0.1,
		"max_new_tokens": 1024,
		"do_sample": True,
		"return_text": True,
		"return_full_text": True,
		"return_tensors": False,
		"clean_up_tokenization_spaces": True
	}
})

print(output)