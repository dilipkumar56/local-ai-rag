import requests

url = "https://cricclubs.com/LansingCricketLeagueLCL/user/GzmneeKMSIAGdfMUCnsNGg?playerName=Salman+Salarzai"

response = requests.get(
    url,
    headers={
        "User-Agent": "Mozilla/5.0"
    },
    timeout=15
)

print("Status:", response.status_code)
print("Content-Type:", response.headers.get("content-type"))
print("Length:", len(response.text))

print("\nFirst 2000 characters:")
print(response.text[:2000])