"""
Temporary debug script — shows the RAW response from Endee's create_index
endpoint, since the SDK hides the real status code/body behind a generic
"Unknown Error" message.

Run this instead of test_endee.py to see what's actually happening.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

ENDEE_TOKEN = os.environ.get("ENDEE_TOKEN")
if not ENDEE_TOKEN:
    raise RuntimeError("ENDEE_TOKEN not set in .env")

# Reconstruct the base_url exactly the way the SDK does:
# base_url = f"https://{token_parts[2]}.endee.io/api/v1"
token_parts = ENDEE_TOKEN.split(":")
if len(token_parts) < 3:
    raise RuntimeError(
        f"Token doesn't look like the expected 3-part format "
        f"('xxxx:yyyy:region'). Got {len(token_parts)} parts."
    )

region = token_parts[2]
base_url = f"https://{region}.endee.io/api/v1"
print(f"Using base_url: {base_url}")

url = f"{base_url}/index/create"
headers = {"Authorization": ENDEE_TOKEN, "Content-Type": "application/json"}
data = {
    "index_name": "medical",
    "dim": 384,
    "space_type": "cosine",
    "M": 16,
    "ef_con": 200,
    "checksum": -1,
    "precision": "int8",
    "version": None,
}

print("Sending request...")
response = requests.post(url, headers=headers, json=data)

print("\n--- RAW RESPONSE ---")
print("Status code:", response.status_code)
print("Body:", response.text)
print("--------------------")