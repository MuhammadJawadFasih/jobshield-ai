import os
import httpx
from dotenv import load_dotenv

load_dotenv()
r = httpx.get(
    "
    headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
)
for m in r.json()["data"]:
    print(m["id"])