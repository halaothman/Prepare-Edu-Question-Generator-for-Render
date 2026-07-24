import re
import urllib.request
from pathlib import Path

secrets = Path(".streamlit/secrets.toml").read_text(encoding="utf-8")
key = re.search(r'DEEPSEEK_API_KEY = "([^"]*)"', secrets).group(1)
req = urllib.request.Request(
    "https://api.deepseek.com/user/balance",
    headers={"Authorization": f"Bearer {key}"},
)
with urllib.request.urlopen(req, timeout=15) as response:
    print(response.read().decode())
