"""Set HuggingFace Space secrets (environment variables).
Usage: Set your secrets as env vars first, then run this script.
  export HF_TOKEN=your_token_here
  python set_secrets.py
"""
import os
from huggingface_hub import HfApi

api = HfApi()
SPACE_ID = "degree-checker-01/meta-new-space"
TOKEN = os.getenv("HF_TOKEN", "")

secrets = {
    "API_BASE_URL": "https://api-inference.huggingface.co/v1",
    "MODEL_NAME": "degree-checker-01/edupath-grpo-tutor",
    "HF_TOKEN": os.getenv("HF_TOKEN", ""),
    "SUPABASE_URL": os.getenv("SUPABASE_URL", ""),
    "SUPABASE_KEY": os.getenv("SUPABASE_KEY", ""),
}

for key, value in secrets.items():
    if value:
        api.add_space_secret(SPACE_ID, key, value, token=TOKEN)
        print(f"Set secret: {key}")
    else:
        print(f"Skipped {key} (not set in environment)")

print("\nAll secrets configured!")
print(f"Space: https://huggingface.co/spaces/{SPACE_ID}")
