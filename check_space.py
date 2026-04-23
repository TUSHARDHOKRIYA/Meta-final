"""Check the status of the HuggingFace Space."""
import os
import json
from huggingface_hub import HfApi

api = HfApi()
SPACE_ID = "degree-checker-01/meta-new-space"

try:
    info = api.space_info(SPACE_ID, token=os.getenv("HF_TOKEN", ""))
    print(f"Space: {info.id}")
    print(f"Status: {info.runtime}")
    print(f"SDK: {info.sdk}")
    print(f"URL: https://huggingface.co/spaces/{SPACE_ID}")
except Exception as e:
    print(f"Error: {e}")
