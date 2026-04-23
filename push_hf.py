"""Push the project to HuggingFace Spaces."""
import os
from huggingface_hub import HfApi

api = HfApi()
SPACE_ID = "degree-checker-01/meta-new-space"
HF_TOKEN = os.getenv("HF_TOKEN", "")

api.upload_folder(
    folder_path=".",
    repo_id=SPACE_ID,
    token=HF_TOKEN,
    repo_type="space",
    ignore_patterns=[
        "frontend/*",
        ".git/*",
        "__pycache__/*",
        "*.pyc",
        "hf-deploy-temp/*",
        "uv.lock",
        "push_error*.txt",
        "push_result*.txt",
        "git_*.txt",
        "uv_error.txt",
        "debug_*",
    ],
)
print(f"Pushed to https://huggingface.co/spaces/{SPACE_ID}")
