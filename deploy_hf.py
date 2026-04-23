"""Deploy project to HuggingFace Spaces."""
import os
from huggingface_hub import HfApi

api = HfApi()

api.create_repo(
    "degree-checker-01/meta-new-space",
    repo_type="space",
    space_sdk="docker",
    token=os.getenv("HF_TOKEN", ""),
    exist_ok=True,
)

api.upload_folder(
    folder_path=".",
    repo_id="degree-checker-01/meta-new-space",
    token=os.getenv("HF_TOKEN", ""),
    repo_type="space",
    ignore_patterns=[
        "frontend/*",
        ".git/*",
        "__pycache__/*",
        "*.pyc",
        "hf-deploy-temp/*",
    ],
)
print("Deployed to https://huggingface.co/spaces/degree-checker-01/meta-new-space")
