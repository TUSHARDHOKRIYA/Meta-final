"""Make the HuggingFace Space public."""
import os
from huggingface_hub import HfApi

api = HfApi()
api.update_repo_visibility(
    "degree-checker-01/meta-new-space",
    private=False,
    repo_type="space",
    token=os.getenv("HF_TOKEN", ""),
)
print("Space is now public!")
