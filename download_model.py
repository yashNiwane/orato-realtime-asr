"""
Robust model downloader using huggingface_hub with hf_hub_download and hf_transfer.
"""
import os
import sys
import time

# Enable fast transfer if available
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import HfApi, hf_hub_download
import config

def download_model(repo_id: str = config.MODEL_NAME_OR_PATH, token: str = config.HF_TOKEN):
    print(f"[*] Connecting to Hugging Face repository '{repo_id}'...")
    api = HfApi(token=token)
    info = api.model_info(repo_id)
    files = [s.rfilename for s in (info.siblings or [])]
    print(f"[*] Found {len(files)} files to verify/download:")
    
    for f in files:
        if f.startswith("."):
            continue
        print(f"    - Downloading {f}...", end=" ", flush=True)
        t0 = time.time()
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=f,
                token=token,
                resume_download=True
            )
            sz_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"DONE ({sz_mb:.2f} MB in {time.time() - t0:.2f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            raise
            
    print("\n[+] All model files verified and ready in HuggingFace cache!")

if __name__ == "__main__":
    download_model()
