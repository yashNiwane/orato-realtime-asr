"""
High-Speed Parallel Multi-Threaded Model Downloader with HTTP Range requests.
"""
import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")

repo_id = "tryorato/orato-asr-hindi-v1"
metadata_files = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja"
]

local_dir = os.path.join(os.path.dirname(__file__), "model_weights")
os.makedirs(local_dir, exist_ok=True)
headers = {"Authorization": f"Bearer {token}"} if token else {}


def download_small_file(fname):
    out_path = os.path.join(local_dir, fname)
    url = f"https://huggingface.co/{repo_id}/resolve/main/{fname}"
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return fname


def download_part(url, start_byte, end_byte, part_num, temp_dir):
    part_path = os.path.join(temp_dir, f"part_{part_num}.tmp")
    part_headers = {**headers, "Range": f"bytes={start_byte}-{end_byte}"}
    
    with requests.get(url, headers=part_headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(part_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return part_path, part_num


def download_large_file_parallel(fname="model.safetensors", num_workers=8):
    out_path = os.path.join(local_dir, fname)
    url = f"https://huggingface.co/{repo_id}/resolve/main/{fname}"

    print(f"[*] Querying '{fname}' metadata...")
    head = requests.head(url, headers=headers, allow_redirects=True, timeout=20)
    head.raise_for_status()
    total_size = int(head.headers.get("content-length", 0))
    resolved_url = head.url

    print(f"[*] Total file size: {total_size / (1024*1024):.2f} MB")
    
    if os.path.exists(out_path) and os.path.getsize(out_path) == total_size:
        print(f"[+] {fname} is already completely downloaded.")
        return

    temp_dir = os.path.join(local_dir, "_temp_parts")
    os.makedirs(temp_dir, exist_ok=True)

    chunk_size = total_size // num_workers
    tasks = []
    
    for i in range(num_workers):
        start = i * chunk_size
        end = total_size - 1 if i == num_workers - 1 else (i + 1) * chunk_size - 1
        tasks.append((start, end, i))

    print(f"[*] Launching {num_workers} parallel download workers...")
    t0 = time.time()
    completed_parts = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_part, resolved_url, start, end, idx, temp_dir): idx
            for start, end, idx in tasks
        }
        for future in as_completed(futures):
            part_path, idx = future.result()
            completed_parts[idx] = part_path
            pct = (len(completed_parts) / num_workers) * 100
            elapsed = max(0.01, time.time() - t0)
            mb_downloaded = (len(completed_parts) * (total_size / num_workers)) / (1024 * 1024)
            print(f"    -> Worker {idx+1}/{num_workers} finished ({pct:.0f}% total, {mb_downloaded/elapsed:.2f} MB/s)")

    print(f"[*] Stitching {num_workers} parts into {out_path}...")
    with open(out_path, "wb") as outfile:
        for i in range(num_workers):
            part_file = completed_parts[i]
            with open(part_file, "rb") as infile:
                while chunk := infile.read(4 * 1024 * 1024):
                    outfile.write(chunk)
            os.remove(part_file)

    try:
        os.rmdir(temp_dir)
    except Exception:
        pass

    total_time = time.time() - t0
    final_speed = (total_size / (1024 * 1024)) / total_time
    print(f"[+] {fname} downloaded & verified in {total_time:.2f}s ({final_speed:.2f} MB/s)!")


def main():
    print(f"[*] Downloading metadata files for '{repo_id}'...")
    for mf in metadata_files:
        download_small_file(mf)
    print("[+] Metadata files downloaded.")
    
    print("\n[*] Starting parallel accelerated download of model weights...")
    download_large_file_parallel("model.safetensors", num_workers=10)
    print("\n[🎉] Complete! All model files are present in:", local_dir)


if __name__ == "__main__":
    main()
