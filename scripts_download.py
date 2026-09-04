import os
from huggingface_hub import snapshot_download
tok = os.environ["HF_TOKEN"]
targets = [
    ("Qwen/Qwen2.5-7B-Instruct", "models/Qwen2.5-7B-Instruct"),
    ("ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice", "models/adapter_risky-financial"),
]
for repo, dest in targets:
    print(f"=== {repo} -> {dest}", flush=True)
    p = snapshot_download(repo_id=repo, local_dir=dest, token=tok,
                          ignore_patterns=["*.pth","*.msgpack","*.h5","original/*"],
                          max_workers=8)
    print(f"    done: {p}", flush=True)
print("ALL DOWNLOADS COMPLETE", flush=True)
