"""Verify the configured SmolVLA checkpoint exists without downloading weights.

Free metadata-only gate. This is NOT a model-load, hook, or rollout test.
"""
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[1]
model = yaml.safe_load((root / "configs/models.yaml").read_text())["models"]["smolvla"]
repo = model["checkpoint"]
url = "https://huggingface.co/api/models/" + urllib.parse.quote(repo, safe="/")
request = urllib.request.Request(url, headers={"User-Agent": "vla-assure-preflight/1.0"})
try:
    with urllib.request.urlopen(request, timeout=25) as response:
        metadata = json.load(response)
except urllib.error.HTTPError as exc:
    print(f"FAIL: checkpoint {repo}: Hugging Face HTTP {exc.code}")
    sys.exit(2)
except (urllib.error.URLError, TimeoutError) as exc:
    print(f"INCONCLUSIVE: Hugging Face unavailable: {exc}")
    sys.exit(3)
actual = metadata.get("id")
if actual != repo:
    print(f"FAIL: expected {repo}, got {actual!r}")
    sys.exit(2)
print(f"PASS: public model metadata found: {actual}")
print("LIMITATION: no weights downloaded, no model loaded, no hooks or rollouts tested")
