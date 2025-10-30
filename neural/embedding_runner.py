#!/usr/bin/env python3
"""
embedding_runner.py
Reads a JSON array of texts from stdin and writes JSON array of embeddings to stdout.
Runs SentenceTransformer in a dedicated process with thread limits to avoid macOS deadlocks.
"""

import sys
import json
import os

# safety: restrict threads in this child process
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from sentence_transformers import SentenceTransformer
    import torch
    import warnings
    warnings.filterwarnings("ignore")
except Exception as e:
    print(json.dumps({"error": f"import_failed: {e}"}))
    sys.exit(1)

# set torch single-threaded for safety
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def load_model():
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(MODEL_NAME)
        except Exception as e:
            print(json.dumps({"error": f"model_load_failed: {e}"}))
            sys.exit(1)
    return _model

def main():
    # read stdin
    raw = sys.stdin.buffer.read()
    try:
        payload = json.loads(raw.decode("utf8"))
        texts = payload.get("texts") if isinstance(payload, dict) else payload
    except Exception as e:
        print(json.dumps({"error": f"bad_input: {e}"}))
        return

    try:
        model = load_model()
        embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
        # convert to lists
        out = [list(map(float, e)) for e in embeddings]
        sys.stdout.write(json.dumps({"embeddings": out}))
        sys.stdout.flush()
    except Exception as e:
        print(json.dumps({"error": f"encode_failed: {e}"}))

if __name__ == "__main__":
    main()