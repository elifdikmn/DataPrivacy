"""Detect stale retrieval indices after source/analysis changes."""
import hashlib
import json
from . import config

MANIFEST_PATH = config.INDEX_DIR / 'manifest.json'

# (stat signature, digest) of the last computed fingerprint. index_is_current() runs on
# every /ask request; re-hashing ~9 MB of sources each time is unnecessary when no file's
# size or modification time has changed.
_cache: tuple | None = None

def _source_paths():
    paths = [config.DATA_PATH, config.APP_DIR/'project_facts.json',
             config.APP_DIR/'domain.py', config.APP_DIR/'indexing.py', config.APP_DIR/'embeddings.py']
    paths += sorted(config.KNOWLEDGE_DIR.glob('rq*.md'))
    paths += sorted(config.FINAL_RESULTS_DIR.glob('*.json'))
    return paths

def source_fingerprint():
    global _cache
    paths = _source_paths()
    signature = (config.EMBEDDING_MODEL_NAME, str(config.BACKEND_DIR),
                 tuple((str(p), st.st_mtime_ns, st.st_size) for p in paths for st in [p.stat()]))
    if _cache is not None and _cache[0] == signature:
        return _cache[1]
    digest = hashlib.sha256(config.EMBEDDING_MODEL_NAME.encode())
    for path in paths:
        digest.update(str(path.relative_to(config.BACKEND_DIR)).encode())
        digest.update(path.read_bytes())
    _cache = (signature, digest.hexdigest())
    return _cache[1]

def index_is_current():
    required = [config.FAISS_RECORDS_PATH, config.FAISS_KNOWLEDGE_PATH,
                config.FAISS_AUDIT_PATH, config.DOCUMENTS_PATH, MANIFEST_PATH]
    if not all(p.is_file() for p in required):
        return False
    try:
        return json.loads(MANIFEST_PATH.read_text())['source_sha256'] == source_fingerprint()
    except (OSError, ValueError, KeyError):
        return False
