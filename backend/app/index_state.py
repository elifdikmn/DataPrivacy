"""Detect stale retrieval indices after source/analysis changes."""
import hashlib
import json
from . import config

MANIFEST_PATH = config.INDEX_DIR / 'manifest.json'

def source_fingerprint():
    paths = [config.DATA_PATH, config.APP_DIR/'project_facts.json',
             config.APP_DIR/'domain.py', config.APP_DIR/'indexing.py', config.APP_DIR/'embeddings.py']
    paths += sorted(config.KNOWLEDGE_DIR.glob('rq*.md'))
    paths += sorted(config.FINAL_RESULTS_DIR.glob('*.json'))
    digest = hashlib.sha256(config.EMBEDDING_MODEL_NAME.encode())
    for path in paths:
        digest.update(str(path.relative_to(config.BACKEND_DIR)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()

def index_is_current():
    required = [config.FAISS_RECORDS_PATH, config.FAISS_KNOWLEDGE_PATH,
                config.FAISS_AUDIT_PATH, config.DOCUMENTS_PATH, MANIFEST_PATH]
    if not all(p.is_file() for p in required):
        return False
    try:
        return json.loads(MANIFEST_PATH.read_text())['source_sha256'] == source_fingerprint()
    except (OSError, ValueError, KeyError):
        return False
