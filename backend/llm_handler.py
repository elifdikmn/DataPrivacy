# llm_handler.py — RAG (FAISS) • Dual-Mode • Treemap • Policies Audit
# ---------------------------------------------------------------
# Features:
# - RAG over catalog (data_entries_final.json) with dual-mode routing
# - Treemap (Main % + Main→DataType %) HTML endpoints
# - Load many privacy policies from a folder (default: ./sentences)
# - Extract declared data categories from policies (lexicon + SBERT)
# - Compare catalog vs declared policies => /audit
# - Reindex & rebuild viz endpoints
# ---------------------------------------------------------------

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json, re, hashlib
from pathlib import Path

import time
from typing import List, Dict, Any, Tuple, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse,JSONResponse
from pydantic import BaseModel

# .env
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))  # <— ÖNEMLİ
except Exception:
    pass

# LLM (Together API - OpenAI compatible)
from openai import OpenAI

# Embeddings / Vector index
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import Counter, defaultdict

# Data viz
import pandas as pd
import plotly.express as px

def _html_nocache(p: Path) -> HTMLResponse:
    body = p.read_text(encoding="utf-8")
    etag = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "ETag": etag,
        "Last-Modified": time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(p.stat().st_mtime)),
    }
    return HTMLResponse(content=body, headers=headers)

# -------------------- Config --------------------
# Catalog (ground truth of collected data)
DATA_FINAL = Path(os.getenv("DATA_PATH_FINAL", "data_entries_final.json")).expanduser().resolve()

# Visualization output paths — dosya isimleri senin iki HTML’inle eşlendi
VIZ_MAIN_HTML = Path(os.getenv(
    "VIZ_MAIN_HTML",
    "main_data_type_treemap_grouped.html"   # “What data are collected…” -> /viz/main
)).expanduser().resolve()

VIZ_HIER_HTML = Path(os.getenv(
    "VIZ_HIER_HTML",
    "main_data_type_treemap_grouped_with_subtypes_no_zero.html"  # “Which sensitive…” -> /viz/hier
)).expanduser().resolve()

# Policies folder (YOUR folder with many JSON privacy policies)
POLICY_DIR = Path(os.getenv("POLICY_DIR", "sentences")).expanduser().resolve()

# --- Game config (card game) ---
GAME_DATA_DIR = Path(os.getenv("GAME_DATA_DIR", "final_results")).expanduser().resolve()

# Runtime cache
_GAME: Dict[str, Any] = {"items": None, "folder_sig": None}

def _folder_signature(dir_path: Path) -> str:
    if not dir_path.exists():
        return "absent"
    parts = []
    for p in sorted(dir_path.glob("*.json")):
        try:
            st = p.stat()
            parts.append(f"{p.name}:{int(st.st_mtime)}:{st.st_size}")
        except Exception:
            continue
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

# --- Label normalize + final label karar kuralları ---

LABEL_ALIASES = {
    "CLEAR": {"CLEAR", "MENTIONED", "DECLARED", "YES", "EVET"},
    "OMITTED": {"OMITTED", "NOT MENTIONED", "NO", "HAYIR"},
    "AMBIGUOUS": {"AMBIGUOUS", "UNCLEAR", "BELIRSIZ", "INCONCLUSIVE"},
}

def _norm_label_once(s: Any) -> str:
    t = str(s or "").strip().upper()
    if not t:
        return ""
    for canon, variants in LABEL_ALIASES.items():
        if t in variants or canon in t:
            return canon
    if "OMIT" in t: return "OMITTED"
    if "MENTION" in t or "DECLAR" in t: return "CLEAR"
    if "UNCLEAR" in t or "AMBIGU" in t or "BELIRS" in t: return "AMBIGUOUS"
    return ""

def _final_label_from_collection(coll: list) -> str:
    labs = set()
    for c in (coll or []):
        labs.add(_norm_label_once(c.get("label")))
    labs.discard("")
    if not labs: return "AMBIGUOUS"
    if labs == {"CLEAR"}: return "CLEAR"
    if labs == {"OMITTED"}: return "OMITTED"
    if "CLEAR" in labs and "OMITTED" in labs: return "AMBIGUOUS"
    if labs == {"AMBIGUOUS"}: return "AMBIGUOUS"
    return "AMBIGUOUS"

# -------------------- Minimal chat memory + summarize helpers --------------------
_CHAT_STATE: Dict[str, str] = {"last_answer": ""}

SUMMARY_PATTERNS = [
    r"\bsummary\b", r"\bsummarize\b", r"\bsummarise\b",
    r"\bshort(er|est)?\b", r"\bbrief(er|ly)?\b",
    r"\bözet(le|)\b", r"\bkısalt\b"
]
_SUMMARY_RE = re.compile("|".join(SUMMARY_PATTERNS), re.IGNORECASE)

def _looks_like_summary_request(q: str) -> bool:
    return bool(_SUMMARY_RE.search((q or "").strip()))

def _summarize_text(text: str, max_sents: int = 3) -> str:
    if not text:
        return ""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    if len(sents) <= max_sents:
        return text.strip()
    return " ".join(sents[:max_sents]).strip()

# -------------------- llm_handler.py helpers --------------------

def _synth_desc(data_name: str, data_type: str) -> str:
    """1 cümlelik açıklama üret. LLM varsa kullan, yoksa şablon."""
    base = (data_type or data_name or "").strip()
    if not base:
        return "This card refers to a data field collected by the API."

    # LLM varsa: kısa, tek cümle, jargon yok.
    if llm_client:
        try:
            prompt = (
                "Write ONE short sentence (max 20 words) that explains what this data field is for.\n"
                f"data_name: {data_name}\n"
                f"data_type: {data_type}\n"
                "Avoid policy language; describe the field's purpose plainly."
            )
            resp = llm_client.chat.completions.create(
                model=TOGETHER_MODEL,
                messages=[{"role":"user","content":prompt}],
                temperature=0.2, max_tokens=40,
            )
            txt = (resp.choices[0].message.content or "").strip()
            # güvenli kırpma
            return (txt[:160] or f"This card refers to the {base} field.")
        except Exception:
            pass

    # Fallback (LLM yoksa)
    if data_name and data_type:
        return f"This card refers to the '{data_name}' field, categorized as {data_type}."
    if data_name:
        return f"This card refers to the '{data_name}' field."
    return f"This card refers to the {data_type} category."

def _synth_explanation(label: str, data_name: str, data_type: str, source: str = "") -> str:
    """
    Return ONE sentence (max ~22 words) explaining why the correct label applies.
    If LLM not available, return a short fallback.
    """
    label = (label or "AMBIGUOUS").upper()
    dn = (data_name or "").strip()
    dt = (data_type or "").strip()

    if llm_client:
        try:
            prompt = (
                "Write ONE short sentence (<=22 words) explaining why the given label fits for this data field, in plain English.\n"
                f"label: {label}\n"
                f"data_name: {dn}\n"
                f"data_type: {dt}\n"
                f"policy_snippet: {source[:300]}\n"
                "Do not add headings or bullets."
            )
            resp = llm_client.chat.completions.create(
                model=TOGETHER_MODEL,
                messages=[{"role":"user","content": prompt}],
                temperature=0.2, max_tokens=50,
            )
            txt = (resp.choices[0].message.content or "").strip()
            return txt[:220]
        except Exception:
            pass

    # Fallback (LLM yoksa)
    if label == "CLEAR":
        return "The privacy policy explicitly mentions this data is collected."
    if label == "OMITTED":
        return "The privacy policy does not mention collecting this data."
    return "The privacy policy is unclear about this data being collected."

def _norm_item(r: dict) -> dict:
    coll = r.get("collection") if isinstance(r.get("collection"), list) else []
    final_label = _final_label_from_collection(coll)

    # explanation: final etiketle eşleşen ilk snippet
    explanation = ""
    for c in coll or []:
        if _norm_label_once(c.get("label")) == final_label and c.get("sentence"):
            explanation = str(c["sentence"]).strip()
            break
    if not explanation and coll:
        explanation = str(coll[0].get("sentence","")).strip()

    raw_name = (r.get("data_name") or r.get("name") or r.get("title") or "").strip()
    raw_type = (r.get("data_type") or r.get("schema_type") or r.get("type") or "").strip()
    desc = (r.get("description") or "").strip()

    # normalize + türetme
    name_derived = _is_garbage(raw_name)
    type_derived = _is_garbage(raw_type)

    norm_name = _derive_name_from_text(raw_name, desc, explanation) if name_derived \
        else _caps(re.sub(r"[_-]+", " ", raw_name).strip() or "Unknown Data")
    norm_type = _sanitize_type(raw_type)

    # açıklama yoksa kısa bir tanım üret
    if not desc:
        desc = _synth_desc(norm_name or raw_name or "Unknown", norm_type or raw_type or "Other")

    # kısa snippet (BACK yüzü için güvenli)
    snippet = (explanation or desc or "")[:400]
    snippet = re.sub(r"\s+#\s*", " ", snippet)
    snippet = re.sub(r"\n{2,}", "\n", snippet)

    return {
        # orijinal alanlar
        "name": (r.get("name") or r.get("title") or "(unnamed)").strip(),
        "data_name": raw_name or "Unknown",
        "data_type": raw_type or "Other info",
        "description": desc,

        # normalized alanlar (oyun/dash bu alanlara bakabilir)
        "norm_data_name": norm_name or "Unknown Data",
        "norm_data_type": norm_type or "Other",
        "name_derived": bool(name_derived),
        "type_derived": bool(type_derived),
        "snippet": snippet,

        # oyun yüzleri
        "label": final_label,                 # CLEAR / OMITTED / AMBIGUOUS
        "question_desc": desc,                # FRONT
        "question_type": norm_type,           # FRONT (normalize edilmiş)
        "explanation": explanation,           # BACK (politikadan parça)
    }



def load_game_items() -> List[dict]:
    sig = _folder_signature(GAME_DATA_DIR)
    if _GAME["items"] is not None and _GAME["folder_sig"] == sig:
        return _GAME["items"]

    items: List[dict] = []
    if not GAME_DATA_DIR.exists():
        print(f"[game] data dir not found: {GAME_DATA_DIR}")
        _GAME.update({"items": [], "folder_sig": sig})
        return _GAME["items"]

    for p in GAME_DATA_DIR.glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            records = obj if isinstance(obj, list) else [obj]
            for r in records:
                try:
                    items.append(_norm_item(r))
                except Exception as ie:
                    print(f"[game] skip record in {p.name}: {ie}")
        except Exception as e:
            print(f"[game] skip file {p.name}: {e}")

    _GAME.update({"items": items, "folder_sig": sig})
    print(f"[game] loaded {len(items)} items from {GAME_DATA_DIR}")
    return items

# --- Heuristics: garbage detection, name derivation, type mapping -------------
import string

STOPWORDS = set("""
the a an and or of to in on for with by at from we our you your is are be as that this it they them
their may can will data information info veri bilgi which what when how where any all such other
""".split())

def _is_garbage(s: Optional[str]) -> bool:
    if not s: return True
    t = str(s).strip()
    if not t: return True
    if len(t) < 2: return True
    # çok sembol/rakam
    non_alpha = "".join(ch for ch in t if ch not in string.ascii_letters + "ğüşöçıİĞÜŞÖÇ ")
    if len(non_alpha) / max(1, len(t)) > 0.7: return True
    # url / hash
    if re.search(r"https?://", t, re.I): return True
    if re.fullmatch(r"[0-9a-f]{16,}", t, re.I): return True
    return False

def _caps(s: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in re.split(r"\s+", s.strip()) if w)

def _topk(text: str, k: int = 3) -> list:
    words = re.sub(r"[^A-Za-zğüşöçıİĞÜŞÖÇ\s]", " ", text or "").lower().split()
    words = [w for w in words if len(w) > 2 and w not in STOPWORDS]
    freq = Counter(words)
    return [w for w, _ in freq.most_common(k)]

def _derive_name_from_text(*cands: str) -> str:
    txt = " ".join([c for c in cands if c])
    kws = _topk(txt, 3)
    if kws:
        return _caps(" ".join(kws))
    # fallback: ilk 3 kelime
    s = " ".join((txt or "").split()[:3])
    return _caps(s) or "Unknown Data"

def _sanitize_type(t: Optional[str]) -> str:
    if _is_garbage(t): return "Other"
    lower = (t or "").lower()
    if re.search(r"(email|e-mail|mail)", lower): return "Identifier"
    if re.search(r"(phone|tel|telephone)", lower): return "Identifier"
    if re.search(r"(location|geo|city|country|address|lat|lon|gps)", lower): return "Location"
    if re.search(r"(credential|password|secret|token|api ?key)", lower): return "Credentials"
    if re.search(r"(usage|analytics|log|telemetry|event)", lower): return "Usage"
    cleaned = re.sub(r"[^A-Za-zğüşöçıİĞÜŞÖÇ\s]", " ", t or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return _caps(cleaned) or "Other"


# LLM
TOGETHER_API_KEY = (os.getenv("TOGETHER_API_KEY") or "").strip()
TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1").strip()

# RAG params
TOPK = int(os.getenv("RAG_TOPK", "6"))
RELEVANCE_MIN = float(os.getenv("RELEVANCE_MIN", "0.28"))
RELEVANCE_HIT_MIN = int(os.getenv("RELEVANCE_HIT_MIN", "2"))

# Policy extraction params
POLICY_THRESHOLD = float(os.getenv("POLICY_THRESHOLD", "0.38"))  # embedding threshold

# -------------------- App --------------------
app = FastAPI(title="RAG + Treemap + Policies Audit")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -------------------- LLM client --------------------
llm_client: Optional[OpenAI] = None
if TOGETHER_API_KEY:
    try:
        llm_client = OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1")
    except Exception as e:
        print(f"[llm] init error: {e}")
        llm_client = None
else:
    print("[llm] TOGETHER_API_KEY missing — LLM answers will be skipped or mocked.")

# -------------------- Catalog dataset + embeddings --------------------
_DATA: Dict[str, Any] = {"mtime": None, "data": None, "sha256": None}
_EMB: Dict[str, Any] = {"model": None, "index": None, "metas": None}

def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _load_json_list(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of objects.")
    return data

def load_dataset() -> List[Dict[str, Any]]:
    st = DATA_FINAL.stat()
    if _DATA["data"] is None or _DATA["mtime"] != st.st_mtime:
        data = _load_json_list(DATA_FINAL)
        _DATA.update({"mtime": st.st_mtime, "data": data, "sha256": _file_sha256(DATA_FINAL)})
        _EMB.update({"index": None, "metas": None})
        print(f"[dataset] loaded {DATA_FINAL} sha={_DATA['sha256'][:12]}… size={st.st_size}")
    return _DATA["data"]


def _normalize_rows(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in raw:
        rows.append({
            "name": r.get("name","") or "",
            "description": r.get("description","") or "",
            "main_data_type": r.get("main_data_type","") or "",
            "data_type": r.get("data_type","") or "",
        })
    return rows

def _ensure_model():
    if _EMB["model"] is None:
        _EMB["model"] = SentenceTransformer("all-MiniLM-L6-v2")
        print("[emb] sentence-transformers model loaded.")

def _embed_texts(texts: List[str]) -> np.ndarray:
    _ensure_model()
    vecs = _EMB["model"].encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype("float32")

def build_index(rows: List[Dict[str, Any]]):
    texts = [
        " | ".join([
            f"name: {r['name']}",
            f"description: {r['description']}",
            f"main_data_type: {r['main_data_type']}",
            f"data_type: {r['data_type']}",
        ])
        for r in rows
    ]
    embs = _embed_texts(texts)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors + inner product
    index.add(embs)
    _EMB.update({"index": index, "metas": rows})
    print(f"[emb] index built. dim={dim}, rows={len(rows)}")

def ensure_index():
    data = load_dataset()
    if _EMB["index"] is None or _EMB["metas"] is None:
        build_index(_normalize_rows(data))

def retrieve_with_scores(question: str, topk: int = TOPK) -> Tuple[List[Dict[str, Any]], List[float]]:
    ensure_index()
    qv = _embed_texts([question])
    D, I = _EMB["index"].search(qv, topk)
    hits, scores = [], []
    for idx, sc in zip(I[0], D[0]):
        if idx < 0:
            continue
        r = _EMB["metas"][idx]
        hits.append({
            "name": r["name"],
            "description": r["description"],
            "main_data_type": r["main_data_type"],
            "data_type": r["data_type"],
        })
        scores.append(float(sc))
    return hits, scores

# -------------------- Taxonomy helpers --------------------
def build_taxonomy(rows: List[Dict[str, Any]]) -> Dict[str, Counter]:
    tax = defaultdict(Counter)
    for r in rows:
        m = (r.get("main_data_type") or "Other").strip()
        d = (r.get("data_type") or "").strip()
        if d:
            tax[m][d] += 1
    return tax

def _nl_join(items: List[str]) -> str:
    if not items: return ""
    if len(items) == 1: return items[0]
    if len(items) == 2: return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"

def collected_sentence_from_taxonomy(tax: Dict[str, Counter], max_mains: int = 3) -> str:
    if not tax: return ""
    mains_sorted = sorted(tax.items(), key=lambda kv: -sum(kv[1].values()))[:max_mains]
    main_names = [m for m, _ in mains_sorted]
    examples = []
    for m, counter in mains_sorted:
        child = next((dt for dt, _ in counter.most_common(1)), "")
        if child:
            examples.append(child)
    left = _nl_join(main_names)
    right = _nl_join(examples)
    if right:
        return f"Across the catalog, apps often collect {left} — for example: {right}."
    else:
        return f"Across the catalog, apps often collect {left}."

# -------------------- Visualization: Treemap builders --------------------
def _rows_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)[["name","description","main_data_type","data_type"]]

def build_treemap_main_percent(rows: List[Dict[str, Any]], out_path: Path = VIZ_MAIN_HTML) -> Path:
    df = _rows_df(rows)
    grp = df["main_data_type"].fillna("Unknown").value_counts().rename_axis("Main Data Type").reset_index(name="Count")
    grp["Percent"] = (grp["Count"] / grp["Count"].sum() * 100).round(1)
    grp["Label"] = grp["Main Data Type"] + "<br>" + grp["Count"].astype(str) + " (" + grp["Percent"].astype(str) + "%)"
    fig = px.treemap(
        grp, path=["Label"], values="Count",
        title="Main Data Type Distribution (%)",
        color="Count", color_continuous_scale="Viridis"
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path

def build_treemap_hier_percent(rows: List[Dict[str, Any]], out_path: Path = VIZ_HIER_HTML) -> Path:
    df = _rows_df(rows).fillna("Unknown")
    grp = df.groupby(["main_data_type","data_type"]).size().reset_index(name="Count")
    grp["Percent"] = (grp["Count"] / grp["Count"].sum() * 100).round(1)
    grp["Label"] = grp["data_type"] + " (" + grp["Count"].astype(str) + ", " + grp["Percent"].astype(str) + "%)"
    fig = px.treemap(
        grp, path=["main_data_type","Label"], values="Count",
        title="Main Data Type → Data Type (%)",
        color="main_data_type"
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path

def build_all_treemaps(rows: List[Dict[str, Any]]) -> Tuple[Path, Path]:
    p1 = build_treemap_main_percent(rows)
    p2 = build_treemap_hier_percent(rows)
    print(f"[viz] treemaps written: {p1.name}, {p2.name}")
    return p1, p2

# -------------------- Broad intent & prompts --------------------
BROAD_PATTERNS = [
    r"\bwhat\s+(data|information)\s+are\s+collected\b",
    r"\bwhich\s+(data|information)\s+.*(collected|captured|used)\b",
    r"\bdata\s+types?\b", r"\bmain\s+data\s+type\b", r"\boverview\b", r"\bbreakdown\b",
    r"\bdağılım\b", r"\bhangi veri(ler|)\b", r"\bhangi veri türleri\b", r"\bgenel bakış\b",
]
def is_broad_overview(q: str) -> bool:
    if not q: return False
    ql = q.lower()
    return any(re.search(p, ql) for p in BROAD_PATTERNS)

_NUM_RE = re.compile(r"(?<![\w-])([0-9]+(?:[.,][0-9]+)?)(?![\w-])")

PROMPT_GROUNDED = """You are a helpful analyst.
Write a short, story-like explanation in **simple English**.
Constraints:
- No section headers or bullet lists.
- Stay grounded in the provided context.
- Do not invent new categories or field names.
- Do not include raw counts or statistics.
- At the end, add **one or two sentences of your own thoughtful comment**
  (for example, why this might be sensitive, risky, or important for users).

User question:
{question}

Context (for grounding only):
{context}
"""

PROMPT_FREE = """You are a helpful assistant focused on data, privacy, and applications that process user information.

When answering:
- Use plain English in one or two short paragraphs. Avoid bullet lists unless the user asks.
- If the question is unrelated to data, privacy, security, or application behavior, say so briefly and ask ONE targeted follow-up question to clarify the user’s intent.
- Do NOT invent facts, product specs, API parameters, URLs, or statistics. If you are not reasonably certain, say you don’t have enough information and ask for a specific detail you need.
- Prefer practical guidance and clear next steps over theory. Keep it grounded and avoid speculation.

User question:
{question}
"""

def context_from_hits(hits: List[Dict[str, Any]]) -> str:
    lines = []
    for h in hits:
        desc = (h.get("description") or "").strip()
        if len(desc) > 220: desc = desc[:220] + "…"
        lines.append(
            f"- name: {h.get('name','')}; main_data_type: {h.get('main_data_type','')}; "
            f"data_type: {h.get('data_type','')}; description: {desc}"
        )
    return "\n".join(lines) if lines else "- (no relevant context found)"

def _call_llm(prompt: str) -> str:
    if not llm_client:
        return ""
    try:
        resp = llm_client.chat.completions.create(
            model=TOGETHER_MODEL,
            messages=[
                {"role": "system", "content": "Be precise, grounded when given context, and answer in English."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.32, max_tokens=900,
        )
        msg = resp.choices[0].message.content if resp and resp.choices else ""
        msg = _NUM_RE.sub(" many ", msg or "")
        return (msg or "").strip()
    except Exception as e:
        return f"(LLM error) {e}"

def is_dataset_relevant(scores: List[float]) -> bool:
    if not scores:
        return False
    best = max(scores)
    good = sum(1 for s in scores if s >= RELEVANCE_MIN)
    return (best >= RELEVANCE_MIN) and (good >= RELEVANCE_HIT_MIN)

# -------------------- Viz intent --------------------
VIZ_KEYWORDS = [
    "visualize","visualise","chart","plot","graph","treemap",
    "distribution","share","percent","percentage","top types","data types",
    "what data are collected","which data types","overview","breakdown",
    "görselleştir","görselleştirme","grafik","çiz","treemap",
    "dağılım","oran","yüzde","hangi veriler","hangi veri türleri","özet","genel bakış"
]
def is_viz_relevant(q: str) -> bool:
    if not q: return False
    ql = q.lower()
    if any(k in ql for k in VIZ_KEYWORDS): return True
    if re.search(r"\b(data\s*types?|main\s*data\s*type|distribution|breakdown)\b", ql): return True
    return False

# -------------------- Policies: load many JSON files --------------------
CANDIDATE_TEXT_FIELDS = ["text", "content", "body", "raw_text", "policy_text"]
CANDIDATE_NAME_FIELDS = ["action_name", "name", "title"]
CANDIDATE_URL_FIELDS  = ["source_url", "url", "link", "policy_url"]
CANDIDATE_VER_FIELDS  = ["policy_version", "version", "last_updated", "updated_at", "date"]
CANDIDATE_LANG_FIELDS = ["language", "lang", "locale"]

def _first(d: dict, keys: List[str], default=""):
    for k in keys:
        if k in d and d[k]:
            return str(d[k])
    return default

def _guess_text(d: dict) -> str:
    t = _first(d, CANDIDATE_TEXT_FIELDS, "")
    if isinstance(t, list):
        t = "\n".join(str(x) for x in t)
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    return (t or "").strip()

def _guess_name(d: dict) -> str:
    return _first(d, CANDIDATE_NAME_FIELDS, "Unknown Action").strip()

def _guess_url(d: dict) -> str:
    return _first(d, CANDIDATE_URL_FIELDS, "").strip()

def _guess_ver(d: dict) -> str:
    return _first(d, CANDIDATE_VER_FIELDS, "").strip()

def _guess_lang(d: dict) -> str:
    return _first(d, CANDIDATE_LANG_FIELDS, "").strip() or "en"

_POLICY_DOCS: List[Dict[str, Any]] = []

def load_policies_from_folder() -> List[Dict[str, Any]]:
    global _POLICY_DOCS
    docs = []
    if not POLICY_DIR.exists():
        print(f"[policy] directory not found: {POLICY_DIR}")
        _POLICY_DOCS = []
        return _POLICY_DOCS

    for p in POLICY_DIR.glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            records = obj if isinstance(obj, list) else [obj]
            for r in records:
                text = _guess_text(r)
                if not text:
                    continue
                docs.append({
                    "doc_id": r.get("doc_id", p.stem),
                    "action_name": _guess_name(r),
                    "source_url": _guess_url(r),
                    "policy_version": _guess_ver(r),
                    "language": _guess_lang(r),
                    "text": text
                })
        except Exception as e:
            print(f"[policy] skip {p.name}: {e}")
    _POLICY_DOCS = docs
    print(f"[policy] loaded {len(docs)} docs from {POLICY_DIR}")
    return _POLICY_DOCS

# -------------------- Policy category extraction (lexicon + SBERT) --------------------
CANON_MAP = {
  "Location": ["location","geolocation","gps","latitude","longitude","konum","coğrafi konum"],
  "App usage data": ["usage data","app activity","interaction","clickstream","kullanım verisi","etkileşim"],
  "Identifiers": ["device id","advertising id","idfa","gaid","email address","tanımlayıcı","cihaz kimliği"],
  "Security credentials": ["password","auth token","session token","credential","şifre","kimlik bilgisi"],
  "Query": ["search query","query string","request parameters","arama sorgusu","sorgu parametresi"],
  "Payment": ["payment","card","credit card","billing","ödeme","kart","fatura"],
  "Contact": ["contact info","phone","email","adres","iletişim"],
  "Cookies": ["cookie","cookies","çerez","çerezler","tracking technologies","izleme teknolojileri"],
}

_model_policy = None
CANON_MAIN = list(CANON_MAP.keys())
CANON_MAIN_EMB = None

def _ensure_policy_model():
    global _model_policy
    if _model_policy is None:
        _model_policy = SentenceTransformer("all-MiniLM-L6-v2")

def _prep_canon_emb():
    global CANON_MAIN_EMB
    _ensure_policy_model()
    if CANON_MAIN_EMB is None:
        CANON_MAIN_EMB = _model_policy.encode(CANON_MAIN, normalize_embeddings=True)

def extract_policy_categories(policy_text: str, threshold: float = POLICY_THRESHOLD):
    _prep_canon_emb()
    sents = re.split(r'(?<=[.!?])\s+', policy_text)
    if not sents: return {"by_main": {}, "all_spans": []}
    sent_emb = _model_policy.encode(sents, normalize_embeddings=True)
    sims = np.matmul(sent_emb, CANON_MAIN_EMB.T)

    found = {k: [] for k in CANON_MAIN}
    spans = []
    for i, s in enumerate(sents):
        low = s.lower()
        # rule-based
        rule_mains = [m for m, kws in CANON_MAP.items() if any(k in low for k in kws)]
        # embedding
        j = int(np.argmax(sims[i]))
        best_main, best_score = CANON_MAIN[j], float(sims[i, j])

        mains = set(rule_mains)
        if not mains and best_score >= threshold:
            mains.add(best_main)

        for m in mains:
            rec = {"span": s[:350], "score": best_score}
            found[m].append(rec)
            spans.append({"text": s[:350], "main": m, "score": best_score})

    by_main = {m: v for m, v in found.items() if v}
    return {"by_main": by_main, "all_spans": spans}



# -------------------- Game: AI explanation endpoint --------------------
# -------------------- Game: definition-anchored AI explanation --------------------
class ExplainReq(BaseModel):
    data_name: str = ""
    description: str = ""
    data_type: str = ""
    label: str = ""          # CLEAR | AMBIGUOUS | OMITTED
    lang: str = "en"         # keep "en" as you requested

class ExplainResp(BaseModel):
    explanation: str

_DEF_TEXT = {
    "CLEAR": (
        "CLEAR = The privacy policy explicitly states that this data is collected."
    ),
    "OMITTED": (
        "OMITTED = The data is collected in practice, but the privacy policy does not mention it."
    ),
    "AMBIGUOUS": (
        "AMBIGUOUS = The policy is vague or indirect: it mentions related concepts but does not clearly state the data is collected."
    ),
}

def _base_reason_sentence(label: str, data_name: str) -> str:
    """Deterministic, single-sentence reason matching the paper-style definitions."""
    ln = (data_name or "this data").strip()
    L = (label or "AMBIGUOUS").upper()
    if L == "CLEAR":
        return f"The policy explicitly states the collection of {ln}, so the label is CLEAR."
    if L == "OMITTED":
        return f"The app collects {ln}, yet the privacy policy does not mention it, so the label is OMITTED."
    # AMBIGUOUS
    return f"The policy refers to related concepts without clearly stating collection of {ln}, so the label is AMBIGUOUS."



def _llm_reason_english(label: str, data_name: str, description: str) -> str:
    """
    If Together LLM is available, rewrite a grounded one/two-sentence reason,
    but keep it aligned with the strict definitions.
    """
    base = _base_reason_sentence(label, data_name)

    if not llm_client:
        return base + " " 

    try:
        prompt = (
            "You write short, precise rationales for labeling policy snippets.\n"
            "Definitions (do not deviate):\n"
            f"- { _DEF_TEXT['CLEAR'] }\n"
            f"- { _DEF_TEXT['OMITTED'] }\n"
            f"- { _DEF_TEXT['AMBIGUOUS'] }\n\n"
            "Constraints:\n"
            "- 1–2 sentences, 28 words max per sentence.\n"
            "- Use the given label strictly according to the definitions.\n"
            "- Ground on the provided data_name and policy description; do not invent facts.\n"
            "- Return in English.\n\n"
            f"Label: {label}\n"
            f"Data Name: {data_name}\n"
            f"Policy Description: {description[:400]}\n\n"
            f"Rewrite this base reason naturally, preserving its meaning: {base}\n"
            
        )
        resp = llm_client.chat.completions.create(
            model=TOGETHER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=120,
        )
        txt = (resp.choices[0].message.content or "").strip()
        # Lightweight guardrail: if model drifts, fall back
        if not txt or any(bad in txt.lower() for bad in ["might be wrong", "cannot", "no info"]):
            return base + " " 
        return txt
    except Exception:
        return base + " " 

@app.post("/game/explain", response_model=ExplainResp)
def game_explain(req: ExplainReq):
    """
    Return a reason aligned with the CLEAR / AMBIGUOUS / OMITTED definitions:
      - CLEAR: policy explicitly states collection.
      - OMITTED: collected in practice but not mentioned.
      - AMBIGUOUS: vague/indirect mention; not clearly stated.
    """
    label = (req.label or "AMBIGUOUS").upper()
    data_name = req.data_name or req.data_type or "this data"
    description = req.description or ""

    reason = _llm_reason_english(label, data_name, description)
    return {"explanation": reason}



# -------------------- Catalog vs Policy comparison --------------------
def collected_sets(catalog_rows: List[Dict[str, Any]]):
    mains, pairs = set(), set()
    for r in catalog_rows:
        m = (r.get("main_data_type") or "").strip()
        d = (r.get("data_type") or "").strip()
        if m: mains.add(m)
        if m and d: pairs.add((m, d))
    return mains, pairs

def declared_from_policies(docs: List[Dict[str, Any]]):
    declared_mains = set()
    evidence = {}
    for d in docs:
        ex = extract_policy_categories(d["text"])
        for m, spans in ex["by_main"].items():
            declared_mains.add(m)
            for sp in spans[:2]:  # keep 2 snippets per main per doc
                evidence.setdefault(m, []).append({
                    "action_name": d["action_name"],
                    "policy_version": d["policy_version"],
                    "source_url": d["source_url"],
                    "snippet": sp["span"]
                })
    return declared_mains, evidence

def compare_catalog_vs_policy_folder(catalog_rows: List[Dict[str, Any]]):
    docs = _POLICY_DOCS if _POLICY_DOCS else load_policies_from_folder()
    if not docs:
        return {"error": "No policy files found."}

    declared_mains, evidence = declared_from_policies(docs)
    collected_mains, _ = collected_sets(catalog_rows)

    overlap = sorted(collected_mains & declared_mains)
    under  = sorted(collected_mains - declared_mains)  # collected but not declared
    over   = sorted(declared_mains - collected_mains)  # declared but not in catalog

    coverage = round(100.0 * len(overlap) / max(1, len(collected_mains)), 1)

    return {
        "coverage_percent": coverage,
        "overlap": overlap,
        "under_disclosure": under,
        "over_disclosure": over,
        "evidence": evidence,
        "docs_count": len(docs)
    }

# -------------------- API Models --------------------
class AskReq(BaseModel):
    question: str

class AskResp(BaseModel):
    text: str
    viz_relevant: bool
    viz_path: Optional[str] = None   # "main" | "hier" | None

class AuditResp(BaseModel):
    coverage_percent: float
    overlap: List[str]
    under_disclosure: List[str]
    over_disclosure: List[str]
    docs_count: int
    evidence: Dict[str, List[Dict[str, str]]]

# -------------------- Startup --------------------
@app.on_event("startup")
def _warmup():
    try:
        data = load_dataset()
        rows = _normalize_rows(data)
        build_index(rows)
       
        load_policies_from_folder()
        print("[startup] warmup complete.")
    except Exception as e:
        print(f"[startup] warmup error: {e}")

# -------------------- Health/Debug --------------------
@app.get("/health")
def health():
    try:
        _ = load_dataset()
        return {"ok": True, "dataset": str(DATA_FINAL), "policy_dir": str(POLICY_DIR)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/debug/config")
def debug_config():
    return {
        "DATA_FINAL": str(DATA_FINAL),
        "VIZ_MAIN_HTML": str(VIZ_MAIN_HTML),
        "VIZ_HIER_HTML": str(VIZ_HIER_HTML),
        "POLICY_DIR": str(POLICY_DIR),
        "GAME_DATA_DIR": str(GAME_DATA_DIR), 
        "TOGETHER_MODEL": TOGETHER_MODEL,
        "HAS_API_KEY": bool(TOGETHER_API_KEY),
        "TOPK": TOPK,
        "RELEVANCE_MIN": RELEVANCE_MIN,
        "RELEVANCE_HIT_MIN": RELEVANCE_HIT_MIN,
        "POLICY_THRESHOLD": POLICY_THRESHOLD,
    }

# -------------------- Visualization endpoints --------------------
@app.get("/viz/main", response_class=HTMLResponse)
def viz_main():
    p = VIZ_MAIN_HTML
    if not p.exists():
        rows = _normalize_rows(load_dataset())
        build_treemap_main_percent(rows, p)
    return _html_nocache(p)
@app.get("/viz/hier", response_class=HTMLResponse)
def viz_hier():
    p = VIZ_HIER_HTML
    if not p.exists():
        rows = _normalize_rows(load_dataset())
        build_treemap_hier_percent(rows, p)
    return _html_nocache(p)

@app.post("/viz/rebuild")
def viz_rebuild():
    rows = _normalize_rows(load_dataset())
    p1, p2 = build_all_treemaps(rows)
    return {"ok": True, "main": str(p1), "hier": str(p2)}

# -------------------- Viz path chooser --------------------
def pick_viz_path(q: str) -> Optional[str]:
    ql = (q or "").strip().lower()
    # Tam eşleşmeler
    if ql == "which sensitive data types appear most often?":
        return "hier"   # with_subtypes_no_zero.html
    if ql == "what data are collected by gpt actions?":
        return "main"   # grouped.html
    # Esnek kurallar
    if "sensitive" in ql or "subtype" in ql or "breakdown" in ql:
        return "hier"
    if "what data are collected" in ql or "overview" in ql:
        return "main"
    return None

# -------------------- Category-aware Q&A (deterministic, no numbers) --------------------
MAIN_ALIASES = {
    # English
    "Location": ["location","geo","geolocation","gps","lat lon","latitude longitude","address","country","city","place"],
    "Identifiers": ["identifier","identifiers","id","ids","user id","device id","advertising id","gaid","idfa"],
    "App usage data": ["usage","app usage","analytics","telemetry","events","interaction","clickstream","activity"],
    "Security credentials": ["credential","credentials","password","token","api key","secret","auth","session"],
    "Contact": ["contact","email","phone","address book","contact info"],
    "Query": ["query","search query","parameters","param","filters","request query"],
    "Payment": ["payment","billing","card","credit card"],
    "Cookies": ["cookie","cookies","tracking technologies","web cookies"],
    # Turkish
    "Location": ["konum","coğrafi konum","yer","adres"],
    "Identifiers": ["tanımlayıcı","cihaz kimliği","reklam kimliği"],
    "App usage data": ["kullanım verisi","etkileşim","telemetri"],
    "Security credentials": ["kimlik bilgisi","şifre","jeton","erişim anahtarı"],
    "Contact": ["iletişim","telefon","e-posta","e posta"],
    "Query": ["arama sorgusu","sorgu parametresi"],
    "Payment": ["ödeme","fatura","kart"],
    "Cookies": ["çerez","izleme teknolojileri"],
}
_ALIAS_TO_MAIN = {}
for _m, _arr in MAIN_ALIASES.items():
    for _a in _arr:
        _ALIAS_TO_MAIN[_a.lower()] = _m

def _norm_main_label(s: str) -> str:
    s = (s or "").strip()
    return s[:1].upper() + s[1:] if s else ""

def _safe_join(items: List[str]) -> str:
    items = [i for i in items if i]
    if not items: return ""
    if len(items) == 1: return items[0]
    if len(items) == 2: return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"

def _norm_cat(s: Optional[str]) -> str:
    return _caps((s or "").strip()) if s else ""

def _detect_main_from_question(q: str) -> str:
    ql = (q or "").lower()
    for alias, main in _ALIAS_TO_MAIN.items():
        if alias in ql:
            return main
    for main in MAIN_ALIASES.keys():
        if main.lower() in ql:
            return main
    return ""

EXAMPLE_PER_CATEGORY = 4

def _collect_subtype_examples(rows: List[Dict[str, Any]], main: str, k: int = EXAMPLE_PER_CATEGORY) -> List[str]:
    seen = set()
    out = []
    for r in rows:
        if _norm_cat(r.get("main_data_type")) == main:
            dt = _norm_cat(r.get("data_type"))
            if dt and dt.lower() != "other" and dt not in seen:
                seen.add(dt)
                out.append(dt)
                if len(out) >= k:
                    break
    return out

def _privacy_note_for_main(main: str) -> str:
    m = (main or "").lower()
    if "location" in m:
        return "Location signals can reveal movement patterns and real-world presence, so limit precision, retention, and sharing."
    if "security credential" in m:
        return "Credentials are highly sensitive. Prefer short-lived tokens, strict storage rules, and least-privilege access."
    if "identifier" in m:
        return "Identifiers link activity back to people or devices. Minimize linkage and rotate identifiers where possible."
    if "app usage" in m:
        return "Usage data supports quality and personalization but can still profile behavior. Collect what is necessary and document purpose."
    if "contact" in m:
        return "Use contact data for clear, user-expected purposes and honor opt-out preferences."
    if "payment" in m:
        return "Payment data requires strong encryption and vendor segregation; avoid storing full card details unless essential."
    if "cookies" in m:
        return "Cookie-based tracking should respect consent and provide a simple way to adjust preferences."
    if "query" in m:
        return "Search and query strings can reveal intentions. Avoid logging full queries when not needed."
    return "Apply purpose limitation and retention controls, and keep access narrow and auditable."

def build_main_category_answer(main: str, rows: List[Dict[str, Any]]) -> str:
    main = _norm_main_label(main)
    examples = _collect_subtype_examples(rows, main, EXAMPLE_PER_CATEGORY)

    if examples:
        para1 = (
            f"Under {main}, the catalog includes fields such as "
            f"{_safe_join(examples)}. These appear in different actions and connectors to support features like context, personalization, and routing."
        )
    else:
        para1 = (
            f"Under {main}, the catalog contains various fields used by actions and connectors to enable core functionality and context."
        )

    para2 = _privacy_note_for_main(main)
    return f"{para1}\n\n{para2}"

# -------------------- Policies management --------------------
@app.post("/policies/reindex")
def policies_reindex():
    docs = load_policies_from_folder()
    return {"ok": True, "count": len(docs), "dir": str(POLICY_DIR)}

@app.post("/policies/upload")
async def upload_policy(file: UploadFile = File(...)):
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    out = POLICY_DIR / file.filename
    out.write_bytes(await file.read())
    load_policies_from_folder()
    return {"ok": True, "saved": str(out)}

# -------------------- Audit endpoint --------------------
@app.post("/audit", response_model=AuditResp)
def audit():
    rows_all = _normalize_rows(load_dataset())
    if not rows_all:
        raise HTTPException(status_code=400, detail="Catalog dataset is empty.")
    result = compare_catalog_vs_policy_folder(rows_all)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

# -------------------- Chat (dual-mode RAG + simple intents) --------------------
CONSISTENCY_PATTERNS = [
  r"\b(consisten|align|match)\b.*\bpolicy\b",
  r"\bunder[- ]?disclosure\b|\bover[- ]?collection\b",
  r"\bprivacy policy\b.*\b(compared|vs|against)\b",
  r"\b(policy|gizlilik).*\b(tutuyor|uyuyor|uyum)\b",
  r"\b(gather|collect|toplan).*\b(more|fazla)\b"
]
def is_consistency_intent(q: str) -> bool:
    if not q: return False
    ql = q.lower()
    return any(re.search(p, ql) for p in CONSISTENCY_PATTERNS)

@app.post("/chat", response_model=AskResp)
def chat(req: AskReq):
    try:
        question = (req.question or "").strip()
        rows_all = _normalize_rows(load_dataset())
        ensure_index()

        # ---- If it's a summarize/shorten follow-up, shrink the last answer (no new LLM/RAG) ----
        if _looks_like_summary_request(question):
            prev = _CHAT_STATE.get("last_answer", "")
            if prev:
                short = _summarize_text(prev, max_sents=3)
                _CHAT_STATE["last_answer"] = short
                return {"text": short, "viz_relevant": False, "viz_path": None}
            else:
                return {"text": "There’s no previous answer to summarize yet.", "viz_relevant": False, "viz_path": None}

        # ---- Category-aware answers: e.g., Location / Identifiers / Security credentials
        asked_main = _detect_main_from_question(question)
        if asked_main:
            txt = build_main_category_answer(asked_main, rows_all)
            _CHAT_STATE["last_answer"] = txt
            return {"text": txt, "viz_relevant": True, "viz_path": "hier"}

        # ---- consistency inquiry shortcut: call audit
        if is_consistency_intent(question):
            res = compare_catalog_vs_policy_folder(rows_all)
            if "error" in res:
                return {"text": f"(Audit) {res['error']}", "viz_relevant": False, "viz_path": None}
            def fmt(lst): return ", ".join(lst) if lst else "—"
            text = (
              f"Policy consistency check across {res['docs_count']} files:\n"
              f"- Coverage (declared vs collected): {res['coverage_percent']:.1f}%\n"
              f"- Overlap: {fmt(res['overlap'])}\n"
              f"- Under-disclosure (collected but not declared): {fmt(res['under_disclosure'])}\n"
              f"- Over-disclosure (declared but not in catalog): {fmt(res['over_disclosure'])}\n\n"
              "Sample evidence:\n"
            )
            shown = 0
            for m, evs in res["evidence"].items():
                for e in evs[:1]:
                    text += f"• [{m}] {e['action_name']} ({e.get('policy_version','')}): {e['snippet']}  <{e.get('source_url','')}>\n"
                    shown += 1
                    if shown >= 6: break
                if shown >= 6: break
            _CHAT_STATE["last_answer"] = text.strip()
            return {"text": text.strip(), "viz_relevant": True, "viz_path": "main"}

        # ---- Regular dual-mode flow
        hits, scores = retrieve_with_scores(question, topk=TOPK)
        dataset_ok = is_dataset_relevant(scores)
        broad = is_broad_overview(question)

        if dataset_ok or broad:
            if broad:
                # global context from all rows
                by_main = defaultdict(list)
                for r in rows_all: by_main[r["main_data_type"] or "Other"].append(r["data_type"])
                # quick textual context (contains counts but LLM output strips numbers via _NUM_RE)
                lines = []
                for m, lst in sorted(by_main.items(), key=lambda kv: -len(kv[1])):
                    cnt = len(lst)
                    sample = ", ".join([x for x, _ in Counter(lst).most_common(3) if x])
                    lines.append(f"- {m}: includes e.g., {sample} (n={cnt})")
                ctx = "\n".join(lines) if lines else "- (no context)"
            else:
                ctx = context_from_hits(hits)

            prompt = PROMPT_GROUNDED.format(question=question, context=ctx)
            llm_txt = _call_llm(prompt)

            if not llm_txt:
                tax = build_taxonomy(rows_all if broad else hits)
                collected_line = collected_sentence_from_taxonomy(tax, max_mains=3)
                fallback = ("Here is a concise view based on the dataset context:\n" + ctx[:1500])
                final_text = (fallback + ("\n\n" + collected_line if collected_line else "")).strip()
            else:
                tax = build_taxonomy(rows_all if broad else hits)
                collected_line = collected_sentence_from_taxonomy(tax, max_mains=3)
                final_text = (llm_txt.strip() + (" " + collected_line if collected_line else "")).strip()
        else:
            prompt = PROMPT_FREE.format(question=question)
            llm_txt = _call_llm(prompt)
            final_text = llm_txt or "(No dataset match) and LLM unavailable — provide API key or ask a dataset-related question."

        # ---- decide viz ----
        chosen = pick_viz_path(question)
        viz_ok = bool(chosen) or is_viz_relevant(question)

        # ---- remember last answer for future "summarize" ----
        _CHAT_STATE["last_answer"] = final_text

        return {"text": final_text, "viz_relevant": viz_ok, "viz_path": chosen}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")


@app.get("/game/data")
def game_data():
    items = load_game_items()
    payload = {"count": len(items), "items": items, "dir": str(GAME_DATA_DIR)}
    return JSONResponse(
        content=payload,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
