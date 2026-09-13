# Backend App

RAG chatbot'un yeni backend kodu. Adım adım doldurulacak:

- `main.py` — FastAPI giriş noktası
- `config.py` — `.env` okuma, ayarlar
- `indexing.py` — veri (`backend/data/`) + bilgi tabanı (`backend/knowledge/`) metinlerini embedding'e çevirip FAISS index kurma
- `retrieval.py` — kullanıcı sorusuna göre en alakalı kayıtları/bulguları bulma
- `llm.py` — Together API ile RAG cevabı üretme
- `viz.py` — Plotly treemap/bar chart üretimi
