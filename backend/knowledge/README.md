# Knowledge

Notebook analizlerinin (`notebooks/bolum1_eda.ipynb`, `bolum2_modelleme.ipynb`, `bolum3_uygulama.ipynb`, `bolum4_dogrulama.ipynb`) bulgularını özetleyen metin dosyaları. Bunlar ham veri kayıtlarının yanında RAG'ın bilgi tabanına eklenecek — böylece chatbot "ki-kare testi ne buldu", "model doğruluğu ne kadar" gibi meta-analitik sorulara da cevap verebilecek, sadece tek tek kayıtları arayan bir sistem olmayacak.

- `rq1_kategori_dagilimi.md` — kategori dağılımı ve hassas kategori payı
- `rq2_aciklama_orani.md` — açıklama yazma oranı ki-kare testi
- `rq3_model_performansi.md` — TF-IDF/embedding model karşılaştırması, feature importance
- `rq4_kumeleme.md` — K-Means kümeleme sonuçları
- `rq5_other_siniflandirma.md` — "Other" kayıtlarının yeniden sınıflandırılması
- `rq6_policy_audit.md` — gizlilik politikalarında açıklanma durumu (ayrı denetim veri seti)
- `rq7_hassas_ve_ifsa_edilmemis.md` — hassas ve aynı zamanda açıklanmamış parametreler

Bu dosyalar elle düzenleniyor. `python -m analysis.rebuild` bunların üzerine yazmaz; otomatik taslakları `analysis/results/generated_knowledge/` altına yazar (üzerine yazmak için `--update-knowledge`).
