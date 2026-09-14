# Araştırma Sorusu 3: Parametre Adından Kategori Tahmini (Model Performansı)

## Soru

Sadece parametre adına (ve varsa açıklamasına) bakarak, o parametrenin hangi kategoriye ait olduğu ne doğrulukla tahmin edilebilir?

## Kurulum

- Girdi (X): `name` + `description` birleştirilmiş metin
- Hedef (y): `main_data_type` (25 sınıf)
- Stratified train/test split: 10.248 train / 2.563 test

## Baseline model: TF-IDF + Lojistik Regresyon

- **Accuracy: %68.9**
- **Macro F1: %46.8**
- Weighted F1: %69.2

Örüntüler:
- Küçük sınıflarda precision yüksek, recall düşük (model bu sınıfları nadiren tahmin ediyor ama tahmin ettiğinde haklı çıkıyor).
- `Other` sınıfı tam tersi: precision 0.36, recall 0.69 — model kararsız kaldığında sistematik olarak "Other"a kaçıyor, bir çöp kutusu gibi davranıyor.
- Hassas kategorilerden Security credentials çok iyi ayırt ediliyor (F1 0.85), Health information de iyi (F1 0.61).

## Embedding tabanlı model

Bu ortamda Hugging Face'e ağ erişimi organizasyon politikası gereği engellendiği için `sentence-transformers` yerine spaCy'nin `en_core_web_md` modeli (GloVe tarzı 300 boyutlu kelime vektörleri, kelime ortalaması) kullanıldı.

- Accuracy: %54.9, Macro F1: %42.6 — **TF-IDF'in gerisinde kaldı**.
- Sebep: parametre adları kısa, spesifik teknik terimlerden oluşuyor (password, api_key gibi); TF-IDF tam kelime eşleşmesini yakalarken, kelime vektörü ortalaması bu keskin sinyali bulanıklaştırıyor.
- İstisna: embedding modeli, **çok az örnekli sınıflarda** (<30 kayıt: Travel, Weather, Real estate, Food and nutrition, E-commerce, Finance information) TF-IDF'i geçti — transfer öğrenme az veride avantaj sağlıyor.

## Feature importance (TF-IDF + LogReg)

Hassas kategorileri belirleyen en önemli kelimeler:
- Security credentials: key, api_key, token, password, apikey, secret
- Personal information: email, gender, age, firstname, lastname, birthday, nickname
- Health information: patient, disease, surgery (ama bazı belirsiz kelimeler de var: does, 30 days)
- Finance information: currency, price, budget, asset, annual (ama "related", "related filter" gibi genel ifadeler de öne çıktı, muhtemelen tekrar eden bir açıklama kalıbından)

Security credentials ve Personal information'da model çok güvenilir; Health ve Finance'ta bazı kalıp ezberleme belirtileri var.

## Confusion matrix bulgusu

Satır bazında normalize edilmiş confusion matrix'te "Other" sütunu neredeyse her satırda belirgin bir şerit oluşturuyor. Sports information'ın %91'i, E-commerce data'nın %85'i yanlış tahmin edildiğinde "Other"a gidiyor. Hassas kategorilerden Finance information ve Health information'ın yanlış tahmin edilen kayıtlarının yarısı da "Other"a düşüyor. Kategoriler birbiriyle neredeyse hiç karışmıyor — tek karışıklık kaynağı "Other".

## data_type (145 ince kategori) üzerinde deneme

145 sınıftan 67'si (%2.2 kayıt) 10'dan az örnekli olduğu için "Nadir tür (birleştirilmiş)" kovasında toplandı, 79 sınıflı bir problem elde edildi.

- Accuracy: %65.4 (main_data_type'a yakın)
- Macro F1: sadece %35.9 (main_data_type'ta %46.8'di)
- 79 sınıfın F1 dağılımı: alt çeyrek (25. persentil) tam olarak 0 — sınıfların dörtte biri kadarında model tamamen başarısız.
- `Other` yine aynı örüntüyü tekrarlıyor: precision 0.49, recall 0.87.

## Genel sonuç

Kategori tahmini orta düzeyde başarıyla yapılabiliyor ama model sistematik olarak "Other" kategorisine kaçma eğiliminde — bu, modelin çıktılarına temkinli yaklaşılması gerektiğini gösteriyor.
