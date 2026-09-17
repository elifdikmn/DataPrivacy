# Güncel analizler + esnek sohbet

Bu sürüm GitHub main dalının 5edef47 sürümünü temel alır. Dört açıklamalı notebook, RQ7, grafikler, analysis klasörü ve project_facts.json dosyası aynen korunmuştur. Sohbet düzeltmesi llm.py ve facts.py dosyalarına uygulanmıştır; testler ve belgeler buna uyarlanmıştır.

## Mevcut güncel GitHub projesine uygulama

1. Backend'i durdurun.
2. Paketteki backend/app/llm.py ve backend/app/facts.py dosyalarını mevcut projenizde aynı konumdaki dosyalarla değiştirin.
3. Mevcut Python ortamınızda backend klasöründen `uvicorn app.main:app --reload` komutuyla yeniden başlatın.

Yalnızca bu iki dosyayı değiştiriyorsanız ve indeksiniz mevcut GitHub analizleriyle güncelse yeniden indeksleme gerekmez. Eski analiz sürümünden geçiyorsanız veya sıfırdan kuruyorsanız README kurulumunu izleyin; backend klasöründe `python -m app.indexing` çalıştırmanız gerekir. Tam kaynak ZIP'i kişisel .env dosyanızı ve oluşturulmuş yerel indeksleri içermez.

## Davranış

Normal Türkçe/İngilizce metin, F1, yüzdeler ve güven aralıkları kabul edilir. Sayı kontrolleri yalnızca log tanısıdır; sohbeti engellemez ve doğruluğu garanti etmez. Gerçek API anahtarıyla canlı yanıt kalitesi ayrıca değerlendirilmelidir. Sohbet geçmişini saklama özelliği eklenmemiştir.
