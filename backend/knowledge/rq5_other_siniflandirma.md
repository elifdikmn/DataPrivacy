# Araştırma Sorusu 5: "Other" Kayıtlarının Yeniden Sınıflandırılması

## Soru

Kurulan model, etiketi belirsiz olan kayıtları (özellikle `data_type` sütununda "Other" kategorisi, 3.544 kayıt) sınıflandırmak için kullanılabilir mi?

## Metodoloji

Bölüm 2'deki `data_type` modeli olduğu gibi kullanılamaz çünkü o model "Other"ı geçerli bir sınıf olarak görmüştü — bu, modelin yine "Other" demesine yol açardı. Bunun yerine model **sadece "Other" olmayan** 9.267 kayıtla, gerçek 144 `data_type` değeri üzerinden (nadir sınıflar gruplanarak, 78 sınıf) yeniden eğitildi. Böylece model "Other" kayıtlarına baktığında gerçek bir kategori önermek zorunda kaldı.

Model her tahmin için bir "güven skoru" (en yüksek olasılık) üretiyor.

## Sonuçlar

- Yeniden eğitilen modelin kendi test setinde performansı: accuracy %69, macro F1 %42.9 (Bölüm 2'deki data_type modeline yakın).
- 3.544 "Other" kaydına uygulandığında güven skoru dağılımı düşük: medyan %20.9, ortalama %28.9.
- **%50 güven eşiğini geçen kayıt sayısı: 411 (%11.6)**. Geri kalan büyük çoğunluk için model de kararsız — bu kayıtlar muhtemelen gerçekten belirsiz/genel amaçlı.
- Güvenli tahminlerin çoğu sezgisel, genel-amaçlı kategoriler: Current session setting, Resource IDs, Search query, Query filter.
- **7 kayıt** güvenli şekilde bir **hassas kategoriye** (main_data_type düzeyinde Security credentials veya Personal information) işaret ediyor. Örnekler: `name="email"` (%91 güvenle Email address), `name="key"`/`name="KEY"` (%82 güvenle API key), `name="token"` (%51 güvenle Access tokens). Bu kayıtlar orijinal veri setinde "Other" olarak etiketlenmiş ama isimlerinden bile belli ki hassas veri.

## Sonuç

Model "Other" etiketli kayıtların tamamını güvenilir şekilde yeniden sınıflandıramaz — çoğu gerçekten belirsiz kalıyor. Ama küçük, yüksek güvenli bir alt kümede gerçekten yanlış etiketlenmiş hassas veri türlerini yakalayabiliyor. Model, otomatik yeniden etiketleme için değil, **insan gözden geçirmesi için öncelik listesi çıkarma** amacıyla kullanılmalı.
