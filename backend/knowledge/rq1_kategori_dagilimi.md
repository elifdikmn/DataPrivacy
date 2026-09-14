# Araştırma Sorusu 1: Kategori Dağılımı ve Hassas Kategorilerin Payı

Veri seti 12.811 parametre kaydından oluşuyor, 4.592 benzersiz GPT eklentisi tarafından toplanıyor. Her kayıt `main_data_type` (25 kaba kategori) ve `data_type` (145 ince kategori) ile etiketlenmiş.

## Genel dağılım

`main_data_type` dağılımı çarpık: en büyük 3 kategori (App usage data: 2.568 kayıt, Identifier: 1.888 kayıt, Other: 1.721 kayıt) tek başına verinin %48.2'sini oluşturuyor. Geri kalan 22 kategori uzun bir kuyruk halinde küçülerek devam ediyor.

Tam sıralama (kayıt sayısı, yüzde):
- App usage data: 2.568 (%20.05)
- Identifier: 1.888 (%14.74)
- Other: 1.721 (%13.43)
- Query: 1.416 (%11.05)
- Time: 1.131 (%8.83)
- Web and network data: 925 (%7.22)
- Location: 701 (%5.47)
- Personal information: 435 (%3.40)
- Files and documents: 412 (%3.22)
- Market data: 367 (%2.86)
- Security credentials: 276 (%2.15)
- Message: 192 (%1.50)
- App metadata: 138 (%1.08)
- Finance information: 138 (%1.08)
- Health information: 82 (%0.64)
- E-commerce data: 66 (%0.52)
- Travel information: 58 (%0.45)
- Sports information: 57 (%0.44)
- Event information: 51 (%0.40)
- Vehicle information: 42 (%0.33)
- Real estate data: 35 (%0.27)
- Food and nutrition information: 33 (%0.26)
- Gaming data: 27 (%0.21)
- Weather information: 26 (%0.20)
- Legal and law enforcement data: 26 (%0.20)

## Hassas kategoriler

Bu projede "hassas kategori" olarak tanımlanan 4 kategori: Security credentials, Personal information, Health information, Finance information. Bu dördü toplamda **931 kayıt (%7.3)** oluşturuyor — veri setinin küçük ama önemli bir dilimi.

## data_type (ince kategori) hakkında

`data_type` sütununda 145 farklı değer var, dağılım çok daha uzun kuyruklu: medyan sınıf boyutu sadece 11 kayıt, 6 sınıfta tek bir kayıt var, 40 sınıfta 5'ten az kayıt var. En büyük `data_type` değeri "Other" — 3.544 kayıt (bu, `main_data_type`'taki "Other" ile aynı şey değil; main_data_type'taki Other 1.721 kayıt).
