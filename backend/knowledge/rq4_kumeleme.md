# Araştırma Sorusu 4: Eklentilerin Kümelenmesi ve Risk Segmentasyonu

## Soru

Eklentileri topladıkları veri türüne göre gruplandırdığımızda doğal riskli/risksiz kümeler ortaya çıkıyor mu?

## Kurulum

Analiz birimi parametreden eklentiye çevrildi. `plugin_id_filenames` listesi patlatılarak (explode) her (eklenti, parametre) çifti kendi satırı yapıldı: 12.811 parametre kaydı → 40.261 (eklenti, parametre) satırı → 4.592 benzersiz eklenti.

En az 3 parametreli eklentiler alındı (3.041 eklenti — daha azı "profil" için anlamsız/şans eseri olurdu). Her eklenti için 25 `main_data_type` kategorisindeki **oranı** (ham sayı değil) hesaplanarak bir "veri toplama profili" matrisi oluşturuldu (3.041 eklenti x 25 kategori).

## K-Means kümeleme

Kategoriler standartlaştırıldıktan sonra K=2..10 arası silhouette skoru denendi. En iyi skor K=10'da (0.254) çıktı, ama genel olarak skorlar düşük (0.17-0.25) — eklentiler net, keskin ayrılmış kümeler oluşturmuyor.

10 kümenin büyüklükleri çok dengesiz: 2 büyük küme (1.523 ve 891 eklenti) çoğunluğu kaplıyor, kalan 8 küme çok küçük (3-19 eklenti).

## Küme profilleri ve hassas kategori payı

| Küme | Eklenti sayısı | Hassas pay | Baskın kategoriler |
|---|---|---|---|
| 1 | 83 | %16.1 | Market data, Time, Finance information |
| 8 | 891 | %12.5 | Identifier, Other, App usage data |
| 5 | 3 | %10.5 | Real estate data, Location |
| 2 | 338 | %9.4 | Message, Files and documents |
| 6 | 5 | %4.2 | Food and nutrition information |
| 4 | 14 | %3.4 | E-commerce data |
| 7 | 11 | %2.3 | Travel information, Time |
| 9 | 19 | %2.2 | Location, Weather information |
| 0 | 1.523 | %1.5 | App usage data, Query |
| 3 | 154 | %0.0 | App metadata, Query |

## Bulgu

Kümeleme net bir "riskli vs risksiz" ikili ayrım üretmiyor. Bunun yerine **fonksiyonel/tematik gruplar** ortaya çıkıyor (finans & pazar, seyahat, e-ticaret, konum & hava durumu, mesajlaşma & dosya, genel amaçlı, sadece meta veri toplayan). Hassas kategori payı bu kümeler arasında kademeli olarak (%0 ile %16.1 arasında) dağılıyor. En büyük iki küme (eklentilerin ~%80'i) zaten düşük-orta hassas paya sahip; yüksek risk küçük, spesifik-amaçlı kümelerde (finans, emlak gibi) yoğunlaşıyor.

Pratik çıkarım: kümeler ikili bir etiket değil, **sıralanabilir bir risk skoru** sağlıyor — Küme 1 ve Küme 8'deki eklentiler incelemeye öncelik verilmesi gereken gruplar olarak işaretlenebilir.
