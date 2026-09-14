# Araştırma Sorusu 2: Hassas Kategorilerde Açıklama Yazma Oranı

## Soru

Hassas kategorilerdeki (Security credentials, Personal information, Health information, Finance information) parametrelerde `description` alanı yazılma oranı, hassas olmayan kategorilere göre farklı mı?

## Eksik veri notu

`description` alanında boşluk `NaN` değil **boş string (`""`)** olarak saklanıyor. Genel olarak 12.811 kayıttan 1.999'unda (%15.6) description boş.

## Bulgu

- Hassas kategorilerde açıklama yazılma oranı: **%81.2**
- Hassas olmayan kategorilerde açıklama yazılma oranı: **%84.65**

Yani hassas kategorilerde açıklama yazılma oranı, beklentinin aksine hafifçe **daha düşük**.

## İstatistiksel test

Ki-kare (chi-square) bağımsızlık testi uygulandı:
- Chi-square istatistiği: 7.514
- p-değeri: ≈0.006 (0.05 eşiğinin altında, istatistiksel olarak anlamlı)
- Cramér's V (etki büyüklüğü): ≈0.024 (çok küçük, ihmal edilebilir)

## Yorum

Fark istatistiksel olarak anlamlı ama etki büyüklüğü pratikte önemsiz düzeyde küçük. 12.811 gibi büyük bir örneklemde çok küçük farklar bile "anlamlı" çıkabiliyor. Sonuç: bir parametrenin hassas olup olmaması, açıklamasının olup olmayacağını pratikte neredeyse hiç öngörmüyor — description eksikliği kategoriden bağımsız, genel bir alışkanlık gibi görünüyor.
