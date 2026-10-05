# Piyasa oranları — 2025-26 held-out dönemi

Bu klasör bahis tavsiyesi değildir ve proje bahis tavsiyesi üretmez. Oranlar
tek bir iş için burada: modelin tahminlerini dışarıdan, bağımsız bir ölçüye
vurmak.

## Dosyalar

`nba_2025_26_heldout_odds.csv` — modelin eğitimde görmediği 308 maç
(2026-03-04 → 2026-04-12, sezonun kronolojik son %25'i). Maç başına:

- `ts_*`: theScore'un kapanış handikabı ve alt/üst çizgisi (bir maçta handikap boş),
- `dk_*`: DraftKings'in açılış ve kapanış handikabı, moneyline'ı, alt/üst çizgisi ve fiyatları (ESPN üzerinden),
- `dk_home_prob_novig_*`: marjı ayıklanmış ev sahibi olasılığı,
- maçın skoru ve `home_won`.

Handikap ev sahibi açısından yazılır: eksi, ev sahibi favori demek.
`prediction_engines/market_check.py` bu dosyayı okur.

`nba_2025_26_heldout_odds.xlsx` — aynı oranlar, kaynakların ve erişilemeyen
sitelerin dökümü ("Kaynaklar ve notlar" sayfası) ve iki "Model vs Piyasa"
sayfası.

## "Model vs Piyasa" sayfaları eskidir

O iki sayfa 13 Eylül 2026'daki modelden üretildi. O model, maçta sahaya
çıkan oyunculardan kurulan kadro feature'ları yüzünden sonucu kısmen
görüyordu (DURUM.md, "Bulunan iki sızıntı"). Sayfalarda görünen handikap
üstünlüğü bu sızıntının ürünüdür, gerçek değildir: sızıntı kapatıldıktan
sonra modelin handikap tarafı %45,1 ± 2,8 tuttu, yani yazı turadan iyi
değil. Excel yeniden üretilmedi.

Güncel karşılaştırma `output/market_check.json` içinde ve DURUM.md'nin
"Piyasa ile karşılaştırma" bölümünde: model kazananı piyasa kadar tutturuyor
(0,773'e 0,776), olasılık, fark ve toplam skorda piyasanın gerisinde.
