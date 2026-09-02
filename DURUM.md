# Durum — db-pipeline dalı

Son güncelleme: 2026-09-02. Bu dosya, çalışmaya ara verildiğinde nerede
kalındığını ve nasıl devam edileceğini anlatır.

## Tek cümlelik özet

Veri katmanı MariaDB'ye taşındı ve bir veri hatası düzeltildi; altı feature
ailesi denendi ve hiçbiri doğruluğu artırmadı; doğruluğu artıran tek şey seed
ensemble oldu (+1.03 puan).

## Dürüst rakamlar

Tek split rakamlarına güvenme — sezonun en kolay beş haftasını ölçüyor ve
yaklaşık 13 puan yüksek okuyor. Bağlayıcı olan walk-forward:

| ölçüm | değer |
|---|---|
| Doğruluk (tek model) | 0.6716 ± 0.0552 |
| Doğruluk (10 seed ensemble) | ~0.6819 |
| Naif taban | 0.5494 |
| Tabana kazanç | +12.2 puan |
| AUC | 0.7376 |
| Brier | 0.2126 (kalibrasyonlu 0.2083) |
| Hedef | 0.70 |

Protokol: 24 aylık genişleyen pencere × 5 seed, burn-in ≥ 10 maç, her katman
kendinden önceki her şeyle eğitiliyor.

## Veri

9 sezon, 10.749 maç, `phonedb` (MariaDB, Tailscale üzerinden 100.101.28.63).

Yerel kopyalar DB olmadan çalışmaya yeter:

| dosya | içerik |
|---|---|
| `output/engineered_dataset_db.pkl` | 10.749 × 274, 190 temel + rest/clutch/avail grupları |
| `game_impact_cache_v4.pkl` | 220.001 oyuncu-maç impact kaydı, person_id anahtarlı |
| `output/walk_forward_7arms.json` | 7 kollu son ölçüm, 105 hücre |
| `output/seed_ensemble.json` | ensemble ölçümü |
| `output/staleness.json` | ağırlık eskimesi ölçümü |

`.env` DB kimlik bilgilerini taşır ve gitignore'dadır.

## Ne denendi, ne çıktı

Hepsi 105 hücrede, A'ya göre eşleştirilmiş fark:

| aile | fark | se | hücre | sonuç |
|---|---|---|---|---|
| müsaitlik (avail) | **+0.0011** | 0.0018 | **315** | sıfır (aşağıya bak) |
| clutch | +0.0006 | 0.0031 | 105 | sıfır |
| kalibrasyon | −0.0003 | 0.0029 | 105 | doğrulukta sıfır, Brier'de −%2 |
| rest / b2b | −0.0015 | 0.0028 | 105 | sıfır |
| rating (Elo/Massey) | −0.0030 | 0.0034 | 105 | sıfır |
| **seed ensemble** | **+0.0103** | **0.0051** | 21 ay | **tek gerçek kazanç** |

Ensemble kazancı N ile tekdüze artıyor (+0.0070 / +0.0088 / +0.0103 at 3/5/10),
ki bu bagging'in öngördüğü doz-yanıt ilişkisi.

### Müsaitlik: güç artınca etki söndü

105 hücrede +0.0032 ± 0.0025 (1.28×) ile en umutlu aileydi. Kol sayısı 7'den
2'ye indirilip seed 5'ten 15'e çıkarılarak 315 hücreye taşındığında etki
**+0.0011 ± 0.0018**'e düştü ve hücrelerin yalnızca 144/315'inde önde çıktı —
yarının altında. Güç arttıkça sıfıra yaklaşan bir etki, gerçek bir etki değil.

Bu, ölçüm gücünü artırmanın neden gerekli olduğunun somut örneği: aynı aile az
hücreyle "en umutlu aday" görünüyordu.

### Toplu sonuç

Altı bağımsız feature ailesi denendi, altısı da sıfır. Bunlardan üçü iki
parçalı tarama testini (tahmin gücü + yenilik) geçmişti. Elimizdeki veriden
türetilen tablo tipi feature mühendisliği tükenmiş durumda.

## Öğrenilen dersler

**İki parçalı tarama testi yeterli değil.** Bir feature'ın işe yaraması için
hem sonucu tahmin etmesi hem yeni bilgi taşıması gerekiyor gibi görünüyordu,
ama clutch ikisini de geçip sıfır getirdi. Ridge R² testi *doğrusal*
fazlalığı ölçüyor; LightGBM'in kurabildiği doğrusal olmayan bileşimler çok
daha geniş. Test bir eleme filtresi olarak kullanılabilir, yeterli koşul değil.

**Elo neden işe yaramadı.** `diff_rating_elo` mevcut feature'lardan %81
tahmin edilebiliyor; en çok örtüştüğü `diff_season_avg_point_margin` (0.842).
Yani Elo aslında "rakip düzeltmeli sezon ortalama marjı" ve model zaten sezon
ortalama marjına sahip.

**Model ağırlıkları eskimiyor.** Test maçları sabit tutulup eğitim kesimi
değiştirildiğinde 150 günlük gecikme −0.0013 ± 0.0090 maliyet veriyor.
Feature'lar zaten maç tarihine göre hesaplandığı için yeni sezonda günlük
yeniden eğitim gerekmiyor; aylık yeter.

**Zayıf aylar veri sorunu değil.** Aralık–Ocak kötü çünkü maçlar gerçekten
daha yakın (%33'ü ≤6 sayı, Nisan'da %27) ve feature'lar o aylarda %39 daha az
sinyal taşıyor. Doğruluk yakın maç oranıyla −0.787, feature-sonuç
korelasyonuyla +0.879 ilişkili.

**Varyansın %86'sı ay etkisi**, %14'ü seed. Ensemble seed kısmına dokunuyor,
o yüzden doğruluğu artırıp tutarlılığı artırmıyor.

## Devam edilecek yer

Bekleyen koşu yok; müsaitlik testi tamamlandı ve sonucu yukarıda.

Production şu an: temel 190 feature, 10 seed ensemble, isotonic kalibratör.
Eklenmiş fazladan feature ailesi yok — çünkü hiçbiri kazanmadı.

## Sonraki adım adayları

Doğruluk cephesinde elimizdeki veri tükendiği için, sıradaki iş "aynı sayıyı
yükseltmek" değil, farklı bir şey üretmek olmalı.

1. **Simülasyon motorunu ürünleştirmek.** Doğruluk artırmıyor ama beş modelin
   çelişkisini bitiriyor (kazanan–marj çelişkisi 28 maçtan 1'e) ve dağılım
   veriyor: %80 aralık gerçekte %78.6 kapsıyor, skor yayılımı 4.5'ten 13.1'e
   çıkıp gerçek 13.7'ye oturuyor. Ürün değeri burada, ve ödül modelinin de
   altyapısı bu.
2. **Ödül modeli (MVP/DPOY).** Altyapı hazır: `player_game_impact` (220.001
   satır, person_id anahtarlı), `pbp_defensive_event` (270.803 satır, blokçu→
   şutör eşleşmesi), `PlayerAwards` endpoint'i All-NBA/All-Defensive
   etiketlerini veriyor (sezon başına ~25 pozitif). Maç sonucu tahmininde tavan
   var; bu problem el değmemiş.
3. **Dışarıdan bilgi.** Sakatlık raporları (maç öncesi, bizim geriye dönük
   yokluk verimizden farklı), bahis çizgisi hareketi. Elimizdeki veriden
   çıkarılabilecekler tükendi; buradan yukarısı yeni bilgi gerektiriyor.

Hedefe uzaklık: ensemble'lı walk-forward ~0.682, hedef 0.70, kalan 0.018.
Karşılaştırma için bahis piyasasının kapanış çizgisi düz kazanan tahmininde
kabaca %68-72 tutar — yani hedef, piyasayı yakalamak demek.

## Çalıştırma

```
py prediction_engines/build_dataset_db.py        # DB gerekir, ~5 dk
py prediction_engines/retrain_production.py      # yerel, 10 seed ensemble
py prediction_engines/walk_forward.py --dataset output/engineered_dataset_db.pkl
py app.py                                        # Tailscale IP'sine bind eder
```

Otomatik push 5 dakikada bir yalnızca veri dizinlerini commit eder; kod elle
commit edilir (`schtasks /Delete /TN "NBA repo auto-push" /F` ile kaldırılır).
