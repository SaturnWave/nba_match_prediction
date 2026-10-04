# Durum — db-pipeline dalı

Son güncelleme: 2026-10-04 (sezon simülatörü; öncesinde güç sıralaması, değer skoru, fiyatlama 5.1). Bu dosya, çalışmaya ara verildiğinde nerede
kalındığını ve nasıl devam edileceğini anlatır.

## Tek cümlelik özet

Modelin iki sızıntısı bulundu ve kapatıldı (kadro feature'ları maçta sahaya
çıkanlardan hesaplanıyordu; üretim modelleri test setiyle erken duruyordu);
dürüst feature'lar üzerinde düzenlileştirme + monoton kısıt olasılık kalitesini
ve iç tutarlılığı artırdı, isabeti değiştirmedi; üretim bu tarifle yeniden
kuruldu.

## Dürüst rakamlar

Bağlayıcı olan walk-forward (18 ay × 3 seed = 54 hücre, 2023-11 → 2026-04,
burn-in ≥ 10 maç, `output/impact_v5_lab.json`, kol `candidate` = üretimdeki
tarif ve fiyatlama 5.1):

| ölçüm | değer |
|---|---|
| Kazanan isabeti (sınıflandırıcı) | 0,666 (ay sapması 0,066) |
| Kazanan isabeti (blend, dashboard'un gösterdiği) | 0,687 |
| AUC | 0,742 |
| Brier | 0,211 (blend 0,204) |
| Sayı farkı MAE | 11,2 |
| Kazanan–marj çelişkisi | %10,5 (eski tarifte %13,0) |
| Naif taban | 0,555 |

Tek split (son 308 maç, 2026-03-04 sonrası, hiçbir fit'e girmedi):
blend isabet 0,773, AUC 0,829, Brier 0,167; marj MAE 11,49; toplam MAE 15,22
(impact v5 modelleri; v4 modelleriyle 0,786 / 0,165 / 11,37 idi — 308 maçta
gürültü sınırında, bağlayıcı olan walk-forward değişmedi).
Sezonun son beş haftası kolay okur; bu rakama değil üsttekine güven.

## Bulunan iki sızıntı

**1. Kadro maçın kendisinden okunuyordu.** `roster_impact_*` ve `matchup_*`
feature'ları oyuncu değerlerini maçtan önce hesaplıyordu (bu doğrulandı), ama
hangi oyuncuların toplanacağını o maçın impact cache'inden / matchup
dosyasından, yani **sahaya çıkanlardan** alıyordu. Sahaya çıkan oyuncu sayısı
maçın |farkı| ile 0,64 korelasyonlu (25 sayılık maçta iki bank da boşalır).
`roster_impact_l10_sum` sayı farkı modelinin 1 numaralı feature'ıydı.

Ölçülen büyüklük (eşleştirilmiş, 54 hücre, `output/consistency_lab.json`):
isabet +0,8 ± 0,4 puan, AUC +0,014, Brier −0,005, **sayı farkı MAE −0,80 sayı
(54 hücrenin 54'ünde)**. Blend isabetine etkisi yok, olasılık kalitesine var.

Çözüm: `prediction_engines/pregame_roster.py`. Kadro = maçın forma listesi
eksi sakatlık raporu işaretleri (`DND`, `NWT`, `Injury`, `Rest`...; "DNP -
Coach's Decision" dahil çünkü o oyuncu formayı giymiş). Ağırlık = önceki
maçlardaki dakika ortalaması. Yeni kadro sayısı 12,5 ± 1,4; |fark| ile
korelasyonu −0,015. Ek iki meşru sinyal: `roster_avail_minutes` (normal
rotasyonun ne kadarı sahada) ve `roster_missing_impact` (eksik oyuncuların
değeri). Veri seti: `output/engineered_dataset_pregame.pkl`
(`blob["roster"] == "pregame"`).

**2. Test seti ağaç sayısını seçiyordu.** `retrain_production.py` her üyeyi
308 maçlık held-out dilimiyle erken durduruyordu. Etkisi küçük (isabet
~0,3 puan, marj MAE ~0,1) ama dashboard'daki "gerçek out-of-sample" etiketi
doğru değildi. Şimdi: eğitim satırlarının son %10'u ağaç sayısını seçiyor,
üye tüm eğitim satırlarıyla o sayıda yeniden kuruluyor; held-out yalnızca
skorlanıyor.

Üçüncü bir uyumsuzluk da düzeltildi: simülatör eski CSV veri setinden
eğitilmişti, DB veri setiyle servis ediliyordu. `simulation.py` artık üretim
veri setini ve `models/feature_list_2025_26.json`'daki listeyi kullanır.

## Ne denendi, ne çıktı (dürüst feature'lar üzerinde)

1. tur, `pregame`'e göre eşleştirilmiş (`output/consistency_lab.json`):

| kol | isabet | AUC | Brier | sonuç |
|---|---|---|---|---|
| +eksik oyuncu (`pregame+out`) | −0,005 ± 0,006 | +0,004 | +0,001 | tek başına sıfır, blend'de +0,4 |
| sadece `diff_` (66 feature) | +0,002 ± 0,006 | +0,006 | −0,0035 ± 0,0015 | olasılık kalitesi |
| düzenlileştirilmiş öğrenici | −0,006 ± 0,007 | **+0,013** | −0,0026 | sıralama kalitesi |
| monoton kısıt | +0,001 ± 0,005 | +0,008 | −0,0034 ± 0,0013 | olasılık kalitesi |
| sezon fazı (gp) | −0,004 ± 0,006 | +0,006 | +0,001 | sıfır |

2. tur, `pregame+out`'a göre, hepsi blend ile (`output/consistency_lab_round2.json`):

| kol | isabet | Brier | çelişki | blend isabet | blend Brier |
|---|---|---|---|---|---|
| **reg+mono+out** | **+0,011 ± 0,005** | **−0,0061 ± 0,0014** | %8,9 | +0,001 | −0,0013 ± 0,0004 |
| diff_mono | +0,011 ± 0,006 | −0,0043 | %10,7 | **+0,006 ± 0,002** | −0,0010 |
| reg+out | −0,001 | −0,0034 | %11,5 | +0,001 | −0,0010 |
| diff_reg_mono | −0,003 | −0,0030 | %12,7 | +0,004 | −0,0010 |

Üretim tarifi `reg+mono+out`: `num_leaves=15, min_child_samples=60,
reg_lambda=5, colsample=0.5, subsample=0.8 (freq 1), lr=0.02` + her `diff_`
sütununda eğitim satırlarından işaretlenen monoton kısıt; kazanan ve marj
modellerinde. Skor modelleri ölçülmediği için eski parametrelerde.
`diff_mono` blend isabetinde daha iyi ama ay sapması daha yüksek; isabet
farkları seed gürültüsü sınırında, Brier/çelişki farkları değil.

## Piyasa ile karşılaştırma (308 maç, DraftKings kapanış, `output/market_check.json`)

| soru | model | piyasa |
|---|---|---|
| kazanan isabeti | 0,773 | 0,776 |
| Brier | 0,167 | 0,153 |
| sayı farkı MAE | 11,49 | 10,58 |
| toplam MAE | 15,22 | 14,33 |
| handikapta modelin tarafı | %45,1 ± 2,8 | başabaş %52,4 |

Önceki oturumda görülen "handikap %57 / çizgiden 3+ sapınca %61" sinyali
**kadro sızıntısının ürünüydü**; dürüst modelde kayboldu. Piyasa marjda ve
toplamda açıkça daha iyi; kazananda berabere. Bahis tavsiyesi yok, olmayacak;
piyasa yalnızca dış ölçüt.

## Güç sıralaması ve OVR (`prediction_engines/power_rankings.py`)

Dashboard'da "Güç Sıralaması" görünümü: hafta / ay / sezon pencereleri,
30 takım logolu, oyuncu OVR listesi. Hepsi oynanmış maçların özeti, tahmin
değil. Birimler maç başına sayı (lig ortalaması rakibe, nötr sahada):

| sayı | ne |
|---|---|
| sezon gücü | sezon-bugüne Massey çözümü (ridge, eşit ağırlık); sezonun ilk ~100 maçı dolmadan Elo'dan çevrilir (28 Elo ≈ 1 sayı) |
| form | pencerenin maçlarında sezon gücü + ev avantajı çıkarıldıktan sonra kalan marjın ridge çözümü; az maçta sıfıra çekilir |
| güç | hafta/ay: sezon gücü + form; sezon: sezon gücü (form ayrı gösterilir) |
| takım OVR | 85 + 1,1 × güç, 60–99 |
| oyuncu OVR | penceredeki impact/maç ortalamasının ligdeki yüzdeliği: 65 + 34 × p^2,6 (ortanca ~71, en iyi %3 96+); ≥12 dk ve yeterli maç şartı |
| kadro OVR | sezon oyuncu OVR'larının en çok oynayan 8 oyuncuda dakika ağırlıklı ortalaması |

API: `/api/rankings/periods`, `/api/rankings?window=&key=`,
`/api/rankings/players?window=&key=&team=`. Haftalık sorgu ~0,2 sn, sonuçlar
önbellekte. Trend oku bir önceki dönemle (sezon görünümünde 28 gün önceyle)
karşılaştırır.

## Değer skoru ve "boş istatistik" (`prediction_engines/value_engine.py`)

Kullanıcının sorusu: Westbrook 2017-18'de Curry'den yüksek impact alıyor; stat
padder nasıl ayırt edilir? Teşhis: impact skoru üretimi sayar, bedelini saymaz.

| impact skorunun açığı | etkisi |
|---|---|
| kaçan şut sıfır | 21 şut/maç %45 ile atan her isabetten puan alır, hiçbir kaçırmadan kaybetmez |
| her ribaunt 0,9 taban (+bonuslar); "Off" kontrolü açıklamadaki "Off:0 Def:1" sayaçlarıyla eşleştiği için **her ribaunt hücum ribaundu sayılıyor** | takımın zaten alacağı savunma ribaundu basketin üçte biri ediyor |
| asist ve serbest atış yok | Harden/Curry'nin değeri eksik |
| çöp zaman indirimi yok | 25 farkla biten maçın son çeyreği tam puan |

2017-18'de 166 düzenli oyuncu: impact/36 dk ile oyuncunun sahadaki +/−'si
arasındaki korelasyon **0,14**. Üst sıralar ribaunt alan uzunlar (Drummond
43,8 ile 5., +/− −0,6).

Değer skoru: her oyun sayı cinsinden, 1,00 sayı/pozisyon tabanına göre
fiyatlanır (basket = sayı − 1; kaçan şut −0,74; TOV −1,15; STL +1,15; ORB
+0,74; DRB +0,26; blok +0,6; asist ve serbest atış için aşağıdaki "Fiyatlama
5.1" bölümü; faul −0,3/−0,6)
ve maçın o anki açıklığıyla ağırlıklanır (25+ fark Q4 → 0,3; son 5 dk ≤5
fark → 1,25). Stil bonusu yok.

Sonuç (2017-18): +/− ile korelasyon **0,14 → 0,42** (maç başına 0,28 → 0,48).
Curry 9,9/maç, Westbrook 6,7 (üretimde 32,0 vs 40,7). Westbrook'un
bileşenleri: skor +10,6, kaçan şut −8,6, top kaybı −5,3. "Boş istatistik"
listesi: Nurkić, Josh Jackson, Schröder, Dennis Smith Jr.; "sessiz değer":
Curry, Collison, Chris Paul, Otto Porter.

Sıralama ekranında oyuncu OVR artık değerden; üretim OVR yanında, fark
"boş istatistik" rozeti (≥ +8). Önbellek: `output/value_cache_v1.pkl`
(`value_engine.py --all`, ~7 dk). Hâlâ görmediği: blok/top çalmaya
dönüşmeyen savunma, perde, alan açma.

## Sezon simülatörü (`prediction_engines/season_sim.py`) — kum havuzu

Bir sezonu, gerçek veriye dokunmadan, pozisyon pozisyon oynatır. Her pozisyon
çekilir (kim bitirdi, şut mu top kaybı mı serbest atış mı, girdi mi, kim pas
verdi, kim ribaunt aldı) ve **gerçek play-by-play'in sütun düzeni ve
ifadeleriyle** satır olarak yazılır; böylece impact motoru (`value_engine.py`,
fiyatlama 5.1) simüle maçı gerçek maç gibi okur. Çıktılar yalnızca
`sim_sandbox/<koşu>/` altına yazılır (git'te yok sayılır); her koşu proje
verisinin parmak izini önce ve sonra alıp değişip değişmediğini raporlar.

Maçı ne belirler: oyuncu profilleri (önceki sezonların kutu skorlarından
dakika başına şut, serbest atış gidişi, top kaybı, asist, ribaunt, top çalma,
blok, faul oranları ve şut yüzdeleri; az oynayan oyuncu yedek seviyesine
çekilir) + kutu skorunda bireysel ölçüsü olmayan üç takım etkisi (tempo,
rakibin şut yüzdesi, zorlanan top kaybı; tahmin modunda oyuncularla birlikte
taşınır) + ev sahibi avantajı + skor etkisi (öndeki takım gevşer). **Eğitilmiş
tahmin modelleri maç üretmek için kullanılmaz**: modelin ürettiği bir dünya
ancak modelle aynı fikirde olabilirdi; bu dünya modeli sınamak için var.

Üç mod:

| mod | ne yapar | ne söyler |
|---|---|---|
| `forecast` | oynanmamış sezon: yayınlanmış fikstür + güncel kadrolar (`nba_api`), son iki sezonun profilleri | simüle sezon |
| `mechanics` | oynanmış sezon, her maçın gerçek kadrosu ve dakikalarıyla | üreteç doğru mu |
| `backtest` | oynanmış sezon, yalnızca öncesindeki bilgiyle | bir tahmine ne kadar güvenilir |

Üretecin doğruluğu (`mechanics`, 2025-26, altı tohum ortalaması; gerçek / simüle):
sayı 115,6 / 116,0; pozisyon 101,8 / 101,7; şut %47,1 / %47,3; ev sahibi farkı
1,74 / 1,70; sayı farkı sapması 16,4 / 16,6; toplam sapması 19,9 / 20,1.
Takım galibiyetleri korelasyon 0,86–0,88, takım sayı farkı 0,91–0,93.
Bilinen sapmalar: takımlar arası fark gerçeğin ~%80'i (5,2 / 6,2), uzatma oranı
%2–3 (gerçek %4,4).

Lig oranları ölçüldü, varsayılmadı (2025-26, 200 maç): asistli ikilik %54,1,
üçlük %85,5; kaçan ikiliğin %20,1'i blok; oyuncu top kaybının %61'i top çalma;
hücum ribaundu %26; serbest atış gidişlerinin %3,8'i üçlük, and-one isabetli
ikiliklerin %7,5'i.

**Tahmin gücü sınırlı ve bu ölçüldü** (`backtest`, beş sezon, sezon öncesi
bilgiyle): gerçek galibiyetle korelasyon ~0,53, ortalama mutlak hata ~8,7 maç;
"geçen sezonun aynısı" 0,58 / 8,8. Yani simüle puan durumu makul bir dünyadır,
güvenilecek bir tahmin değil. Takım etkilerini hiç taşımamak belirgin kötü
(0,36 / 9,7). Galibiyet aralıkları (`--replications`) her tekrarda takım gücünü
de çeker (`TEAM_SEASON_SD = 0,11`), öyle ki %10–%90 aralığı geçmiş sezonlarda
gerçeği ~%80 kapsasın; aralıklar ~30 maç genişliğinde.

2026-27 koşusu (`sim_sandbox/forecast_2026_2027/`, tohum 20261020): 1.200 maç
(takım başına 80; NBA Kupası eleme maçlarının 6'sının takımı belli değil,
24 maç henüz takvimde yok), 620 oyuncu, 84 sn. Motorun simüle satırlardan
okuduğu asistler üretecin dağıttığıyla 1.200 maçın tamamında aynı. Dosyalar:
`games.csv`, `player_games.csv`, `player_values.csv`, `players.csv`,
`standings.csv`, `win_ranges.csv`, `pbp.csv.gz` (tüm satırlar),
`impact_cache.pkl` (gerçek önbellekle aynı düzen), `value_cache.pkl`,
`summary.md` (puan durumu, projenin güç sıralaması motoru simüle sezonda,
oyuncu OVR, sayı kralları).

Simüle edilmeyenler: oyuncu değişiklikleri (kadro dakika ağırlıklı toplam),
ikincil ve serbest atış asisti, teknik ve flagrant fauller, çaylakların gerçek
seviyesi (hepsi yedek profili, az dakika), sakatlıklar yalnızca geçen sezonun
oynama oranından.

Sıradaki olası adım: eğitilmiş modeli simüle dünyanın içinde çalıştırmak (her
simüle maçtan önce feature üretip tahmin almak). Bunun için oynanmamış maça
feature üreten artımlı bir yol gerekiyor; gerçek sezon için de aynı parça.

```
py prediction_engines/season_sim.py forecast --season 2026_2027 --replications 30
py prediction_engines/season_sim.py mechanics --season 2025_2026 --fast
py prediction_engines/season_sim.py backtest --season 2025_2026 --fast
py prediction_engines/season_sim.py show --run forecast_2026_2027 --game 0022600003
```

## Fiyatlama 5.1 — asist ve serbest atış eksiksiz

Kullanıcı: "ilk denememde asist ve serbest atışı hesaba katmıştım; asist pası
bile hesaba katılmalı." İzi sürüldü:

- İlk tek-maç motorunda (`impact_score_calculation/impact_score.py`,
  `add_player_tracking_impact`) **ikincil asist × 0,5** ve pas/asist oranından
  bir bonus vardı. Sezon betikleri ve tahmin motoru `df_player_track`
  vermeden çağırdığı için bu parça sezon ölçeğinde hiç çalışmadı. Ayrıca
  maç toplamı olan bu değer oyuncunun adının geçtiği **her satırda** yeniden
  ekleniyordu.
- Serbest atış hiçbir sürümde puanlanmadı; "Free Throw" yalnızca and-one
  tespiti için okunuyordu.
- v5.0 ikisini de sayıyordu ama eksikti: asistlerin yalnızca **%87–89'u**
  oyuncuya yazılabiliyordu.

Ne değişti (`value_engine.py`, `PRICING_VERSION = "5.1"`):

| konu | önce | şimdi |
|---|---|---|
| pasörü bulma | o ana kadar görülen oyunculara soyadıyla eşleme; %87–89 | tüm maç + kutu skoru kadrosu; aksansız, eksiz ("Butler" = "Butler III"), baş harfli ("G. Antetokounmpo", "Ja. Green"), açıklamadaki sıra numarasıyla; **%99,8–100**, oyuncu-maç bazında %99,7–99,98 birebir |
| asistin değeri | sabit +0,35 | basketin değerinin **%30'u, skorerin üstüne** (ikilik +0,3, üçlük +0,6) |
| ikincil asist | yok | tracking `sast` × 0,15 |
| serbest atış asisti | yok | tracking `ftast` × 0,17 (not: bu sayım 2019-20'den itibaren kabaca iki katı kaydediliyor) |
| serbest atışın pozisyon maliyeti | atış başına sabit 0,44 | n atışlık gidişte atış başına 1/n; and-one, teknik, flagrant, clear path = 0 |

Serbest atış sayımı zaten doğruydu: play-by-play isabet ve deneme kutu
skoruyla her oyuncu-maçta birebir.

Asist kuralı ölçülerek seçildi (düzenli oyuncular, sahadaki +/− /36 dk ile
korelasyon; 2017-18, 2020-21, 2023-24 ortalaması):

| kural | korelasyon |
|---|---|
| asist kredisi yok | 0,394 |
| sabit +0,35, üstüne | 0,442 |
| **basketin %30'u, üstüne** | **0,445** (%20: 0,437; %40: 0,444; %50: 0,436) |
| basketin %30'u, skorerden kesilerek | 0,430 (%20: 0,434; %40: 0,413; %50: 0,389) |

Skorerden kesmek üç sezonda da daha kötü ve kesilen pay arttıkça daha kötü:
şutu sokmak skorerin işi, pas onu daha iyi bir şut yapan şey. İkincil asist
küçük ama her sezon aynı yönde (+0,001…+0,004); serbest atış asisti ölçülemez
(+0,001); **pas başına kredi zarar veriyor** (−0,007 / −0,014), o yüzden ilk
motordaki pas/asist bonusu geri getirilmedi. Potansiyel asist (şut girseydi
asist olacak pas) elimizdeki veride yok; tracking'in `pass` sütunu atılan her
pas. Serbest atışın yeni maliyeti muhasebeyi kesinleştiriyor, +/− ile ilişkiyi
değiştirmiyor (0,431 / 0,433).

Doğrulama, ayrı tutulan altı sezon (ayar üç sezonda yapıldı): +/− ile
korelasyon 0,379 → **0,396**, altı sezonun beşinde daha iyi (2025-26'da
−0,016); ayar sezonlarında 0,430 → 0,448. 2017-18 maç başına değer: Curry 9,9 → 10,6,
Westbrook 6,7 → 7,9, Chris Paul 7,2 → 8,4 (OVR 97, üretim OVR'ı 86).

Model tarafı (`impact_v5_lab.py`, 54 hücre, önceki üretime karşı eşleştirilmiş):
blend isabeti +0,0019 ± 0,0013, blend Brier +0,0002 ± 0,0003, sınıflandırıcı
−0,0044 ± 0,0051, AUC aynı, marj MAE 11,22 / 11,21, çelişki %10,5 / %9,3.
Ayırt edilemez; tek fiyatlama projenin tamamında.

Düzeltilen bir ölçüm hatası: karşılaştırma betiği "impact sütunlarını" ada
göre seçiyordu ve `roster_form_l6/l3` (onlar da impact'ten) ile
`roster_avail_minutes` dışarıda kalıyordu; önceki üç koşuda (aşağıdaki
tablo) aday kol bu yüzden melezdi. Artık iki veri seti arasında değeri
gerçekten farklı olan sütunlar (30) seçiliyor. Önceki sonuçların yönü
değişmiyor: v4 ile tam v5.0 arasındaki fark aynı hücrelerde isabet 0,6729 /
0,6699, Brier 0,2101 / 0,2109.

Yeniden kurma: `value_engine.py --all --refresh` (oyuncu, ~4 dk) ve
`--all --refresh --no-leverage --cache output/value_cache_unweighted.pkl`
(model) → `impact_v5.py` → `rebuild_impact_features.py --in
output/engineered_dataset_pregame_v4.pkl` → `retrain_production.py`.

## Impact motoru v5 — açıklar kapatıldı, proje geneline yayıldı

Kullanıcının hatırladığı düzeltmeler hiçbir dalda yoktu (Master, csv-pipeline,
db-pipeline, stash: "Off" testi ilk commit'ten beri aynı, kaçan şut/asist hiç
yazılmamış). Yeniden yapıldı ve bu kez **tüm tüketiciler** değiştirildi:

| ne | nerede |
|---|---|
| fiyatlama motoru | `value_engine.py` (üstteki tablo) |
| v4 biçiminde önbellek | `impact_v5.py` → `game_impact_cache_v5.pkl`; takım farkı ↔ gerçek marj korelasyonu **0,80 → 0,90** |
| veri setinin 24 impact sütunu | `rebuild_impact_features.py`: takım toplamları + 12 rolling sütun (FeatureEngineer tarifi eski sütunları birebir üretti, fark 0,00) + maç öncesi kadro ailesi |
| model tarafı tüketiciler | `build_dataset_db`, `db_source`, `pregame_roster`, `player_source`, `player_simulation`, `db_build_derived`, `predict_2025_2026` → v5 |
| bilerek v4 kalan | `app.py` (sıralamadaki "üretim" sütunu eski motorun saydığı şey olmalı ki "boş istatistik" farkı anlamlı kalsın), `value_engine.py` (karşılaştırma hedefi) |

Walk-forward etkisi (`impact_v5_lab.py`, aynı 54 hücre, reg+mono+out tarifi,
v4 ile eşleştirilmiş):

| modelin impact feature'ları | isabet | Brier | marj MAE | çelişki | seed yayılımı |
|---|---|---|---|---|---|
| v4 (eski üretim skoru) | 0,673 | 0,2101 | 11,20 | %8,9 | 0,043 |
| değer, oyun ağırlıklı (`impact_v5_leverage_lab.json`) | −0,7 ±0,7 | **+0,0030 ±0,0015** | **+0,04 ±0,01** | %12,4 | 0,045 |
| v4 + ağırlıklı değer birlikte (`impact_v5_lab_both.json`) | **−2,2 ±0,7** | +0,0051 | +0,02 | %14,9 | 0,054 |
| **değer, ağırlıksız (`impact_v5_lab.json`, üretimde)** | −0,15 ±0,5 | +0,0009 ±0,0013 | +0,015 ±0,013 | %9,6 | **0,034** |

Okuma: oyun (çöp zaman) ağırlığı oyuncuya hak vermek için doğru, takım
feature'ı için yanlış — bir sonraki maçı tahmin ederken farkın büyüklüğü
bilgidir (MOV'lu Elo gibi). Ağırlıksız değer, eski skorla her ölçütte ayırt
edilemez ve seed'e daha az duyarlı. Karar: **model `game_impact_cache_v5.pkl`
= ağırlıksız değer** (`value_engine.py --all --no-leverage` →
`impact_v5.py`), **oyuncu OVR = ağırlıklı değer** (`output/value_cache_v1.pkl`).
Aynı fiyatlama, iki kullanım. Takım farkı ↔ gerçek marj: v4 0,80, v5 0,94.

## Veri

DB (telefon) kapalı; her şey yerel: `phonedb_cache/*.pkl` (12 tablo, 12 Eylül
çekimi), `game_impact_cache_v4.pkl`, `matchup_cache_v1.pkl`, `nba_data/`
(CSV). Yeniden kurmak için DB gerekmez.

| dosya | içerik |
|---|---|
| `output/engineered_dataset_pregame.pkl` | 10.749 × 316, üretim veri seti |
| `output/engineered_dataset_db.pkl` | eski (sahaya çıkan kadro) — aynı sütun adları, farklı anlam; modellerle karıştırma |
| `models/feature_list_2025_26.json` | üretimin 196 feature'ı, tarif, kadro türü |
| `models/backup_observed_roster_2026-10-03/` | eski modeller |
| `output/consistency_lab*.json` | iki tur deney |
| `output/market_check.json` | piyasa karşılaştırması |

## Çalıştırma

```
py prediction_engines/pregame_roster.py          # eski veri setinden mac-oncesi kadro, ~1 dk
py prediction_engines/retrain_production.py      # --feature-set base+out --recipe reg+mono (varsayilan), ~2 dk
py prediction_engines/simulation.py              # simulator, ayni veri seti
py prediction_engines/market_check.py            # 308 mac piyasa karsilastirmasi
py prediction_engines/consistency_lab.py         # deney harness'i (--arms, --all-margin, --base)
py app.py                                        # Tailscale IP'sine bind eder
```

Dashboard'un doğruluk rakamı `output/metrics_2025_26.json` →
`honest_walk_forward` bloğundan okunur (yeniden eğitim lab raporundan kopyalar).

## Sonraki adım adayları

1. **Maç öncesi sakatlık raporu.** `roster_missing_impact` bunun geriye dönük
   vekili; canlı sezonda gerçek rapor beslenebilir.
2. **Gelecek maç için feature üretimi.** Dashboard yalnızca veri setindeki
   maçları tahmin ediyor; oynanmamış bir maç için maç öncesi feature kuran yol
   yok. `pregame_roster.build_pregame_features` zaten tarih sırasıyla çalışıyor,
   "bugün itibarıyla" durumunu vermesi küçük bir ek.
3. **Çok sezonlu çizgi arşivi.** Piyasa karşılaştırması 308 maçla sınırlı.

## Bekleyenler

- Hiçbir şey commit edilmedi (dal `db-pipeline`): yeni modüller, veri seti,
  raporlar, model dosyaları.
- `odds_data/*.xlsx` içindeki model sütunları eski (sızıntılı) modelden.
- `DB_KURTARMA.md` eski host/şemayı anlatıyor.
