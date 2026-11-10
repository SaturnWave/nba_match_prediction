# Durum — db-pipeline dalı

Son güncelleme: 2026-10-27 (maç öncesi veri hattı: tablolar telefon kopyası yerine CSV'den, oynanmamış maçlar veri setine giriyor, 2026-27'nin ilk tahminleri; öncesinde 2026-27 fikstürü ve ilk maçları, CSV'lerin D: diskine taşınması, graphify, sezon simülatörü, güç sıralaması, değer skoru, fiyatlama 5.1). Bu dosya, çalışmaya ara verildiğinde nerede
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

Dashboard'da: "Sim Sezon" sekmesi (mor SİM etiketi, açılır uyarı bandı: gerçek
sonuç değil, tahmin gücü sınırlı). Dört alt görünüm: **Puan Durumu** (lig / doğu /
batı, ev-deplasman, son 10, seri, 30 tekrarın ortalaması ve %10–%90 aralığı),
**Lig Liderleri** (12 kategori: sayı, ribaunt, asist, top çalma, blok, üçlük,
şut %, üçlük %, serbest atış %, TS%, değer, dakika; ligin kendi barajı ölçekli:
80 maçta en az 57 maç, yüzdelerde isabet barajı; takım seçilince takım liderleri),
**Maçlar** (kutu skoru + çeyrek çeyrek play-by-play), **Güç Sıralaması** (gerçek
sezonlarla aynı motor ve ekran, sim sezonun haftaları/ayları). Rotalar
`/api/sim/*`, okuyucu `season_sim.SimSeason`, hangi koşunun gösterileceği
`app.SIM_RUN`. Yalnızca okur; gerçek veriyle hiçbir yerde karışmaz. Sayfanın
betiği tarayıcısız bir DOM taklidiyle canlı sunucuya karşı çalıştırıldı (16
kontrol); gerçek tarayıcıda görsel kontrol yapılmadı.

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

## Yeni klonda kurulum

2026-10-05'te yerel çalışma klasörü silinmeden önce, yalnızca yerelde duran
her şey depoya alındı. Bir klondan sonra yapılacak tek şey:

```
py tools/local_backup.py restore
```

Bu, git'in yok saydığı önbellekleri yerlerine açar. Onlar depoda sıkıştırılmış
halde `local_backup/` altında durur (37 dosya, 791 MB → 53 MB, her birinin
boyutu ve özeti `MANIFEST.json` içinde): `phonedb_cache/` (telefon
veritabanının 12 tablosu, 12 Eylül 2026 çekimi), `matchup_cache_v1.pkl`
(398 MB), `team_box_cache_v1.pkl`, `defensive_tracking_cache_v1.pkl`,
`graphify-out/` ve koşu günlükleri. Ham hallerinin depoya girmemesinin
nedeni boyut: ikisi (398 MB ve 106 MB) GitHub'ın 100 MB'lık dosya sınırının
üstünde. Boş bir klasöre geri yükleme denendi; 37 dosyanın 37'si aslıyla
birebir aynı çıktı.

Yedekte olmayan tek önbellek `graphify-out/cache/` (bilgi grafiğinin AST
önbelleği): `/graphify` onu bedavaya yeniden kurar, ve 64 karakterlik dosya
adları depodaki en uzun yolu 81'den 116 karaktere çıkarıp derin bir klasörde
Windows'ta checkout'u bozuyordu.

Doğrudan depoda duranlar (geri yükleme gerekmez): `nba_data/` (107.421 ham
CSV), modeller, veri setleri, `game_impact_cache_v4/v5.pkl`, iki değer
önbelleği (`output/value_cache_v1.pkl` oyuncu, `value_cache_unweighted.pkl`
model), simüle sezon (`sim_sandbox/forecast_2026_2027/`), `odds_data/`.

**Depoda bilerek olmayan tek şey `.env`**: telefon veritabanının şifresini
taşır ve depo herkese açıktır. `.env.example` gereken beş anahtarı listeler.
Veritabanı olmadan da her şey çalışır; okuyucular önce `phonedb_cache/`
önbelleğine bakar. Aynı nedenle `local_backup.py pack`, içinde `.env`
değerlerinden biri geçen dosyayı paketlemeyi reddeder. İlk denemede bunu
yakaladı: `phonedb_cache/manifest.json` her tablonun çekildiği veritabanı
adresini taşıyordu. O alanı okuyan kod yoktu; yedekten önce silindi ve
`phonedb_source.py` artık yazmıyor.

`experiments/2026-10/` rakamların çıktığı tek seferlik analiz betiklerini
tutar (kendi README'si var). Claude'un proje notları (hafıza) bu klasörde
değil, kullanıcı profilinde durur; proje aynı yola klonlanırsa bulunur.

## Bu makinede: veri D: diskinde

2026-10-24'te C: diskinde 5 GB yer kalmıştı. `nba_data/` (107.421 CSV,
1,48 GB) ve `phonedb_cache/` (377 MB) `D:\nba_match_prediction_data\`
altına taşındı; depodaki iki klasörün yerinde birer junction var. Git ve kod
onları sıradan klasör gibi görür: git status temiz, yeni dosyalar normal
commit'lenir, gerçek yolu çözen kod yok. C: kopyası, taşınan her dosya
junction üzerinden okunup git'teki kaydıyla (`phonedb_cache` için yedeğin
özetleriyle) birebir eşleştikten sonra silindi. Bu yerel bir düzenleme;
başka bir klon normal klasörlerle gelir.

D: çok sayıda küçük dosyada yavaş: kopyalama 26 dakika, ardından her dosyayı
bir kez okuyan içerik doğrulaması 66 dakika sürdü. Büyük önbellekler
(`matchup_cache_v1.pkl` vb.) C:'de kaldı. Dosya sembolik bağlantısı bu
makinede yönetici izni istediği için onlar ancak kodda yapılandırılabilir
bir yolla taşınabilir. Günlük iş önbellekten yürüdüğü için etkilenmez.
Bütün CSV'leri yeniden okuyan işler (önbellek yeniden kurma, tam denetim)
uzun sürer.

## Veri

DB (telefon) kapalı; her şey yerel: `phonedb_cache/*.pkl` (12 tablo, 12 Eylül
çekimi), `game_impact_cache_v4.pkl`, `matchup_cache_v1.pkl`, `nba_data/`
(CSV). Yeniden kurmak için DB gerekmez.

2026-27 (27 Ekim itibarıyla): fikstür `game_ids/schedule_2026_2027.csv`
(1.274 maç: 67 hazırlık, 1.206 normal sezon, NBA Cup finali; Cup eleme
turundaki 7 maçın takımları henüz belli değil), oynanan maçlar
`game_ids/game_id_2026_2027.csv` (20–25 Ekim, 43 maç; fikstürdeki biten 43
maçla birebir aynı). 43 maçın hepsinde on dosyanın onu da var: play-by-play,
geleneksel, ileri ve defansif kutu, tracking, matchup. NBA tracking ve
matchup'ı maçtan bir iki gün sonra yayımlıyor; komutlar kaldığı yerden devam
eder. `phonedb_cache` (12 Eylül çekimi) bu sezonu içermez ve artık hiçbir
hesap onu okumuyor (aşağıda, maç öncesi veri hattı). Telefon veritabanına şu
an erişilmiyor (Tailscale'de çevrimdışı, bu klonda `.env` de yok).

Geçmiş sezonlarda 23 maçta defansif kutu skoru ve matchup verisi kaynakta yok:
24 Ekim'de yeniden denendi, NBA hâlâ boş döndürüyor, telefon önbelleğinde de
yoklar. Liste `defensive_box_score_unavailable_games.txt` içinde. 2024-25'in
tek eksiği (0022400856) o gün tamamlandı. Geri kalan her şey (play-by-play,
geleneksel ve ileri kutu, tracking) dokuz sezonun tüm maçlarında var
(`py tools/check_data.py`).

| dosya | içerik |
|---|---|
| `output/engineered_dataset_pregame.pkl` | 10.749 × 316, üretim veri seti |
| `output/engineered_dataset_db.pkl` | eski (sahaya çıkan kadro) — aynı sütun adları, farklı anlam; modellerle karıştırma |
| `models/feature_list_2025_26.json` | üretimin 196 feature'ı, tarif, kadro türü |
| `models/backup_observed_roster_2026-10-03/` | eski modeller |
| `output/consistency_lab*.json` | iki tur deney |
| `output/market_check.json` | piyasa karşılaştırması |

## Simülasyon ve gerçek: 2026-27'nin ilk 28 maçı (`prediction_engines/sim_vs_real.py`)

Kayıtlı tahmin koşusu (`sim_sandbox/forecast_2026_2027/`) her maç için tek bir
çekiliş tutar. Bu yüzden oynanan maçlar aynı kurulumdan 1.000 kez yeniden
oynatıldı. Kurulum aynı mı diye bakıldı: koşunun kendi tohumu 1.200 maçın
1.200'ünün skorunu birebir geri verdi. 20–23 Ekim, 28 maç:

| | isabet | Brier | log loss |
|---|---|---|---|
| simülasyon (1.000 tekrar) | 18/28 | 0,225 | 0,642 |
| geçen sezonun dereceleri (log5 + ev sahibi) | 16/28 | 0,218 | 0,622 |
| her zaman ev sahibi | 13/28 | 0,257 | 0,707 |

Fark MAE simülasyon 13,6 / naif 15,0; toplam MAE 13,9 / 14,6. Gerçek farkın
simülasyonun %10–%90 aralığında kalma oranı %82 (hedef %80). 28 maçta bu
farkların hepsi gürültü içinde. Sezon öncesi gücü "geçen sezonun aynısı"
seviyesinde, geriye dönük testteki gibi.

Lig düzeyinde simülasyon 2025-26'nın sezon ortalamasını kopyalıyor (takım-maç
başına 116,1 sayı, 22,6 serbest atış, 13,8 top kaybı, üçlük %36,5). İlk hafta
farklı: 113,1 sayı, 27,0 serbest atış, 24,0 faul, 16,4 top kaybı, üçlük %32,9.
Serbest atış ve faul fazlası tamamen açılış haftası etkisi. 2025-26'nın tarihe
göre ilk 28 maçı 27,7 serbest atış, 24,0 faul ve 15,6 top kaybıydı; her ekim
böyle olup ay ay sönüyor, simülatöre bir "sezonun ayı" çarpanı gerekir. Bu
sezona özgü görünen soğuk üçlükler (geçen açılışta %36,1) ve 6 sayı eksik;
28 maçta gürültü de olabilir. (İlk yazımda geçen sezonun "ilk 28 maçı" maç
numarası sırasıyla alınmıştı, bu Ekim–Ocak'a yayılıyordu; düzeltildi.)

Oyuncu düzeyinde dakika korelasyonu 0,67. Kadrolar neredeyse doğru: 13.440
gerçek dakikanın yalnızca 1'i simülasyonun kadrosunda olmayan bir oyuncuya
(Walter Clayton Jr.) gitti. En büyük hata çaylaklarda. NBA geçmişi olmayan oyuncu yedek seviyesinde dakika
alıyor, bu yüzden Boozer, Acuff, Lendeborg, Wilson, Peterson ve Wagler
simülasyonda iki maçta toplam 1–5 dakika oynadı, gerçekte maç başına 26–36.
Sakatlar (Ingram, Knueppel, DiVincenzo, Claxton, Raynaud) simülasyonda
oynuyordu.

## Bahis oranları: modele ne katıyorlar? (`prediction_engines/market_lab.py`)

`odds_data/odds_history.csv` dokuz sezonun 10.777 normal sezon maçının açılış
ve kapanış çizgisini tutar: ev sahibinin handikabı, toplam ve marjı ayıklanmış
ev sahibi olasılığı. 2017-18'den 16 Ocak 2023'e kadar Sportsbook Reviews
arşivinden (robots.txt izin veriyor), sonrası ESPN'in oran servisinden
geliyor (kitapların medyanı). ESPN'in koşulları otomatik erişimi yasaklıyor;
kullanıcının kararıyla, saniyede bir istekle çekildi. İki kaynağın eşleşen
maçlarındaki skorların %100'ü projeninkiyle aynı. Örtüştükleri 118 maçta
kapanış handikabı ortalama 0,10, toplam 0,22 puan farklı. Eylül'de toplanan
308 DraftKings maçıyla birebir aynı.

Üretim tarifinin seçildiği walk-forward'da (Kasım 2023 – Nisan 2026, 18 ay ×
3 tohum, yaklaşık 3.200 maç):

| | isabet | Brier | log loss | fark MAE |
|---|---|---|---|---|
| model (üretim tarifi) | %67,0 (blend %68,9) | 0,212 (blend 0,204) | 0,613 | 11,23 |
| model + oranlar (ağaca feature) | %64,8 (blend %70,2) | 0,219 (blend 0,199) | 0,627 | 10,87 |
| model ve piyasa, öğrenilmiş ağırlıkla | %70,1 | 0,194 | 0,570 | – |
| yalnızca piyasa | %70,2 | 0,194 | 0,569 | 10,72 |

Oranlar tahmini belirgin iyileştiriyor: isabet +3 puan, Brier −0,018 ± 0,002.
Ama işi yapan piyasa. Birleşim piyasanın kendisiyle aynı çıkıyor ve modele
verdiği ağırlık ortalama −0,29. Bu ağırlık kesin değil: düzenlileştirmesiz,
birbirine çok bağlı iki girdiyle uyduruluyor; hücrelerin %80'i −0,9 ile +0,5
arasında, tek tek hücreler −3,2 ile +0,6, aylık ortalamalar −2,1 ile +0,5
arasında. Yine de hiçbir hücre, piyasanın bilmediği bir bilgi bulmadı.
Birleşim ve aşağıdaki ayrışma sayımı sevk edilen blend'i değil
sınıflandırıcıyı kullanıyor. Sınıflandırıcı ile piyasa maçların %15,6'sında
farklı kazanan seçti ve o maçlarda piyasa %60 haklı çıktı (üç tohumun
toplamında 904'e 601, tohum başına yaklaşık 500 maç).
Oranları ağaç modeline doğrudan feature olarak vermek sınıflandırıcıyı
kötüleştirdi; doğru yol birleşim. Kapanış çizgisi ancak maç başlarken belli
olur, bu sonuçlar başlama anındaki bir tahmini anlatır. Bir gün önceki tahmin
için açılış çizgileriyle ayrı bir test gerekir. Oranlar yalnızca bilgi ve ölçü
olarak kullanılıyor; bahis tavsiyesi yok.

## Maç öncesi veri hattı ve ilk 2026-27 tahminleri (`prediction_engines/build_dataset_asof.py`)

Denetim raporunun ilk adımı. Üretim veri seti 2025-26'da bitiyordu. Zincirin
iki halkası (değer motoru, kadro feature'ları) telefon veritabanının 12 Eylül
kopyasını okuyordu. O kopyada 2026-27 maçı yok, veritabanı da kapalı. Bu
yüzden modeller bu sezonun tek maçını tahmin edemiyordu. Şimdi tek komut,
oynanan bütün maçları ve sıradaki günün maçlarını `nba_data/`'dan kuruyor.
Bunun için:

- **Tablolar CSV'den** (`csv_tables.py`). Maç listesi, oyuncu kutu skoru ve
  tracking artık `nba_data/` CSV'lerinden kurulan tablolardan okunuyor. Bu
  tabloları okuyanlar: değer motoru, impact v5, kadro feature'ları, güç
  sıralaması ve dashboard. Tablolar telefon kopyasıyla karşılaştırıldı
  (`csv_tables.py --check`): 10.749 maçta ve 277.127 oyuncu satırında birebir
  aynı. Önbellek `table_cache/csv_tables.pkl` (340 MB, git dışı). İlk kurulum
  bu makinede 38 dakika sürdü (D: yavaş). Sonrakiler yalnızca yeni ya da
  dosyası değişen maçları okuyor; bugün 27 maç, 2,5 dakika. Telefon kopyasını
  okuyan, günlük hatta olmayan betikler kaldı. Sezon simülatörü de onlardan
  biri; kayıtlı koşu tekrarlanabilsin diye bilerek değiştirilmedi.
- **Sözleşmeler.** Tablolardan biri bozuksa kurulum durur:
  - eksik kimlik,
  - iki kez listelenen oyuncu,
  - iki takımı olmayan maç,
  - skoru olmayan maç,
  - oyuncu sayılarının toplamı takım sayısını tutmayan takım-maç.
- **Artımlı değer ve impact.** Değer motoru yalnızca önbellekte olmayan
  maçları hesaplıyor (bugün 43 maç, 19 saniye). Tracking'i sonradan yayımlanan
  maç, tracking gelince yeniden hesaplanıyor. Hangi maçın tracking'le
  hesaplandığı `output/value_cache_unweighted.pkl.tracked.pkl` içinde.
  `game_impact_cache_v5.pkl` her koşuda yeniden yazılıyor (şu an 10.792 maç).
- **Oynanmamış maçlar.** Fikstürde olup `nba_data/`'da olmayan normal sezon
  maçları, en erken tarihten başlayarak. Bir takımın kadro listesi, son
  maçındaki listesi olarak taşınıyor; dakikalar 0. Sakatlık raporu
  arşivlenene kadar böyle.
- **Sert duruşlar.** Kurulum şu durumlarda hata verip duruyor:
  - Skoru olmayan oynanmış maç. Eskiden 0-0 ve ev sahibi yenilgisi olarak
    giriyordu.
  - Kimsenin sınıflamadığı bir kutu skoru yorumu. Yorumlar artık kalıpla
    değil, tam listeyle sınıflanıyor. Eski kalıp 2026-27'de gelen "Inactive -
    G League - Two-Way" ve "Inactive - Coach's Decision" satırlarını forma
    giymiş sayıyordu.
  - Kadro listesi taşınamayan takım.
  - Bitmiş ama `nba_data/`'da olmayan maç: fikstürde final görünüyor ya da
    başlama saatinin üstünden 4 saat geçmiş. O durumda önce veri yenilenmeli.
- **Eşitlik testi** (`--check`, sonucu `output/asof_parity.json`). Yeniden
  kurulan veri seti, üretim veri setinin 10.749 maçında 196 feature'ın ve 5
  etiketin hepsini 1e-9 içinde aynen veriyor. Tek istisna kayıtlı:
  - 0022400856'nın (28 Şubat 2025, NYK@MEM) matchup dosyası 24 Ekim 20:34'te
    indi. Üretim veri seti o gün 12:49'da kurulmuştu.
  - Yeni hat bu dosyayı kullanıyor. Bu yüzden o maçtan sonraki 432 maçta 6
    matchup sütunu 0,04'e kadar farklı.
  - Test, o maçın matchup'larını dışarıda bırakıp kadro taramasını yeniden
    yapıyor. O zaman fark sıfır.
  - Liste `LATE_MATCHUPS` içinde. Üretim bu hattan yeniden kurulunca
    boşaltılmalı.
- **Matchup önbelleği** (`matchup_impact`). Anahtarlar artık göreli yol.
  Başka klasöre klonlanan depo 10.749 dosyayı yeniden okumuyor.
- **`requirements.txt`.** Sürümler sabitlendi (pandas 3.0.2, LightGBM 4.6.0
  …); flask, waitress ve xgboost eklendi.

İlk tahminler `prediction_engines/forecast_upcoming.py` ile üretildi
(`output/forecasts/`): 26 Ekim'in 9 maçı. Dosya, tahminin üretildiği anı ve
her maçın başlama saatini UTC olarak tutuyor:

- Bu ilk dosya ABD doğu saatiyle 19:58'de üretildi, yani 19:00 ve 19:30'da
  başlayan 4 maçtan sonra.
- O 4 maç `before_tipoff = False` olarak işaretli. Tahmin onlardan hiçbir
  bilgi kullanmıyor, ama maç öncesi tahmin de sayılmaz.
- Her koşu ayrı bir dosyaya yazıyor. Sonraki bir koşu, maç öncesi kaydın
  üstüne yazmaz.

Aynı modeller sezonun oynanan 43 maçında: isabet 0,744, Brier 0,199, marj
MAE 11,7. Bu satırlar eğitim satırları gibi kuruldu, her takımın listesi o
maçın kutu skorundan. Yani bunlar geri test koşulları, ve 43 maçta gürültü
geniş.

Sınırlar:

- Taşınan liste, son maçtan sonra sakatlanan oyuncuyu hâlâ sağlam sayıyor.
  Dönen oyuncuyu da hâlâ yok sayıyor. İlacı sakatlık raporu arşivi.
  `pregame_roster.carried_forward_listings` fonksiyonunun
  `unavailable_overrides` parametresi o raporun takılacağı yer.
- `--days 3` gibi birden çok tarih, her tarihi sıradaki tarih gibi kuruyor.
  Aradaki maçlar henüz bilinmediği için ikinci ve üçüncü günün form
  feature'ları bir maç geride kalıyor. Dinlenme günü de ilk günün maçını
  görmüyor. Üretim modelleri dinlenme feature'ı kullanmadığı için bu tahmini
  değiştirmiyor.
- Modeller değişmedi: 2025-26 sonuna kadarki veriyle eğitilmiş halleriyle
  tahmin ediyorlar.

## Çalıştırma

```
py prediction_engines/pregame_roster.py          # eski veri setinden mac-oncesi kadro, ~1 dk
py prediction_engines/retrain_production.py      # --feature-set base+out --recipe reg+mono (varsayilan), ~2 dk
py prediction_engines/simulation.py              # simulator, ayni veri seti
py prediction_engines/market_check.py            # 308 mac piyasa karsilastirmasi
py prediction_engines/consistency_lab.py         # deney harness'i (--arms, --all-margin, --base)
py app.py                                        # Tailscale IP'sine bind eder
```

Süren sezonun verisini güncellemek (2026-27; her komut kaldığı yerden devam eder):

```
py data_retriaval/fetch_schedule.py 2026-27                      # fikstur ve durumlar
py data_retriaval/fetch_season_game_ids.py 2026-27 --refresh     # oynanan maclar
py data_retriaval/retrieve_2025_2026_data.py --season 2026_2027  # pbp + geleneksel + ileri kutu
py data_retriaval/backfill_defensive_tracking.py 2026_2027       # defansif kutu + tracking
py data_retriaval/retrieve_matchups.py 2026_2027                 # matchup'lar
py tools/check_data.py 2026_2027                                 # hangi macta hangi dosya eksik
py prediction_engines/build_dataset_asof.py                      # oynanan maclar + siradaki gunun maclari
py prediction_engines/backfill_impact_from_db.py --source csv --seasons 2026_2027   # panonun kadro/siralama verisi
py prediction_engines/value_engine.py --seasons 2026_2027        # oyuncu OVR'inin deger tarafi
py prediction_engines/forecast_upcoming.py                       # o maclarin tahmini, output/forecasts/
py prediction_engines/forecast_vs_market.py                      # tahminler ve piyasa cizgisi yan yana
py prediction_engines/forecast_scorecard.py                      # tip-off oncesi tahminlerin gercek karnesi
py prediction_engines/sim_vs_real.py                             # simulasyon ile oynanan maclar, ~1 dk
py data_retriaval/odds_run_all.py                                # bahis cizgileri: arsiv + ESPN (kaldigi yerden)
py prediction_engines/market_lab.py                              # oranlar modele ne katiyor
py prediction_engines/confidence_buckets.py                      # guven esigi ne satin aliyor (4 rejim)
py prediction_engines/confidence_buckets.py --list --min 0.60    # esigi gecen maclar tek tek + CSV
```

Pano **takvimle** açılıyor (`/api/calendar`): ayın ızgarası, her gün maçları ve
modelin o gün tuttuğu/tutmadığı. Günler fikstür dosyasından, tahminler veri
setinden geliyor; ikisi ayrı olduğu için gelecek maçlar "tahmin bekliyor" diye
görünür. Daha ileriye tahmin istiyorsan `build_dataset_asof.py --days 3` (üç maç
günü) ile kur. Veri seti günde bir kurulduğu için dün akşamın maçları bir süre
veri setinde fikstür, fikstür dosyasında final olarak durur: pano sonucu
fikstürden okuyup o tahminleri **"tip-off öncesi"** rozetiyle not ediyor — panodaki
tek doğrulanmış tahmin cinsi odur.

İki önbellek bu listede olmasaydı sessizce geride kalırdı: `game_impact_cache_v4.pkl`
(panonun kadro paneli ve güç sıralaması) ve `output/value_cache_v1.pkl` (oyuncu
OVR'ının değer tarafı). `build_dataset_asof.py` yalnızca modelin okuduğu
`value_cache_unweighted.pkl` ile `game_impact_cache_v5.pkl`'i yazar; bu ikisine
hiç dokunmaz. Geride kaldıklarında pano hata vermez, yalnızca daha eski bir ligi
anlatır — o yüzden pano artık ikisinin de kapsamını `/api/meta`'da bildiriyor ve
geride kalınca sayfanın başına uyarı basıyor.

Dashboard'un doğruluk rakamı `output/metrics_2025_26.json` →
`honest_walk_forward` bloğundan okunur (yeniden eğitim lab raporundan kopyalar).

## Sonraki adım adayları

1. **Maç öncesi sakatlık raporu.** `roster_missing_impact` bunun geriye dönük
   vekili; canlı sezonda gerçek rapor beslenebilir.
2. **Dashboard'da oynanmamış maçlar.** Tahminler artık üretiliyor
   (`forecast_upcoming.py`), ama dashboard hâlâ yalnızca üretim veri setindeki
   maçları gösteriyor.

## Bekleyenler

- 2026-10-26'ya kadarki çalışma `Master` ve `db-pipeline` dallarında GitHub'da.
- Derin araştırma raporu: `reports/NBA prediction system audit 2026.md`
  (İngilizce). Notları ve denetim betikleri
  `research_notes/NBA prediction system audit 2026/` altında. Önerdiği sıra:
  1. maç öncesi ("as-of") özellik hattı ve sakatlık raporu arşivi,
  2. küçük kazançları görebilecek bir değerlendirme,
  3. oynama olasılığı × oyuncu reytingi × dakikadan takım gücü,
  4. kalibrasyon.
- Oranlarla ilgili iki adım bekliyor:
  - **Açılış çizgisi testi** (bir gün önceki tahmin). Ancak kullanılabilirlik,
    çizginin açıldığı andaki sakatlık raporundan alınırsa anlamlı. Aksi halde
    maç sonrası kadro bilgisi, çizgiyi oynatan haberleri "önceden bilip"
    modeli iyi gösterir.
  - **Model ile piyasanın ayrıldığı maçların incelenmesi** (tohum başına
    yaklaşık 500). Geri testte model maç sonrası kadroyu zaten gördüğü için
    bu ayrılık "görülmeyen sakatlık haberi" değil. Model, yokları, dakikaları,
    dinlenmeyi ve yeni oyuncuları piyasadan farklı değerlendiriyor.
- Üretim veri setinin bu hattan yeniden kurulması ve modellerin yeniden
  eğitilmesi. 2026-27 maçları eğitime girer, `LATE_MATCHUPS` boşalır. Dürüst
  rakamlar walk-forward'la yeniden ölçülmeli.
- Git'in takip ettiği iki önbellek her koşuda değişiyor:
  `game_impact_cache_v5.pkl` (11 MB) ve `output/value_cache_unweighted.pkl`
  (30 MB). Her gün commit'lemek depoyu şişirir; git dışına alınmaları
  düşünülmeli.
- Bu klonda `.env` yok; telefon veritabanına bağlanmak için `.env.example`
  kopyalanıp doldurulmalı.
- Simülatörde çaylaklar yedek seviyesinde dakika alıyor. Draft sırası ya da
  hazırlık maçı dakikası ilk tahmin olarak kullanılabilir. Sakatlık listesi de
  simülasyona girmiyor.
- `odds_data/*.xlsx` içindeki "Model vs Piyasa" sayfaları eski (sızıntılı)
  modelden; oradaki handikap üstünlüğü sızıntının ürünüydü. Güncel
  karşılaştırma `output/market_check.json`. Excel yeniden üretilmedi.
- `models/backup_observed_roster_2026-10-03/` (eski modeller) depoda ayrı bir
  klasör olarak yok; aynı dosyalar git geçmişinde `ace99a7cf` commit'inde.
- `DB_KURTARMA.md` eski host/şemayı anlatıyor.
