# Durum — db-pipeline dalı

Son güncelleme: 2026-10-03 (güç sıralaması eklendi). Bu dosya, çalışmaya ara verildiğinde nerede
kalındığını ve nasıl devam edileceğini anlatır.

## Tek cümlelik özet

Modelin iki sızıntısı bulundu ve kapatıldı (kadro feature'ları maçta sahaya
çıkanlardan hesaplanıyordu; üretim modelleri test setiyle erken duruyordu);
dürüst feature'lar üzerinde düzenlileştirme + monoton kısıt olasılık kalitesini
ve iç tutarlılığı artırdı, isabeti değiştirmedi; üretim bu tarifle yeniden
kuruldu.

## Dürüst rakamlar

Bağlayıcı olan walk-forward (18 ay × 3 seed = 54 hücre, 2023-11 → 2026-04,
burn-in ≥ 10 maç, `output/consistency_lab_round2.json`, kol `reg+mono+out`):

| ölçüm | değer |
|---|---|
| Kazanan isabeti (sınıflandırıcı) | 0,673 (ay sapması 0,061) |
| Kazanan isabeti (blend, dashboard'un gösterdiği) | 0,685 |
| AUC | 0,743 |
| Brier | 0,210 (blend 0,204) |
| Sayı farkı MAE | 11,2 |
| Kazanan–marj çelişkisi | %8,9 (eski tarifte %13,0) |
| Naif taban | 0,555 |

Tek split (son 308 maç, 2026-03-04 sonrası, hiçbir fit'e girmedi):
blend isabet 0,786, AUC 0,837, Brier 0,165; marj MAE 11,37; toplam MAE 15,23.
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
| kazanan isabeti | 0,786 | 0,776 |
| Brier | 0,165 | 0,153 |
| sayı farkı MAE | 11,37 | 10,58 |
| toplam MAE | 15,23 | 14,33 |
| handikapta modelin tarafı | %45,8 ± 2,8 | başabaş %52,4 |

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
