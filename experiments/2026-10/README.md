# Tek seferlik analiz betikleri — Ekim 2026

DURUM.md, README.md ve kod içi açıklamalardaki rakamların çoğu bu betiklerden
çıktı. Çalışma klasörünün dışında, geçici bir dizinde duruyorlardı; klasör
silinmeden önce kayıt olsun diye buraya alındılar. Düzenli bakımı yapılan
araçlar değiller: çoğu yazıldığı gün bir soruyu cevaplamak için yazıldı ve
öyle bırakıldı.

Hepsi proje kökünden çalıştırılır (`py experiments/2026-10/<betik>.py`) ve
yerel önbellekleri ister (`py tools/local_backup.py restore`).

## Sızıntı araştırması

| betik | sorduğu soru | bulduğu |
|---|---|---|
| `verify_shift.py` | Rolling, sezon ve seri feature'ları yalnızca önceki maçlardan mı geliyor? Ham skorlardan yeniden hesaplayıp karşılaştırır | 21.468 takım-maçın tamamında birebir; sızıntı yok |
| `probe_roster_leak.py` | Maçta sahaya çıkan oyuncu sayısı sonucu taşıyor mu? | Sayı ile \|fark\| korelasyonu 0,64; o günkü modellerde kadro ve matchup feature'ları önem payının üçte biri |
| `verify_pregame.py` | Maç öncesi forma listesinden kurulan kadro feature'ları maç bilgisi taşıyor mu? | Forma giyen sayısının \|fark\| ile korelasyonu −0,015 |
| `probe_features.py` | Feature listesi ve tek feature'ın tek başına AUC'si | En yükseği 0,69; 0,75'i geçen yok |
| `lab_paired.py` | `output/consistency_lab.json` içinden eşleştirilmiş farklar | DURUM.md'deki 1. tur tablosu |
| `wf_summary.py` | Eski walk-forward raporlarının ay bazında özeti | Ay etkisi tohum etkisinden çok büyük |

`probe_boxplayer.py`, `probe_minutes.py`, `probe_matchup_cache.py` ve
`probe_fit_time.py` şema ve süre yoklamaları: önbelleklerde hangi sütun var,
dakika nasıl yazılmış, bir LightGBM eğitimi kaç saniye.

## Impact motoru (fiyatlama 5.1)

| betik | sorduğu soru | bulduğu | ne ister |
|---|---|---|---|
| `diag_ast_ft.py` | 5.1 öncesi motor asistlerin ve serbest atışların ne kadarını yakalıyordu? Eski eşleştirme mantığı betiğin içinde yeniden yazılıdır | Asist %87–89, serbest atış %100 | — |
| `tune_assists.py` | Asist kredisi nasıl verilmeli? Kuralları sahadaki +/− ile korelasyona göre karşılaştırır | "Basketin %30'u, skorerin üstüne" 0,445; skorerden kesmek 0,430; pas başına kredi zarar veriyor | Argüman olarak bir klasör; içinde `value_engine.py --seasons X --refresh --raw --cache <klasör>/tune_X.pkl` ile üretilmiş üç sezon (2017_2018, 2020_2021, 2023_2024). Serbest atış karşılaştırması `output/value_cache_v1.pkl` dosyasını "eski" sayar; o gün öyleydi, bugün değil |
| `check_final.py` | Seçilen sabitlerle üç ayar sezonunda önce ve sonra | 0,421 → 0,428, 0,482 → 0,494, 0,388 → 0,420 | Aynı klasör |
| `confirm_seasons.py` | Ayar yapılmayan altı sezonda da tutuyor mu? | 0,379 → 0,396, altısının beşinde daha iyi | Klasörde `value_cache_v1_old.pkl`: `git show 772e4c9da:output/value_cache_v1.pkl` |

## Sezon simülatörü

| betik | sorduğu soru | bulduğu |
|---|---|---|
| `probe_pbp_shapes.py` | Gerçek play-by-play satırları nasıl görünüyor ve lig oranları ne? | `season_sim.py` başındaki sabitler: asistli ikilik %54,1, üçlük %85,5, kaçan ikiliğin %20,1'i blok, vb. |
| `sweep_knobs.py` | Tempo, ev sahibi avantajı, skor etkisi hangi değerde gerçek sezonu tutturuyor? Altı tohum | Dosyadaki son iki ayar; seçilenler `season_sim.py` içinde |
| `sweep_carried.py` | Takım etkilerinin ne kadarı sonraki sezona taşınmalı? Beş sezon | Hiç taşımamak 0,36 / 9,7; yarısı ve tamamı 0,51 civarı |
| `backtest_trust.py` | Sezon öncesi bilgiyle galibiyet tahmini ne kadar tutuyor? | "Geçen sezonun aynısı" ile aynı seviyede |
| `calib_ranges.py` | %10–%90 galibiyet aralığı gerçeği ne sıklıkla kapsıyor? | Takım sapması 0,10'da %77; 0,11 seçildi |

`sweep_knobs.py`, `sweep_carried.py` ve `backtest_trust.py` simülatörün ilk
çalıştırıcısına göre yazılmıştı; bugünkü `prepare_season` / `simulate_season`
arayüzüne bağlandılar, başka bir şeyleri değişmedi. Ayar listeleri son
çalıştırıldıkları haliyle duruyor, ilk denenen değerler değil.
