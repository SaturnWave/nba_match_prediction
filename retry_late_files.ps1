# NBA'in gec yayinladigi mac dosyalarini tamamlar.
#
# Savunma kutu skoru (defensive, defensive_team), oyuncu/takim takip
# (player_tracking, team_tracking) ve matchup dosyalari mac bittikten 1-2 gun
# sonra yayinlaniyor. Mac cekildiginde bu bes dosya bir sure eksik kalabiliyor;
# play-by-play ve traditional/advanced kutu skorlari ayni gun hazir oluyor.
#
# 9 Kasim 2026, ilk kosus: 0022600198 (8 Kasim) bes dosyasiyla geldi. Geriye
# 0022600189 kaldi - DEN@IND, 7 Kasim, Meksika'da oynanan mac. Onda endpointler
# bos govde donuyor (DefensiveV2/PlayerTrackV3 'NoneType has no attribute get',
# MatchupsV3 'list index out of range'), yani dosya henuz yok. Ayni mac
# LeagueGameFinder'in listesinde de yoktu; tarafsiz sahada oynanan maclarda
# NBA'in yayini duzensiz. Yayinlanmasi gecikmis olabilir, hic yayinlanmayabilir
# de: NBA'in hic vermedigi 23 macin listesi
# defensive_box_score_unavailable_games.txt - birkac gun deneyip gelmezse oraya.
#
# Eksik dosya yoksa API'ye hic gidilmez: script saniyeler icinde cikar.
# Dosyalar geldikten sonra veri seti yeniden kurulmali - bu script kurmaz,
# cunku 10 dakikalik bir is ve ne zaman yapilacagi insanin karari.
#
# Elle:           powershell -NoProfile -File retry_late_files.ps1
# Gorevi gor:     schtasks /Query /TN "NBA eksik dosya tamamlama"
# Gorevi kaldir:  schtasks /Delete /TN "NBA eksik dosya tamamlama" /F

$ErrorActionPreference = "Stop"

# Kendi bulundugu klasor. auto_push.ps1 repo yolunu sabit yazmis, repo tasinince
# aylarca sessizce hicbir sey yapmadi; sabit yol yazmamanin sebebi bu.
$repo = $PSScriptRoot
$season = "2026_2027"
$log = Join-Path $repo "retry_late_files.log"

function Write-Log($msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Add-Content -Path $log -Encoding utf8
}

Set-Location $repo

# -not ($dizi -match ...) ile yazilmasinin sebebi: PowerShell'de bir dizi uzerinde
# -notmatch bool degil, esleSmeyen satirlari dondurur; "eksik" gecmeyen baslik
# satiri yuzunden kosul her zaman dogru cikiyordu.
$before = & py tools/check_data.py $season
if (-not ($before -match "eksik")) {
    Write-Log "atlandi: $season tam, istenecek dosya yok"
    exit 0
}
foreach ($line in $before) { if ($line -match "eksik") { Write-Log "once:  $($line.Trim())" } }

# Her iki cekici de var olan dosyalari atlar, yani yalnizca eksikleri ister.
# Yonlendirme cmd uzerinden: PowerShell 5.1 yerel bir exe'nin stderr'ini kendi
# icinde yonlendirdiginde her satiri hata kaydina cevirip exit 0'i basarisiz
# gosteriyor.
foreach ($script in @("data_retriaval\backfill_defensive_tracking.py",
                      "data_retriaval\retrieve_matchups.py")) {
    Write-Log "calisiyor: $script $season"
    cmd /c "py $script $season >> `"$log`" 2>&1"
}

$after = & py tools/check_data.py $season
foreach ($line in $after) { Write-Log "sonra: $($line.Trim())" }
if (-not ($after -match "eksik")) {
    Write-Log "tamam: $season butun dosyalariyla tam. Veri seti yeniden kurulmali: py prediction_engines/build_dataset_asof.py"
} else {
    Write-Log "hala eksik: NBA henuz yayinlamamis. Ayni script yarin tekrar kosturulabilir."
}
