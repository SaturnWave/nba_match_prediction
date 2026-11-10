# Gunluk veri guncellemesi: dun oynanan maclari ceker, veri setini yeniden kurar,
# onbellekleri tazeler ve siradaki mac gunlerinin tahminini yazar.
#
# NEDEN VAR
#   Bu zincir DURUM.md'de yaziliydi ve elle kosuluyordu. Elle kosulmadigi her gun
#   pano bir gun once dunyayi anlatiyor: veri seti dun aksamin maclarini hala
#   "fikstur" sayiyor, guc siralamasi ve kadro paneli eski, bugunun maclarinda
#   hic tahmin olmuyor. Ağırlıkların eskimesi degil, veri setinin kurulmamasi
#   panonun gercek bayatlama sebebi.
#
# NE YAPAR (sirayla)
#   1-5  NBA'den: fikstur, oynanan mac listesi, play-by-play + kutu skorlari,
#        savunma/takip kutulari, matchup'lar. Hepsi kaldigi yerden devam eder;
#        var olan dosyayi yeniden indirmez.
#     6  Eksik dosya raporu (yalnizca gunluge yazilir).
#     7  Veri setini yeniden kurar - KRITIK adim, basarisiz olursa script durur.
#   8-9  Panonun kadro/siralama onbellekleri (impact v4, value v1). Bunlari
#        build_dataset_asof.py yazmaz; yazilmazsa pano sessizce eski ligi anlatir.
#    10  Oynanmamis maclarin tahmini -> output/forecasts/
#    11  Bahis cizgisi arsivi (olcut olarak; tahmin degil)
#    12  Tahmin ve cizgi yan yana
#    13  Tip-off oncesi yazilmis tahminlerin karnesi
#
# NE YAPMAZ
#   Commit ve push yapmaz (-Commit ile acilir). Panoyu yeniden baslatmaz
#   (-RestartDashboard ile acilir) - pano acilista veriyi bir kez okur, yani
#   yeniden baslatilmadan yeni veriyi gormez.
#
# Elle:          powershell -NoProfile -ExecutionPolicy Bypass -File daily_update.ps1
# Hizli deneme:  ... -File daily_update.ps1 -NoFetch        (API'ye gitmez)
# Gorevi gor:    schtasks /Query /TN "NBA gunluk guncelleme"
# Gorevi kaldir: schtasks /Delete /TN "NBA gunluk guncelleme" /F

[CmdletBinding()]
param(
    # Bos birakilirsa game_ids/schedule_*.csv dosyalarinin en yenisinden okunur.
    # Sabit yazmamanin sebebi: auto_push.ps1 repo yolunu sabit yazmisti ve repo
    # tasininca gorev aylarca sessizce hicbir sey yapmadi.
    [string]$Season = "",
    # Kac maç gunu ileriye tahmin uretilsin. 2 = bugun + yarin.
    # DIKKAT: ikinci gunun form feature'lari bir mac geride kalir, cunku aradaki
    # maclar henuz oynanmamistir (DURUM.md, "Sinirlar").
    [int]$Days = 2,
    [switch]$NoFetch,
    [switch]$SkipCaches,
    [switch]$Commit,
    [switch]$RestartDashboard
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
Set-Location $repo

$log = Join-Path $repo "daily_update.log"
$lock = Join-Path $repo "daily_update.lock"

function Write-Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg"
    $line | Add-Content -Path $log -Encoding utf8
    Write-Host $line
}

# ---------------------------------------------------------------------------
# Ayni anda iki kopya calismasin: veri seti yazimi atomik ama iki cekici ayni
# dosyalara yazarsa yarim dosya kalir. 3 saatten eski kilit, cokmus bir kosudan
# arta kalmistir; onu yok sayar.
# ---------------------------------------------------------------------------
if (Test-Path $lock) {
    $age = (Get-Date) - (Get-Item $lock).LastWriteTime
    if ($age.TotalHours -lt 3) {
        Write-Log "atlandi: $([int]$age.TotalMinutes) dk once baslamis bir kosu hala calisiyor ($lock)"
        exit 0
    }
    Write-Log "uyari: $([math]::Round($age.TotalHours,1)) saatlik eski kilit bulundu, yok sayiliyor"
}
Get-Date -Format "o" | Set-Content -Path $lock -Encoding utf8

try {
    # -----------------------------------------------------------------------
    # Sezon: verilmediyse en yeni fikstur dosyasindan. "2026_2027" ve "2026-27"
    # biciminin ikisi de gerekiyor - cekiciler birini, isleyiciler otekini ister.
    # -----------------------------------------------------------------------
    if (-not $Season) {
        $newest = Get-ChildItem (Join-Path $repo "game_ids") -Filter "schedule_*.csv" -ErrorAction SilentlyContinue |
                  Sort-Object Name | Select-Object -Last 1
        if (-not $newest) {
            Write-Log "HATA: game_ids/schedule_*.csv yok ve -Season verilmedi"
            exit 1
        }
        $Season = $newest.BaseName -replace "^schedule_", ""
    }
    if ($Season -notmatch "^\d{4}_\d{4}$") {
        Write-Log "HATA: sezon bicimi 2026_2027 olmali, gelen: $Season"
        exit 1
    }
    $dashed = "{0}-{1}" -f $Season.Substring(0, 4), $Season.Substring(7, 2)

    Write-Log "=========================================================="
    Write-Log "baslangic  sezon $Season ($dashed)  gun $Days  repo $repo"

    # Veri seti kurulumu birkac GB istiyor; pano da 1-2 GB tutuyor. Ikisi ayni
    # anda calisirken bir kez "paging file too small" alindi, o yuzden uyari.
    $dash = Get-CimInstance Win32_Process -Filter "Name='py.exe' OR Name='python.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -like "*app.py*" }
    if ($dash) {
        Write-Log "uyari: pano calisiyor (PID $($dash.ProcessId -join ', ')) - veri seti kurulumu bellekte ona rakip olur"
    }

    # -----------------------------------------------------------------------
    # Adim calistirici. Yonlendirme cmd uzerinden yapiliyor: PowerShell 5.1 yerel
    # bir exe'nin stderr'ini kendi icinde yonlendirdiginde her satiri ErrorRecord'a
    # cevirip exit 0'i bile basarisiz gosteriyor.
    # -----------------------------------------------------------------------
    $failed = @()
    function Invoke-Step {
        param([string]$Name, [string]$Command, [switch]$Critical)
        Write-Log "---- $Name"
        $started = Get-Date
        "" | Add-Content -Path $log -Encoding utf8
        cmd /c "py -X utf8 $Command >> `"$log`" 2>&1"
        $code = $LASTEXITCODE
        $secs = [math]::Round(((Get-Date) - $started).TotalSeconds)
        if ($code -eq 0) {
            Write-Log "     tamam ($secs sn)"
        } else {
            Write-Log "     BASARISIZ (cikis $code, $secs sn)"
            $script:failed += $Name
            if ($Critical) {
                Write-Log "KRITIK adim basarisiz, duruldu. Sonraki adimlar eski veriyle calisirdi."
                throw "kritik adim basarisiz: $Name"
            }
        }
    }

    # ---- 1-5: NBA'den veri ------------------------------------------------
    # Hicbiri kritik degil: NBA'in bir ucu cevap vermezse elimizdekiyle devam
    # etmek, hic guncellememekten iyi. Eksik kalan dosya yarin kaldigi yerden
    # alinir, cekiciler var olani yeniden indirmez.
    if ($NoFetch) {
        Write-Log "---- 1-5 atlandi (-NoFetch)"
    } else {
        Invoke-Step "1/13 fikstur"            "data_retriaval\fetch_schedule.py $dashed"
        Invoke-Step "2/13 oynanan mac listesi" "data_retriaval\fetch_season_game_ids.py $dashed --refresh"
        Invoke-Step "3/13 pbp + kutu skorlari" "data_retriaval\retrieve_2025_2026_data.py --season $Season"
        # Bu ikisi macin ertesi gunu yayinlaniyor; bugun eksik cikmasi normaldir.
        Invoke-Step "4/13 savunma + takip"     "data_retriaval\backfill_defensive_tracking.py $Season"
        Invoke-Step "5/13 matchup"             "data_retriaval\retrieve_matchups.py $Season"
        Invoke-Step "6/13 eksik dosya raporu"  "tools\check_data.py $Season"
    }

    # ---- 7: veri seti (kritik) --------------------------------------------
    Invoke-Step "7/13 veri seti" "prediction_engines\build_dataset_asof.py --days $Days" -Critical

    # ---- 8-9: panonun onbellekleri ----------------------------------------
    # build_dataset_asof.py yalnizca modelin okudugu v5/unweighted onbellekleri
    # yazar. Panonun guc siralamasi ve kadro paneli bu ikisinden okur ve kimse
    # yazmazsa sessizce geride kalir.
    if ($SkipCaches) {
        Write-Log "---- 8-9 atlandi (-SkipCaches)"
    } else {
        Invoke-Step "8/13 impact onbellegi" "prediction_engines\backfill_impact_from_db.py --source csv --seasons $Season"
        Invoke-Step "9/13 deger onbellegi"  "prediction_engines\value_engine.py --seasons $Season"
    }

    # ---- 10-13: tahmin ve karne -------------------------------------------
    Invoke-Step "10/13 tahminler"      "prediction_engines\forecast_upcoming.py"
    if (-not $NoFetch) {
        Invoke-Step "11/13 bahis cizgisi arsivi" "data_retriaval\odds_run_all.py"
    }
    Invoke-Step "12/13 tahmin - piyasa" "prediction_engines\forecast_vs_market.py"
    Invoke-Step "13/13 tahmin karnesi"  "prediction_engines\forecast_scorecard.py"

    # ---- ozet -------------------------------------------------------------
    $asof = Join-Path $repo "output\engineered_dataset_asof.pkl"
    if (Test-Path $asof) {
        Write-Log "veri seti: $((Get-Item $asof).LastWriteTime.ToString('yyyy-MM-dd HH:mm'))"
    }
    $newestForecast = Get-ChildItem (Join-Path $repo "output\forecasts") -Filter "forecast_*.csv" -ErrorAction SilentlyContinue |
                      Sort-Object LastWriteTime | Select-Object -Last 1
    if ($newestForecast) {
        $rows = (Get-Content $newestForecast.FullName | Measure-Object -Line).Lines - 1
        Write-Log "tahmin dosyasi: $($newestForecast.Name) ($rows mac)"
    }

    # ---- commit (varsayilan kapali) ---------------------------------------
    # Veri otomatik commit edilir, kod elle - ama bunu sessizce yapmak dogru
    # degil, o yuzden anahtara bagli. Yalnizca veri yollari eklenir.
    if ($Commit) {
        Write-Log "---- commit"
        cmd /c "git add nba_data game_ids odds_data output/forecasts >> `"$log`" 2>&1"
        cmd /c "git commit -q -m `"Daily data update $(Get-Date -Format 'yyyy-MM-dd')`" >> `"$log`" 2>&1"
        if ($LASTEXITCODE -eq 0) { Write-Log "     commit edildi" } else { Write-Log "     commit yok (degisiklik olmayabilir)" }
    }

    # ---- panoyu yeniden baslat (varsayilan kapali) ------------------------
    # Pano veriyi acilista bir kez okur. Yeniden baslatilmazsa bu kosunun
    # urettigi hicbir sey ekranda gorunmez.
    if ($RestartDashboard) {
        Write-Log "---- pano yeniden baslatiliyor"
        $running = Get-CimInstance Win32_Process -Filter "Name='py.exe' OR Name='python.exe'" -ErrorAction SilentlyContinue |
                   Where-Object { $_.CommandLine -like "*app.py*" }
        if ($running) {
            # Calisan kopyanin kendi argumanlariyla geri gelsin: --dev, --port,
            # --host ne verildiyse korunur.
            $cmdline = ($running | Select-Object -First 1).CommandLine
            # $args degil: $args PowerShell'in otomatik degiskeni, uzerine yazmak
            # fonksiyon icinde sessiz hatalara yol acar.
            $appArgs = ""
            if ($cmdline -match "app\.py(.*)$") { $appArgs = $Matches[1].Trim() }
            $running | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            Start-Sleep -Seconds 3
            Start-Process -FilePath "py" -ArgumentList (@("app.py") + ($appArgs -split "\s+" | Where-Object { $_ })) `
                          -WorkingDirectory $repo -WindowStyle Hidden
            Write-Log "     yeniden baslatildi: py app.py $appArgs"
        } else {
            Write-Log "     pano calismiyordu, baslatilmadi"
        }
    } else {
        Write-Log "not: pano acilista veriyi bir kez okur; yeni veriyi gormesi icin yeniden baslatilmali"
    }

    if ($failed.Count) {
        Write-Log "bitti, $($failed.Count) adim basarisiz: $($failed -join ', ')"
    } else {
        Write-Log "bitti, hepsi tamam"
    }
}
catch {
    Write-Log "DURDU: $($_.Exception.Message)"
    exit 1
}
finally {
    Remove-Item $lock -ErrorAction SilentlyContinue
}
