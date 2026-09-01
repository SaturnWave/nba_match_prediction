# Otomatik commit + push, 5 dakikada bir zamanlanmis gorevden.
#
# YALNIZCA VERI VE URETILEN CIKTILAR commit edilir. Kod dosyalari (*.py, *.ps1,
# templates/, .gitignore, README) kasitli olarak disarida birakilir: 5 dakikalik
# dongu, anlamli commit mesaji yazilmadan once kodu "auto:" mesajiyla kapiyordu.
# Kodu elle commit edin; bu script veriyi yedekler.
#
# Kaldirmak icin:  schtasks /Delete /TN "NBA repo auto-push" /F
# Gecici durdurmak: schtasks /Change /TN "NBA repo auto-push" /DISABLE

$ErrorActionPreference = "Stop"
$repo = "C:\Users\arcan\OneDrive\Desktop\NBA\nba_match_prediction"
$log  = Join-Path $repo "auto_push.log"

# Otomatik commit edilecek yollar - veri ve uretilmis ciktilar.
$dataPaths = @("nba_data", "output", "models", "game_ids", "game_impact_cache_v3.pkl")

function Write-Log($msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Add-Content -Path $log -Encoding utf8
}

Set-Location $repo

# Kendi kilidimiz: onceki calisma hala surerken ikincisi baslamasin. Buyuk bir
# indirme sirasinda 'git add' 5 dakikayi asabiliyor, o zaman iki ornek ayni
# index uzerinde carpisir.
$lockFile = Join-Path $repo ".git\auto_push.lock"
if (Test-Path $lockFile) {
    $age = (Get-Date) - (Get-Item $lockFile).LastWriteTime
    if ($age.TotalMinutes -lt 60) {
        Write-Log "atlandi: onceki calisma hala suruyor ($([int]$age.TotalMinutes) dk)"
        exit 0
    }
    Remove-Item $lockFile -Force   # 1 saati gecmisse kalinti sayilir
}

if (Test-Path (Join-Path $repo ".git\index.lock")) {
    Write-Log "atlandi: git index.lock mevcut"
    exit 0
}

foreach ($marker in @(".git\MERGE_HEAD", ".git\rebase-merge", ".git\rebase-apply")) {
    if (Test-Path (Join-Path $repo $marker)) {
        Write-Log "atlandi: devam eden merge/rebase var ($marker)"
        exit 0
    }
}

New-Item -ItemType File -Path $lockFile -Force | Out-Null
try {
    $existing = $dataPaths | Where-Object { Test-Path (Join-Path $repo $_) }
    if (-not $existing) {
        Write-Log "atlandi: veri yolu bulunamadi"
        return
    }

    git add -- $existing
    if (-not $?) { Write-Log "HATA: git add basarisiz"; return }

    $staged = git diff --cached --name-only
    if (-not $staged) {
        # Veri degismemis. Bekleyen kod degisikligi varsa gorunur yap - yoksa
        # elle commit bekleyen dosyalar sessizce unutuluyor.
        $pending = git status --porcelain | Where-Object { $_ -notmatch ' (nba_data|output|models|game_ids)/' }
        if ($pending) {
            Write-Log "veri degismedi; elle commit bekleyen $(($pending | Measure-Object).Count) dosya var"
        }
        return
    }

    $count = ($staged | Measure-Object).Count
    $branch = git branch --show-current
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm"

    git commit -m "auto(veri): $stamp ($count dosya)" | Out-Null
    if (-not $?) { Write-Log "HATA: git commit basarisiz"; return }

    git push origin $branch 2>&1 | Out-Null
    if ($?) {
        Write-Log "push edildi: $branch, $count veri dosyasi"
    } else {
        Write-Log "HATA: push basarisiz ($branch) - commit yerelde duruyor"
    }
}
finally {
    Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
}
