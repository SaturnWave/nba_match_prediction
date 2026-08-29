# Otomatik commit + push. Zamanlanmis gorev tarafindan 5 dakikada bir calistirilir.
# Degisiklik yoksa hicbir sey yapmaz; bos commit atmaz.
#
# Kaldirmak icin:  schtasks /Delete /TN "NBA repo auto-push" /F

$ErrorActionPreference = "Stop"
$repo = "C:\Users\arcan\OneDrive\Desktop\NBA\nba_match_prediction"
$log  = Join-Path $repo "auto_push.log"

function Write-Log($msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Add-Content -Path $log -Encoding utf8
}

Set-Location $repo

# Kendi kilidimiz: bir onceki calisma hala surerken ikincisi baslamasin.
# Buyuk bir indirme sirasinda 'git add -A' 5 dakikayi asabiliyor, o zaman iki
# ornek ayni index uzerinde carpisir.
$lockFile = Join-Path $repo ".git\auto_push.lock"
if (Test-Path $lockFile) {
    $age = (Get-Date) - (Get-Item $lockFile).LastWriteTime
    if ($age.TotalMinutes -lt 60) {
        Write-Log "atlandi: onceki calisma hala suruyor ($([int]$age.TotalMinutes) dk)"
        exit 0
    }
    Remove-Item $lockFile -Force   # 1 saati gecmisse artik kalinti sayilir
}

# git baska bir islemin ortasindaysa (bizim disimizda) dokunma.
if (Test-Path (Join-Path $repo ".git\index.lock")) {
    Write-Log "atlandi: git index.lock mevcut"
    exit 0
}

# Bir merge/rebase yarim kalmissa dokunma - otomatik commit isi daha da karistirir.
foreach ($marker in @(".git\MERGE_HEAD", ".git\rebase-merge", ".git\rebase-apply")) {
    if (Test-Path (Join-Path $repo $marker)) {
        Write-Log "atlandi: devam eden merge/rebase var ($marker)"
        exit 0
    }
}

New-Item -ItemType File -Path $lockFile -Force | Out-Null
try {

    $changes = git status --porcelain
    if (-not $changes) {
        return          # degisiklik yok, sessizce cik
    }

    $count = ($changes | Measure-Object).Count
    $branch = git branch --show-current

    git add -A
    if (-not $?) { Write-Log "HATA: git add basarisiz"; return }

    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
    git commit -m "auto: $stamp ($count degisiklik)" | Out-Null
    if (-not $?) { Write-Log "HATA: git commit basarisiz"; return }

    git push origin $branch 2>&1 | Out-Null
    if ($?) {
        Write-Log "push edildi: $branch, $count degisiklik"
    } else {
        Write-Log "HATA: push basarisiz ($branch) - commit yerelde duruyor"
    }
}
finally {
    Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
}
