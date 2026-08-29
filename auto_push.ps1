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

# Bir merge/rebase yarim kalmissa dokunma - otomatik commit isi daha da karistirir.
foreach ($marker in @(".git\MERGE_HEAD", ".git\rebase-merge", ".git\rebase-apply")) {
    if (Test-Path (Join-Path $repo $marker)) {
        Write-Log "atlandi: devam eden merge/rebase var ($marker)"
        exit 0
    }
}

$changes = git status --porcelain
if (-not $changes) {
    exit 0          # degisiklik yok, sessizce cik
}

$count = ($changes | Measure-Object).Count
$branch = git branch --show-current

git add -A
if (-not $?) { Write-Log "HATA: git add basarisiz"; exit 1 }

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
git commit -m "auto: $stamp ($count degisiklik)" | Out-Null
if (-not $?) { Write-Log "HATA: git commit basarisiz"; exit 1 }

git push origin $branch 2>&1 | Out-Null
if ($?) {
    Write-Log "push edildi: $branch, $count degisiklik"
} else {
    Write-Log "HATA: push basarisiz ($branch) - commit yerelde duruyor"
    exit 1
}
