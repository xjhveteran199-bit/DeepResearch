Get-ChildItem 'C:\Users\XJH\DeepResearch\DeepPredict\' -Include *.py -Recurse | ForEach-Object {
    $content = Get-Content $_.FullName -Raw -Encoding UTF8
    if ($content -match 'zip|export|download|forecast|save.*result|打包|result_dir') {
        Write-Host "=== $($_.FullName) ===" -ForegroundColor Cyan
        $matches = [regex]::Matches($content, '.{0,50}(zip|export|download|forecast|save.*result|打包|result_dir).{0,100}', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
        foreach ($m in $matches) {
            Write-Host $m.Value -ForegroundColor Yellow
        }
        Write-Host ""
    }
}
