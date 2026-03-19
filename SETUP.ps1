# ============================================================
# InferBench Blog Setup Script (PowerShell)
# Run this in PowerShell from your projects directory
# ============================================================

Write-Host "=== Step 1: Create GitHub repo ===" -ForegroundColor Cyan
Write-Host @"

1. Go to https://github.com/new
2. Repository name: kr-lsb.github.io
3. Public, NO README (we'll push our own)
4. Create repository

Press Enter when done...
"@
Read-Host

Write-Host "=== Step 2: Clone and setup ===" -ForegroundColor Cyan

# Init local repo
git init kr-lsb.github.io
Set-Location kr-lsb.github.io

# Add PaperMod theme as submodule
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod

Write-Host "`n=== Step 3: Copy blog files ===" -ForegroundColor Cyan
Write-Host @"

Now copy the downloaded blog files into this directory:
  - hugo.yaml
  - README.md
  - .github/workflows/hugo.yaml
  - content/  (entire folder)
  - static/   (entire folder)

Press Enter when done...
"@
Read-Host

Write-Host "=== Step 4: Test locally ===" -ForegroundColor Cyan
Write-Host @"

Run: hugo server -D
Open: http://localhost:1313

If it looks good, press Enter to continue to deploy...
"@
Read-Host

Write-Host "=== Step 5: Push to GitHub ===" -ForegroundColor Cyan
git add -A
git commit -m "feat: initial blog setup with Hugo + PaperMod"
git branch -M main
git remote add origin https://github.com/KR-LSB/kr-lsb.github.io.git
git push -u origin main

Write-Host @"

=== Step 6: Enable GitHub Pages ===

1. Go to https://github.com/KR-LSB/kr-lsb.github.io/settings/pages
2. Source: GitHub Actions
3. Wait 1-2 minutes for the first deploy

Your blog will be live at: https://kr-lsb.github.io/

"@ -ForegroundColor Green
