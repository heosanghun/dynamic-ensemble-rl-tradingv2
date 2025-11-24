# GitHub 업로드 스크립트
# 사용법: .\upload_to_github.ps1 YOUR-GITHUB-USERNAME

param(
    [Parameter(Mandatory=$true)]
    [string]$GitHubUsername
)

$repoName = "dynamic-ensemble-rl-trading"
$repoUrl = "https://github.com/$GitHubUsername/$repoName.git"

Write-Host "GitHub 업로드 시작..." -ForegroundColor Green
Write-Host "Repository: $repoUrl" -ForegroundColor Cyan

# Git 초기화 확인
if (-not (Test-Path .git)) {
    Write-Host "Git 초기화 중..." -ForegroundColor Yellow
    git init
}

# Git 설정
git config user.name "Anonymous"
git config user.email "anonymous@example.com"

# 파일 추가
Write-Host "파일 추가 중..." -ForegroundColor Yellow
git add .

# 커밋
Write-Host "커밋 생성 중..." -ForegroundColor Yellow
git commit -m "Initial commit: Dynamic Ensemble RL Trading System"

# 브랜치 설정
git branch -M main

# 원격 저장소 추가
Write-Host "원격 저장소 설정 중..." -ForegroundColor Yellow
git remote remove origin 2>$null
git remote add origin $repoUrl

# 푸시
Write-Host "GitHub에 업로드 중..." -ForegroundColor Yellow
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n업로드 완료!" -ForegroundColor Green
    Write-Host "Repository URL: $repoUrl" -ForegroundColor Cyan
    
    # 링크 업데이트
    Write-Host "`n링크 업데이트 중..." -ForegroundColor Yellow
    python scripts/update_links.py --github $repoUrl
    
    Write-Host "`n완료! 다음 단계:" -ForegroundColor Green
    Write-Host "1. GitHub 저장소 확인: $repoUrl"
    Write-Host "2. README.md에서 링크가 올바르게 업데이트되었는지 확인"
    Write-Host "3. 변경사항 커밋: git add . && git commit -m 'Update repository links' && git push"
} else {
    Write-Host "`n업로드 실패. 오류를 확인하세요." -ForegroundColor Red
}

