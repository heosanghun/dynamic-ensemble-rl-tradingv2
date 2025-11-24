# 빠른 업로드 가이드

## GitHub에 업로드하기

### 1단계: GitHub 저장소 생성

1. https://github.com/new 접속
2. Repository name: `dynamic-ensemble-rl-trading`
3. Description: "A Robust Dynamic Ensemble Reinforcement Learning Trading System for Responding to Market Regimes"
4. Public 또는 Private 선택
5. "Add a README file" 체크 해제
6. "Create repository" 클릭

### 2단계: PowerShell에서 업로드

**옵션 A: 자동 스크립트 사용 (권장)**

```powershell
cd D:\EWSA\dynamic_ensemble_rl_trading
.\upload_to_github.ps1 YOUR-GITHUB-USERNAME
```

**옵션 B: 수동 명령 실행**

```powershell
cd D:\EWSA\dynamic_ensemble_rl_trading

# Git 초기화 (이미 되어있으면 생략)
git init

# Git 설정
git config user.name "Anonymous"
git config user.email "anonymous@example.com"

# 파일 추가 및 커밋
git add .
git commit -m "Initial commit: Dynamic Ensemble RL Trading System"
git branch -M main

# 원격 저장소 추가 (YOUR-USERNAME을 실제 사용자명으로 교체)
git remote add origin https://github.com/YOUR-USERNAME/dynamic-ensemble-rl-trading.git

# 업로드
git push -u origin main
```

### 3단계: 링크 업데이트

업로드가 완료되면 받은 링크로 파일들을 업데이트합니다:

```powershell
# GitHub 링크인 경우
python scripts/update_links.py --github https://github.com/YOUR-USERNAME/dynamic-ensemble-rl-trading

# 또는 4open.science 링크인 경우
python scripts/update_links.py YOUR-ANONYMOUS-LINK-ID
```

### 4단계: 업데이트된 링크 커밋

```powershell
git add .
git commit -m "Update repository links"
git push
```

## 완료!

이제 저장소 링크를 논문에 포함할 수 있습니다.

- GitHub: `https://github.com/YOUR-USERNAME/dynamic-ensemble-rl-trading`
- 또는 4open.science: `https://anonymous.4open.science/r/YOUR-ANONYMOUS-LINK-ID`

## 문제 해결

**오류: "remote origin already exists"**
```powershell
git remote remove origin
git remote add origin https://github.com/YOUR-USERNAME/dynamic-ensemble-rl-trading.git
```

**오류: "Authentication failed"**
- GitHub Personal Access Token이 필요할 수 있습니다
- Settings > Developer settings > Personal access tokens에서 생성

**오류: "Large files"**
- .gitignore가 올바르게 설정되어 있는지 확인
- 큰 데이터 파일은 제외되어야 합니다

