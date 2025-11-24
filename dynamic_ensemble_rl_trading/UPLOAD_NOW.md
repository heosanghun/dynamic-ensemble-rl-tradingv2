# 지금 바로 업로드하기

## 빠른 업로드 가이드

### 옵션 1: GitHub에 업로드 (가장 빠름)

1. **GitHub 웹사이트에서 새 저장소 생성**
   - https://github.com/new 접속
   - Repository name: `dynamic-ensemble-rl-trading`
   - Description: "A Robust Dynamic Ensemble Reinforcement Learning Trading System for Responding to Market Regimes"
   - Public 또는 Private 선택
   - "Add a README file" 체크 해제 (이미 있음)
   - "Create repository" 클릭

2. **PowerShell에서 다음 명령 실행:**

```powershell
cd D:\EWSA\dynamic_ensemble_rl_trading
git init
git config user.name "Anonymous"
git config user.email "anonymous@example.com"
git add .
git commit -m "Initial commit: Dynamic Ensemble RL Trading System"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/dynamic-ensemble-rl-trading.git
git push -u origin main
```

**주의:** `YOUR-USERNAME`을 실제 GitHub 사용자명으로 교체하세요.

3. **링크 업데이트:**
```powershell
python scripts/update_links.py --github https://github.com/YOUR-USERNAME/dynamic-ensemble-rl-trading
```

### 옵션 2: 4open.science에 업로드

1. **프로젝트 압축:**
```powershell
cd D:\EWSA
Compress-Archive -Path dynamic_ensemble_rl_trading -DestinationPath dynamic_ensemble_rl_trading.zip -Force
```

2. **4open.science에 업로드**
   - https://4open.science/ 접속
   - 업로드 페이지에서 `dynamic_ensemble_rl_trading.zip` 업로드
   - 익명 링크 ID 받기 (예: `r/abc123xyz`)

3. **링크 업데이트:**
```powershell
cd D:\EWSA\dynamic_ensemble_rl_trading
python scripts/update_links.py abc123xyz
```

## 업로드 후 확인

업로드가 완료되면:
1. 저장소 링크를 확인
2. README.md에서 링크가 올바르게 업데이트되었는지 확인
3. 논문에 링크 포함

## 도움이 필요하신가요?

- 자세한 가이드: `UPLOAD_INSTRUCTIONS.md` 참조
- 익명 업로드 가이드: `ANONYMOUS_UPLOAD.md` 참조

