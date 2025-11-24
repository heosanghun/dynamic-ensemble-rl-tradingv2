# 업로드 가이드 (Upload Instructions)

## 방법 1: 4open.science에 직접 업로드 (권장)

### 단계별 가이드

1. **4open.science 접속**
   - https://4open.science/ 접속
   - "Submit" 또는 "Upload" 메뉴 선택

2. **프로젝트 압축**
   ```powershell
   # PowerShell에서 실행
   cd D:\EWSA
   Compress-Archive -Path dynamic_ensemble_rl_trading -DestinationPath dynamic_ensemble_rl_trading.zip
   ```

3. **업로드**
   - 압축 파일을 4open.science에 업로드
   - 익명 링크 ID를 받습니다 (예: `r/abc123xyz`)

4. **링크 업데이트**
   - 받은 ID를 사용하여 다음 명령 실행:
   ```powershell
   cd D:\EWSA\dynamic_ensemble_rl_trading
   python scripts/update_links.py YOUR-ANONYMOUS-LINK-ID
   ```

## 방법 2: GitHub에 익명 저장소 생성

### GitHub CLI 사용 (gh 명령어)

1. **GitHub CLI 설치 확인**
   ```powershell
   gh --version
   ```

2. **로그인**
   ```powershell
   gh auth login
   ```

3. **익명 저장소 생성 및 업로드**
   ```powershell
   cd D:\EWSA\dynamic_ensemble_rl_trading
   git init
   git config user.name "Anonymous"
   git config user.email "anonymous@example.com"
   git add .
   git commit -m "Initial commit: Dynamic Ensemble RL Trading System"
   
   # GitHub에 저장소 생성 (Private으로 생성)
   gh repo create dynamic-ensemble-rl-trading --private --source=. --remote=origin --push
   ```

4. **저장소를 Public으로 변경 (선택사항)**
   - GitHub 웹사이트에서 Settings > Change visibility > Make public

5. **링크 받기**
   - 저장소 URL을 받습니다 (예: `https://github.com/username/dynamic-ensemble-rl-trading`)

## 방법 3: 수동 GitHub 업로드

1. **GitHub 웹사이트에서 새 저장소 생성**
   - https://github.com/new 접속
   - Repository name: `dynamic-ensemble-rl-trading`
   - Description: "A Robust Dynamic Ensemble Reinforcement Learning Trading System"
   - Public 또는 Private 선택
   - "Create repository" 클릭

2. **로컬에서 Git 초기화 및 업로드**
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

## 업로드 후 링크 업데이트

업로드 후 받은 링크 ID를 사용하여 파일들을 업데이트하세요.

### 4open.science 링크인 경우:
```powershell
python scripts/update_links.py YOUR-ANONYMOUS-LINK-ID
```

### GitHub 링크인 경우:
```powershell
python scripts/update_links.py --github https://github.com/username/dynamic-ensemble-rl-trading
```

## 확인 사항

업로드 전에 다음을 확인하세요:
- [ ] 모든 개인 정보가 제거되었는지 확인
- [ ] .gitignore가 올바르게 설정되었는지 확인
- [ ] 큰 데이터 파일이 제외되었는지 확인
- [ ] README.md가 완성되었는지 확인

