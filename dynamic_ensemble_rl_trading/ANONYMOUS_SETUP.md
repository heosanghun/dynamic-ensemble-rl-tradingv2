# 익명 업로드 설정 가이드

## 현재 상황

GitHub 저장소가 공개되어 있고 사용자명이 노출되어 있습니다:
- 현재 링크: https://github.com/heosanghun/dynamic-ensemble-rl-trading
- 문제: `heosanghun` 사용자명이 노출되어 익명이 아님

## 즉시 조치 사항

### 1단계: GitHub 저장소를 Private으로 변경

```powershell
# GitHub 웹사이트에서 직접 변경:
# 1. https://github.com/heosanghun/dynamic-ensemble-rl-trading/settings 접속
# 2. "Danger Zone" 섹션으로 스크롤
# 3. "Change visibility" 클릭
# 4. "Make private" 선택
```

또는 GitHub API로 변경:
```powershell
$headers = @{ "Authorization" = "token YOUR_TOKEN"; "Accept" = "application/vnd.github.v3+json" }
$body = @{ private = $true } | ConvertTo-Json
Invoke-RestMethod -Uri "https://api.github.com/repos/heosanghun/dynamic-ensemble-rl-trading" -Method PATCH -Headers $headers -Body $body
```

### 2단계: 4open.science에 익명 업로드

1. **프로젝트 압축**
   ```powershell
   cd D:\EWSA
   Compress-Archive -Path dynamic_ensemble_rl_trading -DestinationPath dynamic_ensemble_rl_trading.zip -Force
   ```

2. **4open.science 업로드**
   - https://4open.science/ 접속
   - "Submit" 또는 "Upload" 메뉴
   - `dynamic_ensemble_rl_trading.zip` 업로드
   - 익명 링크 ID 받기

3. **링크 업데이트**
   ```powershell
   cd D:\EWSA\dynamic_ensemble_rl_trading
   python scripts/update_links.py YOUR-ANONYMOUS-LINK-ID
   ```

### 3단계: 문서 업데이트

받은 익명 링크로 다음 파일들이 자동 업데이트됩니다:
- README.md
- setup.py
- docs/REPRODUCTION.md

## 중요 사항

- **GitHub 저장소는 Private으로 유지**하거나 삭제
- **논문에는 4open.science 링크만 포함**
- GitHub 링크는 논문에 포함하지 않음

## 확인 체크리스트

- [ ] GitHub 저장소를 Private으로 변경
- [ ] 4open.science에 업로드 완료
- [ ] 익명 링크 ID 받음
- [ ] 링크 업데이트 스크립트 실행
- [ ] README.md에 익명 링크만 포함
- [ ] 논문에 4open.science 링크만 포함

