# 익명 업로드 가이드

현재 GitHub 저장소는 사용자명(`heosanghun`)이 포함되어 있어 익명이 아닙니다.

## 해결 방법

### 방법 1: 4open.science에 익명 업로드 (권장)

4open.science는 논문 제출을 위한 익명 코드 공유 플랫폼입니다.

#### 단계:

1. **프로젝트 압축**
   ```powershell
   cd D:\EWSA
   Compress-Archive -Path dynamic_ensemble_rl_trading -DestinationPath dynamic_ensemble_rl_trading.zip -Force
   ```

2. **4open.science 접속 및 업로드**
   - https://4open.science/ 접속
   - "Submit" 또는 "Upload" 메뉴 선택
   - `dynamic_ensemble_rl_trading.zip` 파일 업로드
   - 익명 링크 ID 받기 (예: `r/abc123xyz`)

3. **링크 업데이트**
   ```powershell
   cd D:\EWSA\dynamic_ensemble_rl_trading
   python scripts/update_links.py abc123xyz
   ```

4. **GitHub 저장소 처리**
   - GitHub 저장소를 삭제하거나
   - Private으로 변경하여 개인 백업용으로만 사용

### 방법 2: GitHub 저장소를 Private으로 변경

1. **GitHub에서 저장소 설정 변경**
   - https://github.com/heosanghun/dynamic-ensemble-rl-trading/settings 접속
   - "Danger Zone" 섹션으로 스크롤
   - "Change visibility" 클릭
   - "Make private" 선택

2. **4open.science에 업로드**
   - 위의 방법 1과 동일하게 진행

### 방법 3: 새 익명 GitHub 계정 생성 (비권장)

- 새로운 익명 GitHub 계정 생성
- 해당 계정으로 저장소 생성
- 하지만 여전히 GitHub 사용자명이 노출됨

## 권장 사항

**4open.science 사용을 강력히 권장합니다:**
- 논문 제출에 최적화된 플랫폼
- 완전한 익명성 보장
- 학술 논문 제출에 널리 사용됨
- 심사위원들이 익숙한 플랫폼

## 현재 GitHub 저장소 처리

익명 업로드 후:
1. GitHub 저장소를 Private으로 변경하거나
2. 저장소를 삭제

이렇게 하면 논문 제출 시 익명성이 보장됩니다.

