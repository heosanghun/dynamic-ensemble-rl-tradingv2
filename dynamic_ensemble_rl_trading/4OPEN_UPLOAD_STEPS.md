# 4open.science 익명 업로드 단계별 가이드

## 현재 상황

- GitHub 저장소가 Private으로 변경되었습니다
- 이제 4open.science에 익명으로 업로드해야 합니다

## 단계별 가이드

### 1단계: 압축 파일 확인

압축 파일이 생성되었습니다:
- 위치: `D:\EWSA\dynamic_ensemble_rl_trading.zip`
- 크기: 확인 필요

### 2단계: 4open.science 접속 및 익명화

#### 2-1. 웹사이트 접속
- https://anonymous.4open.science/ 접속

#### 2-2. GitHub로 로그인
- 페이지 오른쪽 상단의 **GitHub 로고/버튼** 클릭
- GitHub 계정으로 로그인
- 권한 승인 (이전에 이미 승인하셨다면 이 단계는 건너뛰기)

#### 2-3. "Anonymize" 메뉴 클릭
- 상단 네비게이션 바에서 **"Anonymize"** 메뉴 클릭
- 또는 대시보드로 이동

#### 2-4. 저장소 선택
- 대시보드에서 익명화할 저장소 선택
  - `heosanghun/dynamic-ensemble-rl-trading` 선택
- 또는 저장소 URL 직접 입력

#### 2-5. 익명화 설정 완료
- 익명화할 용어 목록 확인 및 완성
- 개인 정보가 제거될 항목 확인
- 설정 확인 후 익명화 실행

#### 2-6. 익명 링크 받기
- 익명화 완료 후 익명 링크 ID를 받습니다
- 형식: `r/abc123xyz` 또는 `abc123xyz`
- 전체 링크: `https://anonymous.4open.science/r/abc123xyz`
- **이 링크를 복사해두세요!**

### 3단계: 링크 업데이트

받은 익명 링크 ID로 파일들을 업데이트합니다:

```powershell
cd D:\EWSA\dynamic_ensemble_rl_trading
python scripts/update_links.py YOUR-ANONYMOUS-LINK-ID
```

예시:
```powershell
python scripts/update_links.py abc123xyz
# 또는
python scripts/update_links.py r/abc123xyz
```

### 4단계: 확인

업데이트된 파일 확인:
- README.md
- setup.py
- docs/REPRODUCTION.md

모든 파일에서 `YOUR-ANONYMOUS-LINK-ID`가 실제 익명 링크로 변경되었는지 확인하세요.

## 중요 사항

1. **GitHub 저장소는 Private으로 유지**
   - 논문에는 GitHub 링크를 포함하지 않음
   - 4open.science 링크만 포함

2. **익명성 보장**
   - 모든 개인 정보가 제거되었는지 확인
   - 코드에 작성자 이름이 없는지 확인

3. **논문 제출 시**
   - 논문에 4open.science 링크만 포함
   - GitHub 링크는 포함하지 않음

## 문제 해결

### ❌ "does not exist or is not accessible" 오류 해결

**원인:** 저장소가 Private이어서 4open.science가 접근할 수 없습니다.

**해결 방법 1: 저장소를 임시로 Public으로 변경 (권장)**

1. GitHub 저장소 페이지로 이동
   - https://github.com/heosanghun/dynamic-ensemble-rl-trading 접속
2. Settings 탭 클릭
3. 맨 아래로 스크롤하여 "Danger Zone" 섹션 찾기
4. "Change visibility" 클릭
5. "Make public" 선택하고 확인
6. 4open.science에서 다시 시도
7. **익명화 완료 후 다시 Private으로 변경** (중요!)

**해결 방법 2: GitHub OAuth 권한 재설정**

1. GitHub Settings → Applications → Authorized OAuth Apps
   - https://github.com/settings/applications 접속
2. "anonymous Github" 앱 찾기
3. 권한 확인: "repo" 스코프가 있는지 확인
4. 없다면 4open.science에서 다시 로그인하여 권한 재승인

**해결 방법 3: ZIP 파일 직접 업로드 (대안)**

4open.science에서 ZIP 파일 업로드 기능이 있다면:
1. `D:\EWSA\dynamic_ensemble_rl_trading.zip` 파일 준비
2. 4open.science의 다른 업로드 방법 확인
3. ZIP 파일 직접 업로드

**4open.science 접속이 안 될 때:**
- 다른 브라우저 시도
- VPN 사용 고려
- 또는 다른 익명 코드 공유 플랫폼 사용 (Zenodo 등)

**업로드 실패 시:**
- 파일 크기 확인 (너무 크면 문제 가능)
- 네트워크 연결 확인
- 브라우저 캐시 삭제 후 재시도

## 다음 단계

익명 링크를 받으신 후:
1. 링크 업데이트 스크립트 실행
2. 논문에 링크 포함
3. 제출 준비 완료!

