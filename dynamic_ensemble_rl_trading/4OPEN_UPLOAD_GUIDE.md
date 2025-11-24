# 4open.science 업로드 완전 가이드

## 준비 완료 상태

✅ GitHub 저장소: Private으로 전환 완료
✅ 압축 파일: `D:\EWSA\dynamic_ensemble_rl_trading.zip` 생성 완료
✅ 모든 문서: 익명 링크 형식으로 업데이트 완료

## 4open.science 업로드 단계

### 1단계: 4open.science 접속

1. 웹 브라우저에서 다음 주소 접속:
   - https://4open.science/
   - 또는 https://anonymous.4open.science/

2. 페이지에서 "Submit" 또는 "Upload" 버튼 찾기
   - 메인 페이지 상단 메뉴 확인
   - 또는 "New Submission" 링크 클릭

### 2단계: 파일 업로드

1. **업로드 페이지에서:**
   - "Choose File" 또는 "Browse" 버튼 클릭
   - `D:\EWSA\dynamic_ensemble_rl_trading.zip` 파일 선택
   - 파일이 선택되면 업로드 시작

2. **업로드 진행:**
   - 파일 크기: 약 0.1 MB (빠른 업로드 예상)
   - 업로드 완료까지 대기

3. **메타데이터 입력 (필요한 경우):**
   - Title: "Dynamic Ensemble Reinforcement Learning Trading System"
   - Description: "A Robust Dynamic Ensemble Reinforcement Learning Trading System for Responding to Market Regimes"
   - Keywords: "reinforcement learning", "trading system", "market regimes", "ensemble learning"

### 3단계: 익명 링크 받기

업로드 완료 후:
- 익명 링크 ID를 받습니다
- 형식 예시: `abc123xyz` 또는 `r/abc123xyz`
- 전체 링크: `https://anonymous.4open.science/r/abc123xyz`

**중요:** 링크 ID를 복사해서 저장하세요!

### 4단계: 링크 업데이트 (자동)

받은 익명 링크 ID를 알려주시면 자동으로 업데이트하겠습니다:

```powershell
cd D:\EWSA\dynamic_ensemble_rl_trading
python scripts/update_links.py YOUR-ANONYMOUS-LINK-ID
```

예시:
```powershell
python scripts/update_links.py abc123xyz
```

## 업로드 후 확인 사항

업데이트된 파일 확인:
- ✅ README.md
- ✅ setup.py
- ✅ docs/REPRODUCTION.md

모든 파일에서 `YOUR-ANONYMOUS-LINK-ID`가 실제 익명 링크로 변경되었는지 확인하세요.

## 문제 해결

### 4open.science 접속이 안 될 때:
- 다른 브라우저 시도 (Chrome, Firefox, Edge)
- VPN 사용 고려
- 네트워크 연결 확인

### 업로드 실패 시:
- 파일 크기 확인 (너무 크면 문제 가능)
- 네트워크 연결 확인
- 브라우저 캐시 삭제 후 재시도
- 다른 시간대에 재시도

### 링크를 받지 못한 경우:
- 이메일 확인 (등록한 경우)
- 업로드 완료 페이지 다시 확인
- 4open.science 지원팀에 문의

## 다음 단계

익명 링크를 받으신 후:
1. 링크 ID를 알려주세요
2. 자동으로 모든 파일 업데이트
3. 논문에 링크 포함
4. 제출 준비 완료!

## 참고

- GitHub 저장소는 Private으로 유지 (논문에 포함하지 않음)
- 논문에는 4open.science 링크만 포함
- 모든 개인 정보 제거 완료

