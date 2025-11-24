# 익명 업로드 완료 가이드

## 완료된 작업

1. ✅ GitHub 저장소를 **Private**으로 변경
2. ✅ 프로젝트 압축 파일 생성 (`D:\EWSA\dynamic_ensemble_rl_trading.zip`)
3. ✅ README.md를 익명 링크 형식으로 업데이트
4. ✅ setup.py를 익명 링크 형식으로 업데이트
5. ✅ docs/REPRODUCTION.md를 익명 링크 형식으로 업데이트

## 다음 단계: 4open.science 업로드

### 1. 4open.science 접속
- https://4open.science/ 또는 https://anonymous.4open.science/ 접속

### 2. 파일 업로드
- `D:\EWSA\dynamic_ensemble_rl_trading.zip` 파일 업로드
- 업로드 완료 후 익명 링크 ID 받기 (예: `abc123xyz`)

### 3. 링크 업데이트
받은 익명 링크 ID로 업데이트:

```powershell
cd D:\EWSA\dynamic_ensemble_rl_trading
python scripts/update_links.py YOUR-ANONYMOUS-LINK-ID
```

예시:
```powershell
python scripts/update_links.py abc123xyz
```

### 4. 확인
다음 파일들이 자동으로 업데이트됩니다:
- README.md
- setup.py  
- docs/REPRODUCTION.md

## 중요 사항

- ✅ GitHub 저장소는 **Private**으로 유지 (논문에 포함하지 않음)
- ✅ 논문에는 **4open.science 링크만** 포함
- ✅ 모든 개인 정보 제거 완료

## 현재 상태

- GitHub: Private 저장소 (백업용)
- 4open.science: 업로드 대기 중
- 압축 파일: `D:\EWSA\dynamic_ensemble_rl_trading.zip` 준비 완료

## 익명 링크 받은 후

익명 링크를 받으시면 알려주세요. 자동으로 모든 파일을 업데이트하겠습니다!

