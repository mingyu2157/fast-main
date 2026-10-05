# 진행 상황

> 단계가 끝날 때마다 체크. 새 세션을 열면 여기부터 확인하고 이어서 진행한다.

## 작업 A — Speech 모델 개선  (브랜치: `speech-model`)
- [x] 1. 현재 모델 분석 — 10 feature(DTW5+slope+ZCR/RMS4) → RF. 학습 스크립트 없음, 태그 S5_voiced~10s_sr16000_cal에서 레시피 복원
- [x] 2. 장단점 진단 — feature 병목(8000값→10숫자), train/serve skew, 후처리 가드 250줄. 핵심 근거는 "아티팩트 부재 → 재현·측정 불가"
- [x] 3. 방향 결정 → **선택: C** (새 데이터셋 + 딥러닝)
- [ ] 4. 새 데이터셋 탐색 (후보: TORGO/Kaggle 유력, AI Hub 71434 신청 대기)
- [ ] 5. EDA + 전처리
- [ ] 6. 모델링
- [ ] 7. 학습 & 튜닝
- [ ] 8. baseline 비교 평가 (재학습한 RF vs 새 모델, 동일 test set)
- [ ] 9. 서비스 적용 (FastAPI 연동)
- [ ] ✅ 전체 동작 테스트 → `main` 병합

### 메모 — baseline 전략 (2026-09-16 변경)
- ❌ 기존 모델 아티팩트(`back-end/app/assets/models/speech/`) **확보 불가**. 옛 모델을 그대로 돌리는 비교는 포기.
- ✅ 대신 **baseline을 직접 재학습**한다: `preprocess.py`의 기존 feature 레시피(16kHz → voiced → MFCC+CMVN → ZCR/RMS → DTW) + RandomForest를 **새 데이터셋에 그대로 학습**.
  → 같은 데이터 / 같은 split / 같은 seed로 비교해야 공정하다. 이게 오히려 더 설득력 있는 비교.
- 기존 Accuracy 0.854는 "이전 프로젝트 보고값"으로만 인용. 데이터가 달라 새 모델과 직접 비교 불가함을 포트폴리오에 명시.
- 1단계(코드 분석)는 건너뛰지 않는다 — `preprocess.py`가 baseline의 설계도이기 때문.

## 작업 B — 배포 파이프라인  (작업 A 완료 후 `main`에서 시작)
- [ ] 1. Docker 최적화 (백엔드/프론트 Dockerfile 신규 작성 포함)
- [ ] 2. 단위 테스트 (pytest)
- [ ] 3. 통합 테스트 (TestClient + DB 격리 → 방식: ____)
- [ ] 4. 브랜치 전략 + 보호 규칙
- [ ] 5. 로컬 k8s (minikube / kind → 선택: ____)
- [ ] 6. compose → k8s 매니페스트
- [ ] 7. CI (PR 테스트 자동화)
- [ ] 8. CD (GHCR 푸시 → 롤아웃)
- [ ] 9. 실제 클라우드 (GKE / EKS → 선택: ____)
