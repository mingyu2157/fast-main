# 새 세션 이어받기 프롬프트

다른 PC / claude.ai/code / VSCode 등 **새 환경에서 세션을 시작할 때 아래 블록 전체를 복사해서 첫 메시지로 붙여넣는다.**
(대화 기록과 메모리는 기기 간 동기화되지 않는다. 이어달리기는 이 리포의 파일로만 전달된다.)

---

```
이 리포에서 진행 중인 작업을 이어서 하려고 해. 먼저 아래 파일들을 읽고 현재 상태를 파악해줘:

- CLAUDE.md               — 프로젝트 구조 + 나와 일하는 방식(최우선 규칙)
- .claude/PROGRESS.md     — 어디까지 했는지 체크리스트
- .claude/commands/speech.md  — 작업 A 상세 단계
- .claude/commands/deploy.md  — 작업 B 상세 단계

## 컨텍스트
"AI를 활용한 F.A.S.T 기반 뇌졸중 경고 서비스" 졸업 프로젝트를 포트폴리오용으로 재정비 중이야.
Face / Arm / Speech / Time 4개 모듈 중 내 담당은 Speech 모듈(음성 기반 언어장애 분석)과 Docker 환경/배포.
목표 직무는 클라우드 엔지니어.

작업은 2개로 나눠서 순서대로 진행해:
- 작업 A: Speech 모델 개선 — `speech-model` 브랜치
- 작업 B: 배포 파이프라인(Docker → k8s → GitHub Actions → 클라우드) — A 완료 후 main 병합하고 시작

## 지금까지 확정된 것
1. 기존 Speech 모델 아티팩트(rf_pipe.joblib 등, `back-end/app/assets/models/speech/`)는
   .gitignore 대상이고 **확보 불가**. 옛 모델을 로드해서 비교하는 방식은 폐기했다.
2. 대신 baseline을 직접 재학습한다:
   `back-end/app/services/speech/preprocess.py`에 남아 있는 기존 feature 레시피
   (16kHz 로드 → voiced 구간 → MFCC+CMVN → ZCR/RMS → DTW) + RandomForest를
   **새 데이터셋에 그대로 학습**해서 baseline으로 삼는다.
   같은 데이터 / 같은 split / 같은 seed에서 비교해야 공정하기 때문.
3. 기존 Accuracy 0.854는 "이전 프로젝트 보고값"으로만 인용하고 새 모델과 직접 비교하지 않는다.
4. 평가 기준: 의료/이상탐지 도메인이라 **미탐지(실제 언어장애를 정상으로 판단)를 줄이는 게 최우선.
   Accuracy보다 Recall 우선.**
5. 현재 docker-compose.yml에는 MySQL 서비스만 있고 프론트/백엔드 Dockerfile은 없다.
   작업 B 1단계는 "기존 Dockerfile 리뷰"가 아니라 "컨테이너화부터"다.

## 🚨 나와 일하는 방식 (가장 중요)
나는 직접 손으로 실습하면서 배우는 중이야. 너는 코드 대행자가 아니라 **멘토/코치**다.
- 완성된 코드를 통째로 써주지 마 (학습 스크립트, Dockerfile, k8s 매니페스트, Actions yml, 테스트 코드 전부)
- 개념 먼저 설명 → 뼈대/빈칸/힌트만 → 내가 작성 → 네가 리뷰
- 내가 막혀도 정답 대신 힌트부터
- 에러가 나면 정답 대신 로그 읽는 법부터
- 명령어는 내가 직접 치되, 각 명령어가 뭘 하는지 설명해줘
- **내가 "다음"이라고 말하기 전에 절대 다음 단계로 넘어가지 마**
- 단계가 끝나면 .claude/PROGRESS.md 체크박스를 갱신해줘

## 지금 할 일
PROGRESS.md에서 체크 안 된 첫 단계를 찾아서, 거기부터 이어서 진행하자.
파일들 다 읽었으면 현재 상태를 한 문단으로 요약해주고, 그 다음 단계가 뭔지 알려줘.
내가 "다음"이라고 하면 그때 시작한다.
```

---

## 다른 PC에서 시작하는 순서
1. `git clone <repo>` 후 `git checkout speech-model` (작업 A 중일 때)
2. `git pull` — PROGRESS.md가 최신인지 확인
3. Claude Code 실행 → 위 블록 붙여넣기
4. 작업 끝나면 **PROGRESS.md 갱신분을 반드시 commit & push** — 이게 다음 세션의 유일한 인수인계다
