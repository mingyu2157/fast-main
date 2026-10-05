---
description: 작업 B — 클라우드 배포 파이프라인 (멘토 모드, 작업 A 완료 후 main에서)
---

# 작업 B — 배포 파이프라인 (Docker → k8s → GitHub Actions → 클라우드)

**전제: 작업 A(Speech 모델 개선)가 끝나고 `main`에 병합된 상태.** 아니면 먼저 알려줘.

## 컨텍스트
React(Vite) 프론트 + FastAPI 백엔드 + MySQL.
나는 Docker 환경/배포 담당이고 **목표 직무는 클라우드 엔지니어**.

⚠️ 현재 리포 실상: `docker-compose.yml`에 **MySQL 서비스만** 있고 프론트/백엔드 Dockerfile이 없다.
1단계는 "기존 Dockerfile 리뷰"가 아니라 "프론트/백엔드 컨테이너화부터" 시작해야 한다.

## 최종 목표 워크플로우
1. feature 브랜치에서 작업
2. develop으로 병합 시도 시 자동 테스트(단위/통합) 실행, 통과해야 병합 가능
3. develop에서 문제없으면 main으로 병합
4. main push → 빌드 → 이미지 푸시 → 배포 자동 실행

## 기본 전제 (바꾸고 싶으면 말한다)
- 레지스트리: GitHub Container Registry(GHCR)
- 테스트 범위: 백엔드(FastAPI) 집중. 프론트 테스트 이번엔 제외.
- 배포 타겟: 먼저 로컬 k8s(minikube 또는 kind) 자동 배포 완성 → 그다음 실제 클라우드(GKE 또는 EKS) 하나만 골라 깊게.

## 진행 방식
`CLAUDE.md`의 "함께 일하는 방식" 규칙을 그대로 따른다. 실습 코치 역할.
**설정 파일(Dockerfile, k8s 매니페스트, Actions 워크플로우, 테스트 코드)을 통째로 써주지 말 것.**
개념 설명 → 뼈대/빈칸/힌트 → 내가 작성 → 네가 리뷰.
명령어는 내가 직접 치되 각 명령어가 뭘 하는지 설명. 에러 나면 로그 읽고 디버깅하는 법부터.
**각 단계는 내가 "다음"이라고 할 때까지 절대 넘어가지 말 것.**
단계가 끝나면 `.claude/PROGRESS.md`의 작업 B 체크박스를 갱신해줘.

## 단계
1. **[Docker 최적화]** 현재 `docker-compose.yml`을 같이 리뷰하고, 백엔드/프론트 Dockerfile을 내가 직접 작성. 멀티스테이지 빌드 / 이미지 경량화 / 빌드 캐시 개선점 포함.
2. **[테스트 - 단위]** FastAPI 백엔드에 pytest 단위 테스트. "무엇을 단위 테스트로 잡아야 하는지" 기준부터. pytest, pytest-cov를 왜 쓰는지도 설명. 첫 테스트는 내가 직접 작성. (현재 `back-end/app/tests/test_auth.py` 하나만 존재)
3. **[테스트 - 통합]** API 엔드포인트를 실제 호출하는 통합 테스트. FastAPI TestClient(httpx) 사용. DB 격리 방법 비교(테스트용 MySQL 컨테이너 / testcontainers / SQLite 대체)의 장단점 → 내가 골라서 직접 구현 → 로컬에서 직접 실행.
4. **[브랜치 전략]** main / develop / feature 구조 세팅, develop·main에 브랜치 보호 규칙(테스트 통과해야 병합).
5. **[로컬 k8s 세팅]** minikube(또는 kind) 설치·실행을 내 손으로. 노드/파드/클러스터 개념도 같이.
6. **[compose → k8s 매니페스트]** 서비스 하나씩 이관. Deployment / Service / Ingress / ConfigMap / Secret이 각각 왜 필요한지 설명 → 내가 하나씩 작성 → 네가 리뷰 → 로컬 클러스터에 배포해서 동작 확인.
7. **[CI - 테스트 자동화]** GitHub Actions로 "develop / main 향하는 PR 열리면 테스트 자동 실행". yml은 내가 작성하도록 구조와 힌트만. 통과/실패가 PR에 어떻게 표시되는지도 설명.
8. **[CD - 배포 자동화]** main 반영 시 이미지 빌드 → GHCR 푸시 → k8s 롤아웃. 시크릿(레지스트리 인증, kubeconfig)을 GitHub에 안전하게 넣는 법. 내가 작성하고 실제 커밋해서 파이프라인 도는 것 확인.
9. **[실제 클라우드 배포]** GKE vs EKS의 난이도·비용·이력서 관점 장단점 정리 → 하나 선택. 로컬 매니페스트에서 무엇을 바꿔야 클라우드에 올라가는지(레지스트리, 인증, LoadBalancer/Ingress) 단계별로 내가 직접 적용.
