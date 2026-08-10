# 카테부 4기 AI 성능 개선 대회 준비

## 요약

### 1일차

- 제출일: 2일차 13:00까지 제출
- 목표
  - Baseline 만들기 (T4 GPU 1회 최대 사용량 파악하기)
  - Upgrade 개선하기
  - 정량 평가하기

- 준비할 것
  1. 정확도를 어떻게 평가할 것인가
  2. n차 시도(실험 문서 양식)
  3. 답변 스키마 생성

  ```json
  {
      "team": "3",
      "answers": [
          {
              "qid": str,
              "retrieved": [
                  ["약관명", 10]
              ],
              "answer": str
          },
      ]
  }
  ```

  4. 약관 데이터 다운로드 및 저장 위치 파악해서 문서 로딩 단계에 연결하기
  - 외부인이 접근했을 때 VectorDB로 접근 가능한지 파악해서 평가시 문제 없도록

## 인덱싱

### 로딩 및 가져오기

- Baseline
- Upgrade
  - 파일 포맷별 파서 분기, 메타데이터 스키마 사전 정의 (doc_name은 카카오계정 약관/카카오 통합서비스약관/카카오 통합 약관/카카오 위치정보 이용약관 고정)

### 파싱

- Baseline
  - 원문 텍스트 그대로 사용, 조 번호(제n조) 정규식 추출
- Upgrade
  - 구조화 파싱(제목/조/항/호 계층 인식), 표·리스트 등 비정형 요소 별도 처리

### 청킹

- Baseline
  - 조(article) 단위 청킹
  - 청크 메타데이터에 doc_name·article_number 필수 포함(채점 스키마 gold_articles와 직접 대응)
- Upgrade
  - semantic chunking, 항/호 세분화 시 상위 조 정보 유지(parent-child chunking)

### 임베딩

- Baseline
  - 사전 학습 임베딩 모델(API 또는 경량 로컬 모델) 1종 고정 적용
- Upgrade
  - 한국어 법률 도메인 특화 임베딩(bge-m3, ko-sroberta 등) 비교 실험

### 저장

- Baseline
  - in-memory 벡터 저장소(FAISS)
- Upgrade
  - HNSW 인덱싱, 문서명 기반 메타데이터 필터링(4종 문서 스코프 제한)

## 검색

### 임베딩

- Baseline
  - bge-m3
- Upgrade
  - multilingual-e5-large
  - ko-sroberta-multitask

### 유사도 검색

- Baseline
  - cosine similarity top-k(k=4, 채점 스키마 상한과 일치)
- Upgrade
  - 하이브리드 검색(BM25+dense)
  - reranking(cross-encoder)

## 증강

### 컨텍스트 조합

- Baseline
  - [문서명, 조번호, 본문] 형식으로 프롬프트 입력
- Upgrade
  - Structured Outputs(JSON Schema)로 답변 형식을 answers_public 스키마와 동일하게 강제 생성
  - few-shot 예시 포함

## 생성

### 사용 모델

- Baseline
  - 7B
- Upgrade
  - 3B

### 품질 결과

- Baseline
  - gold_questions_public10 기준 자체 채점 — article exact match율(근거 조항 정확 일치율), key_facts 커버리지(정답 핵심 사실 포함 비율)
  - (수동 대조)
- Upgrade
  - p50/p95 latency 및 throughput_rps를 공식 프로토콜(requests=12, concurrency=2, repetitions=3)과 동일 조건으로 자체 측정해 제출 전 사전 검증

## 파인 튜닝

- 파인 튜닝 방법
