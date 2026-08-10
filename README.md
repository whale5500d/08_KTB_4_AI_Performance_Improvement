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

### 파싱

- Baseline
- Upgrade

### 청킹

- Baseline
- Upgrade

### 임베딩

- Baseline
- Upgrade

### 저장

- Baseline
- Upgrade

## 검색

### 임베딩

- Baseline
- Upgrade

### 유사도 검색

- Baseline
- Upgrade

## 증강

### 컨텍스트 조합

- Baseline
- Upgrade

## 생성

### 사용 모델

- Baseline
- Upgrade

### 품질 결과

## 파인 튜닝

- 파인 튜닝 방법
