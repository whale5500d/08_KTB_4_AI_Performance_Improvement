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

- 질문지
  1. RAG는 결과기의 answer_question 안에 모두 만드는 것인지? 아니면 기존 셀만 그대로 두고 자체적으로 작성해야 할 코드는 다른 셀로 분리해도 되는지? 그대로 유지

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
  - ⭐️ semantic chunking, 항/호 세분화 시 상위 조 정보 유지(parent-child chunking)

### 임베딩

- Baseline
  - 사전 학습 임베딩 모델(API 또는 경량 로컬 모델) 1종 고정 적용
- Upgrade
  - ⭐️ 한국어 법률 도메인 특화 임베딩(bge-m3, ko-sroberta 등) 비교 실험

### 저장

- Baseline
  - in-memory 벡터 저장소(FAISS)
- Upgrade
  - ⭐️ HNSW 인덱싱, 문서명 기반 메타데이터 필터링(4종 문서 스코프 제한)

## 검색

### 임베딩

- Baseline
  - bge-m3
- Upgrade
  - ⭐️ multilingual-e5-large
  - ko-sroberta-multitask

### 유사도 검색

- Baseline
  - cosine similarity top-k(k=4, 채점 스키마 상한과 일치)
- Upgrade
  - ⭐️ 하이브리드 검색(BM25+dense)
  - reranking(cross-encoder)

## 증강

### 컨텍스트 조합

- Baseline
  - [문서명, 조번호, 본문] 형식으로 프롬프트 입력
- Upgrade
  - ⭐️ Structured Outputs(JSON Schema)로 답변 형식을 answers_public 스키마와 동일하게 강제 생성
  - few-shot 예시 포함

## 생성

### 사용 모델

- Baseline
  - Qwen 2.5-instruct-7B
- Upgrade
  - Qwen 2.5-instruct-7B-양자화
  - ⭐️ Qwen 2.5-instruct-3B

### 품질 결과

- Baseline
  - gold_questions_public10 기준 자체 채점 — article exact match율(근거 조항 정확 일치율), key_facts 커버리지(정답 핵심 사실 포함 비율)
  - (수동 대조)
- Upgrade
  - p50/p95 latency 및 throughput_rps를 공식 프로토콜(requests=12, concurrency=2, repetitions=3)과 동일 조건으로 자체 측정해 제출 전 사전 검증

## 파인 튜닝

- 파인 튜닝 방법

## 1차 테스트 결과 - 제출 전

```
[로딩] 카카오계정 약관 · 11847자 · https://www.kakao.com/policy/terms?lang=ko
[로딩] 카카오 위치정보 이용약관 · 5951자 · https://www.kakao.com/policy/location?lang=ko
[로딩] 카카오 통합서비스약관 · 18262자 · https://www.kakao.com/policy/terms?type=ts&lang=ko
[로딩] 카카오 통합 약관 · 18643자 · https://www.kakao.com/policy/kakaoTerms?lang=ko
[파싱] 카카오계정 약관 · 제1조~제17조 (17개)
[파싱] 카카오 위치정보 이용약관 · 제1조~제16조 (16개)
[파싱] 카카오 통합서비스약관 · 제1조~제18조 (18개)
[파싱] 카카오 통합 약관 · 제1조~제21조 (21개)
[청킹] 72개 조항 → 45개 청크
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
modules.json: 100%
 349/349 [00:00<00:00, 34.8kB/s]
config_sentence_transformers.json: 100%
 123/123 [00:00<00:00, 10.9kB/s]
README.md: 100%
 15.8k/15.8k [00:00<00:00, 1.49MB/s]
sentence_bert_config.json: 100%
 54.0/54.0 [00:00<00:00, 4.28kB/s]
config.json: 100%
 687/687 [00:00<00:00, 75.3kB/s]
pytorch_model.bin: reconstructing file: 100%
 2.27GB / 2.27GB,  136MB/s  
pytorch_model.bin: downloading bytes: 
 1.52GB,  105MB/s  
Loading weights: 100%
 391/391 [00:00<00:00, 12390.90it/s]
model.safetensors: reconstructing file: 100%
 2.27GB / 2.27GB,  106MB/s  
model.safetensors: downloading bytes: 
 1.52GB, 85.8MB/s  
tokenizer_config.json: 100%
 444/444 [00:00<00:00, 40.9kB/s]
sentencepiece.bpe.model: reconstructing file: 100%
 5.07MB / 5.07MB,  502kB/s  
sentencepiece.bpe.model: downloading bytes: 
 3.62MB,  359kB/s  
tokenizer.json: reconstructing file: 100%
 17.1MB / 17.1MB, 1.51MB/s  
tokenizer.json: downloading bytes: 
 5.88MB,  566kB/s  
special_tokens_map.json: 100%
 964/964 [00:00<00:00, 11.3kB/s]
config.json: 100%
 191/191 [00:00<00:00, 17.4kB/s]
/tmp/ipykernel_570/317778062.py:666: FutureWarning: The `get_sentence_embedding_dimension` method has been renamed to `get_embedding_dimension`.
  dim = int(model.get_sentence_embedding_dimension())
[임베딩 모델] BAAI/bge-m3 · device=cuda · dim=1024
[임베딩] 45개 청크 · shape=(45, 1024)
[저장] FAISS IndexFlatIP · 45개 벡터 · dim=1024
[인덱싱 완료] {'n_documents': 4, 'n_articles': 72, 'n_chunks': 45, 'dim': 1024, 'per_document': {'카카오계정 약관': 7, '카카오 위치정보 이용약관': 16, '카카오 통합서비스약관': 11, '카카오 통합 약관': 11}, 'elapsed_s': 109.9893}
config.json: 100%
 663/663 [00:00<00:00, 67.7kB/s]
tokenizer_config.json: 100%
 7.30k/7.30k [00:00<00:00, 818kB/s]
vocab.json: 100%
 2.78M/2.78M [00:00<00:00, 52.5MB/s]
merges.txt: 100%
 1.67M/1.67M [00:00<00:00, 50.6MB/s]
tokenizer.json: 100%
 7.03M/7.03M [00:00<00:00, 93.6MB/s]
model.safetensors.index.json: 100%
 27.8k/27.8k [00:00<00:00, 2.76MB/s]
Download complete: : 
 13.1GB, 53.4MB/s  
Reconstruction complete: 100%
 15.2GB / 15.2GB, 56.4MB/s  
Fetching 4 files: 100%
 4/4 [07:46<00:00, 466.57s/it]
Loading weights: 100%
 339/339 [01:01<00:00,  7.74it/s]
generation_config.json: 100%
 243/243 [00:00<00:00, 23.4kB/s]
[생성 모델] Qwen/Qwen2.5-7B-Instruct · 4bit=True
[부팅 완료] answer_question() 호출 준비됨
======================================================================
[answer_question 반환] {"answer": "사업자/단체 카카오계정은 계정 정보에 등록된 담당자 1인만 이용할 수 있습니다. (제17조 ②)", "retrieved": [["카카오계정 약관", 17]]}
======================================================================
[완료] 스텁 잔여 호출 0회
```

- 조 파싱은 정확함.
- 청크가 72개 조항 중 45개만 만듦. 27개 조항이 사라짐.

## 1차 테스트 결과(2) - 제출 전

```json
    {
      "qid": "P01",
      "retrieved": [],
      "answer": "",
      "error": "Timeout: 120초 안에 응답하지 않았습니다."
    },
    {
      "qid": "P02",
      "retrieved": [],
      "answer": "",
      "error": "Timeout: 120초 안에 응답하지 않았습니다."
    },
    {
      "qid": "P03",
      "retrieved": [],
      "answer": "",
      "error": "Timeout: 120초 안에 응답하지 않았습니다."
    },
```

- Timeout 에러 발생

## 2차 테스트 결과 - 제출 완료

baseline 구현과 결과기 템플릿 통합을 완료함. 공개 10문항 채점 결과, 검색 단계 10개 모두 정답 도출함.

생성 단계에서 아래 3건의 문제가 확인됨.

- 자기모순 답변: P08에서 정답 근거를 정확히 인용하고도 결론은 "확인할 수 없음"으로 답변함
- 중국어 출력: P10에서 검색은 정답을 찾았으나 최종 답변 전체가 중국어로 출력됨
- 느린 성능 지표: p50 19.6초, p95 39.0초, throughput 0.091 req/s로 스텁 기준 예시(p50 7.0초) 대비 저하됨
