# -*- coding: utf-8 -*-
"""pipeline.py 회귀 테스트 · GPU/네트워크 없이 도는 구간만 검증한다.

실행: python tests_pipeline.py
검증 대상: 파싱, 항 분할, 청킹, RRF 융합, 하이브리드 검색 순서, 프롬프트 조립,
          답변 후처리, 추출식 폴백, 근거 선택, 반환 계약.
"""

import sys
import types

# langgraph는 Colab에서만 설치하므로 테스트 환경에서는 최소 스텁으로 대체한다.
if "langgraph" not in sys.modules:
    langgraph = types.ModuleType("langgraph")
    graph_module = types.ModuleType("langgraph.graph")

    class _StubGraph:
        def __init__(self, *args, **kwargs):
            self.nodes = {}

        def add_node(self, name, fn):
            self.nodes[name] = fn

        def set_entry_point(self, name):
            self.entry = name

        def add_edge(self, *args):
            pass

        def compile(self):
            return self

    graph_module.StateGraph = _StubGraph
    graph_module.END = "__end__"
    langgraph.graph = graph_module
    sys.modules["langgraph"] = langgraph
    sys.modules["langgraph.graph"] = graph_module

import numpy as np  # noqa: E402

import pipeline as P  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS · {name}")
    else:
        print(f"  FAIL · {name} {detail}")
        FAILURES.append(name)


SAMPLE_TEXT = """카카오 통합 약관
제1조(목적)
제2조(정의)
제3조(약관 외 준칙)
제1조(목적)
① 본 약관은 회사가 제공하는 서비스의 이용조건을 정합니다.
② 여러분과 회사의 권리와 의무를 규정합니다.
제2조(정의)
① 이 약관에서 사용하는 용어의 뜻은 다음과 같습니다.
1. 서비스란 회사가 제공하는 모든 서비스를 말합니다.
2. 회원이란 서비스에 가입한 자를 말합니다.
제3조(약관 외 준칙)
본 약관에 규정되지 않은 사항에 대해서는 관련법령 또는 회사가 정한 서비스의 개별 이용약관, 운영정책 및 규칙 등(이하 세부지침)의 규정에 따릅니다. 또한 본 약관과 세부지침의 내용이 충돌할 경우 세부지침에 따릅니다.
부칙
이 약관은 2022년 8월 25일부터 시행합니다.
"""


def test_parse_and_chunk():
    print("[1] 파싱·항 분할·청킹")
    document = P.RawDocument(
        doc_name="카카오 통합 약관",
        text=SAMPLE_TEXT,
        source_url="test://sample",
        fetched_at="2026-08-11T00:00:00+00:00",
        char_len=len(SAMPLE_TEXT),
    )
    articles = P.parse_articles(document)
    check("조 3개 추출", len(articles) == 3, f"(실제 {len(articles)})")
    check("제목 추출", articles[2].article_title == "약관 외 준칙", articles[2].article_title)
    check("목차가 아닌 본문 채택", "세부지침" in articles[2].body)
    check("부칙 제거", "시행합니다" not in articles[2].body, articles[2].body[-40:])

    paragraphs = P.split_article_paragraphs(articles[0].body, 10)
    check("항 기호 분할", len(paragraphs) == 2, f"(실제 {len(paragraphs)})")
    check("항 라벨 유지", paragraphs[0][0] == "①", paragraphs[0][0])
    merged = P.split_article_paragraphs(articles[0].body, 200)
    check("짧은 항은 앞 항에 병합", len(merged) == 1, f"(실제 {len(merged)})")
    check("병합 시 내용 보존", "권리와 의무" in merged[0][1])

    config = P.IndexConfig(sources=P.DEFAULT_SOURCES)
    chunks = P.chunk_articles(articles, config)
    check("항 단위 청크 수 >= 조 수", len(chunks) >= len(articles), f"(실제 {len(chunks)})")
    check("레지스트리 등록", ("카카오 통합 약관", 3) in P.ARTICLE_REGISTRY)
    check("청크에 문서명 머리말", chunks[0].text.startswith("[카카오 통합 약관]"))
    return articles, chunks


def test_rrf():
    print("[2] RRF 융합")
    fused = P.reciprocal_rank_fusion([[5, 1, 2], [1, 5, 9]], rrf_k=60)
    check("양쪽 상위 문서가 최고점", max(fused, key=fused.get) in (1, 5))
    check("한쪽에만 있는 문서도 포함", 9 in fused)
    single = P.reciprocal_rank_fusion([[7]], rrf_k=60)
    check("단일 목록 처리", abs(single[7] - 1 / 61) < 1e-9)


class _FakeIndex:
    """FAISS 대체 · 질의 벡터와의 내적 상위를 돌려준다."""

    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype="float32")

    def search(self, query, top_k):
        scores = self.vectors @ query[0]
        order = np.argsort(scores)[::-1][:top_k]
        return np.asarray([scores[order]]), np.asarray([order])


class _FakeEmbedderModel:
    """청크 텍스트의 문자 바이그램 해시를 벡터로 쓰는 결정적 더미 임베더."""

    dim = 64

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            vector = np.zeros(self.dim, dtype="float32")
            for bigram in P.character_bigrams(text):
                vector[hash(bigram) % self.dim] += 1.0
            norm = np.linalg.norm(vector) or 1.0
            vectors.append(vector / norm)
        return np.asarray(vectors, dtype="float32")

    def get_sentence_embedding_dimension(self):
        return self.dim


def test_retrieve(chunks):
    print("[3] 하이브리드 검색")
    embedder = P.EmbedderHandle(model_name="fake", dim=64, device="cpu", model=_FakeEmbedderModel())
    vectors = embedder.model.encode([c.text for c in chunks])
    store = P.VectorStoreHandle(
        dim=64, n_vectors=len(chunks), chunks=chunks, index=_FakeIndex(vectors)
    )
    P._LEXICAL_INDEX = P.build_lexical_index(chunks)
    config = P.RetrievalConfig(use_reranker=False)

    output = P.retrieve("약관과 세부지침의 내용이 충돌하면 무엇에 따르나요?", store, embedder, config)
    check("검색 결과 존재", len(output.hits) >= 1)
    check(
        "정답 조가 1순위",
        output.hits[0].chunk.article_number == 3,
        f"(실제 제{output.hits[0].chunk.article_number}조)",
    )
    check("조 단위 중복 없음", len({(h.chunk.doc_name, h.chunk.article_number) for h in output.hits})
          == len(output.hits))

    evidence = P.select_evidence(output, 4)
    check("근거 1~4개", 1 <= len(evidence) <= 4, f"(실제 {len(evidence)})")
    payload = P.AnswerPayload(answer="테스트 답변입니다.", retrieved=evidence)
    contract = payload.to_contract()
    check(
        "반환 계약 형식",
        isinstance(contract["retrieved"][0][0], str) and isinstance(contract["retrieved"][0][1], int),
        str(contract["retrieved"][0]),
    )
    return output


def test_prompt(output):
    print("[4] 프롬프트 조립")
    generation_config = P.GenerationConfig()
    bundle = P.build_prompt(output, generation_config)
    check("컨텍스트 중복 삽입 없음", bundle.user_prompt.count("[근거 1]") == 1)
    check("조 전체 확장", "세부지침에 따릅니다" in bundle.context_block)
    check("예산 준수", len(bundle.context_block) <= generation_config.max_context_chars)
    citation = P.primary_citation(bundle.context_block)
    check("대표 인용 추출", citation.startswith("(카카오 통합 약관 제"), citation)


def test_polish():
    print("[5] 답변 후처리")
    config = P.GenerationConfig()
    citation = "(카카오 통합 약관 제3조)"

    text = (
        "답변: 아니오. 본 약관과 세부지침의 내용이 충돌할 경우 세부지침에 따릅니다. "
        + P.ABSTENTION_SENTENCE
    )
    polished = P.polish_answer(text, citation, config)
    check("메타 접두 제거", not polished.startswith("답변:"), polished[:20])
    check("모순 회피 문구 제거", P.ABSTENTION_SENTENCE not in polished, polished)
    check("인용 표기 부착", polished.endswith(citation), polished[-30:])

    duplicated = "같은 문장입니다. 같은 문장입니다. 다른 문장입니다."
    check("중복 문장 제거", P.polish_answer(duplicated, "", config).count("같은 문장입니다.") == 1)

    only_abstain = P.polish_answer(P.ABSTENTION_SENTENCE, "", config)
    check("근거 없을 때 회피 문구 유지", P.ABSTENTION_SENTENCE in only_abstain, only_abstain)

    check("한자 비율 계산", P.han_ratio("根据条款") > 0.9)
    check("한글 비율 계산", P.hangul_ratio("한국어 문장입니다") > 0.7)


def test_fallback(output):
    print("[6] 추출식 폴백")
    config = P.GenerationConfig()
    answer = P.build_extractive_answer(output, config)
    check("폴백 비어 있지 않음", len(answer) > 20, answer[:40])
    check("원문 문장 사용", "세부지침" in answer, answer[:80])
    check("한국어", P.hangul_ratio(answer) > 0.4)
    check("인용 표기 포함", "제3조" in answer, answer[-40:])


def test_document_hint():
    print("[7] 문서명 힌트")
    check(
        "약관명 직접 지목 감지",
        P.detect_document_hint("카카오계정 약관에서 정한 내용은?") == "카카오계정 약관",
    )
    check("공백 변형 감지", P.detect_document_hint("카카오 통합서비스 약관의 정의는?") is not None)
    check("무관한 질문은 None", P.detect_document_hint("서비스 중단 시 어떻게 알리나요?") is None)


if __name__ == "__main__":
    articles, chunks = test_parse_and_chunk()
    test_rrf()
    output = test_retrieve(chunks)
    test_prompt(output)
    test_polish()
    test_fallback(output)
    test_document_hint()

    print()
    if FAILURES:
        print(f"실패 {len(FAILURES)}건: {FAILURES}")
        sys.exit(1)
    print("전체 통과")