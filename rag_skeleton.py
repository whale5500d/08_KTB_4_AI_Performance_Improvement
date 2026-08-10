# -*- coding: utf-8 -*-
"""
카카오 약관 RAG 결과기 — 워킹 스켈레톤 (LangGraph)

목적
  단계별 함수 시그니처와 Pydantic 입출력 스키마를 먼저 고정하고,
  인덱싱 → 검색 → 증강 → 생성 → 품질 결과까지 한 번 흐르는 것만 확인한다.
  각 단계 내부는 [STUB] 로그만 남기는 스텁이며, GPU·네트워크 없이 실행된다.

구현 순서(스텁을 하나씩 실제 구현으로 교체)
  1) load_documents      2) parse_articles     3) chunk_articles
  4) embed_chunks        5) build_vector_store 6) retrieve
  7) build_prompt        8) generate           9) score_*

Baseline 목표 구성 (스텁 교체 시 채울 값)
  임베딩 bge-m3 / 저장 FAISS IndexFlatIP / 검색 cosine top-k(k=4)
  청킹 조(article) 단위 / 생성 Qwen2.5-7B-Instruct 4bit
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, Field, field_validator

# =====================================================================================
# 0. 스텁 헬퍼
# =====================================================================================

STUB_LOG: List[str] = []


def stub(stage: str, detail: str = "") -> None:
    """스텁 실행을 기록하고 출력한다. 실제 구현으로 교체할 지점 표시."""
    line = f"[STUB] {stage}" + (f" — {detail}" if detail else "")
    STUB_LOG.append(line)
    print(line)


# =====================================================================================
# 1. 고정 상수
# =====================================================================================

DocName = Literal[
    "카카오계정 약관",
    "카카오 위치정보 이용약관",
    "카카오 통합서비스약관",
    "카카오 통합 약관",
]

OFFICIAL_DOCUMENT_NAMES: Tuple[str, ...] = (
    "카카오계정 약관",
    "카카오 위치정보 이용약관",
    "카카오 통합서비스약관",
    "카카오 통합 약관",
)

REQUIRED_GENERATION_MODEL_FAMILY = "Qwen2.5-Instruct"


def normalize_doc_name(value: Any) -> str:
    """공통 러너 _sp_norm_doc과 동일 규칙 — NFC 정규화 + 공백 제거."""
    import re
    import unicodedata

    return re.sub(r"\s+", "", unicodedata.normalize("NFC", str(value)))


ALLOWED_DOCS_NORM = {normalize_doc_name(d) for d in OFFICIAL_DOCUMENT_NAMES}


# =====================================================================================
# 2. 스키마 — 설정
# =====================================================================================

class DocumentSource(BaseModel):
    """약관 1종의 원문 취득 경로."""

    doc_name: DocName
    urls: List[str] = Field(min_length=1)
    effective_date: str
    note: str = ""


class IndexConfig(BaseModel):
    sources: List[DocumentSource]
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_batch_size: int = 8
    max_chunk_chars: int = 1800
    request_timeout_s: float = 20.0


class RetrievalConfig(BaseModel):
    top_k: int = 4  # 채점 스키마 retrieved 상한과 일치
    query_prefix: str = ""


class GenerationConfig(BaseModel):
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    load_in_4bit: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.0
    max_context_chars: int = 6000


class PipelineConfig(BaseModel):
    index: IndexConfig
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()


# =====================================================================================
# 3. 스키마 — 인덱싱
# =====================================================================================

class RawDocument(BaseModel):
    """취득한 약관 원문 1건 (평문 텍스트)."""

    doc_name: DocName
    text: str
    source_url: str
    fetched_at: str
    char_len: int


class Article(BaseModel):
    """조(條) 단위로 분해된 약관 조항."""

    doc_name: DocName
    article_number: int
    article_title: str = ""
    body: str

    @property
    def citation(self) -> str:
        """골드셋 gold_articles[].citation 과 같은 표기."""
        head = f"{self.doc_name} 제{self.article_number}조"
        return f"{head}({self.article_title})" if self.article_title else head


class Chunk(BaseModel):
    """벡터 저장소 최소 단위. 메타데이터가 채점 스키마와 직접 대응한다."""

    chunk_id: str
    doc_name: DocName
    article_number: int
    article_title: str = ""
    text: str


class EmbeddingBundle(BaseModel):
    """청크 목록과 대응 임베딩 (행 순서 일치)."""

    chunks: List[Chunk]
    vectors: List[List[float]]
    model_name: str
    dim: int


class IndexStats(BaseModel):
    """인덱싱 결과 요약 — 수동 점검용."""

    n_documents: int
    n_articles: int
    n_chunks: int
    dim: int
    per_document: Dict[str, int]
    elapsed_s: float


# =====================================================================================
# 4. 스키마 — 검색 / 증강 / 생성
# =====================================================================================

class RetrievedChunk(BaseModel):
    rank: int
    score: float
    chunk: Chunk


class RetrievalOutput(BaseModel):
    question: str
    hits: List[RetrievedChunk]
    top_k: int
    elapsed_s: float


class PromptBundle(BaseModel):
    """[문서명, 조번호, 본문] 형식으로 조합된 프롬프트."""

    system_prompt: str
    user_prompt: str
    context_block: str
    n_context_chunks: int


class GenerationOutput(BaseModel):
    answer_text: str
    n_new_tokens: int
    elapsed_s: float


class Evidence(BaseModel):
    """retrieved 항목 1건 — [문서명, 조번호] 2원소로 직렬화된다."""

    doc_name: DocName
    article_number: int

    def to_pair(self) -> List[Any]:
        return [self.doc_name, int(self.article_number)]


class AnswerPayload(BaseModel):
    """answer_question()의 최종 반환값. 공통 러너 형식 검사와 1:1 대응."""

    answer: str
    retrieved: List[Evidence] = Field(min_length=1, max_length=4)

    @field_validator("retrieved")
    @classmethod
    def _allowed_docs(cls, v: List[Evidence]) -> List[Evidence]:
        for item in v:
            if normalize_doc_name(item.doc_name) not in ALLOWED_DOCS_NORM:
                raise ValueError(f"허용 목록 밖 문서명: {item.doc_name}")
        return v

    def to_contract(self) -> Dict[str, Any]:
        """공통 러너가 기대하는 순수 dict로 변환."""
        return {"answer": self.answer, "retrieved": [e.to_pair() for e in self.retrieved]}


# =====================================================================================
# 5. 스키마 — 품질 결과 (골드셋 / 제출 파일)
# =====================================================================================

class GoldArticle(BaseModel):
    doc: DocName
    article: int
    citation: str


class GoldQuestion(BaseModel):
    id: str
    question: str
    ptype: str
    difficulty: str
    gold_articles: List[GoldArticle]
    key_facts: List[str]


class GoldSet(BaseModel):
    questions: List[GoldQuestion]
    meta: Dict[str, Any] = Field(default_factory=dict, alias="_meta")

    model_config = ConfigDict(populate_by_name=True)


class SubmissionAnswer(BaseModel):
    """answers_public_<팀>.json 의 answers[] 항목."""

    qid: str
    retrieved: List[List[Any]]
    answer: str
    error: Optional[str] = None


class SubmissionFile(BaseModel):
    """answers_public_example.json 과 동일 구조."""

    team: str
    answers: List[SubmissionAnswer]
    meta: Dict[str, Any] = Field(default_factory=dict)


class ArticleScore(BaseModel):
    """근거 조항 정확 일치 채점 결과."""

    qid: str
    predicted: List[List[Any]]
    gold: List[List[Any]]
    hit_at_1: bool
    hit_at_k: bool
    n_gold_matched: int
    n_gold_total: int


class KeyFactScore(BaseModel):
    """정답 핵심 사실 포함 여부. covered는 수동 대조로 확정한다."""

    qid: str
    n_key_facts: int
    n_covered_auto: int
    coverage_auto: float
    per_fact: List[Dict[str, Any]]
    needs_manual_review: bool = True


class ItemReport(BaseModel):
    qid: str
    question: str
    difficulty: str
    ptype: str
    article: ArticleScore
    key_fact: KeyFactScore
    answer_text: str


class EvalReport(BaseModel):
    """공개 10문항 자체 채점 종합."""

    n_items: int
    article_hit_at_1_rate: float
    article_hit_at_k_rate: float
    key_fact_coverage_mean: float
    items: List[ItemReport]


class PerfProtocol(BaseModel):
    """공식 프로토콜과 동일 조건 (Upgrade 단계에서 사용)."""

    requests_per_run: int = 12
    concurrency: int = 2
    warmup_requests: int = 2
    repetitions: int = 3


class PerfReport(BaseModel):
    protocol: PerfProtocol
    success_rate: float
    throughput_rps: float
    p50_latency_s: Optional[float]
    p95_latency_s: Optional[float]


# =====================================================================================
# 6. 인덱싱 — 로딩 및 가져오기  [STUB]
# =====================================================================================

DEFAULT_SOURCES: List[DocumentSource] = [
    DocumentSource(
        doc_name="카카오계정 약관",
        urls=["https://www.kakao.com/policy/terms?lang=ko"],
        effective_date="2026-05-29",
        note="URL 실제 접근 가능 여부 확인 필요",
    ),
    DocumentSource(
        doc_name="카카오 위치정보 이용약관",
        urls=["https://www.kakao.com/policy/location?lang=ko"],
        effective_date="2026-07-16",
    ),
    DocumentSource(
        doc_name="카카오 통합서비스약관",
        urls=["https://www.kakao.com/policy/servterms?lang=ko"],
        effective_date="2026-05-29",
        note="URL 실제 접근 가능 여부 확인 필요",
    ),
    DocumentSource(
        doc_name="카카오 통합 약관",
        urls=["https://www.kakao.com/policy/total?lang=ko"],
        effective_date="2022-08-25",
        note="운영진 배포 아카이브 링크로 교체할 것",
    ),
]


def fetch_document(source: DocumentSource, timeout_s: float = 20.0) -> RawDocument:
    """DocumentSource → RawDocument.

    TODO(baseline): requests.get + BeautifulSoup 로 HTML → 평문 변환.
    """
    stub("fetch_document", f"{source.doc_name} ← {source.urls[0]}")
    return RawDocument(
        doc_name=source.doc_name,
        text=f"제1조(목적) {source.doc_name} 더미 본문",
        source_url=source.urls[0],
        fetched_at="1970-01-01T00:00:00Z",
        char_len=0,
    )


def load_documents(
    sources: List[DocumentSource],
    timeout_s: float = 20.0,
) -> List[RawDocument]:
    """List[DocumentSource] → List[RawDocument]."""
    stub("load_documents", f"{len(sources)}종 약관")
    return [fetch_document(s, timeout_s) for s in sources]


# =====================================================================================
# 7. 인덱싱 — 파싱  [STUB]
# =====================================================================================

def parse_articles(document: RawDocument) -> List[Article]:
    """RawDocument → List[Article].

    TODO(baseline): r"제\\s*(\\d+)\\s*조\\s*(?:\\(([^)]*)\\))?" 로 조 경계 분할.
    """
    stub("parse_articles", document.doc_name)
    return [
        Article(
            doc_name=document.doc_name,
            article_number=n,
            article_title=f"더미조항{n}",
            body=f"{document.doc_name} 제{n}조 더미 본문",
        )
        for n in (1, 2)
    ]


# =====================================================================================
# 8. 인덱싱 — 청킹  [STUB]
# =====================================================================================

def chunk_articles(articles: List[Article], config: IndexConfig) -> List[Chunk]:
    """List[Article] → List[Chunk]. 조 단위 1청크, 초과분만 길이 분할.

    TODO(baseline): max_chunk_chars 초과 시 분할, chunk_id 규칙 확정.
    """
    stub("chunk_articles", f"{len(articles)}개 조항 → 청크")
    return [
        Chunk(
            chunk_id=f"{normalize_doc_name(a.doc_name)}-{a.article_number:03d}-00",
            doc_name=a.doc_name,
            article_number=a.article_number,
            article_title=a.article_title,
            text=f"[{a.doc_name}] 제{a.article_number}조({a.article_title})\n{a.body}",
        )
        for a in articles
    ]


# =====================================================================================
# 9. 인덱싱 — 임베딩  [STUB]
# =====================================================================================

class EmbedderHandle(BaseModel):
    """임베딩 모델 핸들. 실제 구현에서는 SentenceTransformer 인스턴스를 보유한다."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str
    dim: int
    device: str = "cpu"
    model: Any = None


def build_embedder(config: IndexConfig) -> EmbedderHandle:
    """IndexConfig → EmbedderHandle.

    TODO(baseline): SentenceTransformer(config.embedding_model_name, device="cuda").
    """
    stub("build_embedder", config.embedding_model_name)
    return EmbedderHandle(model_name=config.embedding_model_name, dim=8)


def embed_chunks(
    chunks: List[Chunk],
    embedder: EmbedderHandle,
    config: IndexConfig,
) -> EmbeddingBundle:
    """List[Chunk] → EmbeddingBundle. L2 정규화하여 내적=cosine이 되게 한다.

    TODO(baseline): embedder.encode(..., normalize_embeddings=True).
    """
    stub("embed_chunks", f"{len(chunks)}개 청크 · dim={embedder.dim}")
    return EmbeddingBundle(
        chunks=chunks,
        vectors=[[0.0] * embedder.dim for _ in chunks],
        model_name=embedder.model_name,
        dim=embedder.dim,
    )


# =====================================================================================
# 10. 인덱싱 — 저장  [STUB]
# =====================================================================================

class VectorStoreHandle(BaseModel):
    """FAISS in-memory 저장소 핸들."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: str = "faiss.IndexFlatIP"
    dim: int
    n_vectors: int
    chunks: List[Chunk]
    index: Any = None

    def search(self, query_vector: List[float], top_k: int) -> List[RetrievedChunk]:
        """질의 벡터 → 상위 top_k 청크.

        TODO(baseline): self.index.search(query_vector, top_k).
        """
        stub("VectorStoreHandle.search", f"top_k={top_k}")
        selected = self.chunks[:top_k]
        return [
            RetrievedChunk(rank=i, score=round(1.0 - 0.1 * i, 3), chunk=c)
            for i, c in enumerate(selected, start=1)
        ]


def build_vector_store(bundle: EmbeddingBundle) -> VectorStoreHandle:
    """EmbeddingBundle → VectorStoreHandle.

    TODO(baseline): faiss.IndexFlatIP(dim) 생성 후 add.
    """
    stub("build_vector_store", f"{len(bundle.chunks)}개 벡터 · dim={bundle.dim}")
    return VectorStoreHandle(
        dim=bundle.dim, n_vectors=len(bundle.chunks), chunks=bundle.chunks
    )


def build_index(config: IndexConfig) -> Tuple[VectorStoreHandle, EmbedderHandle, IndexStats]:
    """인덱싱 오케스트레이션: 로딩 → 파싱 → 청킹 → 임베딩 → 저장."""
    started = time.perf_counter()
    documents = load_documents(config.sources, config.request_timeout_s)

    articles: List[Article] = []
    for document in documents:
        articles.extend(parse_articles(document))

    chunks = chunk_articles(articles, config)
    embedder = build_embedder(config)
    bundle = embed_chunks(chunks, embedder, config)
    store = build_vector_store(bundle)

    per_document: Dict[str, int] = {}
    for chunk in chunks:
        per_document[chunk.doc_name] = per_document.get(chunk.doc_name, 0) + 1

    stats = IndexStats(
        n_documents=len(documents),
        n_articles=len(articles),
        n_chunks=len(chunks),
        dim=store.dim,
        per_document=per_document,
        elapsed_s=round(time.perf_counter() - started, 4),
    )
    return store, embedder, stats


# =====================================================================================
# 11. 검색  [STUB]
# =====================================================================================

def embed_query(
    question: str,
    embedder: EmbedderHandle,
    config: RetrievalConfig,
) -> List[float]:
    """질문 문자열 → 정규화된 질의 벡터.

    TODO(baseline): embedder.encode([prefix+question], normalize_embeddings=True)[0].
    """
    stub("embed_query", question[:24] + "…")
    return [0.0] * embedder.dim


def retrieve(
    question: str,
    store: VectorStoreHandle,
    embedder: EmbedderHandle,
    config: RetrievalConfig,
) -> RetrievalOutput:
    """질문 → RetrievalOutput (cosine top-k)."""
    started = time.perf_counter()
    query_vector = embed_query(question, embedder, config)
    hits = store.search(query_vector, config.top_k)
    return RetrievalOutput(
        question=question,
        hits=hits,
        top_k=config.top_k,
        elapsed_s=round(time.perf_counter() - started, 4),
    )


# =====================================================================================
# 12. 증강  [STUB]
# =====================================================================================

SYSTEM_PROMPT = (
    "당신은 카카오 약관 전문 어시스턴트입니다. 제공된 약관 조문만을 근거로 답하십시오. "
    "근거에 없는 내용은 지어내지 말고, 숫자·기간·조문 번호는 근거 그대로 옮기십시오."
)


def build_prompt(retrieval: RetrievalOutput, config: GenerationConfig) -> PromptBundle:
    """RetrievalOutput → PromptBundle. [문서명, 조번호, 본문] 형식으로 조합.

    TODO(baseline): max_context_chars 예산 안에서 근거 블록 절단.
    """
    stub("build_prompt", f"{len(retrieval.hits)}개 근거")
    blocks = [
        f"[근거 {h.rank}] 문서명: {h.chunk.doc_name} / 조번호: 제{h.chunk.article_number}조\n"
        f"본문: {h.chunk.text}"
        for h in retrieval.hits
    ]
    context_block = "\n\n".join(blocks)
    return PromptBundle(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"{context_block}\n\n질문: {retrieval.question}",
        context_block=context_block,
        n_context_chunks=len(blocks),
    )


# =====================================================================================
# 13. 생성  [STUB]
# =====================================================================================

class GeneratorHandle(BaseModel):
    """Qwen2.5-Instruct 로컬 생성 모델 핸들."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str
    load_in_4bit: bool
    model: Any = None
    tokenizer: Any = None


def load_generator(config: GenerationConfig) -> GeneratorHandle:
    """GenerationConfig → GeneratorHandle.

    TODO(baseline): AutoModelForCausalLM + BitsAndBytesConfig(load_in_4bit=True).
    """
    if REQUIRED_GENERATION_MODEL_FAMILY.split("-")[0] not in config.model_name:
        raise ValueError(f"생성 모델은 {REQUIRED_GENERATION_MODEL_FAMILY} 계열이어야 합니다.")
    stub("load_generator", f"{config.model_name} (4bit={config.load_in_4bit})")
    return GeneratorHandle(model_name=config.model_name, load_in_4bit=config.load_in_4bit)


def generate(prompt: PromptBundle, generator: GeneratorHandle) -> GenerationOutput:
    """PromptBundle → GenerationOutput.

    TODO(baseline): apply_chat_template → model.generate → decode.
    """
    started = time.perf_counter()
    stub("generate", f"{generator.model_name} · 근거 {prompt.n_context_chunks}개")
    return GenerationOutput(
        answer_text="[STUB 답변] 생성 모델 연결 전 더미 응답입니다.",
        n_new_tokens=0,
        elapsed_s=round(time.perf_counter() - started, 4),
    )


def select_evidence(retrieval: RetrievalOutput, max_items: int = 4) -> List[Evidence]:
    """RetrievalOutput → 관련도 순 Evidence 1~4개. (doc, article) 중복 제거."""
    evidence: List[Evidence] = []
    seen: set[Tuple[str, int]] = set()
    for hit in retrieval.hits:
        key = (hit.chunk.doc_name, hit.chunk.article_number)
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            Evidence(doc_name=hit.chunk.doc_name, article_number=hit.chunk.article_number)
        )
        if len(evidence) >= max_items:
            break
    if not evidence:  # retrieved는 최소 1개여야 형식 검사를 통과한다
        evidence.append(Evidence(doc_name="카카오 통합 약관", article_number=1))
    return evidence


# =====================================================================================
# 14. LangGraph 오케스트레이션
# =====================================================================================

class RagState(BaseModel):
    """LangGraph 상태. 각 노드가 부분 갱신을 반환한다."""

    question: str
    retrieval: Optional[RetrievalOutput] = None
    prompt: Optional[PromptBundle] = None
    generation: Optional[GenerationOutput] = None
    payload: Optional[AnswerPayload] = None


class PipelineContext(BaseModel):
    """그래프 노드가 공유하는 런타임 리소스."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: VectorStoreHandle
    embedder: EmbedderHandle
    generator: GeneratorHandle
    config: PipelineConfig


def make_retrieve_node(ctx: PipelineContext):
    def retrieve_node(state: RagState) -> Dict[str, Any]:
        return {"retrieval": retrieve(state.question, ctx.store, ctx.embedder, ctx.config.retrieval)}

    return retrieve_node


def make_augment_node(ctx: PipelineContext):
    def augment_node(state: RagState) -> Dict[str, Any]:
        assert state.retrieval is not None
        return {"prompt": build_prompt(state.retrieval, ctx.config.generation)}

    return augment_node


def make_generate_node(ctx: PipelineContext):
    def generate_node(state: RagState) -> Dict[str, Any]:
        assert state.prompt is not None
        return {"generation": generate(state.prompt, ctx.generator)}

    return generate_node


def make_finalize_node(ctx: PipelineContext):
    def finalize_node(state: RagState) -> Dict[str, Any]:
        assert state.retrieval is not None and state.generation is not None
        payload = AnswerPayload(
            answer=state.generation.answer_text or "제공된 약관 조문에서 확인할 수 없습니다.",
            retrieved=select_evidence(state.retrieval, ctx.config.retrieval.top_k),
        )
        return {"payload": payload}

    return finalize_node


def build_graph(ctx: PipelineContext):
    """PipelineContext → 컴파일된 LangGraph 그래프."""
    graph = StateGraph(RagState)
    graph.add_node("retrieve", make_retrieve_node(ctx))
    graph.add_node("augment", make_augment_node(ctx))
    graph.add_node("generate", make_generate_node(ctx))
    graph.add_node("finalize", make_finalize_node(ctx))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "augment")
    graph.add_edge("augment", "generate")
    graph.add_edge("generate", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


# =====================================================================================
# 15. 부팅 + 고정 진입점
# =====================================================================================

PIPELINE_CONFIG = PipelineConfig(index=IndexConfig(sources=DEFAULT_SOURCES))

_STORE: Optional[VectorStoreHandle] = None
_EMBEDDER: Optional[EmbedderHandle] = None
_GENERATOR: Optional[GeneratorHandle] = None
_GRAPH: Optional[Any] = None
INDEX_STATS: Optional[IndexStats] = None


def bootstrap(config: PipelineConfig = PIPELINE_CONFIG) -> None:
    """인덱스 + 생성 모델 + 그래프를 준비한다. 셀 실행 시 1회 호출."""
    global _STORE, _EMBEDDER, _GENERATOR, _GRAPH, INDEX_STATS

    _STORE, _EMBEDDER, INDEX_STATS = build_index(config.index)
    print(f"[인덱싱 완료] {INDEX_STATS.model_dump()}")

    _GENERATOR = load_generator(config.generation)
    ctx = PipelineContext(
        store=_STORE, embedder=_EMBEDDER, generator=_GENERATOR, config=config
    )
    _GRAPH = build_graph(ctx)
    print("[부팅 완료] answer_question() 호출 준비됨")


def answer_question(question: str) -> Dict[str, Any]:
    """공통 러너 고정 진입점.

    Input : question (비어 있지 않은 str)
    Output: {"answer": str, "retrieved": [[문서명, 조번호], ...]}  (retrieved 1~4개)
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question은 비어 있지 않은 문자열이어야 합니다.")
    if _GRAPH is None:
        raise RuntimeError("bootstrap()이 완료되지 않았습니다.")

    final_state = _GRAPH.invoke(RagState(question=question.strip()))
    payload = final_state["payload"] if isinstance(final_state, dict) else final_state.payload
    if isinstance(payload, dict):
        payload = AnswerPayload(**payload)
    return payload.to_contract()


# =====================================================================================
# 16. 품질 결과 — 골드셋 채점  [STUB]
# =====================================================================================

def load_gold_set(path: str | Path) -> GoldSet:
    """gold_questions_public10.json → GoldSet."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    gold = GoldSet(questions=data["questions"], _meta=data.get("_meta", {}))
    print(f"[골드셋] {len(gold.questions)}문항 로드")
    return gold


def load_submission(path: str | Path) -> SubmissionFile:
    """answers_public_<팀>.json → SubmissionFile."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sub = SubmissionFile(**data)
    print(f"[제출본] team={sub.team} · {len(sub.answers)}문항 로드")
    return sub


def score_article_exact_match(gold: GoldQuestion, answer: SubmissionAnswer) -> ArticleScore:
    """근거 조항 정확 일치 채점.

    TODO(baseline): normalize_doc_name 기준으로 (doc, article) 집합 대조,
                    hit@1 / hit@k / 매칭 개수 산출.
    """
    stub("score_article_exact_match", gold.id)
    return ArticleScore(
        qid=gold.id,
        predicted=answer.retrieved,
        gold=[[g.doc, g.article] for g in gold.gold_articles],
        hit_at_1=False,
        hit_at_k=False,
        n_gold_matched=0,
        n_gold_total=len(gold.gold_articles),
    )


def score_key_fact_coverage(gold: GoldQuestion, answer: SubmissionAnswer) -> KeyFactScore:
    """정답 핵심 사실 포함 비율.

    TODO(baseline): 키팩트별 핵심 토큰/수치 포함 여부를 자동 판정하고,
                    per_fact에 근거 문장을 담아 수동 대조용으로 남긴다.
    """
    stub("score_key_fact_coverage", f"{gold.id} · 키팩트 {len(gold.key_facts)}개")
    return KeyFactScore(
        qid=gold.id,
        n_key_facts=len(gold.key_facts),
        n_covered_auto=0,
        coverage_auto=0.0,
        per_fact=[{"fact": f, "covered": None} for f in gold.key_facts],
        needs_manual_review=True,
    )


def evaluate(gold: GoldSet, submission: SubmissionFile) -> EvalReport:
    """GoldSet + SubmissionFile → EvalReport."""
    by_qid = {a.qid: a for a in submission.answers}
    items: List[ItemReport] = []

    for question in gold.questions:
        answer = by_qid.get(question.id)
        if answer is None:
            print(f"[경고] {question.id} 답변 누락")
            continue
        items.append(
            ItemReport(
                qid=question.id,
                question=question.question,
                difficulty=question.difficulty,
                ptype=question.ptype,
                article=score_article_exact_match(question, answer),
                key_fact=score_key_fact_coverage(question, answer),
                answer_text=answer.answer,
            )
        )

    n = max(1, len(items))
    return EvalReport(
        n_items=len(items),
        article_hit_at_1_rate=round(sum(i.article.hit_at_1 for i in items) / n, 4),
        article_hit_at_k_rate=round(sum(i.article.hit_at_k for i in items) / n, 4),
        key_fact_coverage_mean=round(sum(i.key_fact.coverage_auto for i in items) / n, 4),
        items=items,
    )


def render_manual_review(report: EvalReport) -> str:
    """EvalReport → 수동 대조용 마크다운 표.

    TODO(baseline): 문항별 gold citation·키팩트·생성 답변을 나란히 배치.
    """
    stub("render_manual_review", f"{report.n_items}문항")
    lines = [
        "| qid | 난이도 | hit@1 | hit@k | 키팩트 커버리지 |",
        "|---|---|---|---|---|",
    ]
    for item in report.items:
        lines.append(
            f"| {item.qid} | {item.difficulty} | {item.article.hit_at_1} | "
            f"{item.article.hit_at_k} | {item.key_fact.coverage_auto} |"
        )
    return "\n".join(lines)


# =====================================================================================
# 17. 품질 결과 — 성능 사전 검증  [STUB · Upgrade]
# =====================================================================================

def measure_performance(protocol: PerfProtocol = PerfProtocol()) -> PerfReport:
    """공식 프로토콜과 동일 조건으로 p50/p95/throughput 자체 측정.

    TODO(upgrade): ThreadPoolExecutor로 concurrency만큼 동시 POST /answer 호출,
                   repetitions 회 반복 후 중앙값 집계.
    """
    stub("measure_performance", protocol.model_dump_json())
    return PerfReport(
        protocol=protocol,
        success_rate=0.0,
        throughput_rps=0.0,
        p50_latency_s=None,
        p95_latency_s=None,
    )


# =====================================================================================
# 18. 고정 FastAPI 연결 영역
# =====================================================================================

def create_app():
    """전역 FastAPI app 생성. GET /health, POST /answer 고정."""
    from fastapi import FastAPI, HTTPException

    fastapi_app = FastAPI(title="KTB AI Performance Result Generator")
    lock = threading.Lock()

    @fastapi_app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @fastapi_app.post("/answer")
    def answer_api(payload: dict) -> Dict[str, Any]:
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            raise HTTPException(status_code=400, detail="question must be a non-empty string")
        with lock:
            return answer_question(question.strip())

    return fastapi_app


try:
    app = create_app()
except ImportError:
    app = None
    print("[주의] fastapi 미설치 — 스켈레톤 검증 모드로 app=None")


# =====================================================================================
# 19. 스켈레톤 흐름 확인
# =====================================================================================

def smoke_test(gold_path: Optional[str] = None, answers_path: Optional[str] = None) -> None:
    """인덱싱 → 검색 → 증강 → 생성 → 품질 결과까지 한 번 흘려본다."""
    print("=" * 70)
    bootstrap()

    print("=" * 70)
    result = answer_question("사업자/단체 카카오계정은 담당자 몇 명이 이용할 수 있나요?")
    print(f"[answer_question 반환] {json.dumps(result, ensure_ascii=False)}")

    if gold_path and answers_path:
        print("=" * 70)
        report = evaluate(load_gold_set(gold_path), load_submission(answers_path))
        print(render_manual_review(report))
        print("=" * 70)
        print(measure_performance().model_dump())

    print("=" * 70)
    print(f"[스켈레톤 완료] 스텁 호출 {len(STUB_LOG)}회 — 위 [STUB] 지점을 순서대로 구현하세요.")


if __name__ == "__main__":
    import sys as _sys

    smoke_test(*(_sys.argv[1:3] if len(_sys.argv) >= 3 else (None, None)))
