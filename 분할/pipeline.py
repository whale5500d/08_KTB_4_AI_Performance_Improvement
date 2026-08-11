# -*- coding: utf-8 -*-
"""
카카오 약관 RAG 결과기 — Baseline 구현본 (LangGraph)

스켈레톤의 모든 [STUB] 지점을 실제 로직으로 교체했다.
함수 시그니처와 스키마는 스켈레톤 원본을 그대로 유지하며, 각 함수가 필요로 하는
import는 모두 함수 본문 안에 둔다.

Baseline 구성
  로딩 requests + BeautifulSoup / 파싱 조 번호 정규식(순번 검증)
  청킹 조(article) 단위 / 임베딩 bge-m3 / 저장 FAISS IndexFlatIP
  검색 cosine top-k(k=4) / 생성 Qwen2.5-7B-Instruct 4bit
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    """약관 1종의 원문 취득 경로. local_path가 있으면 로컬 파일을 우선 사용한다."""

    doc_name: DocName
    urls: List[str] = Field(default_factory=list)
    local_path: Optional[str] = None
    effective_date: str
    note: str = ""

    @model_validator(mode="after")
    def _require_source(self) -> "DocumentSource":
        if not self.urls and not self.local_path:
            raise ValueError(f"[{self.doc_name}] urls 또는 local_path 중 하나는 필요합니다.")
        return self


class IndexConfig(BaseModel):
    sources: List[DocumentSource]
    embedding_model_name: str = "intfloat/multilingual-e5-large"
    embedding_batch_size: int = 8
    max_chunk_chars: int = 1800
    request_timeout_s: float = 20.0


class RetrievalConfig(BaseModel):
    top_k: int = 4  # 채점 스키마 retrieved 상한과 일치
    query_prefix: str = "query"


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
    """run_rag_pipeline()의 최종 반환값. 공통 러너 형식 검사와 1:1 대응."""

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
# 6. 인덱싱 — 로딩 및 가져오기
# =====================================================================================

DEFAULT_SOURCES: List[DocumentSource] = [
    DocumentSource(
        doc_name="카카오계정 약관",
        urls=[
            "https://www.kakao.com/policy/terms?lang=ko",
            "https://qr.kakao.com/policy/terms?lang=ko",
            "https://t1.kakaocdn.net/kakaocorp/pw/policy/files/카카오계정약관.pdf",
        ],
        effective_date="2026-05-29",
        note="본문 컨테이너는 div.wrap_terms.wrap_policy로 확인됨. PDF는 미검증 폴백.",
    ),
    DocumentSource(
        doc_name="카카오 위치정보 이용약관",
        urls=[
            "https://www.kakao.com/policy/location?lang=ko",
            "https://qr.kakao.com/policy/location?lang=ko",
        ],
        effective_date="2026-07-16",
        note="본문 컨테이너는 div.wrap_terms(단독)로 확인됨. PDF 폴백 경로는 미확보.",
    ),
    DocumentSource(
        doc_name="카카오 통합서비스약관",
        urls=[
            "https://www.kakao.com/policy/terms?type=ts&lang=ko",
            "https://qr.kakao.com/policy/terms?type=ts&lang=ko",
        ],
        effective_date="2026-05-29",
        note="본문 컨테이너는 div.wrap_terms.wrap_policy로 확인됨. PDF 폴백 경로는 미확보.",
    ),
    DocumentSource(
        doc_name="카카오 통합 약관",
        urls=[
            "https://www.kakao.com/policy/kakaoTerms?lang=ko",
        ],
        effective_date="2022-08-25",
    ),
]


def _load_local_file_text(local_path: str) -> Tuple[str, str]:
    """로컬 PDF/DOCX 경로 → (원문 텍스트, 실제로 읽은 경로).

    Colab에서 노트북과 같은 디렉터리 또는 /content 아래에 파일이 있다고 가정한다.
    """
    from pathlib import Path

    candidates = [Path(local_path), Path("/content") / local_path]
    resolved = next((path for path in candidates if path.exists()), None)
    if resolved is None:
        tried = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"로컬 파일을 찾을 수 없습니다. 시도한 경로: {tried}")

    suffix = resolved.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(resolved))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix == ".docx":
        from docx import Document

        document = Document(str(resolved))
        text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    else:
        raise ValueError(f"지원하지 않는 로컬 파일 형식: {suffix} ({resolved})")

    return text, str(resolved)


def fetch_document(source: DocumentSource, timeout_s: float = 20.0) -> RawDocument:
    """DocumentSource → RawDocument.

    local_path가 있으면 로컬 PDF/DOCX를 우선 사용한다. 없으면 urls를 순서대로
    시도해 조 구조(제N조)가 확인되는 첫 응답을 채택한다. HTML은 script/style
    제거 후 블록 경계를 개행으로 살려 평문화한다.
    """
    import datetime
    import re

    article_pattern = re.compile(r"제\s*\d+\s*조")

    def finalize(raw_text: str) -> str:
        cleaned = raw_text.replace("\u00a0", " ")
        lines = [line.strip() for line in cleaned.split("\n")]
        return "\n".join(line for line in lines if line)

    if source.local_path:
        raw_text, resolved_path = _load_local_file_text(source.local_path)
        text = finalize(raw_text)
        if len(article_pattern.findall(text)) < 3:
            raise RuntimeError(
                f"[{source.doc_name}] 조 구조 미검출 · 경로={resolved_path} · len={len(text)}"
            )
        return RawDocument(
            doc_name=source.doc_name,
            text=text,
            source_url=resolved_path,
            fetched_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            char_len=len(text),
        )

    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
    failures: List[str] = []

    def is_pdf_response(url: str, response: "requests.Response") -> bool:
        content_type = response.headers.get("Content-Type", "").lower()
        return "application/pdf" in content_type or url.lower().endswith(".pdf")

    for url in source.urls:
        try:
            response = requests.get(url, headers=headers, timeout=timeout_s)
            response.raise_for_status()

            if is_pdf_response(url, response):
                from io import BytesIO

                from pypdf import PdfReader

                reader = PdfReader(BytesIO(response.content))
                raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
            else:
                response.encoding = response.apparent_encoding or "utf-8"
                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                    tag.decompose()

                # 실제 약관 본문 컨테이너를 찾는다. 템플릿에 따라 클래스 조합이 다를 수 있어
                # 구체적인 선택자부터 순서대로 시도한다.
                container_selectors = [
                    # "div.wrap_terms.wrap_policy",
                    "div.wrap_terms",
                ]
                content = None
                for selector in container_selectors:
                    content = soup.select_one(selector)
                    if content is not None:
                        break
                if content is None:
                    # 어떤 선택자도 없는 템플릿이면 경고를 남기고 페이지 전체로 폴백한다
                    print(
                        f"[경고][로딩] {source.doc_name} — "
                        f"본문 컨테이너({', '.join(container_selectors)})를 찾지 못해 "
                        "전체 페이지에서 추출합니다(목차 혼입 가능, span 휴리스틱에 의존)."
                    )
                    content = soup

                raw_text = content.get_text(separator="\n")

            text = finalize(raw_text)

            if len(article_pattern.findall(text)) < 3:
                failures.append(f"{url}: 조 구조 미검출(len={len(text)})")
                continue

            return RawDocument(
                doc_name=source.doc_name,
                text=text,
                source_url=url,
                fetched_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                char_len=len(text),
            )
        except Exception as exc:  # noqa: BLE001 — URL 후보를 끝까지 시도한다
            failures.append(f"{url}: {type(exc).__name__}: {exc}")

    raise RuntimeError(f"[{source.doc_name}] 원문 취득 실패 — " + " | ".join(failures))


def load_documents(
    sources: List[DocumentSource],
    timeout_s: float = 20.0,
) -> List[RawDocument]:
    """List[DocumentSource] → List[RawDocument]."""
    documents: List[RawDocument] = []
    for source in sources:
        document = fetch_document(source, timeout_s)
        print(
            f"[로딩] {document.doc_name} · {document.char_len}자 · {document.source_url}"
        )
        documents.append(document)
    return documents


# =====================================================================================
# 7. 인덱싱 — 파싱
# =====================================================================================

def parse_articles(document: RawDocument) -> List[Article]:
    """RawDocument → List[Article].

    조 번호는 1부터 순차 증가하는 것만 헤더 후보로 인정한다. 다만 목차(TOC)처럼
    "제1조 ... 제N조"가 본문보다 먼저 촘촘하게 나열되는 페이지에서는 이 조건을
    만족하는 1..N 구간이 문서 안에 두 번(목차 1회, 본문 1회) 나타날 수 있다.
    이 경우 각 구간을 독립적으로 모은 뒤, 전체 글자 폭(span)이 가장 큰 구간을
    실제 본문으로 채택한다. 목차는 항목 간 간격이 좁고, 본문은 조마다 실제
    내용이 있어 간격이 훨씬 크다는 점을 이용한다.

    표기 형태 두 가지를 모두 처리한다.
      · "제 1 조 (목적)"  — 공백 있음, 제목 괄호
      · "제1조 목적"      — 공백 없음, 제목 괄호 없음
    """
    import re

    text = document.text
    header_pattern = re.compile(r"제\s*(\d+)\s*조")
    reference_suffix = ("에", "의", "와", "과", "및", "부터", "까지", "에서", ",", "제")

    candidates: List[Tuple[int, int, int, str]] = []  # (시작, 헤더끝, 조번호, 제목)

    for match in header_pattern.finditer(text):
        number = int(match.group(1))

        tail = text[match.end():match.end() + 90]
        stripped_tail = tail.lstrip()
        if stripped_tail and stripped_tail[0] in reference_suffix:
            continue

        title_match = re.match(r"[ \t]*\(([^)\n]{1,60})\)", tail)
        if title_match:
            title = title_match.group(1).strip()
            header_end = match.end() + title_match.end()
        else:
            line_match = re.match(r"[ \t]*([^\n]{0,60})", tail)
            candidate = line_match.group(1).strip() if line_match else ""
            candidate = candidate.split("\n")[0].strip()
            title = candidate if 0 < len(candidate) <= 40 else ""
            header_end = match.end() + (len(line_match.group(0)) if title else 0)

        candidates.append((match.start(), header_end, number, title))

    # 1부터 연속 증가하는 구간(run) 단위로 후보를 분리한다.
    # 목차와 본문이 각각 독립된 1..N 시퀀스를 형성하므로, run이 여러 개 나올 수 있다.
    runs: List[List[Tuple[int, int, int, str]]] = []
    current_run: List[Tuple[int, int, int, str]] = []
    expected_number = 1

    for candidate in candidates:
        _, _, number, _ = candidate
        if number == expected_number:
            current_run.append(candidate)
            expected_number += 1
        elif number == 1:
            if current_run:
                runs.append(current_run)
            current_run = [candidate]
            expected_number = 2
        # 그 외(기대값도 1도 아님)는 잡음으로 보고 무시한다.

    if current_run:
        runs.append(current_run)

    if not runs:
        raise ValueError(f"[{document.doc_name}] 조 구조 파싱 실패 — 정규식 재검토 필요")

    def run_span(run: List[Tuple[int, int, int, str]]) -> int:
        return run[-1][0] - run[0][0]  # 마지막 헤더 시작 - 첫 헤더 시작

    headers = max(runs, key=run_span)

    if len(runs) > 1:
        detail = ", ".join(f"{len(run)}개조/span={run_span(run)}자" for run in runs)
        print(
            f"[경고][파싱] {document.doc_name} 조 시퀀스 {len(runs)}개 발견"
            f"(목차 등 중복 가능성) — {detail} · 가장 긴 구간을 본문으로 채택"
        )

    chapter_pattern = re.compile(r"^제\s*\d+\s*장.*$", re.MULTILINE)
    articles: List[Article] = []

    for index, (_, header_end, number, title) in enumerate(headers):
        body_end = headers[index + 1][0] if index + 1 < len(headers) else len(text)
        body = text[header_end:body_end]
        body = chapter_pattern.sub("", body)
        body = re.sub(r"\n{2,}", "\n", body).strip()

        articles.append(
            Article(
                doc_name=document.doc_name,
                article_number=number,
                article_title=title,
                body=body,
            )
        )

    if not articles:
        raise ValueError(f"[{document.doc_name}] 조 구조 파싱 실패 — 정규식 재검토 필요")

    suspicious = [
        f"제{a.article_number}조(len={len(a.body)})"
        for a in articles
        if len(a.body) < 20
    ]
    if suspicious:
        print(
            f"[경고][파싱] {document.doc_name} 본문이 20자 미만인 조 {len(suspicious)}건 "
            f"— 제목 추출 로직이 본문을 흡수했을 가능성: " + ", ".join(suspicious)
        )

    print(f"[파싱] {document.doc_name} · 제1조~제{articles[-1].article_number}조 "
          f"({len(articles)}개)")
    return articles


# =====================================================================================
# 8. 인덱싱 — 청킹
# =====================================================================================

def chunk_articles(articles: List[Article], config: IndexConfig) -> List[Chunk]:
    """List[Article] → List[Chunk]. 조 단위 1청크, max_chunk_chars 초과분만 분할.

    분할된 조각에도 문서명·조번호 머리말을 붙여 검색 시 조 소속을 유지한다.
    """
    chunks: List[Chunk] = []
    empty_body_articles: List[str] = []

    for article in articles:
        header_line = f"[{article.doc_name}] 제{article.article_number}조"
        if article.article_title:
            header_line += f"({article.article_title})"

        body = article.body
        if not body:
            empty_body_articles.append(f"제{article.article_number}조")
            chunks.append(
                Chunk(
                    chunk_id=(
                        f"{normalize_doc_name(article.doc_name)}"
                        f"-{article.article_number:03d}-00"
                    ),
                    doc_name=article.doc_name,
                    article_number=article.article_number,
                    article_title=article.article_title,
                    text=f"{header_line}\n(본문 파싱 실패 — 원문 확인 필요)",
                )
            )
            continue

        budget = max(200, config.max_chunk_chars - len(header_line) - 1)

        if len(body) <= budget:
            segments = [body]
        else:
            segments = []
            cursor = 0
            while cursor < len(body):
                window_end = min(cursor + budget, len(body))
                if window_end < len(body):
                    boundary = body.rfind("\n", cursor + budget // 2, window_end)
                    if boundary == -1:
                        boundary = body.rfind(". ", cursor + budget // 2, window_end)
                    if boundary != -1:
                        window_end = boundary + 1
                segments.append(body[cursor:window_end].strip())
                cursor = window_end

        segments = [s for s in segments if s] or [""]

        for part_index, segment in enumerate(segments):
            chunks.append(
                Chunk(
                    chunk_id=(
                        f"{normalize_doc_name(article.doc_name)}"
                        f"-{article.article_number:03d}-{part_index:02d}"
                    ),
                    doc_name=article.doc_name,
                    article_number=article.article_number,
                    article_title=article.article_title,
                    text=f"{header_line}\n{segment}",
                )
            )

    if empty_body_articles:
        print(
            f"[경고][청킹] {articles[0].doc_name} 본문 파싱 실패 {len(empty_body_articles)}건: "
            + ", ".join(empty_body_articles)
        )

    print(f"[청킹] {len(articles)}개 조항 → {len(chunks)}개 청크")
    return chunks


# =====================================================================================
# 9. 인덱싱 — 임베딩
# =====================================================================================

class EmbedderHandle(BaseModel):
    """임베딩 모델 핸들. 실제 구현에서는 SentenceTransformer 인스턴스를 보유한다."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str
    dim: int
    device: str = "cpu"
    model: Any = None


def build_embedder(config: IndexConfig) -> EmbedderHandle:
    """IndexConfig → EmbedderHandle. bge-m3를 로컬 로드한다."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(config.embedding_model_name, device=device)
    dim = int(model.get_sentence_embedding_dimension())

    print(f"[임베딩 모델] {config.embedding_model_name} · device={device} · dim={dim}")
    return EmbedderHandle(
        model_name=config.embedding_model_name, dim=dim, device=device, model=model
    )


def embed_chunks(
    chunks: List[Chunk],
    embedder: EmbedderHandle,
    config: IndexConfig,
) -> EmbeddingBundle:
    """List[Chunk] → EmbeddingBundle. L2 정규화하여 내적=cosine이 되게 한다."""
    texts = [chunk.text for chunk in chunks]
    vectors = embedder.model.encode(
        texts,
        batch_size=config.embedding_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    print(f"[임베딩] {len(chunks)}개 청크 · shape={tuple(vectors.shape)}")
    return EmbeddingBundle(
        chunks=chunks,
        vectors=vectors.astype("float32").tolist(),
        model_name=embedder.model_name,
        dim=int(vectors.shape[1]),
    )


# =====================================================================================
# 10. 인덱싱 — 저장
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
        """질의 벡터 → 상위 top_k 청크. 정규화 벡터이므로 내적=cosine."""
        import numpy as np

        query = np.asarray([query_vector], dtype="float32")
        scores, indices = self.index.search(query, min(top_k, self.n_vectors))

        hits: List[RetrievedChunk] = []
        for rank, (score, position) in enumerate(zip(scores[0], indices[0]), start=1):
            if position < 0:
                continue
            hits.append(
                RetrievedChunk(
                    rank=rank,
                    score=round(float(score), 4),
                    chunk=self.chunks[int(position)],
                )
            )
        return hits


def build_vector_store(bundle: EmbeddingBundle) -> VectorStoreHandle:
    """EmbeddingBundle → VectorStoreHandle. FAISS IndexFlatIP 생성 후 add."""
    import faiss
    import numpy as np

    vectors = np.asarray(bundle.vectors, dtype="float32")
    index = faiss.IndexFlatIP(bundle.dim)
    index.add(vectors)

    print(f"[저장] FAISS IndexFlatIP · {index.ntotal}개 벡터 · dim={bundle.dim}")
    return VectorStoreHandle(
        dim=bundle.dim,
        n_vectors=int(index.ntotal),
        chunks=bundle.chunks,
        index=index,
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
# 11. 검색
# =====================================================================================

def embed_query(
    question: str,
    embedder: EmbedderHandle,
    config: RetrievalConfig,
) -> List[float]:
    """질문 문자열 → 정규화된 질의 벡터."""
    vector = embedder.model.encode(
        [config.query_prefix + question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    return vector.astype("float32").tolist()


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
# 12. 증강
# =====================================================================================

SYSTEM_PROMPT = (
    "당신은 카카오 약관 전문 어시스턴트입니다. 제공된 약관 조문만을 근거로 답하십시오. "
    "근거에 없는 내용은 지어내지 말고, 숫자·기간·조문 번호는 근거 그대로 옮기십시오."
)

FEW_SHOT_EXAMPLES = (
    "다음은 답변 방식을 보여주는 예시입니다.\n\n"
    "[예시 근거] 제5조: 회원이 탈퇴하면 본인이 작성한 게시물은 삭제된다. 다만 제3자가\n"
    "공유하거나 댓글을 단 게시물은 삭제되지 않는다.\n"
    "[예시 질문] 탈퇴하면 제가 쓴 글이 전부 삭제되나요?\n"
    "[예시 답변] 아니오. 제3자가 공유하거나 댓글을 단 게시물은 삭제되지 않습니다. (제5조)\n\n"
    "[예시 근거] 제9조: 회사는 서비스 개선을 위해 필요한 경우 약관을 개정할 수 있다.\n"
    "[예시 질문] 약관 개정 시 회원에게 별도의 금전적 보상을 지급하나요?\n"
    "[예시 답변] 제공된 약관 조문에서 확인할 수 없습니다."
)

def build_prompt(retrieval: RetrievalOutput, config: GenerationConfig) -> PromptBundle:
    """RetrievalOutput → PromptBundle. [문서명, 조번호, 본문] 형식으로 조합.

    max_context_chars 예산 안에서 관련도 상위 근거부터 담고, 예산을 넘기는
    근거는 제외한다. 최소 1개는 반드시 포함한다.
    """
    blocks: List[str] = []
    used_chars = 0

    for hit in retrieval.hits:
        block = (
            f"[근거 {hit.rank}] 문서명: {hit.chunk.doc_name} / "
            f"조번호: 제{hit.chunk.article_number}조\n"
            f"본문: {hit.chunk.text}"
        )
        if blocks and used_chars + len(block) > config.max_context_chars:
            break
        if not blocks and len(block) > config.max_context_chars:
            block = block[: config.max_context_chars]
        blocks.append(block)
        used_chars += len(block)

    context_block = "\n\n".join(blocks)
    user_prompt = (
        f"{context_block}\n\n"
        f"{context_block}\n\n"
        f"{FEW_SHOT_EXAMPLES}\n\n"
        f"질문: {retrieval.question}\n\n"
        "답변 안에 근거가 된 조문 번호를 함께 밝히십시오. "
        "근거 조문에서 질문에 답이 되는 문장을 인용했다면, 그 인용문이 뜻하는 결론을 "
        "그대로 따르십시오. 인용한 내용과 다른 결론(예: 인용문이 답을 담고 있는데도 "
        "\"확인할 수 없습니다\"라고 답하는 것)을 내리지 마십시오. "
        "답변의 결론은 하나여야 합니다. 이미 답을 제시했다면 답변 끝에 "
        "\"제공된 약관 조문에서 확인할 수 없습니다\"를 다시 덧붙이지 마십시오. "
        "이 문구는 근거에 답이 전혀 없을 때만, 답변 전체에서 정확히 한 번만 사용하십시오. "
        "근거에서 확인되지 않는 내용은 \"제공된 약관 조문에서 확인할 수 없습니다\"라고 답하십시오."
    )

    return PromptBundle(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        context_block=context_block,
        n_context_chunks=len(blocks),
    )


# =====================================================================================
# 13. 생성
# =====================================================================================

class GeneratorHandle(BaseModel):
    """Qwen2.5-Instruct 로컬 생성 모델 핸들."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str
    load_in_4bit: bool
    model: Any = None
    tokenizer: Any = None


def load_generator(config: GenerationConfig) -> GeneratorHandle:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if REQUIRED_GENERATION_MODEL_FAMILY.split("-")[0] not in config.model_name:
        raise ValueError(f"생성 모델은 {REQUIRED_GENERATION_MODEL_FAMILY} 계열이어야 합니다.")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    load_kwargs: Dict[str, Any] = {"device_map": "auto"}
    if config.load_in_4bit and torch.cuda.is_available():
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)
    model.eval()

    print(f"[생성 모델] {config.model_name} · 4bit={config.load_in_4bit}")
    return GeneratorHandle(
        model_name=config.model_name,
        load_in_4bit=config.load_in_4bit,
        model=model,
        tokenizer=tokenizer,
    )


def generate(prompt: PromptBundle, generator: GeneratorHandle) -> GenerationOutput:
    """PromptBundle → GenerationOutput. chat template 적용 후 greedy decoding."""
    import torch

    started = time.perf_counter()
    tokenizer = generator.tokenizer

    messages = [
        {"role": "system", "content": prompt.system_prompt},
        {"role": "user", "content": prompt.user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(generator.model.device)

    with torch.inference_mode():
        generated = generator.model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    input_length = model_inputs["input_ids"].shape[1]
    new_token_ids = generated[0][input_length:]
    answer_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    return GenerationOutput(
        answer_text=answer_text,
        n_new_tokens=int(new_token_ids.shape[0]),
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

    if _GRAPH is not None:
        return

    _STORE, _EMBEDDER, INDEX_STATS = build_index(config.index)
    print(f"[인덱싱 완료] {INDEX_STATS.model_dump()}")

    _GENERATOR = load_generator(config.generation)
    ctx = PipelineContext(
        store=_STORE, embedder=_EMBEDDER, generator=_GENERATOR, config=config
    )
    _GRAPH = build_graph(ctx)
    print("[부팅 완료] run_rag_pipeline() 호출 준비됨")


def run_rag_pipeline(question: str) -> Dict[str, Any]:
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