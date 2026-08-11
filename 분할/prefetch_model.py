# -*- coding: utf-8 -*-
# =====================================================================================
#  0-1번 셀 · 생성 모델 가중치 선다운로드 (prefetch)
#
#  실행 순서: 0번(installation) → 0-1번(이 셀) → 1번(pipeline) → 2번(공통 러너)
#
#  왜 분리하는가
#    pipeline의 load_generator()는 from_pretrained()를 호출하는데, 이 호출 안에서
#    가중치 다운로드와 모델 적재가 한 덩어리로 일어난다. 다운로드가 전체 부팅 시간을
#    지배하므로 다운로드만 앞 셀로 떼어 두면, 1번 셀은 Hugging Face 캐시에서 읽기만 한다.
#    이 셀은 파일을 캐시에 내려받기만 하고 GPU에는 아무것도 올리지 않는다.
#    두 번째 실행부터는 캐시 적중으로 즉시 끝난다(멱등).
#
#  주의
#    · REPO_ID 는 pipeline.py 의 GenerationConfig.model_name 기본값과 반드시 같아야 한다.
#      한쪽만 바꾸면 1번 셀이 캐시를 못 찾고 다시 내려받는다.
#    · 이 셀은 pipeline.py 를 import 하지 않는다(1번 셀보다 먼저 실행되기 때문).
#    · 드라이브 마운트나 사전 업로드 파일을 쓰지 않는다. 실행 중 원격 저장소에서만 받는다.
# =====================================================================================
import os
import subprocess
import sys
import time

# ── 여기만 바꾼다 ────────────────────────────────────────────────────────────────────
REPO_ID = "Qwen/Qwen2.5-3B-Instruct"          # pipeline.GenerationConfig.model_name 과 일치
PREFETCH_FALLBACK = False                      # True 로 두면 폴백 모델(1.5B)도 미리 받는다
FALLBACK_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PREFETCH_RETRIEVAL_MODELS = False              # True 로 두면 임베딩·재순위 모델도 미리 받는다
RETRIEVAL_REPO_IDS = ("BAAI/bge-m3", "BAAI/bge-reranker-v2-m3")
MAX_WORKERS = 8                                # 샤드 병렬 다운로드 수
# ────────────────────────────────────────────────────────────────────────────────────

# 가중치 파일만 받는다. 학습용 옵티마이저 상태, gguf, 중복 포맷은 제외해 전송량을 줄인다.
ALLOW_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "*.model",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/*",
]
IGNORE_PATTERNS = ["*.pth", "*.gguf", "*.onnx", "*.msgpack", "*.h5", "original/*"]


def _enable_fast_transfer():
    """hf_transfer(Rust 다운로더)를 켠다. 설치 실패 시 기본 다운로더로 진행한다."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "hf_transfer"],
            check=True,
        )
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("[프리페치] hf_transfer 활성화")
    except Exception as exc:  # noqa: BLE001 · 가속은 선택 사항이다
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        print(f"[프리페치] hf_transfer 미사용({type(exc).__name__}) · 기본 다운로더로 진행")


def prefetch(repo_id):
    """repo_id 의 가중치를 Hugging Face 캐시에 내려받고 (경로, 소요초)를 돌려준다."""
    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        max_workers=MAX_WORKERS,
    )
    elapsed = round(time.perf_counter() - started, 1)
    size_gb = sum(
        os.path.getsize(os.path.join(root, name))
        for root, _dirs, files in os.walk(path)
        for name in files
    ) / (1024 ** 3)
    print(f"[프리페치] {repo_id} · {size_gb:.2f}GB · {elapsed}s · {path}")
    return path, elapsed


def _verify_tokenizer(repo_id):
    """토크나이저만 캐시에서 열어 파일 누락을 조기에 잡는다. GPU를 쓰지 않는다."""
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(repo_id)
        print(f"[프리페치] {repo_id} 토크나이저 확인 완료")
    except Exception as exc:  # noqa: BLE001
        print(f"[경고][프리페치] {repo_id} 토크나이저 확인 실패({type(exc).__name__}: {exc})")


_enable_fast_transfer()

_targets = [REPO_ID]
if PREFETCH_FALLBACK:
    _targets.append(FALLBACK_REPO_ID)
if PREFETCH_RETRIEVAL_MODELS:
    _targets.extend(RETRIEVAL_REPO_IDS)

_total = 0.0
for _repo in _targets:
    try:
        _, _elapsed = prefetch(_repo)
        _total += _elapsed
    except Exception as exc:  # noqa: BLE001 · 실패해도 1번 셀이 스스로 다시 받는다
        print(f"[경고][프리페치] {_repo} 실패({type(exc).__name__}: {exc}) · 1번 셀에서 재시도됩니다")

_verify_tokenizer(REPO_ID)
print(f"[0-1번 셀 완료] 총 {round(_total, 1)}s · 다음 셀(pipeline)에서는 캐시에서 즉시 적재됩니다.")