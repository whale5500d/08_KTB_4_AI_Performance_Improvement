# -*- coding: utf-8 -*-
# =====================================================================================
#  0번 셀 · 라이브러리 설치
#  새 Colab T4 런타임에서 가장 먼저 한 번 실행합니다.
#
#  torch 재설치 금지 규정 대응
#    pip는 의존성 해석 과정에서 torch를 조용히 교체할 수 있습니다(peft, sentence-transformers
#    등이 torch를 요구하기 때문). 그래서 현재 설치된 torch 계열 버전을 constraints 파일로
#    고정하고 -c 옵션으로 넘겨, 어떤 경우에도 torch가 교체되지 않게 막습니다.
#    설치 전후 버전을 대조해 변동이 있으면 즉시 경고를 출력합니다.
# =====================================================================================
import subprocess
import sys

CONSTRAINTS_PATH = "/content/torch_constraints.txt"

PACKAGES = [
    # 서빙 계약
    "fastapi",
    "uvicorn",
    # 오케스트레이션 / 스키마
    "langgraph",
    "pydantic>=2",
    # 검색: dense + 희소 + 재순위
    "sentence-transformers",
    "faiss-cpu",
    "rank_bm25",
    # 생성 및 미세조정
    "transformers>=4.44",
    "accelerate",
    "peft",
    "bitsandbytes",
    # 원문 취득
    "pypdf",
    "python-docx",
    "beautifulsoup4",
    "requests",
]


def _pinned_torch_versions():
    """설치된 torch 계열 버전을 (패키지명, 순수버전) 목록으로 돌려준다."""
    pinned = []
    for name in ("torch", "torchvision", "torchaudio"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        version = getattr(module, "__version__", "")
        if version:
            pinned.append((name, version.split("+")[0]))
    return pinned


def _write_constraints(pinned):
    with open(CONSTRAINTS_PATH, "w", encoding="utf-8") as handle:
        for name, version in pinned:
            handle.write(f"{name}=={version}\n")
    return CONSTRAINTS_PATH


def _install_baseline_packages():
    before = dict(_pinned_torch_versions())
    command = [sys.executable, "-m", "pip", "install", "-q", *PACKAGES]
    if before:
        command += ["-c", _write_constraints(list(before.items()))]
    subprocess.run(command, check=True)

    after = dict(_pinned_torch_versions())
    for name, version in before.items():
        if after.get(name) != version:
            print(
                f"[경고] {name} 버전이 {version} → {after.get(name)} 로 바뀌었습니다. "
                "런타임을 초기화하고 다시 실행하십시오(torch 재설치 금지 규정)."
            )
    print(f"[torch 고정] {before if before else 'torch 미검출'}")


_install_baseline_packages()
print("[0번 셀 완료] 라이브러리 설치 완료 · 다음 셀(pipeline)을 실행하세요.")