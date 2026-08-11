# -*- coding: utf-8 -*-
# =====================================================================================
#  0번 셀 — 라이브러리 설치
#  새 Colab T4 런타임에서 가장 먼저 한 번 실행합니다.
#  torch는 Colab 기본 설치본을 그대로 사용하며 재설치하지 않습니다(재설치 금지 규정).
# =====================================================================================
import subprocess
import sys


def _install_baseline_packages():
    packages = [
        "fastapi",
        "uvicorn",
        "langgraph",
        "pydantic>=2",
        "sentence-transformers",
        "faiss-cpu",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "pypdf",
        "python-docx",
        "beautifulsoup4",
        "requests",
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        check=True,
    )


_install_baseline_packages()
print("[0번 셀 완료] 라이브러리 설치 완료 — 다음 셀(결과기 로직)을 실행하세요.")
