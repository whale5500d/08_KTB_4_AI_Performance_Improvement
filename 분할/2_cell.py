# 2번 셀 — 공개 10문항 답변 파일 생성
# 이 셀은 전 팀 공통이며 _SP_TEAM 한 줄 외에는 수정하지 않습니다.
# 새 Google Colab T4 런타임에서 결과기 코드를 먼저 실행한 뒤 이 셀을 실행합니다.
#
# 사용 순서
# 1. 새 Google Colab T4 런타임에서 1번 셀 결과기 코드를 실행합니다.
# 2. 이 공통 러너를 2번 셀에 그대로 둡니다.
# 3. 맨 위 _SP_TEAM에 운영진이 알려준 숫자 팀 식별자를 입력합니다.
# 4. 생성된 answers_public_<팀>.json을 결과기 코랩 파일과 함께 제출합니다.
# 공개 문항 10개 · 실행 방식: http
# ═══════════════════════════════════════════════════════════════
#  ★ 여기 한 줄만 자기 팀으로 바꾸세요. 나머지는 손대지 마세요. ★
# ═══════════════════════════════════════════════════════════════
_SP_TEAM = "3"          # 예: "1"  ← 운영진이 알려준 팀 식별자(숫자)를 그대로 적습니다
# ═══════════════════════════════════════════════════════════════

import builtins as _sp_builtins
import json as _sp_json
import os as _sp_os_rt
import re as _sp_re
import signal as _sp_signal
import socket as _sp_socket
import sys as _sp_sys
import time as _sp_time
import traceback as _sp_traceback
import unicodedata as _sp_unicodedata
import urllib.error as _sp_urlerror
import urllib.request as _sp_urlrequest

_sp_open = _sp_builtins.open
_sp_print = _sp_builtins.print

if "_sp_real_sys_exit" in globals():
    _sp_sys.exit = _sp_real_sys_exit
    if _sp_real_exit is not None:
        _sp_builtins.exit = _sp_real_exit
    if _sp_real_quit is not None:
        _sp_builtins.quit = _sp_real_quit

_SP_OUTPUT_DIR = "/content/"
_SP_OUTPUT_PREFIX = "answers_public_"
_SP_EXPECTED_OUTPUT_PATH = ""
_SP_TEAM_ALLOWED = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_-"
_SP_TEAM_MAX_LEN = 32
_SP_TEAM_NUMERIC_ONLY = True

def _sp_team_howto(head):
    """중단 사유 + 학생이 바로 고칠 수 있는 안내를 한 덩어리로 만든다."""
    rule = (
        "1 이상의 정수를 문자열로 입력합니다. 예: 1, 2, 17"
        if _SP_TEAM_NUMERIC_ONLY
        else "영문·숫자·밑줄(_)·하이픈(-) 1~" + str(_SP_TEAM_MAX_LEN) + "자"
    )
    return (
        head
        + "\n"
        + "\n  [고치는 법] 이 셀 맨 위 ★ 상자 안의 한 줄을 이렇게 바꾸세요."
        + '\n      _SP_TEAM = "1"      ← 운영진이 알려준 팀 식별자(숫자)를 따옴표 안에 그대로'
        + "\n  [쓸 수 있는 값] " + rule
        + "\n                 띄어쓰기와 / \\ . : 같은 경로 문자는 파일 이름을 깨뜨려 쓸 수 없습니다."
        + "\n  [왜] 결과 파일 이름이 " + _SP_OUTPUT_PREFIX + "<팀>.json 이고, 채점은 이 이름으로"
        + "\n       어느 팀 답안인지 가립니다. 비워 두면 채점 자체가 되지 않습니다."
    )

def _sp_resolve_team(value):
    """_SP_TEAM 을 검사·정리해 돌려준다. 쓸 수 없는 값이면 RuntimeError 로 즉시 중단."""
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(_sp_team_howto(
            "★ 팀 식별자(_SP_TEAM)가 비어 있어 실행을 중단했습니다. 결과 파일은 만들지 않았습니다."))
    team = value.strip()
    if _SP_TEAM_NUMERIC_ONLY and not _sp_re.fullmatch(r"[1-9][0-9]*", team):
        raise RuntimeError(_sp_team_howto(
            "★ 팀 식별자(_SP_TEAM)는 운영진이 알려준 숫자여야 합니다. 지금 값: " + repr(value)))
    if len(team) > _SP_TEAM_MAX_LEN:
        raise RuntimeError(_sp_team_howto(
            "★ 팀 식별자(_SP_TEAM)가 너무 깁니다(" + str(len(team)) + "자). 팀 이름이 아니라 짧은 식별자입니다."))
    _bad = _sp_builtins.sorted(
        _sp_builtins.set(c for c in team if c not in _SP_TEAM_ALLOWED and not ("가" <= c <= "힣")))
    if _bad:
        raise RuntimeError(_sp_team_howto(
            "★ 팀 식별자(_SP_TEAM)에 파일 이름으로 쓸 수 없는 문자가 있습니다: "
            + ", ".join(repr(c) for c in _bad) + "   (지금 값: " + repr(value) + ")"))
    return team

_SP_TEAM = _sp_resolve_team(_SP_TEAM)
if any(ord(c) > 127 for c in _SP_TEAM):
    _sp_print("[주의] 팀 식별자에 한글 등 ASCII 밖 문자가 있습니다: " + _SP_TEAM
              + " — 운영진이 알려준 식별자가 맞는지 확인하세요."
              " 한글 파일 이름은 내려받기·올리기 과정에서 자모 표현이 달라져 팀이 어긋날 수 있습니다.",
              flush=True)

_SP_OUTPUT_PATH = _SP_OUTPUT_DIR.rstrip("/") + "/" + _SP_OUTPUT_PREFIX + _SP_TEAM + ".json"
if _SP_EXPECTED_OUTPUT_PATH and (_sp_os_rt.path.basename(_SP_OUTPUT_PATH)
                                 != _sp_os_rt.path.basename(_SP_EXPECTED_OUTPUT_PATH)):
    raise RuntimeError(
        "이 셀은 " + _sp_os_rt.path.basename(_SP_EXPECTED_OUTPUT_PATH) + " 용으로 생성됐는데 "
        + _sp_os_rt.path.basename(_SP_OUTPUT_PATH) + " 로 저장하려 합니다"
        "(_SP_TEAM 을 손으로 고쳤습니까?). 다른 팀으로 돌리려면 --team 을 바꿔 셀을 다시 생성하세요."
    )
_SP_AUTO_DOWNLOAD = True
_SP_QUESTIONS_JSON = (
    "[[\"P01\", \"사업자/단체 카카오계정은 계정 정보에 등록된 담당자 몇 명이 이용할 수 있으며, 다른 사람과 공유하는 것은 허용되나요?\"], [\"P02\", \"회사가 예측하거나 통제할 수 없는 사유로 서비스가 중단된 경우, 복구가 몇 시간 이상 지연되면 회사는 공지사항에 게시하여 알리나요?\"], [\"P03\", \"카카오계정 약관에서 회사가 개별 서비스와 연동하여 카카오계정에서 제공한다고 열거한 '카카오계정 서비스'의 내용 5가지는 각각 무엇인가요?\"], [\"P04\", \"회사가 위치기반서비스의 이용을 제한하거나 중지한 때에는 이용자에게 무엇을 어떤 방법으로 알리나요?\"], [\"P05\", \"회사가 위치정보 수집·이용·제공사실 확인자료를 기록·보존하는 근거는 위치정보의 보호 및 이용 등에 관한 법률 제 몇 조 제 몇 항이며, 그 자료는 어디에 기록되어 몇 개월간 보관되나요?\"], [\"P06\", \"카카오계정이 없는 사람이 통합서비스에 가입하려면 무엇을 먼저 해야 하며, 통합서비스 이용계약은 동의·확인·승낙의 어떤 순서로 체결되나요?\"], [\"P07\", \"서비스 명칭에 '카카오'가 사용되더라도 카카오 통합서비스약관의 '통합서비스'에 포함되지 않는 서비스는 누가 제공하는 서비스이며, 약관은 그 예로 무엇을 들고 있나요?\"], [\"P08\", \"카카오 통합 약관과 세부지침(회사가 정한 서비스의 개별 이용약관·운영정책·규칙 등)의 내용이 충돌하는 경우"
    ", 본 약관이 세부지침보다 우선하여 적용되나요?\"], [\"P09\", \"이용자가 서비스 사용을 중단하거나 카카오계정 및 Daum 아이디를 탈퇴한 이후, 게시물에 관하여 회사에 부여한 라이선스의 효력은 어떻게 되나요?\"], [\"P10\", \"8세 이하의 아동 등의 생명 또는 신체 보호를 위해 보호의무자가 개인위치정보의 이용 또는 제공에 동의하려면 어떤 서류에 무엇을 첨부하여 어디에 제출해야 하며, 그 동의는 어떤 효력을 갖나요?\"]]"
)
_SP_QUESTIONS = [tuple(_x) for _x in _sp_json.loads(_SP_QUESTIONS_JSON)]
_SP_ALLOWED_DOCS = _sp_json.loads("[\"카카오계정 약관\", \"카카오 통합서비스약관\", \"카카오 통합 약관\", \"카카오 위치정보 이용약관\"]")
_SP_PER_Q_TIMEOUT_S = 120
_SP_TRANSPORT = "http"
_SP_HTTP_HOST = "127.0.0.1"
_SP_HTTP_PORT = 8765
_SP_HTTP_STARTUP_TIMEOUT_S = 30
_SP_HTTP_HEALTH_PATH = "/health"
_SP_HTTP_ANSWER_PATH = "/answer"
_SP_PERFORMANCE_REQUESTS = 12
_SP_PERFORMANCE_CONCURRENCY = 2
_SP_PERFORMANCE_REPETITIONS = 3
_SP_PERFORMANCE_WARMUP_REQUESTS = 2

_sp_fn = globals().get("answer_question")
if not callable(_sp_fn):
    raise RuntimeError(
        "팀 코드에 answer_question(question) 함수가 없습니다(규정 ②). 실행을 중단합니다."
    )

_sp_doc_warnings = []
_sp_timeouts = []
_sp_http_server = None
_sp_http_thread = None

class _SpHttpTimeout(Exception):
    """HTTP 요청 시간 초과. 품질 추출에서는 timeout_qids로 기록한다."""

def _sp_http_url(path):
    return "http://" + _SP_HTTP_HOST + ":" + str(_SP_HTTP_PORT) + path

def _sp_http_json(method, path, payload=None, timeout_s=None):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = _sp_json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = _sp_urlrequest.Request(
        _sp_http_url(path), data=data, headers=headers, method=method
    )
    try:
        with _sp_urlrequest.urlopen(req, timeout=timeout_s or _SP_PER_Q_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8")
            if resp.status != 200:
                raise RuntimeError("HTTP " + str(resp.status) + ": " + raw[:500])
    except (_sp_socket.timeout, TimeoutError) as exc:
        raise _SpHttpTimeout(str(timeout_s or _SP_PER_Q_TIMEOUT_S) + "초 안에 응답하지 않았습니다.") from exc
    except _sp_urlerror.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("HTTP " + str(exc.code) + ": " + raw[:500]) from exc
    except _sp_urlerror.URLError as exc:
        if isinstance(exc.reason, (_sp_socket.timeout, TimeoutError)):
            raise _SpHttpTimeout(
                str(timeout_s or _SP_PER_Q_TIMEOUT_S) + "초 안에 응답하지 않았습니다."
            ) from exc
        raise RuntimeError("HTTP 연결 실패: " + str(exc.reason)) from exc
    try:
        return _sp_json.loads(raw)
    except _sp_json.JSONDecodeError as exc:
        raise TypeError("HTTP 응답이 JSON이 아닙니다: " + raw[:500]) from exc

def _sp_start_http_server():
    global _sp_http_server, _sp_http_thread
    _sp_app = globals().get("app")
    if _sp_app is None:
        raise RuntimeError(
            "HTTP 실행 모드에는 전역 FastAPI app과 GET /health, POST /answer가 필요합니다."
        )
    try:
        import threading as _sp_threading
        import uvicorn as _sp_uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "HTTP 실행 모드에는 fastapi와 uvicorn이 필요합니다. 팀 설치 목록에 추가하세요."
        ) from exc
    _sp_config = _sp_uvicorn.Config(
        _sp_app,
        host=_SP_HTTP_HOST,
        port=_SP_HTTP_PORT,
        workers=1,
        log_level="warning",
        access_log=False,
    )
    _sp_http_server = _sp_uvicorn.Server(_sp_config)
    _sp_http_thread = _sp_threading.Thread(
        target=_sp_http_server.run, name="ktb-fastapi", daemon=True
    )
    _sp_http_thread.start()
    _sp_deadline = _sp_time.time() + _SP_HTTP_STARTUP_TIMEOUT_S
    _sp_last = None
    while _sp_time.time() < _sp_deadline:
        if not _sp_http_thread.is_alive():
            raise RuntimeError("FastAPI 서버가 준비되기 전에 종료됐습니다.")
        try:
            health = _sp_http_json("GET", _SP_HTTP_HEALTH_PATH, timeout_s=1)
            if isinstance(health, dict):
                _sp_print("[서버] FastAPI /health 준비 완료: " + _sp_http_url(_SP_HTTP_HEALTH_PATH))
                return
        except Exception as exc:
            _sp_last = exc
        _sp_time.sleep(0.2)
    _sp_stop_http_server()
    raise RuntimeError(
        "FastAPI 서버가 " + str(_SP_HTTP_STARTUP_TIMEOUT_S)
        + "초 안에 준비되지 않았습니다: " + str(_sp_last)
    )

def _sp_stop_http_server():
    if _sp_http_server is not None:
        _sp_http_server.should_exit = True
    if _sp_http_thread is not None and _sp_http_thread.is_alive():
        _sp_http_thread.join(timeout=5)

def _sp_invoke(question):
    if _SP_TRANSPORT == "http":
        return _sp_http_json(
            "POST", _SP_HTTP_ANSWER_PATH, {"question": question},
            timeout_s=_SP_PER_Q_TIMEOUT_S,
        )
    return _sp_call_with_timeout(_sp_fn, question, _SP_PER_Q_TIMEOUT_S)

if _SP_TRANSPORT == "http":
    _sp_start_http_server()

_sp_env_warnings = []
for _sp_d in ("/content/drive", "/content/gdrive", "/gdrive"):
    if _sp_os_rt.path.ismount(_sp_d):
        _sp_env_warnings.append(_sp_d + " 가 마운트되어 있습니다")
if _sp_env_warnings:
    _sp_print("", flush=True)
    _sp_print("!" * 86, flush=True)
    _sp_print("[규정 ③ 경고] 이 세션은 운영진 실행 환경과 다릅니다.", flush=True)
    for _sp_w in _sp_env_warnings:
        _sp_print("  · " + _sp_w, flush=True)
    _sp_print("  운영진은 드라이브가 연결되지 않은 새 세션에서 실행합니다. 드라이브에 둔 약관·인덱스를", flush=True)
    _sp_print("  읽고 있다면 본선에서 전량 실패합니다. 약관은 실행 중 내려받거나 셀 안에 포함하세요.", flush=True)
    _sp_print("  확인 방법: 새 노트북을 열어 코드와 이 셀만 붙여 넣고 실행해 보세요.", flush=True)
    _sp_print("!" * 86, flush=True)
    _sp_print("", flush=True)

class _SpTimeout(BaseException):
    """문항 단위 시간 초과.

    **BaseException 을 상속하는 것이 핵심이다.** 팀 코드가 `try/except Exception` 으로
    넓게 감싸는 일은 흔한데, Exception 을 상속하면 그 handler 가 시간 초과를 삼켜
    상한이 무력화된다(그대로 다음 루프를 돌며 계속 매달린다).
    """

def _sp_call_with_timeout(fn, arg, seconds):
    """SIGALRM 으로 문항 호출에 상한을 건다.

    메인 스레드가 아니거나 SIGALRM 이 없는 환경(윈도 등)에서는 signal 설정이
    실패하므로, 그때는 상한 없이 그대로 호출한다 — 상한을 못 걸었다고 해서
    채점 자체를 포기하는 편이 더 나쁘다.

    웹 Colab 셀은 IPython 이 메인 스레드에서 실행하므로 정상 동작한다.
    """
    if not seconds or seconds <= 0:
        return fn(arg)
    _sp_secs = max(1, int(seconds))     # alarm() 은 정수만 받는다. 0 은 '취소' 라 최소 1초.

    def _sp_on_alarm(signum, frame):
        raise _SpTimeout(str(_sp_secs) + "초 안에 응답하지 않았습니다.")

    try:
        _sp_prev = _sp_signal.signal(_sp_signal.SIGALRM, _sp_on_alarm)
        _sp_signal.alarm(_sp_secs)
    except (ValueError, AttributeError, OSError):
        return fn(arg)          # 상한을 걸 수 없는 환경 — 그대로 실행
    try:
        return fn(arg)
    finally:
        _sp_signal.alarm(0)
        try:
            _sp_signal.signal(_sp_signal.SIGALRM, _sp_prev)
        except Exception:
            pass

def _sp_json_safe_art(art):
    """조번호를 JSON 으로 쓸 수 있는 값으로. 표기는 최대한 원본을 살린다.

    **여기서 흡수하지 않으면 30문항을 다 돌린 뒤 파일 저장에서 터진다.**
    일부 수치 라이브러리의 정수형은 dict 도 아니고 2원소 검사도 통과하지만
    json.dump 가 거부한다. 이 값을 흡수하지 않으면
    실패 시점이 맨 끝이라 GPU 시간을 다 쓰고 결과 파일이 없는 최악의 형태가 된다.

    '제7조' 같은 문자열은 그대로 둔다 — 채점기 _art_no 가 정수로 읽는다.
    """
    if isinstance(art, bool):        # bool 은 int 의 하위형이라 먼저 걸러 낸다
        return str(art)
    if isinstance(art, (int, str)):
        return art
    try:                              # np.int64 등 정수로 볼 수 있는 것
        return int(art)
    except (TypeError, ValueError):
        return str(art)

def _sp_norm_doc(x):
    """문서명 대조용 정규화 — NFC 통일 + 공백 전부 제거.

    ⚠️ 채점기 judge_service/engine/objective.py 의 `norm_doc` 과 **같은 규칙이어야 한다.**
    러너는 Colab 셀이라 judge_service 를 import 할 수 없어 규칙을 여기에 복제해 둔다.
    한쪽만 바뀌어 어긋나면 곧바로 오탐이 난다 — 예전에 러너가 완전 일치로 대조하던 때
    '카카오계정약관'·'카카오 계정 약관' 은 실제 채점 MRR 이 1.00 인데도 규정 ④ 위반 경고를
    맞았다. 팀은 없는 문제를 고치러 다니고(자가 확인표가 n_doc_violations == 0 을 요구한다),
    정상 팀이 경고를 맞기 시작하면 아무도 경고를 안 보게 된다.
    두 구현의 일치는 submission_pipeline/tests/test_doc_name_normalization.py 가 고정한다.
    """
    return _sp_re.sub(r"\s+", "", _sp_unicodedata.normalize("NFC", str(x)))

_SP_ALLOWED_DOCS_NORM = _sp_builtins.set(_sp_norm_doc(_d) for _d in _SP_ALLOWED_DOCS)

def _sp_normalize_retrieved(qid, value):
    """retrieved 를 근거순 [[문서명, 조번호], ...] 1~4개로 정규화."""
    if not isinstance(value, (list, tuple)):
        raise TypeError(qid + ": retrieved 는 목록이어야 합니다. (실제: " + type(value).__name__ + ")")
    out = []
    for item in value:
        if isinstance(item, dict) and "doc" in item and "article_no" in item:
            doc, art = item["doc"], item["article_no"]
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            doc, art = item
        else:
            raise TypeError(qid + ": retrieved 항목은 [문서명, 조번호] 2원소여야 합니다. (실제: " + repr(item) + ")")
        doc = str(doc)
        if _SP_ALLOWED_DOCS_NORM and _sp_norm_doc(doc) not in _SP_ALLOWED_DOCS_NORM:
            _sp_doc_warnings.append({"qid": qid, "doc": doc})
        out.append([doc, _sp_json_safe_art(art)])
    if not 1 <= len(out) <= 4:
        raise ValueError(
            qid + ": retrieved 는 실제 답변 근거를 관련도 순으로 1~4개 반환해야 합니다. "
            "(실제: " + str(len(out)) + "개)"
        )
    return out

_sp_answers = []
_sp_errors = []
_sp_total = len(_SP_QUESTIONS)
_sp_print(
    "\n========== " + "공개" + " " + str(_sp_total)
    + "문항 실행 · " + _SP_TEAM + "팀 ==========",
    flush=True,
)
_sp_t0 = _sp_time.time()

for _sp_i, (_sp_qid, _sp_q) in enumerate(_SP_QUESTIONS, 1):
    _sp_print("[" + str(_sp_i).zfill(2) + "/" + str(_sp_total) + "] " + _sp_qid + " 실행 중 ...", flush=True)
    _sp_started = _sp_time.time()
    try:
        _sp_out = _sp_invoke(_sp_q)
        if not isinstance(_sp_out, dict):
            raise TypeError(_sp_qid + ": answer_question() 은 딕셔너리를 반환해야 합니다. (실제: "
                            + type(_sp_out).__name__ + ")")
        _sp_retrieved = _sp_normalize_retrieved(_sp_qid, _sp_out.get("retrieved"))
        _sp_answer = _sp_out.get("answer")
        if not isinstance(_sp_answer, str):
            raise TypeError(_sp_qid + ": answer 는 문자열이어야 합니다. (실제: "
                            + type(_sp_answer).__name__ + ")")
        _sp_answers.append({"qid": _sp_qid, "retrieved": _sp_retrieved, "answer": _sp_answer})
    except (_SpTimeout, _SpHttpTimeout) as _sp_exc:  # 한 문항이 세션 전체를 잡아먹지 않도록 끊는다.
        _sp_msg = "Timeout: " + str(_sp_exc)
        _sp_timeouts.append(_sp_qid)
        _sp_errors.append({"qid": _sp_qid, "error": _sp_msg})
        _sp_answers.append({"qid": _sp_qid, "retrieved": [], "answer": "", "error": _sp_msg})
        _sp_print("[시간초과] " + _sp_qid + " — " + _sp_msg, flush=True)
    except Exception as _sp_exc:  # 한 문항 실패로 30문항 전체를 잃지 않는다.
        _sp_msg = type(_sp_exc).__name__ + ": " + str(_sp_exc)
        _sp_errors.append({"qid": _sp_qid, "error": _sp_msg})
        _sp_answers.append({"qid": _sp_qid, "retrieved": [], "answer": "", "error": _sp_msg})
        _sp_print("[오류] " + _sp_qid + " — " + _sp_msg, flush=True)
        _sp_traceback.print_exc()
    finally:
        _sp_print("      (" + str(round(_sp_time.time() - _sp_started, 1)) + "s)", flush=True)

_sp_performance = None
if _SP_TRANSPORT == "http" and _SP_PERFORMANCE_REQUESTS > 0:
    from concurrent.futures import ThreadPoolExecutor as _SpThreadPoolExecutor

    def _sp_perf_one(index):
        _qid, _question = _SP_QUESTIONS[index % len(_SP_QUESTIONS)]
        started = _sp_time.perf_counter()
        try:
            value = _sp_http_json(
                "POST", _SP_HTTP_ANSWER_PATH, {"question": _question},
                timeout_s=_SP_PER_Q_TIMEOUT_S,
            )
            ok = (
                isinstance(value, dict)
                and isinstance(value.get("answer"), str)
                and isinstance(value.get("retrieved"), (list, tuple))
            )
            return {
                "ok": ok,
                "qid": _qid,
                "latency_s": round(_sp_time.perf_counter() - started, 4),
                "error": None if ok else "invalid_schema",
            }
        except Exception as exc:
            return {
                "ok": False,
                "qid": _qid,
                "latency_s": round(_sp_time.perf_counter() - started, 4),
                "error": type(exc).__name__ + ": " + str(exc),
            }

    def _sp_percentile(values, ratio):
        if not values:
            return None
        pos = min(len(values) - 1, max(0, int((len(values) - 1) * ratio)))
        return round(values[pos], 4)

    def _sp_median(values):
        values = sorted(values)
        if not values:
            return None
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) / 2

    def _sp_perf_round(n_requests, repetition):
        started = _sp_time.perf_counter()
        with _SpThreadPoolExecutor(max_workers=max(1, _SP_PERFORMANCE_CONCURRENCY)) as pool:
            rows = list(pool.map(_sp_perf_one, range(n_requests)))
        wall_s = _sp_time.perf_counter() - started
        ok_rows = [row for row in rows if row["ok"]]
        latencies = sorted(row["latency_s"] for row in ok_rows)
        return {
            "repetition": repetition,
            "transport": "http",
            "requests": n_requests,
            "concurrency": _SP_PERFORMANCE_CONCURRENCY,
            "success": len(ok_rows),
            "fail": len(rows) - len(ok_rows),
            "success_rate": round(len(ok_rows) / len(rows), 4),
            "throughput_rps": round(len(ok_rows) / wall_s, 4) if wall_s else 0.0,
            "wall_s": round(wall_s, 4),
            "p50_latency_s": _sp_percentile(latencies, 0.50),
            "p95_latency_s": _sp_percentile(latencies, 0.95),
            "errors": [row for row in rows if not row["ok"]],
        }

    _sp_warmup = None
    if _SP_PERFORMANCE_WARMUP_REQUESTS > 0:
        _sp_print(
            "[성능] 워밍업 " + str(_SP_PERFORMANCE_WARMUP_REQUESTS) + "요청 실행 중 ...",
            flush=True,
        )
        _sp_warmup = _sp_perf_round(_SP_PERFORMANCE_WARMUP_REQUESTS, 0)

    _sp_perf_samples = []
    for _sp_repetition in range(1, _SP_PERFORMANCE_REPETITIONS + 1):
        _sp_print(
            "[성능] 측정 " + str(_sp_repetition) + "/"
            + str(_SP_PERFORMANCE_REPETITIONS) + " 실행 중 ...",
            flush=True,
        )
        _sp_perf_samples.append(
            _sp_perf_round(_SP_PERFORMANCE_REQUESTS, _sp_repetition)
        )

    _sp_success_median = _sp_median([row["success"] for row in _sp_perf_samples])
    _sp_fail_median = _sp_median([row["fail"] for row in _sp_perf_samples])
    _sp_p50_values = [
        row["p50_latency_s"] for row in _sp_perf_samples
        if row["p50_latency_s"] is not None
    ]
    _sp_p95_values = [
        row["p95_latency_s"] for row in _sp_perf_samples
        if row["p95_latency_s"] is not None
    ]
    _sp_performance = {
        "version": 2,
        "transport": "http",
        "requests": _SP_PERFORMANCE_REQUESTS,
        "concurrency": _SP_PERFORMANCE_CONCURRENCY,
        "success": int(_sp_success_median),
        "fail": int(_sp_fail_median),
        "success_rate": round(_sp_median(
            [row["success_rate"] for row in _sp_perf_samples]
        ), 4),
        "throughput_rps": round(_sp_median(
            [row["throughput_rps"] for row in _sp_perf_samples]
        ), 4),
        "wall_s": round(_sp_median(
            [row["wall_s"] for row in _sp_perf_samples]
        ), 4),
        "p50_latency_s": (
            round(_sp_median(_sp_p50_values), 4) if _sp_p50_values else None
        ),
        "p95_latency_s": (
            round(_sp_median(_sp_p95_values), 4) if _sp_p95_values else None
        ),
        "errors": [
            dict(error, repetition=sample["repetition"])
            for sample in _sp_perf_samples
            for error in sample["errors"]
        ],
        "summary_method": "median",
        "protocol": {
            "requests_per_run": _SP_PERFORMANCE_REQUESTS,
            "concurrency": _SP_PERFORMANCE_CONCURRENCY,
            "warmup_requests": _SP_PERFORMANCE_WARMUP_REQUESTS,
            "repetitions": _SP_PERFORMANCE_REPETITIONS,
        },
        "samples": _sp_perf_samples,
    }
    if _sp_warmup is not None:
        _sp_performance["warmup"] = _sp_warmup
    _sp_print(
        "[성능] closed-loop 중앙값 · "
        + str(_SP_PERFORMANCE_REQUESTS) + "요청 × "
        + str(_SP_PERFORMANCE_REPETITIONS) + "회 · 동시성 "
        + str(_SP_PERFORMANCE_CONCURRENCY) + " · 대표 성공 "
        + str(_sp_performance["success"]) + " · "
        + str(_sp_performance["throughput_rps"]) + " req/s · p95 "
        + str(_sp_performance["p95_latency_s"]) + "s",
        flush=True,
    )

_sp_stop_http_server()

_sp_submission = {"team": _SP_TEAM, "answers": _sp_answers}
if _sp_doc_warnings or _sp_timeouts or _sp_env_warnings or _sp_performance:
    _sp_submission["meta"] = {"doc_name_violations": _sp_doc_warnings,
                              "timeout_qids": _sp_timeouts,
                              "env_warnings": _sp_env_warnings,
                              "transport": _SP_TRANSPORT}
    if _sp_performance:
        _sp_submission["meta"]["performance"] = _sp_performance
_sp_text = _sp_json.dumps(_sp_submission, ensure_ascii=False, indent=2, default=str)
with _sp_open(_SP_OUTPUT_PATH, "w", encoding="utf-8") as _sp_f:
    _sp_f.write(_sp_text)

_sp_print("[완료] " + str(len(_sp_answers)) + "문항 저장: " + _SP_OUTPUT_PATH
      + "  (총 " + str(round(_sp_time.time() - _sp_t0, 1)) + "s)", flush=True)
if _sp_errors:
    _sp_print("[경고] 실패 문항 " + str(len(_sp_errors)) + "건: "
          + ", ".join(_e["qid"] for _e in _sp_errors), flush=True)
if _sp_doc_warnings:
    _sp_print("[경고] 규정 ④ 위반 — 허용 목록 밖 문서명 " + str(len(_sp_doc_warnings)) + "건: "
          + ", ".join(sorted(set(_w["doc"] for _w in _sp_doc_warnings)))
          + "  → 해당 항목은 검색 점수가 0으로 채점됩니다. 허용(띄어쓰기 차이는 무관): "
          + ", ".join(_SP_ALLOWED_DOCS), flush=True)

if _SP_AUTO_DOWNLOAD:
    try:
        from google.colab import files as _sp_files
        _sp_files.download(_SP_OUTPUT_PATH)
        _sp_print("[다운로드] 브라우저 다운로드를 시작했습니다: " + _SP_OUTPUT_PATH, flush=True)
    except Exception as _sp_dl_exc:
        _sp_print("[다운로드] 자동 다운로드 실패(" + type(_sp_dl_exc).__name__ + ": " + str(_sp_dl_exc)
                  + ") — 좌측 파일 탭에서 " + _SP_OUTPUT_PATH + " 를 직접 내려받으세요.", flush=True)

_sp_print("SUBMISSION_RUNNER_DONE " + _sp_json.dumps(
    {"team": _SP_TEAM, "output_path": _SP_OUTPUT_PATH, "n_answers": len(_sp_answers),
     "n_errors": len(_sp_errors), "failed_qids": [_e["qid"] for _e in _sp_errors],
     "n_doc_violations": len(_sp_doc_warnings), "timeout_qids": _sp_timeouts,
     "env_warnings": _sp_env_warnings, "transport": _SP_TRANSPORT,
     "performance": _sp_performance},
    ensure_ascii=False), flush=True)
