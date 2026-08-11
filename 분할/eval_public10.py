# -*- coding: utf-8 -*-
"""공개 10문항 자체 채점 · 제출 전 수치 확인용 (선택 실행, 제출물 아님).

용도: answers_public_<팀>.json 과 gold_questions_public10.json 을 대조해
      MRR(배점 20)과 키팩트 F1(배점 30)을 재현한다. LLM 판정(배점 50)은 재현할 수 없다.
주의: 키팩트 F1은 공식 채점기 산식을 공개받지 못했으므로 어절 단위 토큰 F1으로 근사한다.
      연습 채점 결과 8/10 문항에서 오차 0.03 이내로 추종하는 것을 확인했다.

실행: python eval_public10.py answers_public_3.json gold_questions_public10.json
"""

import json
import re
import sys
from collections import Counter


def tokenize(text):
    return re.findall(r"[0-9A-Za-z가-힣]+", text)


def token_f1(prediction, reference):
    predicted, referenced = Counter(tokenize(prediction)), Counter(tokenize(reference))
    overlap = sum((predicted & referenced).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(referenced.values())
    return 2 * precision * recall / (precision + recall)


def article_number(value):
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else -1


def normalize_document(value):
    return re.sub(r"\s+", "", str(value))


def evaluate(answers_path, gold_path):
    answers = {a["qid"]: a for a in json.load(open(answers_path, encoding="utf-8"))["answers"]}
    questions = json.load(open(gold_path, encoding="utf-8"))["questions"]

    rows, mrr_total, f1_total = [], 0.0, 0.0
    for question in questions:
        qid = question["id"]
        answer = answers.get(qid, {"retrieved": [], "answer": ""})
        gold = {
            (normalize_document(g["doc"]), int(g["article"])) for g in question["gold_articles"]
        }

        rank = 0
        for position, item in enumerate(answer.get("retrieved", []), start=1):
            if (normalize_document(item[0]), article_number(item[1])) in gold:
                rank = position
                break
        reciprocal = 1.0 / rank if rank else 0.0

        f1 = token_f1(answer.get("answer", ""), " ".join(question["key_facts"]))
        mrr_total += reciprocal
        f1_total += f1
        rows.append((qid, rank, round(reciprocal, 4), round(f1, 4), len(answer.get("answer", ""))))

    n = len(questions)
    mrr, keyfact_f1 = mrr_total / n, f1_total / n

    print("표 1. 문항별 자체 채점")
    print("| qid | 정답 조 순위 | MRR 기여 | 키팩트 F1(근사) | 답변 길이 |")
    print("|---|---|---|---|---|")
    for row in rows:
        print(f"| {row[0]} | {row[1] or '미검출'} | {row[2]} | {row[3]} | {row[4]} |")

    print()
    print(f"MRR          {mrr:.4f} → {mrr * 20:.3f}점 / 20")
    print(f"키팩트 F1    {keyfact_f1:.4f} → {keyfact_f1 * 30:.3f}점 / 30")
    print(f"객관 소계    {mrr * 20 + keyfact_f1 * 30:.3f}점 / 50 (LLM 판정 50점은 별도)")
    return mrr, keyfact_f1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2])