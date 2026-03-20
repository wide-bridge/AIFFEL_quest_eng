import json
import os
import re
import sys
import traceback
from pathlib import Path


def _sentence_count_heuristic(text: str) -> int:
    """
    Estimate sentence count for Korean text.
    We treat `.?!。` as sentence boundaries.
    """

    if not isinstance(text, str):
        return 0
    parts = [p.strip() for p in re.split(r"[.!?。]+", text) if p.strip()]
    return len(parts)


def _nudge_status(nudge_message: str) -> str:
    """
    Report whether nudge is single-sentence and short enough.
    """

    sent_cnt = _sentence_count_heuristic(nudge_message)
    nudge_len = len(nudge_message or "")
    if sent_cnt != 1:
        return "개선 필요(문장 수 아님)"
    if nudge_len > 60:
        return "개선 필요(너무 김)"
    return "OK"


def main() -> None:
    os.environ["BABYCOACH_LLM_MOCK"] = "1"

    try:
        # Ensure project root is on sys.path so `import app...` works.
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from fastapi.testclient import TestClient

        from app.main import app

        base_dir = Path(__file__).resolve().parents[1]
        sample_path = base_dir / "data" / "sample_input_1.json"
        sample_payload = json.loads(sample_path.read_text(encoding="utf-8"))

        client = TestClient(app)

        health_resp = client.get("/health")
        health = health_resp.json()

        rec_resp = client.post("/recommend", json=sample_payload)
        recommend_status_code = rec_resp.status_code

        rec_data = rec_resp.json()
        final_output = rec_data.get("final_output", {})

        expected_keys = {
            "spoon",
            "play",
            "growth",
            "nudge",
            "explanation",
            "chat_context_summary",
        }

        actual_keys = set(final_output.keys()) if isinstance(final_output, dict) else set()
        schema_matches_expected = actual_keys == expected_keys

        nudge_message = ""
        if isinstance(final_output, dict):
            nudge_message = (final_output.get("nudge") or {}).get("nudge_message", "")

        nudge_message_len = len(nudge_message or "")
        explanation = ""
        if isinstance(final_output, dict):
            explanation = (final_output.get("explanation") or {}).get("explanation", "")

        chat_payload = {
            "final_output": final_output,
            "state_summary": final_output.get("chat_context_summary", ""),
            "user_message": "왜 이런 놀이를 추천했어?",
        }
        chat_resp = client.post("/chat", json=chat_payload)
        chat_status_code = chat_resp.status_code
        chat_data = chat_resp.json()
        assistant_message = chat_data.get("assistant_message", "")

        assistant_message_prefix_200 = assistant_message[:200]

        print("[1] 실행 명령")
        print(r"python scripts\verify_poc.py")

        print("[2] 성공/실패")
        print("성공")

        print("[3] health 응답")
        print(json.dumps(health, ensure_ascii=False))

        print("[4] recommend_status_code")
        print(recommend_status_code)

        print("[5] final_output_keys")
        print(sorted(list(actual_keys)))

        print("[6] schema_matches_expected")
        print(schema_matches_expected)

        print("[7] nudge_message")
        print(nudge_message)

        print("[8] nudge_message_len")
        print(nudge_message_len)

        sent_cnt = _sentence_count_heuristic(nudge_message)
        status = _nudge_status(nudge_message)
        print(f"[추가] nudge_sentence_count={sent_cnt}, status={status}")

        print("[9] explanation")
        print(explanation)

        print("[10] chat_status_code")
        print(chat_status_code)

        print("[11] assistant_message_prefix_200")
        print(assistant_message_prefix_200)

        # 최종 스키마 일치 검증용 추가 정보
        if not schema_matches_expected:
            print("[추가] 스키마 불일치: expected_keys != actual_keys")

    except Exception:
        print("[1] 실행 명령")
        print(r"python scripts\verify_poc.py")
        print("[2] 성공/실패")
        print("실패")
        print("[12] 실패 시 에러 원문")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

