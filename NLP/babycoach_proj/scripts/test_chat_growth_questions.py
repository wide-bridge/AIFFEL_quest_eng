import json
import os
import traceback
from pathlib import Path


def main() -> None:
    os.environ["BABYCOACH_LLM_MOCK"] = "1"

    try:
        import sys

        # Ensure project root is on sys.path so `import app...` works.
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from fastapi.testclient import TestClient

        from app.main import app

        base_dir = Path(__file__).resolve().parents[1]
        sample_path = base_dir / "data" / "sample_input_1.json"
        payload = json.loads(sample_path.read_text(encoding="utf-8"))

        client = TestClient(app)

        rec = client.post("/recommend", json=payload)
        rec_data = rec.json()
        final_output = rec_data["final_output"]

        questions = [
            "잠을 못 잔다는데",
            "밥을 잘 안 먹어요",
            "요즘 놀이를 금방 싫어해요",
            "상호작용을 더 늘리려면?",
        ]

        for q in questions:
            chat_payload = {
                "final_output": final_output,
                "state_summary": final_output.get("chat_context_summary", ""),
                "user_message": q,
            }
            chat_resp = client.post("/chat", json=chat_payload)
            print(f"\n=== Q: {q} ===")
            if not chat_resp.ok:
                print("STATUS:", chat_resp.status_code)
                print("DETAIL:", chat_resp.json().get("detail"))
                continue
            assistant = chat_resp.json().get("assistant_message", "")
            print("A:", assistant)

    except Exception:
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

