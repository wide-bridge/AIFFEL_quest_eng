import json
import os

# Smoke tests should not depend on real OpenAI calls.
os.environ["BABYCOACH_LLM_MOCK"] = "1"

from fastapi.testclient import TestClient

from app.graph import run_recommendation
from app.main import app as fastapi_app


def _read_sample(name: str) -> dict:
    with open(os.path.join(os.path.dirname(__file__), "..", "data", name), "r", encoding="utf-8") as f:
        return json.load(f)


def test_graph_runs_and_formats_final_output():
    payload = _read_sample("sample_input_1.json")
    state = run_recommendation(payload)
    final_output = state.get("final_output")
    assert isinstance(final_output, dict)

    assert final_output.get("spoon", {}).get("title") == "Spoon"
    assert isinstance(final_output.get("spoon", {}).get("suggestions", []), list)
    assert final_output.get("play", {}).get("title") == "Play"
    assert final_output.get("growth", {}).get("title") == "Growth"
    assert isinstance(final_output.get("nudge", {}).get("nudge_message"), str)
    assert isinstance(final_output.get("explanation", {}).get("explanation"), str)

    assert isinstance(final_output.get("chat_context_summary"), str)


def test_recommend_and_chat_endpoints():
    client = TestClient(fastapi_app)
    payload = _read_sample("sample_input_1.json")

    rec = client.post("/recommend", json=payload)
    assert rec.status_code == 200
    rec_data = rec.json()
    assert "final_output" in rec_data
    assert isinstance(rec_data["final_output"], dict)

    chat = client.post(
        "/chat",
        json={
            "final_output": rec_data["final_output"],
            "state_summary": rec_data["final_output"].get("chat_context_summary"),
            "user_message": "왜 이런 놀이를 추천했어?"
        },
    )
    assert chat.status_code == 200
    chat_data = chat.json()
    assert isinstance(chat_data.get("assistant_message"), str)
    assert chat_data["assistant_message"].strip() != ""

