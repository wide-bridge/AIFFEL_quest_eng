from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from .agents.epigenetic_agent import epigenetic_agent
from .agents.explanation_agent import explanation_agent
from .agents.growth_agent import growth_agent
from .agents.interaction_agent import interaction_agent
from .agents.nudge_agent import nudge_agent
from .agents.nutrition_agent import nutrition_agent
from .agents.play_agent import play_agent
from .agents.ranker_agent import ranker_agent
from .formatter import format_final_output
from .state import BabyCoachState


def build_graph():
    """
    LangGraph orchestration:
    Nutrition → Play → Interaction → Epigenetic → Growth → Ranker → Nudge → Explanation → Formatter
    """

    graph = StateGraph(BabyCoachState)

    graph.add_node("nutrition", nutrition_agent)
    graph.add_node("play", play_agent)
    graph.add_node("interaction", interaction_agent)
    graph.add_node("epigenetic", epigenetic_agent)
    graph.add_node("growth", growth_agent)
    graph.add_node("ranker", ranker_agent)
    graph.add_node("nudge", nudge_agent)
    graph.add_node("explanation", explanation_agent)
    graph.add_node("formatter", _formatter_node)

    graph.set_entry_point("nutrition")
    graph.add_edge("nutrition", "play")
    graph.add_edge("play", "interaction")
    graph.add_edge("interaction", "epigenetic")
    graph.add_edge("epigenetic", "growth")
    graph.add_edge("growth", "ranker")
    graph.add_edge("ranker", "nudge")
    graph.add_edge("nudge", "explanation")
    graph.add_edge("explanation", "formatter")
    graph.add_edge("formatter", END)

    return graph.compile()


def _formatter_node(state: BabyCoachState) -> Dict[str, Any]:
    """
    Final node: build `final_output`.
    """

    final_output = format_final_output(state)
    return {"final_output": final_output, "chat_context_summary": final_output.get("chat_context_summary", "")}


_COMPILED_GRAPH = None


def get_compiled_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph()
    return _COMPILED_GRAPH


def run_recommendation(input_state: BabyCoachState) -> BabyCoachState:
    """
    Run the BabyCoach LangGraph pipeline and return final state.
    """

    graph = get_compiled_graph()
    result: Any = graph.invoke(dict(input_state))
    return result

