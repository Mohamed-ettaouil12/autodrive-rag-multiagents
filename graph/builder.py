import logging
from langgraph.graph import StateGraph, END
from graph.state import AgentState
from graph.nodes import (
    superviseur,
    agent_contrats,
    agent_assurances,
    agent_faq,
    agent_vehicules,
    agent_maintenance,
    hors_sujet,
    reformulateur,
)
from config import ENABLE_REACT

LOG_FORMAT = "%(asctime)s [BUILDER] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("builder")


AGENT_PRIORITY = ["contrats", "assurances", "faq", "vehicules", "maintenance"]
AGENT_NODE_MAP = {
    "contrats": "agent_contrats",
    "assurances": "agent_assurances",
    "faq": "agent_faq",
    "vehicules": "agent_vehicules",
    "maintenance": "agent_maintenance",
}


def should_route(state: AgentState) -> str:
    agents = state.get("agents", [])

    if not agents:
        return "hors_sujet"

    if not isinstance(agents, list) or not all(isinstance(a, str) for a in agents):
        logger.warning("Liste d'agents invalide: %s, fallback vers faq", agents)
        return "agent_faq"

    for agent in AGENT_PRIORITY:
        if agent in agents:
            return AGENT_NODE_MAP[agent]

    logger.warning("Aucun agent reconnu dans: %s, fallback vers faq", agents)
    return "agent_faq"


def should_route_react(state: AgentState) -> str:
    """Route for ReAct mode: always goes to agent_react unless hors_sujet."""
    agents = state.get("agents", [])
    if not agents:
        return "hors_sujet"
    return "agent_react"


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("reformulateur", reformulateur)
    workflow.add_node("superviseur", superviseur)
    workflow.add_node("hors_sujet", hors_sujet)

    if ENABLE_REACT:
        # ReAct mode: single agent_react node
        from graph.nodes_react import agent_react
        workflow.add_node("agent_react", agent_react)
        
        workflow.set_entry_point("reformulateur")
        workflow.add_edge("reformulateur", "superviseur")
        
        workflow.add_conditional_edges(
            "superviseur",
            should_route_react,
            {
                "agent_react": "agent_react",
                "hors_sujet": "hors_sujet",
            },
        )
        workflow.add_edge("agent_react", END)
        workflow.add_edge("hors_sujet", END)
        logger.info("Graph built in REACT mode")
    else:
        # Linear mode (current behavior)
        workflow.add_node("agent_contrats", agent_contrats)
        workflow.add_node("agent_assurances", agent_assurances)
        workflow.add_node("agent_faq", agent_faq)
        workflow.add_node("agent_vehicules", agent_vehicules)
        workflow.add_node("agent_maintenance", agent_maintenance)

        workflow.set_entry_point("reformulateur")
        workflow.add_edge("reformulateur", "superviseur")

        workflow.add_conditional_edges(
            "superviseur",
            should_route,
            {
                "agent_contrats": "agent_contrats",
                "agent_assurances": "agent_assurances",
                "agent_faq": "agent_faq",
                "agent_vehicules": "agent_vehicules",
                "agent_maintenance": "agent_maintenance",
                "hors_sujet": "hors_sujet",
            },
        )

        for agent_node in AGENT_NODE_MAP.values():
            workflow.add_edge(agent_node, END)
        workflow.add_edge("hors_sujet", END)
        logger.info("Graph built in LINEAR mode (legacy)")

    return workflow.compile()


if __name__ == "__main__":
    graph = build_graph()
    print("Graph built successfully")
    print(graph.get_graph().draw_ascii())