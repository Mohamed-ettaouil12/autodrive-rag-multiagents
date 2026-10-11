from typing import TypedDict, List, Any, Dict


class AgentState(TypedDict):
    question: str
    question_reformulee: str
    agents: List[str]
    documents: List[str]
    reponse: str
    sources: List[Any]
    session_id: str
    historique: List[Dict[str, str]]