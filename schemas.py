from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class RoutingDecision(BaseModel):
    agents: List[str] = Field(
        default_factory=list,
        description="Liste des agents à activer (contrats, assurances, faq, vehicules, maintenance)"
    )


class ToolCall(BaseModel):
    tool: str = Field(description="Nom de l'outil à appeler")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments de l'outil")


class ToolResult(BaseModel):
    tool: str
    success: bool
    result: Any = None
    error: Optional[str] = None


class ClaimVerdict(BaseModel):
    claim: str = Field(description="L'affirmation extraite de la réponse")
    supported: bool = Field(description="True si l'affirmation est soutenue par les preuves")
    evidence: List[str] = Field(default_factory=list, description="Extraits des contextes qui soutiennent l'affirmation")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0, description="Confiance du verdict")


class TableToolResult(BaseModel):
    value: float = Field(description="Valeur numérique calculée")
    unit: str = Field(description="Unité (MAD, km, %, mois, etc.)")
    source_chunks: List[str] = Field(default_factory=list, description="IDs des chunks sources")
    formula: str = Field(description="Formule ou règle de calcul utilisée")


class ReActStep(BaseModel):
    step: int
    tool: str
    args: Dict[str, Any]
    result: Any
    tokens_used: int = 0


class ReActState(BaseModel):
    steps: List[ReActStep] = Field(default_factory=list)
    total_tokens: int = 0
    start_time: float = 0.0
    finished: bool = False
    final_answer: Optional[str] = None
    abstention_reason: Optional[str] = None
    agent_cible: Optional[str] = None
    last_tool_result: Any = None


class EvaluationMetrics(BaseModel):
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    recall_at_5: Optional[float] = None
    mrr: Optional[float] = None
    ndcg_at_10: Optional[float] = None
    citation_accuracy: Optional[float] = None
    abstention_correct: Optional[bool] = None
    routing_correct: Optional[bool] = None
    elapsed_sec: Optional[float] = None
    tokens_estimate: Optional[int] = None


class ReformulatedQuestion(BaseModel):
    question: str = Field(description="Question reformulée pour être autonome")