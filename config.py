import os
from dotenv import load_dotenv

load_dotenv()

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "autodrive123")

# Feature flags for ablation study (read from env with defaults)
ENABLE_STRUCTURED_OUTPUT = os.getenv("ENABLE_STRUCTURED_OUTPUT", "True").lower() == "true"
ENABLE_TABLE_TOOL = os.getenv("ENABLE_TABLE_TOOL", "True").lower() == "true"
ENABLE_REACT = os.getenv("ENABLE_REACT", "False").lower() == "true"
ENABLE_CLAIM_CHECKER = os.getenv("ENABLE_CLAIM_CHECKER", "False").lower() == "true"

# Structured output settings
MAX_JSON_RETRIES = int(os.getenv("MAX_JSON_RETRIES", "2"))
OLLAMA_JSON_FORMAT = os.getenv("OLLAMA_JSON_FORMAT", "True").lower() == "true"

# ReAct settings
MAX_REACT_STEPS = int(os.getenv("MAX_REACT_STEPS", "3"))
REACT_TOKEN_BUDGET = int(os.getenv("REACT_TOKEN_BUDGET", "4000"))
REACT_TIME_BUDGET_SEC = int(os.getenv("REACT_TIME_BUDGET_SEC", "30"))

# Claim checker settings
MAX_CLAIM_RETRIES = int(os.getenv("MAX_CLAIM_RETRIES", "1"))
CLAIM_CHECKER_TEMPERATURE = float(os.getenv("CLAIM_CHECKER_TEMPERATURE", "0.0"))

# Table calculator settings
TABLE_CALCULATOR_CATEGORIES = ["contrats", "assurances", "maintenance"]