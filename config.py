import os, torch
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ---------------------------
#  LLM & embeddings
# ---------------------------
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_EMBED = "text-embedding-3-small"

# ---------------------------
#  Data / vector store paths
# ---------------------------
REPO_URL  = "https://github.com/Anduin2017/HowToCook"
REPO_DIR  = BASE_DIR / "howtocook_repo"
VECTOR_DB = BASE_DIR / "vector_store"

# ---------------------------
#  Devices (kept like project1)
# ---------------------------
EMBEDDING_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ---------------------------
#  MCP endpoints (env-vars)
# ---------------------------
CV_MCP_URL     = os.getenv("CV_MCP_URL",     "http://cv:8080")
SEARCH_MCP_URL = os.getenv("SEARCH_MCP_URL", "http://search:8082")
