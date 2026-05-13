"""agent-pathologies: trajectory pathologies in multi-turn LLM agents."""

__version__ = "0.2.0"

# Auto-load .env so OPENROUTER_API_KEY (and friends) are picked up by any
# script that imports the package. Silent no-op if dotenv or .env is missing.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass
