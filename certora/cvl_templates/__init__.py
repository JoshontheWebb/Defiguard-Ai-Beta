"""
CVL Template Specifications for common contract patterns.

These templates provide baseline formal verification rules
for standard contract types like ERC20, ERC721, etc.
"""

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent


def load_template(name: str) -> str:
    """Load a CVL template by name."""
    template_path = TEMPLATES_DIR / f"{name}.spec"
    if template_path.exists():
        return template_path.read_text()
    return ""


def list_templates() -> list[str]:
    """List available template names."""
    return [f.stem for f in TEMPLATES_DIR.glob("*.spec")]
