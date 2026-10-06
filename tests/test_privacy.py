from pathlib import Path


FORBIDDEN_PATTERNS = [
    "tumorlocationmaster_export",
    "BreatCancer_ALLDataForML_Numeric",
    "Date of diagnosis",
    "Date of death",
    "Date of last follow up",
]


def test_original_data_references_are_not_present():
    root = Path(__file__).parents[1]
    public_files = [
        *root.glob("*.md"),
        *root.glob("*.toml"),
        *root.glob("*.txt"),
        *root.glob("src/**/*.py"),
        *root.glob("notebooks/*.ipynb"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in public_files)
    for pattern in FORBIDDEN_PATTERNS:
        assert pattern not in combined
