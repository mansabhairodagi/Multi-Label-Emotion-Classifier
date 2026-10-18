"""Checks that project files stay in the expected directories."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_FILES = (
    "src/main.py",
    "src/predict.py",
    "src/app.py",
    "src/emotion_model.py",
    "scripts/DistillBERT.py",
    "scripts/README.md",
    "tests/test_repository_layout.py",
    "docs/ARCHITECTURE.md",
    "data/track-a.csv",
    "data/track-a-test.csv",
    "assets/demo-preview.png",
    "assets/dashboard-preview.png",
    "assets/Text_Emotion_Analytics.pbix",
    "README.md",
    "LICENSE",
    ".gitignore",
    "requirements.txt",
)

ALLOWED_ROOT_NAMES = {
    ".git",
    ".gitignore",
    "LICENSE",
    "README.md",
    "requirements.txt",
    "src",
    "scripts",
    "tests",
    "docs",
    "data",
    "assets",
}


class RepositoryLayoutTest(unittest.TestCase):
    def test_expected_files_exist(self):
        missing = [path for path in EXPECTED_FILES if not (ROOT / path).is_file()]
        self.assertEqual(missing, [])

    def test_root_directories_are_intentional(self):
        unexpected = sorted(
            entry.name
            for entry in ROOT.iterdir()
            if entry.name not in ALLOWED_ROOT_NAMES
            and not entry.name.startswith(".")
        )
        self.assertEqual(unexpected, [])

    def test_gitignore_covers_secrets_and_artifacts(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for token in (".env", "artifacts/", ".venv/", "*.pem", "*.key", "secrets/"):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
