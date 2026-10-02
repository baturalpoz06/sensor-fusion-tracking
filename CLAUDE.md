# Project conventions

## Language
- All code, comments, docstrings, commit messages, README.md, and any other file in the repository: English only, with no Turkish words and no parenthetical translations
- Chat explanations to the user: Turkish, with important terms as English(Türkçe). This format is for chat only and never goes into a file

## Structure
- src/ layout, package name: fusion
- Subpackages: filters, association, sensors, tracker
- Dependencies are declared only in pyproject.toml (no requirements.txt)

## Working style
- Claude Code writes the code, including filter mathematics; the user reviews it
- After each change, explain in Turkish what it does and why, briefly
- Every new function comes with pytest tests

## Before suggesting a commit
- Run `ruff check .` and `pytest`; both must pass
- Stage files with explicit paths (git add src tests scripts), never git add .
- Never commit .claude/ or any personal settings file