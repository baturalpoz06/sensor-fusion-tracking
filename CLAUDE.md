# Project conventions

## Language
- All code, comments, docstrings, commit messages, and string literals in English
- Explanations to the user in Turkish; important terms as English(Türkçe)

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