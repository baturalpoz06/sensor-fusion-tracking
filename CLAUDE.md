# Project conventions

## Language
- All code, comments, docstrings, commit messages, and string literals in English
- Explanations to the user in Turkish; important terms as English(Türkçe)

## Structure
- src/ layout, package name: fusion
- Subpackages: filters, association, sensors, tracker
- Dependencies are declared only in pyproject.toml (no requirements.txt)

## Division of work
- The user writes filter mathematics (KF/EKF/UKF) and their tests personally
- Do not implement or complete filter math unless explicitly asked
- Help with boilerplate, CI configuration, refactoring, docstrings, and debugging

## Before suggesting a commit
- Run `ruff check .` and `pytest`; both must pass