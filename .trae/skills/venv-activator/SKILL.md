---
name: "venv-activator"
description: "Activates project virtual environment before running Python commands. Invoke when running Python scripts or installing dependencies to ensure isolated environment usage."
---

# Virtual Environment Activator

## Purpose

This skill ensures that Python commands are always executed within the project's virtual environment (`venv/`) instead of the global Python environment.

## When to Use

- Before running any Python script in the project
- Before installing Python dependencies with pip
- When working with LangChain or other Python packages to avoid version conflicts

## Usage

**Activation Command:**
```bash
source venv/bin/activate
```

**Verification:**
After activation, the terminal prompt will show `(venv)` prefix:
```bash
(venv) user@host:~/projects/202605-ai-study$
```

**Run Python Script:**
```bash
python src/demos/20260509-demo1-LangChainDemo1/chapter5_memory.py
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

## Workflow

1. Check if virtual environment exists at `venv/`
2. Activate the virtual environment
3. Verify activation by checking `$VIRTUAL_ENV` environment variable
4. Execute Python commands within the activated environment

## Benefits

- Isolated dependency management
- Avoid version conflicts between projects
- Clean global Python environment
- Reproducible development environment
