#!/usr/bin/env python3
"""
Regression test script - runs after each commit to verify nothing is broken.
Tests: main.py (CLI), api.py (FastAPI), evaluate_ragas.py, evaluate_baseline.py, evaluate_extended.py
All run with --skip-llm-eval for speed.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: list, timeout: int = 60, cwd: str = None) -> tuple[bool, str]:
    """Run command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or "/home/tawil/Bureau/stage d'ete"
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"
    except Exception as e:
        return False, str(e)


def test_cli():
    """Test CLI main.py with a simple question."""
    print("Testing CLI (main.py)...")
    # We can't easily test interactive CLI, so test imports instead
    success, output = run_cmd([sys.executable, "-c", "from core import process_question; print('OK')"], timeout=30)
    if success:
        print("  ✓ core imports work")
    else:
        print(f"  ✗ core import failed: {output[:200]}")
    return success


def test_api_imports():
    """Test API imports."""
    print("Testing API imports...")
    success, output = run_cmd([sys.executable, "-c", "from api import app; print('OK')"], timeout=30)
    if success:
        print("  ✓ api imports work")
    else:
        print(f"  ✗ api import failed: {output[:200]}")
    return success


def test_evaluate_ragas():
    """Test evaluate_ragas.py with --skip-llm-eval on 1 question."""
    print("Testing evaluate_ragas.py...")
    success, output = run_cmd([
        sys.executable, "evaluate_ragas.py", "--skip-llm-eval", "--max-questions", "1"
    ], timeout=120)
    if success and "Evaluation complete" in output:
        print("  ✓ evaluate_ragas works")
    else:
        print(f"  ✗ evaluate_ragas failed: {output[-500:]}")
    return success


def test_evaluate_baseline():
    """Test evaluate_baseline.py with --skip-llm-eval on 1 question."""
    print("Testing evaluate_baseline.py...")
    success, output = run_cmd([
        sys.executable, "evaluate_baseline.py", "--skip-llm-eval", "--max-questions", "1"
    ], timeout=120)
    if success and "Baseline evaluation complete" in output:
        print("  ✓ evaluate_baseline works")
    else:
        print(f"  ✗ evaluate_baseline failed: {output[-500:]}")
    return success


def test_evaluate_extended():
    """Test evaluate_extended.py with --skip-llm-eval on 1 question."""
    print("Testing evaluate_extended.py...")
    success, output = run_cmd([
        sys.executable, "evaluate_extended.py", "--skip-llm-eval", "--max-questions", "1"
    ], timeout=120)
    if success and "Extended evaluation complete" in output:
        print("  ✓ evaluate_extended works")
    else:
        print(f"  ✗ evaluate_extended failed: {output[-500:]}")
    return success


def test_supervisor():
    """Test supervisor routing."""
    print("Testing supervisor...")
    success, output = run_cmd([
        sys.executable, "-c",
        "from supervisor.supervisor import route; print(route('Quelle est la franchise ?'))"
    ], timeout=30)
    if success and "agents" in output:
        print("  ✓ supervisor works")
    else:
        print(f"  ✗ supervisor failed: {output[:200]}")
    return success


def test_graph():
    """Test graph building."""
    print("Testing graph builder...")
    success, output = run_cmd([
        sys.executable, "-c", "from graph.builder import build_graph; g = build_graph(); print('OK')"
    ], timeout=30)
    if success:
        print("  ✓ graph builder works")
    else:
        print(f"  ✗ graph builder failed: {output[:200]}")
    return success


def test_hybrid_retrieval():
    """Test hybrid retrieval."""
    print("Testing hybrid retrieval...")
    success, output = run_cmd([
        sys.executable, "-c",
        "from hybrid_retrieval import retrieve_hybrid; docs, meta = retrieve_hybrid('test', 'faq', 1); print(f'OK: {len(docs)} docs')"
    ], timeout=60)
    if success:
        print("  ✓ hybrid retrieval works")
    else:
        print(f"  ✗ hybrid retrieval failed: {output[:200]}")
    return success


def main():
    print("=" * 60)
    print("REGRESSION TEST SUITE")
    print("=" * 60)
    print()

    tests = [
        ("Core imports", test_cli),
        ("API imports", test_api_imports),
        ("Supervisor", test_supervisor),
        ("Graph builder", test_graph),
        ("Hybrid retrieval", test_hybrid_retrieval),
        ("evaluate_ragas", test_evaluate_ragas),
        ("evaluate_baseline", test_evaluate_baseline),
        ("evaluate_extended", test_evaluate_extended),
    ]

    results = []
    for name, test_fn in tests:
        start = time.time()
        try:
            ok = test_fn()
            elapsed = time.time() - start
            results.append((name, ok, elapsed))
        except Exception as e:
            elapsed = time.time() - start
            results.append((name, False, elapsed))
            print(f"  ✗ {name} crashed: {e}")
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, ok, elapsed in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status} | {name:25s} | {elapsed:.1f}s")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("ALL TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())