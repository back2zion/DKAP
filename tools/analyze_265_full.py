#!/usr/bin/env python3
"""
Full DKS analysis of 265 GitHub repos.
Shallow-clones each repo, runs dks_analyzer, saves results.
Resumes from previous progress.

Usage: python3.10 analyze_265_full.py
"""
import json
import os
import subprocess
import sys
import time
import shutil
from pathlib import Path

# Import dks_analyzer
sys.path.insert(0, os.path.dirname(__file__))
from dks_analyzer import analyze_repo, DKSResult
from dataclasses import asdict

REPOS_FILE = "/home/babelai/DKAP/tools/repos_all_265.txt"
CLONE_DIR = "/home/babelai/DKAP/tools/cloned_repos"
OUTPUT_FILE = "/home/babelai/DKAP/tools/dks_265_results.json"
ERRORS_FILE = "/home/babelai/DKAP/tools/dks_265_errors.json"


def load_repos():
    """Load repo URLs from file."""
    with open(REPOS_FILE) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def clone_shallow(url, target_dir):
    """Shallow clone a repo. Returns path or None on failure."""
    repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
    owner = url.rstrip("/").split("/")[-2]
    local_name = f"{owner}__{repo_name}"
    repo_path = os.path.join(target_dir, local_name)

    if os.path.exists(repo_path):
        return repo_path

    try:
        r = subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", "-q", url, repo_path],
            capture_output=True, text=True, timeout=180
        )
        if r.returncode != 0:
            return None
        return repo_path
    except Exception as e:
        return None


def cleanup_repo(repo_path):
    """Remove cloned repo to save disk space."""
    try:
        shutil.rmtree(repo_path, ignore_errors=True)
    except:
        pass


def main():
    urls = load_repos()
    total = len(urls)
    print(f"Total repos to analyze: {total}")

    os.makedirs(CLONE_DIR, exist_ok=True)

    # Load previous progress
    try:
        with open(OUTPUT_FILE) as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} already analyzed")
    except:
        results = {}

    try:
        with open(ERRORS_FILE) as f:
            errors = json.load(f)
    except:
        errors = {}

    for i, url in enumerate(urls):
        owner_repo = "/".join(url.rstrip("/").split("/")[-2:])

        if owner_repo in results or owner_repo in errors:
            continue

        print(f"[{i+1}/{total}] {owner_repo}...", end=" ", flush=True)

        # Clone
        repo_path = clone_shallow(url, CLONE_DIR)
        if not repo_path:
            print("CLONE FAILED")
            errors[owner_repo] = "clone_failed"
            continue

        # Analyze with timeout via signal
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Analysis took too long")

        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(120)  # 2 min timeout per repo
            r = analyze_repo(repo_path)
            signal.alarm(0)
            result_dict = asdict(r)
            result_dict["dks_total"] = r.dks_total
            result_dict["url"] = url
            results[owner_repo] = result_dict
            print(f"DKS={r.dks_total}, LOC={r.python_loc:,}, files={r.python_files}")
        except TimeoutError:
            signal.alarm(0)
            print(f"TIMEOUT (>120s)")
            errors[owner_repo] = "timeout"
        except Exception as e:
            signal.alarm(0)
            print(f"ANALYZE FAILED: {e}")
            errors[owner_repo] = str(e)

        # Cleanup to save disk
        cleanup_repo(repo_path)

        # Save progress every 5 repos
        if (i + 1) % 5 == 0 or i == total - 1:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            with open(ERRORS_FILE, "w") as f:
                json.dump(errors, f, indent=2)
            sys.stdout.flush()

    # Final save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(ERRORS_FILE, "w") as f:
        json.dump(errors, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE: {len(results)} analyzed, {len(errors)} errors")
    print(f"Results: {OUTPUT_FILE}")

    # Quick summary
    dks_scores = [v["dks_total"] for v in results.values()]
    if dks_scores:
        dks_scores.sort()
        print(f"DKS range: {min(dks_scores)} ~ {max(dks_scores)}")
        print(f"DKS median: {dks_scores[len(dks_scores)//2]}")
        print(f"DKS mean: {sum(dks_scores)/len(dks_scores):.1f}")


if __name__ == "__main__":
    main()
