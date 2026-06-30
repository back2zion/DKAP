#!/usr/bin/env python3
"""
Collect detailed metadata for all 265 GitHub repos via API.
Outputs: tools/repos_265_full.json
"""
import json
import subprocess
import sys
import time

INPUT = "/home/babelai/DKAP/tools/github_population.json"
OUTPUT = "/home/babelai/DKAP/tools/repos_265_full.json"

def gh_api(endpoint, jq_filter=None):
    """Call GitHub API via gh CLI."""
    cmd = ["gh", "api", endpoint]
    if jq_filter:
        cmd += ["--jq", jq_filter]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None
        return r.stdout.strip()
    except:
        return None

def gh_api_json(endpoint):
    """Call GitHub API and return parsed JSON."""
    cmd = ["gh", "api", endpoint]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None
        return json.loads(r.stdout)
    except:
        return None

def collect_repo(owner_repo):
    """Collect data for a single repo."""
    result = {}

    # 1. Languages (bytes per language)
    langs = gh_api_json(f"repos/{owner_repo}/languages")
    result["languages"] = langs or {}

    # 2. Top-level directory listing (tree with depth 1)
    tree_data = gh_api_json(f"repos/{owner_repo}/git/trees/HEAD")
    if tree_data and "tree" in tree_data:
        entries = tree_data["tree"]
        result["top_dirs"] = [e["path"] for e in entries if e["type"] == "tree"]
        result["top_files"] = [e["path"] for e in entries if e["type"] == "blob"]
    else:
        # Try main/master branch
        tree_data = gh_api_json(f"repos/{owner_repo}/git/trees/main")
        if not tree_data or "tree" not in tree_data:
            tree_data = gh_api_json(f"repos/{owner_repo}/git/trees/master")
        if tree_data and "tree" in tree_data:
            entries = tree_data["tree"]
            result["top_dirs"] = [e["path"] for e in entries if e["type"] == "tree"]
            result["top_files"] = [e["path"] for e in entries if e["type"] == "blob"]
        else:
            result["top_dirs"] = []
            result["top_files"] = []

    # 3. Key config/domain directories (check if they exist)
    domain_dirs = ["config", "configs", "schema", "schemas", "models",
                   "rules", "ontology", "knowledge", "domain",
                   "locale", "i18n", "translations", "prompts",
                   "data", "sql", "migrations", "tests"]
    result["has_dirs"] = [d for d in domain_dirs if d in result["top_dirs"]]

    # 4. Repo detail (description, default_branch, has_wiki, license, forks, open_issues)
    info = gh_api_json(f"repos/{owner_repo}")
    if info:
        result["description"] = info.get("description", "")
        result["default_branch"] = info.get("default_branch", "")
        result["license"] = info.get("license", {}).get("spdx_id", "") if info.get("license") else ""
        result["forks_count"] = info.get("forks_count", 0)
        result["open_issues_count"] = info.get("open_issues_count", 0)
        result["has_wiki"] = info.get("has_wiki", False)
        result["archived"] = info.get("archived", False)
        result["topics"] = info.get("topics", [])
        result["stars"] = info.get("stargazers_count", 0)
        result["size_kb"] = info.get("size", 0)
        result["created_at"] = info.get("created_at", "")
        result["updated_at"] = info.get("updated_at", "")
        result["pushed_at"] = info.get("pushed_at", "")

    return result


def main():
    with open(INPUT) as f:
        population = json.load(f)

    repos = list(population.keys())
    total = len(repos)
    print(f"Collecting data for {total} repos...")

    # Load existing progress if any
    try:
        with open(OUTPUT) as f:
            results = json.load(f)
        print(f"  Resuming from {len(results)} already collected")
    except:
        results = {}

    for i, owner_repo in enumerate(repos):
        if owner_repo in results:
            continue

        print(f"  [{i+1}/{total}] {owner_repo}...", end=" ", flush=True)
        data = collect_repo(owner_repo)
        results[owner_repo] = data

        lang_count = len(data.get("languages", {}))
        dir_count = len(data.get("top_dirs", []))
        print(f"langs={lang_count}, dirs={dir_count}")

        # Save progress every 10 repos
        if (i + 1) % 10 == 0:
            with open(OUTPUT, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  ... saved progress ({len(results)}/{total})")

        # Rate limit: ~3 calls per repo, stay under 5000/hr
        # 265 * 4 = 1060 calls, well within limit
        time.sleep(0.3)

    # Final save
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDone! {len(results)} repos saved to {OUTPUT}")


if __name__ == "__main__":
    main()
