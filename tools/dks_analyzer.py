#!/usr/bin/env python3
"""
DKS Analyzer — Automated Domain Knowledge Score Proxy Estimator
Analyzes GitHub repositories to estimate DKS component scores automatically.

Usage:
    python dks_analyzer.py https://github.com/org/repo
    python dks_analyzer.py /path/to/local/repo
    python dks_analyzer.py --batch repos.txt        # one URL per line
    python dks_analyzer.py --validate               # compare with manual scores

Author: Dooil Kwak (Soongsil University / Datastreams Corp.)
"""

import os
import sys
import re
import json
import glob
import subprocess
import tempfile
import shutil
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# DKS Component Proxy Definitions
# ---------------------------------------------------------------------------

# D1: Domain lexicon — config/glossary files with domain-specific keys
LEXICON_FILE_PATTERNS = [
    "**/glossary*", "**/lexicon*", "**/vocabulary*", "**/vocab*",
    "**/dictionary*", "**/terms*", "**/taxonomy*",
]
LEXICON_EXTENSIONS = {".json", ".yaml", ".yml", ".csv", ".txt", ".tsv"}

# D2: Ontology/KG — graph-related imports and structures
ONTOLOGY_IMPORTS = [
    r"import\s+networkx", r"from\s+networkx", r"import\s+dgl", r"from\s+dgl",
    r"import\s+neo4j", r"from\s+neo4j", r"import\s+rdflib", r"from\s+rdflib",
    r"import\s+owlready", r"from\s+owlready", r"import\s+pyke", r"from\s+pyke",
    r"knowledge.graph", r"KnowledgeGraph", r"ontology", r"triple",
]
ONTOLOGY_CLASS_PATTERNS = [
    r"class\s+\w*(Graph|Ontology|KG|Knowledge|Entity|Relation|Triple|Node|Edge)\w*",
]

# D3: Cross-ontology — mapping between standards
CROSS_ONTOLOGY_KEYWORDS = [
    "icd", "snomed", "loinc", "fhir", "omop", "umls", "mesh",
    "schema_mapping", "ontology_mapping", "cross_reference", "alignment",
    "mapping_table", "code_mapping", "standard_mapping",
]

# D4: Schema/DDL — database schemas and structured data models
SCHEMA_PATTERNS = [
    r"CREATE\s+TABLE", r"ALTER\s+TABLE", r"FOREIGN\s+KEY",
    r"class\s+\w+\(.*BaseModel\)", r"class\s+\w+\(.*DataClass\)",
    r"@dataclass", r"class\s+\w+\(.*Schema\)", r"class\s+\w+\(.*Model\)",
]
SCHEMA_FILE_PATTERNS = [
    "**/*.sql", "**/schema*", "**/models.py", "**/model.py",
    "**/tables.json", "**/ddl*",
]

# D5: Domain profiles — configuration files
PROFILE_EXTENSIONS = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"}
PROFILE_DIRS = ["config", "configs", "conf", "settings", "recipes", "profiles", "hparams"]

# D6: Rule patterns — regex and business rules in code
RULE_PATTERNS = [
    r"re\.compile\(", r"re\.match\(", r"re\.search\(", r"re\.findall\(",
    r"re\.sub\(", r"pattern\s*=\s*r['\"]", r"regex\s*=",
    r"rule_", r"validate_", r"constraint_",
]

# D7: Multi-language — internationalization
MULTILANG_PATTERNS = [
    r"i18n", r"locale", r"gettext", r"lang_code", r"language_code",
    r"multilingual", r"bilingual", r"translation",
]
MULTILANG_FILE_PATTERNS = ["**/locale*", "**/i18n*", "**/lang*", "**/translations*"]

# D8: Regulatory/compliance
REGULATORY_KEYWORDS = [
    "pii", "gdpr", "hipaa", "compliance", "regulatory",
    "anonymiz", "redact", "pii_mask", "privacy_policy", "consent_form",
    "data_protection_officer", "rbac_policy",
]


@dataclass
class DKSResult:
    """Automated DKS proxy scores for a single repository."""
    repo_name: str
    repo_path: str

    # Raw counts
    d1_lexicon_files: int = 0
    d1_lexicon_entries: int = 0
    d2_ontology_imports: int = 0
    d2_ontology_classes: int = 0
    d3_cross_ontology_hits: int = 0
    d4_schema_patterns: int = 0
    d4_schema_files: int = 0
    d4_dataclass_count: int = 0
    d5_config_files: int = 0
    d5_config_dirs: int = 0
    d6_regex_patterns: int = 0
    d6_rule_functions: int = 0
    d7_multilang_hits: int = 0
    d7_locale_files: int = 0
    d8_regulatory_hits: int = 0

    # Proxy scores (0-15 or 0-10 scale matching DKS rubric)
    d1_score: int = 0
    d2_score: int = 0
    d3_score: int = 0
    d4_score: int = 0
    d5_score: int = 0
    d6_score: int = 0
    d7_score: int = 0
    d8_score: int = 0

    # Code metrics
    python_loc: int = 0
    python_files: int = 0
    total_files: int = 0

    @property
    def dks_total(self) -> int:
        return (self.d1_score + self.d2_score + self.d3_score + self.d4_score +
                self.d5_score + self.d6_score + self.d7_score + self.d8_score)


def clone_repo(url: str, target_dir: str) -> str:
    """Clone a GitHub repository. Returns path to cloned repo."""
    repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_path = os.path.join(target_dir, repo_name)
    if os.path.exists(repo_path):
        return repo_path
    print(f"  Cloning {url}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--single-branch", url, repo_path],
        capture_output=True, timeout=120
    )
    return repo_path


MAX_FILES_SCAN = 5000       # max files to scan per repo (avoid OOM on huge repos)
MAX_FILE_SIZE = 1_000_000   # max file size to read (1MB)
MAX_JSON_SIZE = 500_000     # max JSON file size to parse


def count_python_loc(repo_path: str) -> tuple[int, int]:
    """Count Python lines of code and file count."""
    loc = 0
    file_count = 0
    for py_file in Path(repo_path).rglob("*.py"):
        if file_count >= MAX_FILES_SCAN:
            break
        # Skip tests, examples, docs
        parts = str(py_file).lower()
        if any(skip in parts for skip in ["test", "example", "doc", "benchmark", "demo", "node_modules", ".git"]):
            continue
        try:
            if py_file.stat().st_size > MAX_FILE_SIZE:
                continue
            loc += sum(1 for line in open(py_file, errors="ignore") if line.strip())
            file_count += 1
        except:
            pass
    return loc, file_count


def search_files(repo_path: str, pattern: str) -> list[str]:
    """Find files matching glob pattern."""
    return glob.glob(os.path.join(repo_path, pattern), recursive=True)


def search_content(repo_path: str, patterns: list[str], extensions: set[str] = None) -> int:
    """Count regex pattern matches across Python files."""
    if extensions is None:
        extensions = {".py"}
    total = 0
    scanned = 0
    for ext in extensions:
        for f in Path(repo_path).rglob(f"*{ext}"):
            if scanned >= MAX_FILES_SCAN:
                break
            parts = str(f).lower()
            if any(skip in parts for skip in [".git", "node_modules", "__pycache__"]):
                continue
            try:
                if f.stat().st_size > MAX_FILE_SIZE:
                    continue
                content = f.read_text(errors="ignore")
                for pat in patterns:
                    total += len(re.findall(pat, content, re.IGNORECASE))
                scanned += 1
            except:
                pass
    return total


def count_config_files(repo_path: str) -> tuple[int, int]:
    """Count configuration files and config directories."""
    config_files = 0
    config_dirs = 0
    for ext in PROFILE_EXTENSIONS:
        count = 0
        for f in Path(repo_path).rglob(f"*{ext}"):
            if count >= MAX_FILES_SCAN:
                break
            parts = str(f).lower()
            if any(s in parts for s in ["test", "example", "node_modules", ".git", "__pycache__"]):
                continue
            count += 1
        config_files += count

    for d in PROFILE_DIRS:
        dirs = list(Path(repo_path).rglob(d))
        config_dirs += len(dirs)

    return config_files, config_dirs


def analyze_repo(repo_path: str) -> DKSResult:
    """Analyze a repository and compute DKS proxy scores."""
    repo_name = os.path.basename(repo_path.rstrip("/"))
    result = DKSResult(repo_name=repo_name, repo_path=repo_path)

    print(f"  Analyzing {repo_name}...")

    # Code metrics
    result.python_loc, result.python_files = count_python_loc(repo_path)
    # Count total files with a cap to avoid OOM
    file_count = 0
    for _ in Path(repo_path).rglob("*"):
        if _.is_file():
            file_count += 1
            if file_count >= 50000:
                break
    result.total_files = file_count

    # D1: Domain lexicon
    for pat in LEXICON_FILE_PATTERNS:
        result.d1_lexicon_files += len(search_files(repo_path, pat))
    # Count entries in JSON/YAML config files that look like domain terms
    json_scanned = 0
    for f in Path(repo_path).rglob("*.json"):
        if json_scanned >= 200:
            break
        parts = str(f).lower()
        if any(s in parts for s in [".git", "node_modules", "__pycache__", "test"]):
            continue
        try:
            if f.stat().st_size > MAX_JSON_SIZE:
                continue
            data = json.loads(f.read_text(errors="ignore"))
            if isinstance(data, dict):
                result.d1_lexicon_entries += len(data)
            elif isinstance(data, list):
                result.d1_lexicon_entries += len(data)
            json_scanned += 1
        except:
            pass

    # D2: Ontology/KG
    result.d2_ontology_imports = search_content(repo_path, ONTOLOGY_IMPORTS)
    result.d2_ontology_classes = search_content(repo_path, ONTOLOGY_CLASS_PATTERNS)

    # D3: Cross-ontology
    result.d3_cross_ontology_hits = search_content(
        repo_path, [rf"\b{kw}\b" for kw in CROSS_ONTOLOGY_KEYWORDS],
        {".py", ".yaml", ".yml", ".json", ".md"}
    )

    # D4: Schema/DDL
    result.d4_schema_patterns = search_content(repo_path, SCHEMA_PATTERNS)
    for pat in SCHEMA_FILE_PATTERNS:
        result.d4_schema_files += len(search_files(repo_path, pat))
    result.d4_dataclass_count = search_content(repo_path, [r"@dataclass", r"class\s+\w+\(.*BaseModel\)"])

    # D5: Domain profiles
    result.d5_config_files, result.d5_config_dirs = count_config_files(repo_path)

    # D6: Rule patterns
    result.d6_regex_patterns = search_content(repo_path, RULE_PATTERNS[:7])  # regex patterns
    result.d6_rule_functions = search_content(repo_path, RULE_PATTERNS[7:])  # rule/validate functions

    # D7: Multi-language
    result.d7_multilang_hits = search_content(
        repo_path, MULTILANG_PATTERNS, {".py", ".yaml", ".yml"}
    )
    for pat in MULTILANG_FILE_PATTERNS:
        result.d7_locale_files += len(search_files(repo_path, pat))

    # D8: Regulatory
    result.d8_regulatory_hits = search_content(
        repo_path, [rf"\b{kw}\b" for kw in REGULATORY_KEYWORDS],
        {".py", ".yaml", ".yml", ".md"}
    )

    # --- Compute proxy scores (normalized by code size) ---
    # Normalize: large repos inflate raw counts; use density (per 1K LOC)
    kloc = max(result.python_loc / 1000.0, 0.1)  # avoid div by zero

    # D1 (max 15): lexicon — only dedicated glossary/lexicon FILES count
    # JSON entries are too noisy (any config file has keys)
    if result.d1_lexicon_files >= 3:       result.d1_score = 15
    elif result.d1_lexicon_files >= 1:     result.d1_score = 10 if result.d1_lexicon_entries > 100 else 5
    elif result.d1_lexicon_entries > 200:  result.d1_score = 5   # many entries but no dedicated file
    else:                                   result.d1_score = 0

    # D2 (max 15): ontology/KG — strict: need graph library imports
    graph_imports = result.d2_ontology_imports  # networkx, dgl, neo4j, rdflib
    graph_classes = result.d2_ontology_classes
    if graph_imports >= 5 and graph_classes >= 5:  result.d2_score = 15
    elif graph_imports >= 2 or graph_classes >= 3: result.d2_score = 10
    elif graph_imports >= 1 or graph_classes >= 1: result.d2_score = 5
    else:                                           result.d2_score = 0

    # D3 (max 10): cross-ontology — density-based
    d3_density = result.d3_cross_ontology_hits / kloc
    if d3_density == 0:      result.d3_score = 0
    elif d3_density <= 0.5:  result.d3_score = 5
    else:                    result.d3_score = 10

    # D4 (max 15): schema/DDL — use density to avoid framework inflation
    # Only count SQL patterns heavily; dataclass lightly
    schema_strong = result.d4_schema_patterns  # CREATE TABLE, ALTER TABLE, FK
    schema_weak = result.d4_dataclass_count    # Python dataclass (common in any project)
    schema_score_raw = schema_strong * 2 + schema_weak
    schema_density = schema_score_raw / kloc
    if schema_density == 0:     result.d4_score = 0
    elif schema_density <= 1:   result.d4_score = 5
    elif schema_density <= 3:   result.d4_score = 10
    else:                       result.d4_score = 15

    # D5 (max 10): domain profiles — density-normalized
    config_density = result.d5_config_files / kloc
    if result.d5_config_files == 0:     result.d5_score = 0
    elif config_density < 1:            result.d5_score = 3
    elif config_density <= 3:           result.d5_score = 6
    else:                               result.d5_score = 10

    # D6 (max 10): rule patterns — density-normalized
    rules_density = (result.d6_regex_patterns + result.d6_rule_functions) / kloc
    if rules_density == 0:      result.d6_score = 0
    elif rules_density < 1:     result.d6_score = 3
    elif rules_density <= 3:    result.d6_score = 6
    else:                       result.d6_score = 10

    # D7 (max 10): multi-language — only locale FILES are reliable
    if result.d7_locale_files >= 3:                          result.d7_score = 10
    elif result.d7_locale_files >= 1 or result.d7_multilang_hits >= 10: result.d7_score = 5
    else:                                                     result.d7_score = 0

    # D8 (max 15): regulatory — strict density to avoid false positives
    # "access" is too common; use only strong signals
    d8_density = result.d8_regulatory_hits / kloc
    if d8_density == 0:       result.d8_score = 0
    elif d8_density <= 0.3:   result.d8_score = 0  # below noise floor
    elif d8_density <= 1:     result.d8_score = 5
    elif d8_density <= 3:     result.d8_score = 10
    else:                     result.d8_score = 15

    return result


def print_result(r: DKSResult):
    """Print formatted DKS analysis result."""
    print(f"\n{'='*60}")
    print(f"  {r.repo_name}")
    print(f"{'='*60}")
    print(f"  Python: {r.python_loc:,} LOC, {r.python_files} files ({r.total_files} total)")
    print(f"{'─'*60}")
    print(f"  D1 Lexicon      : {r.d1_score:>2}/15  (files={r.d1_lexicon_files}, entries~{r.d1_lexicon_entries})")
    print(f"  D2 Ontology/KG  : {r.d2_score:>2}/15  (imports={r.d2_ontology_imports}, classes={r.d2_ontology_classes})")
    print(f"  D3 Cross-ontol  : {r.d3_score:>2}/10  (hits={r.d3_cross_ontology_hits})")
    print(f"  D4 Schema/DDL   : {r.d4_score:>2}/15  (patterns={r.d4_schema_patterns}, dataclass={r.d4_dataclass_count})")
    print(f"  D5 Profiles     : {r.d5_score:>2}/10  (configs={r.d5_config_files}, dirs={r.d5_config_dirs})")
    print(f"  D6 Rules        : {r.d6_score:>2}/10  (regex={r.d6_regex_patterns}, rule_fn={r.d6_rule_functions})")
    print(f"  D7 Multi-lang   : {r.d7_score:>2}/10  (hits={r.d7_multilang_hits}, locale_files={r.d7_locale_files})")
    print(f"  D8 Regulatory   : {r.d8_score:>2}/15  (hits={r.d8_regulatory_hits})")
    print(f"{'─'*60}")
    print(f"  DKS Total       : {r.dks_total:>2}/100")
    print()


def validate_against_manual():
    """Compare automated scores with manual DKS scores for 12 external systems."""
    # Manual DKS scores (first rater) from paper
    manual = {
        "MedRAG":      5,
        "LLaVA":       12,
        "crewAI":      15,
        "llama_index":  20,
        "bird":        26,
        "wheatley":    33,
        "FinSQL":      35,
        "speechbrain": 37,
        "Logic-LLM":   41,
        "audiocraft":  42,
        "EmotiVoice":  47,
        "graphrag":    57,
    }

    # Check which repos are available locally
    ext_dir = "/home/babelai/DKAP/external-systems"
    available = {}
    if os.path.isdir(ext_dir):
        for name in os.listdir(ext_dir):
            full_path = os.path.join(ext_dir, name)
            if os.path.isdir(full_path):
                available[name] = full_path

    print(f"\nValidation: {len(available)} local repos found")
    print(f"Manual scores available for: {list(manual.keys())}")

    results = []
    for name, path in sorted(available.items()):
        r = analyze_repo(path)
        print_result(r)
        # Find matching manual score
        manual_dks = None
        for mname, mscore in manual.items():
            if mname.lower() in name.lower() or name.lower() in mname.lower():
                manual_dks = mscore
                break
        if manual_dks is not None:
            results.append((name, r.dks_total, manual_dks))
            print(f"  → Manual DKS: {manual_dks}, Auto DKS: {r.dks_total}, Diff: {r.dks_total - manual_dks:+d}")

    if len(results) >= 3:
        auto_scores = [r[1] for r in results]
        manual_scores = [r[2] for r in results]
        # Spearman rank correlation
        from statistics import mean, stdev
        n = len(results)
        rank_a = sorted(range(n), key=lambda i: auto_scores[i])
        rank_m = sorted(range(n), key=lambda i: manual_scores[i])
        ranks_auto = [0]*n
        ranks_manual = [0]*n
        for i, idx in enumerate(rank_a):
            ranks_auto[idx] = i+1
        for i, idx in enumerate(rank_m):
            ranks_manual[idx] = i+1
        d_sq = sum((a-m)**2 for a,m in zip(ranks_auto, ranks_manual))
        rho = 1 - 6*d_sq / (n*(n**2-1))

        print(f"\n{'='*60}")
        print(f"  VALIDATION SUMMARY ({n} systems)")
        print(f"{'='*60}")
        print(f"  Spearman rho (auto vs manual): {rho:.3f}")
        print(f"  Mean absolute diff: {mean(abs(a-m) for a,m in zip(auto_scores, manual_scores)):.1f}")
        for name, auto, manual in results:
            print(f"    {name:<15} auto={auto:>3}  manual={manual:>3}  diff={auto-manual:>+4}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python dks_analyzer.py <repo_url_or_path>")
        print("  python dks_analyzer.py --batch repos.txt")
        print("  python dks_analyzer.py --validate")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "--validate":
        validate_against_manual()
        return

    if arg == "--batch":
        batch_file = sys.argv[2]
        with open(batch_file) as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        tmp_dir = tempfile.mkdtemp(prefix="dks_")
        print(f"Working directory: {tmp_dir}")
        results = []
        for url in urls:
            try:
                if url.startswith("http"):
                    repo_path = clone_repo(url, tmp_dir)
                else:
                    repo_path = url
                r = analyze_repo(repo_path)
                print_result(r)
                results.append(r)
            except Exception as e:
                print(f"  ERROR: {url}: {e}")
        # Summary table
        print(f"\n{'='*70}")
        print(f"{'System':<20} {'DKS':>4} {'D1':>3} {'D2':>3} {'D3':>3} {'D4':>3} {'D5':>3} {'D6':>3} {'D7':>3} {'D8':>3} {'LOC':>8}")
        print(f"{'─'*70}")
        for r in sorted(results, key=lambda x: x.dks_total):
            print(f"{r.repo_name:<20} {r.dks_total:>4} {r.d1_score:>3} {r.d2_score:>3} {r.d3_score:>3} "
                  f"{r.d4_score:>3} {r.d5_score:>3} {r.d6_score:>3} {r.d7_score:>3} {r.d8_score:>3} "
                  f"{r.python_loc:>8,}")
        # Save JSON
        out_file = "dks_batch_results.json"
        with open(out_file, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved to {out_file}")
        return

    # Single repo
    if arg.startswith("http"):
        tmp_dir = tempfile.mkdtemp(prefix="dks_")
        repo_path = clone_repo(arg, tmp_dir)
    else:
        repo_path = arg

    r = analyze_repo(repo_path)
    print_result(r)


if __name__ == "__main__":
    main()
