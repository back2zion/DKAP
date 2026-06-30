# DKS/RSS Second Rater Scoring Results

**Rater**: Claude (AI second rater, independent scoring)
**Date**: 2026-04-01
**Method**: 5 systems scored from local source code analysis; 7 systems scored from Supplementary Material evidence

## Scoring Summary

| System | R1 DKS | R2 DKS | Diff | R1 RSS | R2 RSS | Diff | R1 Total | R2 Total | Diff |
|--------|--------|--------|------|--------|--------|------|----------|----------|------|
| MedRAG | 5 | 5 | 0 | 5 | 4 | +1 | 10 | 9 | +1 |
| LLaVA | 12 | 8 | +4 | 2 | 0 | +2 | 14 | 8 | +6 |
| CrewAI | 15 | 18 | -3 | 9 | 8 | +1 | 24 | 26 | -2 |
| LlamaIndex | 20 | 18 | +2 | 7 | 7 | 0 | 27 | 25 | +2 |
| BIRD-SQL | 26 | 22 | +4 | 3 | 3 | 0 | 29 | 25 | +4 |
| Wheatley | 33 | 30 | +3 | 2 | 2 | 0 | 35 | 32 | +3 |
| FinSQL | 35 | 32 | +3 | 6 | 5 | +1 | 41 | 37 | +4 |
| SpeechBrain | 37 | 35 | +2 | 4 | 4 | 0 | 41 | 39 | +2 |
| **Logic-LLM** | **41** | **14** | **+27** | 3 | 1 | +2 | **44** | **15** | **+29** |
| AudioCraft | 42 | 52 | -10 | 4 | 0 | +4 | 46 | 52 | -6 |
| EmotiVoice | 47 | 43 | +4 | 3 | 2 | +1 | 50 | 45 | +5 |
| **GraphRAG** | **57** | **30** | **+27** | 8 | 6 | +2 | **65** | **36** | **+29** |

**Mean absolute difference**: DKS=7.4, RSS=1.2, Total=7.8

## Inter-Rater Reliability Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Pearson r (DKS+RSS) | 0.725 | Good agreement on continuous scores |
| Spearman rho (DKS+RSS) | 0.734 | Good rank-order agreement |
| Cohen's Kappa (4-bin) | 0.520 | Moderate agreement |
| Agreement rate (4-bin) | 66.7% | 8/12 systems in same category |
| Rank agreement (within +/-1) | 50.0% | 6/12 systems within 1 rank position |

## Major Disagreements Analysis

### Logic-LLM (R1=44, R2=15, diff=29)
**Root cause**: Interpretation of D2 (Ontology/KG) and D1 (Lexicon).
- R1 scored D2=7 (FOL grammar as ontological structure) and D1=8 (formal logic terms as domain lexicon)
- R2 scored D2=0 (no knowledge graph or ontology file) and D1=5 (logic terms exist but <50)
- **Key disagreement**: Whether formal logic grammar rules and solver DSLs count as "domain knowledge artifacts" or are simply "programming language constructs." R1's broader interpretation includes symbolic AI structures; R2 applied a more literal reading of the rubric criteria.

### GraphRAG (R1=65, R2=36, diff=29)
**Root cause**: Interpretation of D1 (Lexicon), D2 (Ontology), D4 (Schema).
- R1 scored D1=10 (graph-specific vocabulary), D2=15 (KG construction is core purpose), D4=12 (7 data model classes)
- R2 scored D1=0 (no curated glossary file), D2=15 (agreed on KG), D4=5 (data models, not relational DDL)
- **Key disagreement**: R1 counted field constants and data model classes toward D1/D4; R2 required explicit glossary files and relational schemas. Also, R2 scored D4 strictly as relational tables, not Python dataclasses.

### AudioCraft (R1=46, R2=52, diff=-6)
**Note**: R2 scored HIGHER than R1. R2's D1=11 vs R1's D1 implied in component total. RSS disagreement: R1=4, R2=0 (R2 applied strict interpretation that neural conditioning is not "retrieval").

## Implications for Paper

1. **RSS scores show strong agreement** (mean diff=1.2): The retrieval rubric is relatively unambiguous.
2. **DKS scores show moderate agreement** (mean diff=7.4): The domain knowledge rubric needs clearer operationalization, especially for:
   - D1: Does the rubric count in-code vocabulary terms or require explicit glossary files?
   - D2: Do computational graph structures (GNN edges, FOL grammars) count as ontologies?
   - D4: Do Python dataclasses count as schemas, or only SQL DDL?
3. **Rank order is largely preserved**: Spearman rho=0.734 suggests the rubric captures relative ordering even when absolute scores differ.
4. **Cohen's Kappa=0.52 (moderate)**: Below the target of 0.80. The rubric requires refinement before claims of validated scoring can be made.

## Recommendations for Rubric Improvement

1. **D1**: Specify that terms must be in explicit configuration/glossary files, not just variable names in code
2. **D2**: Create sub-criteria distinguishing semantic ontologies from computational graph structures
3. **D4**: Clarify that "tables" includes structured data schemas (YAML, dataclasses) not only SQL DDL
4. **D6**: Distinguish domain business rules from programming pattern matching
5. **Add worked examples**: Include 2-3 scored examples in the rubric definition to anchor interpretations
