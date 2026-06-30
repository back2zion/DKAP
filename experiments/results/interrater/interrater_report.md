# Multi-Agent Inter-Rater Reliability Report

**Date**: 2026-04-03 11:08
**Model**: default-model
**Raters**: 5 agent personas
**Systems**: 12 external systems
**Total scorings**: 60

## 1. Summary Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Fleiss' κ (4-bin) | 0.897 | Substantial agreement |
| Mean pairwise Cohen's κ | 0.971 | Weighted, 4-bin |
| Mean Spearman ρ | 0.956 | Rank-order agreement |
| ICC(2,k) Total | 0.995 | Excellent reliability |
| ICC(2,k) DKS | 0.995 | DKS subscale |
| ICC(2,k) RSS | 0.993 | RSS subscale |

## 2. R1 (Author) vs Agent Mean

| Metric | Value |
|--------|-------|
| Spearman ρ | 0.675 (p=0.0159) |
| Pearson r | 0.783 (p=0.0026) |
| Cohen's κ (4-bin) | 0.712 |
| Mean abs diff | 9.7 |

## 3. Per-System Scores

| System | R1 | SW_Architect | ML_Researcher | Data_Engineer | Domain_Specialist | Junior_Developer | Agent Mean ± Std |
|--------|----|----|----|----|----|----|-----------------|
| MedRAG | 10 | 8 | 8 | 8 | 8 | 8 | 8.0 ± 0.0 |
| LLaVA | 14 | 3 | 8 | 3 | 8 | 3 | 5.0 ± 2.4 |
| CrewAI | 24 | 13 | 14 | 16 | 14 | 14 | 14.2 ± 1.0 |
| LlamaIndex | 27 | 47 | 41 | 39 | 33 | 33 | 38.6 ± 5.3 |
| BIRD-SQL | 29 | 21 | 19 | 21 | 20 | 18 | 19.8 ± 1.2 |
| Wheatley | 35 | 47 | 47 | 47 | 47 | 47 | 47.0 ± 0.0 |
| FinSQL | 41 | 36 | 36 | 36 | 38 | 38 | 36.8 ± 1.0 |
| SpeechBrain | 41 | 46 | 46 | 46 | 51 | 46 | 47.0 ± 2.0 |
| Logic-LLM | 44 | 20 | 15 | 15 | 15 | 10 | 15.0 ± 3.2 |
| AudioCraft | 46 | 34 | 34 | 34 | 34 | 34 | 34.0 ± 0.0 |
| EmotiVoice | 50 | 36 | 46 | 41 | 46 | 46 | 43.0 ± 4.0 |
| GraphRAG | 65 | 64 | 59 | 64 | 57 | 58 | 60.4 ± 3.0 |

## 4. Pairwise Cohen's κ Matrix

- SW_Architect_vs_ML_Researcher: κ = 0.958
- SW_Architect_vs_Data_Engineer: κ = 0.958
- SW_Architect_vs_Domain_Specialist: κ = 0.923
- SW_Architect_vs_Junior_Developer: κ = 0.958
- ML_Researcher_vs_Data_Engineer: κ = 1.0
- ML_Researcher_vs_Domain_Specialist: κ = 1.0
- ML_Researcher_vs_Junior_Developer: κ = 0.972
- Data_Engineer_vs_Domain_Specialist: κ = 0.964
- Data_Engineer_vs_Junior_Developer: κ = 1.0
- Domain_Specialist_vs_Junior_Developer: κ = 0.972

**Mean pairwise κ = 0.971**

## 5. Pairwise Spearman ρ

- SW_Architect_vs_ML_Researcher: ρ = 0.96 (p=0.0)
- SW_Architect_vs_Data_Engineer: ρ = 0.958 (p=0.0)
- SW_Architect_vs_Domain_Specialist: ρ = 0.9 (p=0.0001)
- SW_Architect_vs_Junior_Developer: ρ = 0.902 (p=0.0001)
- ML_Researcher_vs_Data_Engineer: ρ = 0.989 (p=0.0)
- ML_Researcher_vs_Domain_Specialist: ρ = 0.967 (p=0.0)
- ML_Researcher_vs_Junior_Developer: ρ = 0.97 (p=0.0)
- Data_Engineer_vs_Domain_Specialist: ρ = 0.963 (p=0.0)
- Data_Engineer_vs_Junior_Developer: ρ = 0.977 (p=0.0)
- Domain_Specialist_vs_Junior_Developer: ρ = 0.979 (p=0.0)

**Mean ρ = 0.956**

## 6. Per-Rater Summary

| Rater | Mean DKS | Mean RSS | Mean Total | Std Total |
|-------|----------|----------|------------|-----------|
| SW_Architect | 28.8 | 2.5 | 31.2 | 17.7 |
| ML_Researcher | 28.5 | 2.6 | 31.1 | 16.8 |
| Data_Engineer | 27.9 | 2.9 | 30.8 | 17.5 |
| Domain_Specialist | 28.2 | 2.7 | 30.9 | 16.7 |
| Junior_Developer | 27.0 | 2.6 | 29.6 | 17.5 |