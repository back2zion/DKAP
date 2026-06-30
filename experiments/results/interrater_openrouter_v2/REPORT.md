# Cross-Family LLM-as-Judge — Inter-Rater Reliability (DKS/RSS)

- Systems scored (complete cases): **13**
- Judges (families): **7** — OpenAI, Anthropic, Google, Mistral, DeepSeek, Qwen, GLM
- Models:
  - OpenAI: `openai/gpt-5.4`
  - Anthropic: `anthropic/claude-sonnet-4.6`
  - Google: `google/gemini-3.1-pro-preview`
  - Mistral: `mistralai/mistral-medium-3-5`
  - DeepSeek: `deepseek/deepseek-v3.2`
  - Qwen: `qwen/qwen3.6-27b`
  - GLM: `z-ai/glm-5.2`

## Agreement
- Fleiss' kappa (4-bin): **0.367**
- Krippendorff's alpha (interval): **0.718**
- Mean pairwise Cohen's kappa: **0.656**
- Mean pairwise Spearman rho: **0.782**

## Author vs. cross-family ensemble
- n = 13 systems
- Spearman rho = **0.786** (p = 0.0015)
- Pearson r = **0.851** (p = 0.0002)

## Per-system ensemble totals
| System | mean | min | max | std |
|---|---|---|---|---|
| MedRAG | 8.0 | 8.0 | 8.0 | 0.0 |
| CrewAI | 17.1 | 2.0 | 31.0 | 8.3 |
| LlamaIndex | 25.6 | 17.0 | 36.0 | 5.6 |
| BIRD-SQL | 17.9 | 13.0 | 22.0 | 3.5 |
| Wheatley | 29.4 | 19.0 | 37.0 | 6.2 |
| FinSQL | 39.4 | 32.0 | 65.0 | 10.6 |
| SpeechBrain | 39.4 | 10.0 | 53.0 | 13.7 |
| AudioCraft | 20.4 | 10.0 | 32.0 | 8.7 |
| EmotiVoice | 45.0 | 36.0 | 56.0 | 6.5 |
| GraphRAG | 54.3 | 43.0 | 60.0 | 5.3 |
| MetaGPT | 33.4 | 28.0 | 39.0 | 4.3 |
| DeepTutor | 31.7 | 26.0 | 40.0 | 5.0 |
| PulseAI | 21.1 | 18.0 | 23.0 | 1.7 |
