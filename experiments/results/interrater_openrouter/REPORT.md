# Cross-Family LLM-as-Judge — Inter-Rater Reliability (DKS/RSS)

- Systems scored (complete cases): **12**
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
- Fleiss' kappa (4-bin): **0.43**
- Krippendorff's alpha (interval): **0.726**
- Mean pairwise Cohen's kappa: **0.714**
- Mean pairwise Spearman rho: **0.792**

## Author vs. cross-family ensemble
- n = 12 systems
- Spearman rho = **0.732** (p = 0.0068)
- Pearson r = **0.832** (p = 0.0008)

## Per-system ensemble totals
| System | mean | min | max | std |
|---|---|---|---|---|
| MedRAG | 8.0 | 8.0 | 8.0 | 0.0 |
| LLaVA | 5.3 | 0.0 | 14.0 | 5.0 |
| CrewAI | 17.0 | 5.0 | 25.0 | 6.1 |
| LlamaIndex | 20.6 | 5.0 | 28.0 | 7.5 |
| BIRD-SQL | 18.7 | 13.0 | 25.0 | 3.7 |
| Wheatley | 33.0 | 21.0 | 47.0 | 7.7 |
| FinSQL | 37.9 | 33.0 | 50.0 | 5.2 |
| SpeechBrain | 36.0 | 13.0 | 61.0 | 17.3 |
| Logic-LLM | 15.0 | 5.0 | 29.0 | 8.6 |
| AudioCraft | 19.7 | 5.0 | 32.0 | 9.6 |
| EmotiVoice | 42.4 | 36.0 | 55.0 | 6.3 |
| GraphRAG | 53.3 | 46.0 | 58.0 | 4.3 |
