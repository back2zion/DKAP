# Cross-Model Multi-Seed Finance Ablation (B2 / B2' / DKAP)

- Models analyzed: **6** (OpenAI, Anthropic, Google, DeepSeek, Qwen, Mistral) | seeds: **10** @ temp=0.3
  - OpenAI: `openai/gpt-5.4`
  - Anthropic: `anthropic/claude-sonnet-4.6`
  - Google: `google/gemini-3.1-pro-preview`
  - DeepSeek: `deepseek/deepseek-v3.2`
  - Qwen: `qwen/qwen3.6-27b`
  - Mistral: `mistralai/mistral-medium-3-5`

## Per-model execution accuracy (%) and gains
| Family | B2 | B2' | DKAP | Compl. (B2'-B2) | Struct. E+M (DKAP-B2') |
|---|---|---|---|---|---|
| OpenAI | 65.4 | 83.4 | 81.0 | +18.0 | -0.9 |
| Anthropic | 64.6 | 83.4 | 84.6 | +18.8 | +1.2 |
| Google | 60.4 | 70.2 | 73.2 | +9.8 | +4.4 |
| DeepSeek | 53.8 | 75.6 | 77.4 | +21.8 | +0.3 |
| Qwen | 60.0 | 74.6 | 77.0 | +14.6 | +1.9 |
| Mistral | 57.2 | 79.4 | 78.2 | +22.2 | -2.5 |

## Structuring gain (DKAP - B2'), Easy+Medium
- Per-model (pp): [-0.9, 1.2, 4.4, 0.3, 1.9, -2.5]
- Mean **0.73pp** (range -2.5..4.4); models with positive gain: **4/6**
- Pooled mean **0.73pp**, bootstrap 95% CI (-1.0, 3.0)

## Completeness gain (B2' - B2): mean **+17.53pp**
