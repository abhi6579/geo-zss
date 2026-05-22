# GeoPrompt: Zero-Shot Semantic Segmentation for Satellite Imagery

**Abhinav Mishra · 2025-26**

A systematic prompt engineering framework that improves zero-shot semantic segmentation of satellite imagery using CLIP and SAM, without any labeled training data.

## Strategies
- Template engineering (15+ satellite-specific prompts)
- Hierarchical prompting (scene → object → pixel level)
- Ensemble aggregation (confidence-weighted voting)
- Context-aware selection (auto-detects urban/rural/coastal)

## Datasets
- LoveDA (7 classes)
- iSAID (15 classes)
- DOTA (18 classes)

## Setup
```bash
git clone https://github.com/abhi6579/geo-zss.git
cd geo-zss
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## GitHub
https://github.com/abhi6579/geo-zss
