# Human-Centric Reflective Architecture for Human-AI Collaborative Decision-Making

An architecture for collaborative decision-making that integrates reinforcement learning (RL) language agents with human calibrated models in an iterative reflective process

## Quick Start

### Prerequisites
- Python 3.8+
- DeepSeek API key ([Get one here](https://api-docs.deepseek.com/))

### Installation
1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure API key**

Create `.env` file:
```
DEEPS_KEY=your_deepseek_api_key_here
```

3. **Run the framework**

Make sure you are in the project root directory, then run:
```bash
python src/main.py
```

## What it does
The framework performs iterative decision-making across all 42 tourism questions about Athens, Greece. Questions are split into two phases — the first 32 and the remaining 10 — each shuffled randomly before processing.

For each question:
1. The **actor** generates a response with a confidence score
2. The **evaluator** assesses factual correctness and alignment with user preferences
3. The **acceptance model** predicts the probability that a human user would accept the answer
4. The **reflexion agent** analyzes performance and provides feedback to the actor
5. Process repeats until termination condition is met

## Required Files
Ensure these files exist:
- `data/questions_preferences_extented.csv` - Questions dataset
- `data/haiid_dataset.csv` - Demographic data
- `hb_models/*.pth` - Pre-trained models

> Users should train and provide their own hb_models before running the framework.

> The `haiid_dataset.csv` can be downloaded from the [human-ai-interactions](https://github.com/kailas-v/human-ai-interactions) repository.

## Output

The system generates detailed logs showing:
- Actor responses and confidence
- Correctness assessment
- Preference alignment scores
- Predicted human acceptance probability
- Reflection feedback
- Termination analysis

Results are saved as a JSON file in the `logs/` directory.

## Configuration

Adjust parameters in `.env`:
```
ACTOR_TEMP=1.0              # Actor LLM temperature
ACCEPTANCE_THRESHOLD=0.5    # Acceptance probability cutoff
EVALUATOR_TEMP=0.0          # Evaluator LLM temperature
MAX_REFLEXION_TRIALS=10     # Maximum iterations per question
REFLEXION_TEMP=1.0          # Reflexion LLM temperature
```

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).

[![License: CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc/4.0/)
