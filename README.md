# Endoscopic Report Analysis System Using LLMs

This repository contains the implementation of a systematic evaluation framework for analyzing endoscopic examination reports using Large Language Models (LLMs). The system is designed to extract structured information from medical reports through prompt engineering techniques.

## Overview

The system performs three hierarchical tasks:
1. Basic Entity Extraction
2. Pattern Recognition
3. Diagnostic Assessment

It supports evaluation of multiple LLM models including:
- Gemini series 
- Claude series 
- Bailian-LLaMA series 
- Meta-LLaMA series 
- ChatGLM3
- DeepSeek Chat
- Qwen series 
- GPT-4 series 
- Grok-beta
- GLM series 
- And more

## Paper

The full details of our research methodology and findings are available in our paper:
*"Evaluating Large Language Models for Information Extraction from Gastroscopic and Colonoscopic Reports through Multi-strategy Prompting"*

## Repository Structure

```
.
├── run_llm.py              # Main execution script
├── models/                 # Model implementations
│   └── model_tester.py
├── evaluation/            # Evaluation metrics
│   └── enhanced_metrics.py
├── data/                  # Data management
│   └── data_manager.py
└── paper/                 # Research paper and supplementary materials
```

## Requirements

- Python 3.8+
- numpy
- argparse
- sklearn
- Additional requirements in `requirements.txt`

## Installation

```bash
git clone https://github.com/username/endoscopy-llms.git
cd endoscopy-llms
```

## Usage

The system can be run in three modes:
1. Evaluation mode (with ground truth)
2. Test mode (single run)
3. Batch mode (multiple models)

Basic command:
```bash
python run_llm.py --mode evaluate --models gemini-2.0-flash-exp --tasks all
```

### Command Line Arguments

- `--mode`: Select mode [evaluate/test/batch]
- `--models`: Select LLM model(s) to use
- `--shot-mode`: Select shot mode [zero/few]
- `--num-shots`: Number of shots for few-shot evaluation [1/2/5]
- `--tasks`: Select tasks to evaluate [basic/relationship/diagnostic/all]
- Complete list of arguments available in run_llm.py

## Features

- Multi-model evaluation framework
- Support for zero-shot and few-shot learning
- Comprehensive evaluation metrics
- Flexible data management
- Multiple prompting strategies including
  
## Prompting Strategies

Our framework implements multiple prompting strategies for comprehensive model evaluation:

### Direct Prompting
- Basic prompting strategy that provides task instructions and format requirements
- Serves as baseline for comparing with more sophisticated approaches

### Chain-of-Thought (CoT) Variants

1. **Basic Step-by-Step**
   - Breaks down analysis into logical progression
   - Provides systematic guidance through fundamental steps
   - Focuses on clear sequential reasoning

2. **Expert Role**
   - Frames analysis from clinical specialist's perspective
   - Encourages leveraging domain knowledge
   - Promotes professional judgment in medical context

3. **Systematic Analysis**
   - Ensures comprehensive evaluation through methodical documentation
   - Guides through structured analysis processes
   - Maintains consistency with clinical practice standards

4. **Deductive Reasoning**
   - Applies logical elimination and evidence-based conclusions
   - Particularly effective for complex cases
   - Emphasizes careful analysis of multiple factors

5. **Clinical Protocol**
   - Emphasizes adherence to standardized procedures
   - Focuses on documentation requirements
   - Ensures alignment with clinical practice guidelines

Each prompting strategy is implemented across three core tasks (Basic Entity Extraction, Pattern Recognition, and Diagnostic Assessment) with task-specific adaptations.

## Citation
