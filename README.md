# Fairness in AI

## Project Overview
This project explores algorithmic fairness in machine learning, with a focus on the impact of regularization on fairness in both standard and fairness-aware models. Using logistic regression, it examines the trade-offs between model accuracy and fairness, including the evaluation of models trained on Florida data and tested on Texas data for generalizability and robustness.

## Requirements
- Python 3.10
- Dependencies are listed in `requirements.txt`.

## Installation
To install the required dependencies, run:
pip install -r requirements.txt

## Tasks and Methods
- **Task 1 – Standard Model:** Employed logistic regression with variable regularization. Hyperparameters were tuned using a grid search to observe the impact on accuracy and fairness (measured as Equality of Opportunity Difference, EOD).
- **Task 2 – Fairness-Aware Model:** Implemented reweighting in logistic regression to enhance fairness. Similar grid search methodology was applied to find the optimal trade-off.
- **Task 3 – Model Selection Strategy:** Developed a criterion based on accuracy and fairness to select the most balanced model.
- **Additional Exploration:** Analyzed the performance of Florida-trained models on Texas data to evaluate model robustness and generalization.

## Results
- Demonstrated that basic regularization techniques might not achieve adequate fairness in machine learning models.
- Showed the effectiveness of reweighting in training fair models with minimal performance sacrifice.
- Proposed model selection criteria balanced between accuracy and fairness.

## Conclusion
The project highlights the complexity in achieving both high accuracy and fairness in AI models. It suggests that while striving for algorithmic fairness, the trade-off between accuracy and fairness must be carefully managed. Future work includes refining the criterion with weighted accuracy/fairness contributions and exploring non-linear functions for utility assessment.

## How to Use
- Execute scripts for each task to replicate the findings.
- Review the project's Jupyter notebooks for detailed analysis and visualization.
