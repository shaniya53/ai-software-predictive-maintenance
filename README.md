# AI-Assisted Software Evolution and Predictive Maintenance

An AI-assisted software engineering research project for predicting the future fault risk of software changes using software evolution, refactoring, code-quality, and technical-debt information.

## Research Objective

The project investigates whether information available at the time of a software commit can be used to predict whether that commit will later be identified as fault-inducing.

## Current Research Direction

The system is designed around:

Git History
→ Feature Extraction
→ Fault-Risk Prediction
→ Explainable AI
→ Maintenance Recommendations

## Dataset

The primary research dataset is the Technical Debt Dataset v2.

The raw database is not included in this repository because of its size.

Place the database locally at:

`data/raw/td_V2.db`

## Current Status

- Literature review: Completed
- Research gap analysis: Completed
- Dataset selection: Completed
- Dataset exploration: Completed
- Feature engineering: In progress
- ML modelling: Not started
- Explainability: Planned
- Real repository integration: Planned

## Research Approach

The project will use historical fault information as ground truth during research while ensuring that prediction features only contain information available before the predicted commit outcome.

## Planned Experiments

1. Git/change-history baseline
2. Baseline + refactoring information
3. Baseline + software-quality and technical-debt information
4. Explainable AI analysis
5. Real-world repository evaluation