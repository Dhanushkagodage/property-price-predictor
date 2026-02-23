md_content = """Machine Learning Assignment — Guidelines

Objective
The objective of this assignment is to enable students to collect a local dataset, apply a 
existing machine learning algorithm, evaluate and explain the model, and optionally
integrate it into a front-end system.

Task Description
Students must identify a real-world problem, collect or compile a dataset, apply a 
machine learning algorithm, train and evaluate it, and explain the results using XAI
techniques. Avoid developing an image processing application.

Guidelines
1. Problem Definition & Dataset Collection (15 marks)
• Clearly describe the problem and its relevance.
• Explain:
o Data source (how and from where it was collected),
o Features and target variable,
o Size of the dataset,
• Any preprocessing done (cleaning, encoding, normalization).
o Ensure ethical data use (no personal or sensitive data without consent).

2. Selection of a Machine Learning Algorithm (15 marks)
• Choose an algorithm avoid using deep learning models.
• Justify:
o Why was this algorithm selected,
o How it differs from standard models (e.g., decision trees, logistic regression,
k-NN, etc.).

3. Model Training and Evaluation (20 marks)
• Explain:
o Train/validation/test split,
o Hyperparameter choices,
o Performance metrics used (accuracy, F1, RMSE, AUC, etc. depending on
task),
o Results obtained and what they indicate.
• Include tables, graphs, or plots where appropriate.

4. Explainability & Interpretation (20 marks)
• Apply at least one explainability method, such as:
o SHAP
o LIME
o Feature importance analysis
o Partial Dependence Plots (PDP)
• Explain:
o What the model has learned,
o Which features are most influential,
o Whether the model’s behavior aligns with domain knowledge.

5. Critical Discussion (10 marks)
• Limitations of the model,
• Data quality issues,
• Risks of bias or unfairness,
• Potential real-world impact and ethical considerations.

6. Report Quality & Technical Clarity (10 marks)

7. Bonus: Front-End Integration (10 marks)
Bonus marks will be awarded for:
• Integrating the trained model into a front-end system (web app, dashboard, mobile 
app, etc.).
• Allowing users to input data and view predictions/explanations.
• Examples: Streamlit app, Flask + HTML, React frontend, etc.

Submission Requirements
• Written Report (PDF)
Including problem description, methodology, results, interpretation, and discussion.
• Source Code (ZIP / GitHub link)
Including data preprocessing, training, evaluation, and explainability
scripts. • Dataset (if publicly shareable) or a description of how it was
obtained. • Demo video (3–5 minutes) showing the front-end system.
Upload your submission to the Moodle course page.
"""
path = "/mnt/data/ML_Assignment_GuidelinesMSc_in_AI.md"
with open(path, "w", encoding="utf-8") as f:
    f.write(md_content)
path
