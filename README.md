# 🤖 Intelligent Consumer Complaint Classification Using Transformers

## 📌 Project Overview

Consumer complaints contain valuable information about customer experiences, service failures, and product issues. The large volume and unstructured nature of complaint narratives make manual triage and routing difficult.

Team Upemba explored a transformer-based natural language processing (NLP) approach for intelligent complaint classification and routing. The goal was to build a model that can assign a complaint to the most relevant category while supporting human review for cases that need contextual judgment.

For Cohort 10, the team used the benchmark dataset provided through the TRI AI Saturdays / Kaggle challenge rather than the original CFPB Consumer Complaint Database. This allowed the team to prototype the classification workflow within the project timeline while keeping the broader application goal in view.

---

## 🎯 Objectives

The project aims to:

* Build an NLP pipeline for complaint classification.
* Explore the structure and characteristics of the benchmark dataset.
* Establish a baseline for comparison.
* Test a transformer-based architecture for text classification.
* Fine-tune a pretrained language model.
* Evaluate model performance using task-appropriate metrics.
* Analyse errors and difficult categories.
* Consider class imbalance, fairness, and generalisation.
* Support human-in-the-loop review for large-scale complaint handling.

---

## ⚙️ Tools & Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Jupyter Notebook
* Git & GitHub

---

## 📊 Dataset

### Benchmark dataset

This project uses the benchmark corpus provided for the TRI AI Saturdays / Kaggle challenge:

* `data/train_complaints.csv` — training data with complaint text and category labels.
* `data/test_complaints.csv` — held-out test data without labels.
* `data/sample_submission.csv` — template for model submissions.
* `data/baseline_submission.csv` — baseline model output.
* `data/dataset-metadata.json` — metadata describing the challenge dataset and schema.

The dataset is a complaint-routing benchmark focused on multi-class text classification. The metadata indicates a ten-way complaint routing task scored using Kaggle F1 (balanced), with `ComplaintId`, `text`, `Category`, and `FamilyId` fields in the training data.

## 🔄 Project Workflow

The project follows a standard machine-learning workflow:

```text
Data
  ↓
Exploratory Data Analysis
  ↓
Data Preparation
  ↓
Baseline
  ↓
Tokenization
  ↓
Transformer Model
  ↓
Fine-Tuning
  ↓
Prediction
  ↓
Evaluation & Error Analysis
```

## 🔍 Expected Outcomes

The project is intended to provide insight into:

* The structure and characteristics of the benchmark complaint dataset.
* Patterns in complaint categories and text variation.
* The effectiveness of transformer-based models for text classification.
* The impact of fine-tuning on model performance.
* Categories that are harder for the model to classify.
* The influence of class imbalance and data quality.
* The limitations of using a benchmark dataset for a real-world problem.

Expected outcomes include:

* A documented preprocessing and modelling workflow.
* A baseline implementation.
* A transformer-based classification model.
* Evaluation and error analysis.
* A reproducible notebook.
* Responsible-AI documentation and project rationale.

---


---

## 🗂️ Repository Structure

```text
c10-TeamUpemba/
├── README.md
├── data/
│   ├── baseline_submission.csv
│   ├── dataset-metadata.json
│   ├── sample_submission.csv
│   ├── test_complaints.csv
│   └── train_complaints.csv
├── docs/
│   ├── data_card.pdf
│   ├── impact_statement_card.pdf
│   ├── problem_statement.pdf
│   └── stakeholder_engagement.pdf
├── notebooks/
│   └── intelligent-complaint.ipynb
└── .git/
```

### 📓 `notebooks/`

The primary experimental workflow is contained in `notebooks/intelligent-complaint.ipynb`. This notebook documents the dataset exploration, baseline, transformer-based modelling, and evaluation process.

### 📁 `docs/`

The `docs/` folder contains project documents related to the challenge context, data understanding, impact reflection, and stakeholder considerations.

### 📁 `data/`

The `data/` folder contains the benchmark dataset and submission files used during model development and evaluation.

Note: This repository currently contains a single notebook-based workflow and does not include a separate `scripts/` folder or a `requirements.txt` file at the project root.

---

## 🚀 Future Direction

The next stage of the project would be to evaluate the approach on a larger, real-world consumer complaint dataset, subject to the necessary access, privacy, and governance requirements.

Possible future work includes:

* Testing on a broader real-world complaint dataset.
* Improving performance across multiple complaint categories.
* Addressing class imbalance.
* Conducting more detailed error analysis.
* Evaluating fairness across complaint groups and categories.
* Building a human-in-the-loop review workflow.
* Adding explainability features for reviewer-facing decisions.
* Exploring complaint routing and prioritisation beyond basic category prediction.

The Cohort 10 implementation therefore serves as a technical proof of concept for the NLP workflow, while future iterations would focus on validating its usefulness and safety in the intended consumer-complaint domain.

---

## 👥 Team

**Team Upemba**  
TRI AI Saturdays — Cohort 10  
Google DeepMind AI Research Foundations

---

## 🙏 Acknowledgment

We acknowledge the TRI AI Saturdays program, the Cohort 10 facilitators and mentors, and the Google DeepMind AI Research Foundations curriculum for providing the learning environment and resources that supported this project.

We also acknowledge the benchmark dataset and challenge environment that enabled the team to develop and evaluate the technical prototype within the Cohort 10 timeline.
