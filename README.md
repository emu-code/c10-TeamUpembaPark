# ComplaintSense: Consumer Complaint Classification

## Overview

ComplaintSense is a 10-class consumer complaint classification project designed to automatically route short complaints to the most relevant category. The project compares a lightweight TF-IDF baseline with a transformer-based DistilBERT classifier and evaluates both using Balanced F1 (macro F1).

The final approach fine-tunes DistilBERT on the competition training data and generates predictions for the 160-row test set.

---

## Dataset

The project uses the [ComplaintSense benchmark dataset](https://www.kaggle.com/competitions/complaint-sense-consumer-complaint-classification-challenge/overview) supplied through the competition. It contains **380 labeled training complaints** and **160 unlabeled test complaints** across 10 categories.

Training records contain a complaint identifier, complaint text, target `Category`, and `FamilyId`. `FamilyId` identifies complaints derived from the same underlying scenario and was used to prevent related complaint variants from appearing in both training and validation sets.

The 10 categories are:

`account_access`, `billing`, `customer_service`, `delivery_shipping`, `fraud_unauthorized`, `general_inquiry`, `product_defect`, `refund_return`, `subscription_cancel`, and `warranty_repair`.

No external data was added. The competition data was used as supplied.

---

## Training Pipeline

### Preprocessing and validation

The complaint text was tokenized using the DistilBERT tokenizer. Text was truncated to a maximum sequence length of 128 tokens.

A family-aware validation split was used so that related complaint variants would not cross the training/validation boundary. The resulting split contained **285 training rows from 57 families** and **95 validation rows from 19 families**.

### TF-IDF baseline

A TF-IDF-based classifier was implemented as a lightweight baseline. It was evaluated using the same validation data and achieved a local **Balanced F1 of 0.2596**.

The baseline provided a reference point for assessing the transformer-based approach.

### DistilBERT classifier

The final model uses a pretrained DistilBERT transformer with a 10-class sequence-classification head. The model was fine-tuned using the Hugging Face `Trainer` framework and evaluated on the held-out validation set after each training epoch.

---

## Evaluation

Balanced F1 (macro F1) was used because it gives each complaint category equal importance rather than allowing larger categories to dominate the evaluation.

The TF-IDF baseline achieved a **Balanced F1 of 0.2596** on the family-aware validation split.

The DistilBERT model improved substantially, reaching its best validation **Balanced F1 of 0.4380 at epoch 7**. The model's validation loss continued to decrease after this point, but Balanced F1 fluctuated, showing that lower loss did not necessarily translate into better classification performance.

Validation error analysis identified **51 incorrect predictions out of 95 validation examples**, with the largest number of errors occurring in `delivery_shipping`, `subscription_cancel`, and `customer_service`.

The final model was then used to generate predictions for the 160 unlabeled test complaints.

---

## Reproduction

This notebook is written to run inside a Kaggle kernel and reads/writes via
`/kaggle/input` and `/kaggle/working`. The same, unmodified notebook can be
run in two ways:

### Option A — Run on Kaggle (recommended)

1. Upload `notebook.ipynb` to a new Kaggle notebook (or open the existing one).
2. Attach the ComplaintSense competition dataset: **Add Input → Competitions**
   → search for "ComplaintSense" → attach.
3. Attach a DistilBERT model: **Add Input → Models** → search "DistilBERT base
   uncased" → choose a Transformers/PyTorch variant with `config.json`,
   tokenizer files, and `model.safetensors` (or `pytorch_model.bin`).
4. Click **Save Version → Save & Run All** to execute the notebook top to
   bottom in a clean environment.
5. `submission.csv` is written to `/kaggle/working/submission.csv`
   (160 predictions).

### Option B — Run locally

Because the code is unmodified from the Kaggle version, running it locally
means recreating Kaggle's expected folder structure on your own machine,
rather than editing any paths in the notebook.

1. **Clone the repository:**

   `git clone https://github.com/emu-code/c10-team-upembapark.git`

2. **Navigate to the repository folder:**

   `cd c10-team-upembapark`

3. **Create and activate a virtual environment, then install dependencies**

4. **Recreate the Kaggle directory structure locally**

5. **Place the competition data** (`train_complaints.csv`, `test_complaints.csv`)

6. **Download DistilBERT into the same input directory** so the notebook's
   model-search cell can find it:

```python
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id="distilbert-base-uncased",
       local_dir="/kaggle/input/distilbert-base-uncased",
   )
```

7. **Open the notebook** with the `c10-teamupemba` kernel and run the cells
   in order from the beginning.

8. `submission.csv` is written to `/kaggle/working/submission.csv`
   (160 predictions).

N/B: The pretrained DistilBERT model is not stored in the repository due to
its size. It must be attached on Kaggle (Option A) or downloaded via
`huggingface_hub` into the local `/kaggle/input` mirror (Option B, step 6)
before the notebook's model-loading cell will succeed.
   
---

## Appendix

### Team Members

* [Adebobajo Inioluwa](https://github.com/adebobajoinioluwa) — Team Lead
* [Emumena Oweh](https://github.com/emu-code) 
* [Adedotun Onasanya](https://github.com/adedotguy)
* [David Arfo](https://github.com/Daxe5)

* Program: TRI-AI  Lagos
### References

* [ComplaintSense Competition — Kaggle](https://www.kaggle.com/competitions/complaint-sense-consumer-complaint-classification-challenge/overview)
* [Google Skills — DeepMind AI Research Foundations](https://www.skills.google/paths/234)
