# XAI Study – Explainable AI Movie Recommender

**Course:** Designing Intelligent Systems (IS 596) | Group 02 | UIUC

> Do users trust AI recommendations more when explanations are provided?

---

## Folder Structure

```
xai_project/
├── data/
│   └── xai_study_results.csv       # Study results (all 20 participants)
├── notebooks/
│   ├── XAI_Project_Prototype.ipynb # Study prototype – run with participants
│   └── XAI_Study_Analysis.ipynb    # Statistical analysis (RQ1, RQ2, RQ3)
├── dashboard/
│   └── xai_dashboard.py            # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## One-Time Setup

Open Terminal, `cd` into this folder, then run:

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Every time you return**, just re-activate: `source venv/bin/activate`

---

## Running Each File

### Streamlit Dashboard
```bash
cd dashboard
streamlit run xai_dashboard.py
```
Opens at http://localhost:8501  
Reads data from `../data/xai_study_results.csv`

---

### Analysis Notebook (RQ1 / RQ2 / RQ3 stats)
```bash
cd notebooks
jupyter notebook XAI_Study_Analysis.ipynb
```
Reads data from `../data/xai_study_results.csv`

---

### Prototype Notebook (interactive recommender)
```bash
cd notebooks
jupyter notebook XAI_Project_Prototype.ipynb
```
Downloads MovieLens + TMDB datasets on first run (requires internet).

---

## Notes

- The dashboard and analysis notebook both expect `xai_study_results.csv` — keep it in `data/`
- The prototype downloads MovieLens (~6MB) automatically on first run into `notebooks/ml-latest-small/`
- If Jupyter isn't installed: `pip install notebook` inside your venv
