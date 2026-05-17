#  Resume Screening AI

An AI-powered resume screening web app built with **Streamlit**, **TF-IDF**, and **XGBoost** that matches resumes against job descriptions and ranks candidates by match score.

---

##  Features

-  Upload resume as **PDF or DOCX** — text is extracted automatically
-  Paste resume text directly
-  Get an instant **match score (%)** for any job description
-  **Batch ranking** — compare and rank multiple resumes at once
-  Train on your **own CSV dataset**
-  **Export results** as CSV

---

##  Tech Stack

| Layer | Tools |
|---|---|
| Frontend / UI | Streamlit, Custom CSS, HTML |
| ML Model | XGBoost Classifier |
| Text Features | TF-IDF Vectorizer (scikit-learn) |
| Scoring | Cosine Similarity + Keyword Overlap |
| File Parsing | PyPDF2 (PDF), python-docx (DOCX) |
| Data | pandas, NumPy |

---

##  Installation

**1. Clone the repository**
```bash
git clone https://github.com/kanishkaSinghal201/Resume_Screening.git
cd Resume_Screening
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

##  Project Structure

```
Resume_Screening/
│
├── app.py               # Main Streamlit app
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

##  How It Works

1. **Enter a Job Description** in the sidebar
2. **Upload or paste a resume** in the Single Resume tab
3. The app computes:
   - **Cosine Similarity** between resume and job description (TF-IDF vectors)
   - **Keyword Overlap** — % of job keywords found in resume
   - **Skill Bonus** — extra weight for important technical skills
4. A final **Match Score (%)** is calculated and the candidate is either **Shortlisted ✅** or **Rejected ❌**

---

 ## Scoring Formula

```
Final Score = (0.60 × Cosine Similarity) + (0.30 × Keyword Overlap) + Skill Bonus
Threshold   = 35% → Shortlist | Below 35% → Reject
```

---

## Screenshots

> Add screenshots of your app here after running it locally.

---

# Author

**Kanishka Singhal**  
[GitHub](https://github.com/kanishkaSinghal201)

---

# License

This project is open source and available under the [MIT License](LICENSE).
