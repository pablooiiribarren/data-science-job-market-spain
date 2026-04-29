# 📊 Data Science Job Market Analysis — Spain (2025)

## 👨‍💻 Author

**Pablo Iribarren**  
💼 [LinkedIn](https://www.linkedin.com/in/pablo-iribarren-muru-93b547269)  
✉️ [Email me](mailto:pabloiribarren2003@gmail.com)

> End-to-end analysis of the **Data Science / AI / Machine Learning** job market in Spain.  
> Covers real job listing scraping, exploratory data analysis, salary prediction modelling, 
> and an interactive Streamlit dashboard.

---

## 🚀 Live Demo

🔗 **[View Dashboard on Streamlit Cloud](https://data-science-job-market-spain-pablooiiribarren.streamlit.app/)**  
📈 **[View experiments on Weights & Biases](https://wandb.ai/paulsbusiness1111/data-science-job-market-es)**

---

## 🧠 Project Goals

- Analyse the current state of the **Data Science job market in Spain**
- Identify the most in-demand **roles, skills and locations**
- Estimate salary ranges using **predictive models**
- Build an **interactive dashboard** for data exploration

---

## 🗂️ Project Structure

```bash
data-science-job-market-spain/
├── data/
│   ├── raw/              # Raw data (original from API)
│   ├── processed/        # Cleaned and analysis-ready data
│   └── sample/           # Sample or test data
├── src/
│   ├── data_collection.py    # Data collection from Adzuna API
│   ├── data_processing.py    # Cleaning and preprocessing
│   ├── skills_extractor.py   # Skills extraction and analysis
│   ├── model.py              # Predictive model training
│   └── visualization.py      # Plotly visualizations
├── app/
│   └── streamlit_dashboard.py   # Interactive dashboard
├── models/                 # Trained models (.pkl, .json)
├── notebooks/              # Exploratory analysis (Jupyter)
├── images/                 # Charts and screenshots for README
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Technologies & Libraries

| Category | Libraries |
|----------|-----------|
| 📦 Data collection | `requests`, `beautifulsoup4`, `selenium` |
| 🧹 Cleaning / EDA | `pandas`, `numpy` |
| 🤖 Modelling | `scikit-learn`, `xgboost` |
| 🧠 Experiment tracking | `wandb` |
| 📊 Visualisation | `plotly`, `matplotlib`, `seaborn`, `wordcloud` |
| 🖥 Dashboard | `streamlit` |
| 🧰 Utilities | `tqdm`, `dotenv`, `nltk` |

---

## 🧩 Project Pipeline

1. **Data Collection**  
   Scraped over **2,000 real job listings** from the public Adzuna API (Spain).

2. **Cleaning & Feature Engineering**  
   - Normalisation of city names and job roles  
   - Extraction of the 20 most frequent skills  
   - Calculation of average salary (`salary_avg`)

3. **Exploratory Data Analysis (EDA)**  
   Visualisations covering demand, location distribution and key market skills.

4. **Predictive Modelling**  
   Model comparison:
   - Ridge Regression
   - Random Forest
   - Gradient Boosting  
   ✅ Best model: **Ridge Regression** *(R² = 0.008, MAE ≈ €9,734)*

5. **Interactive Dashboard**  
   Tab-based navigation:
   - 🏠 **General overview**
   - 🗺️ **Geographic analysis**
   - 🔥 **Most demanded skills**
   - 💰 **Salaries**
   - 🤖 **AI/ML trends**
   - 🔮 **Salary predictor**

---

## 📈 Key Findings

| Insight | Result |
|---------|--------|
| 📍 Cities with most listings | Madrid, Barcelona, Valencia |
| 💼 Most demanded roles | Data Analyst, Data Engineer, Data Scientist |
| 🔥 Top skills | Python, SQL, Machine Learning, AWS, Power BI |
| 💰 Estimated average salary | ~€42,000/year |
| 🤖 AI/ML share of listings | ~25% of total |

---

## ⚙️ Local Setup

```bash
# 1️⃣ Clone the repository
git clone https://github.com/pablooiiribarren/data-science-job-market-spain.git
cd data-science-job-market-spain

# 2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run dashboard
streamlit run app/streamlit_dashboard.py
```

---

*Developed as part of a Data Science and AI portfolio project (2025).  
Dataset obtained via the public Adzuna API.*
