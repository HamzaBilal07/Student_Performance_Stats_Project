# Statistical Analysis and Prediction of Student Academic Performance Using Python

Semester project topic: **Student Performance & Lifestyle Analysis**

This repository contains a complete Probability and Statistics project built with Python and Streamlit. The project analyzes student academic performance using descriptive statistics, exploratory data analysis, probability, probability distributions, confidence intervals, hypothesis testing, regression modeling, prediction, and automated Word report generation.

## Team

Team name: **HAATS Academic Analytics**

| Roll Number | Name | Section | Role |
| --- | --- | --- | --- |
| 24F-0569 | Muhammad Hamza Bilal | BS(CS)4E | Group Leader |
| 24F-0563 | Aytsamullah | BS(CS)4E | Member |
| 24F-0577 | Ali Haider | BS(CS)4E | Member |
| 24F-3085 | Talha Asif | BS(SE) 4B | Member |
| 24F-3104 | Muhammad Shahab Raheem | BS(SE) 4B | Member |

## Dataset

The project uses one dataset for all analysis and prediction.

| Item | Detail |
| --- | --- |
| Dataset file | `data/student_performance_factors.csv` |
| Records | 6,607 |
| Variables | 17 |
| Target variable | `Exam_Score` |

Important variables include:

- `Hours_Studied`
- `Attendance`
- `Sleep_Hours`
- `Previous_Scores`
- `Motivation_Level`
- `Internet_Access`
- `Tutoring_Sessions`
- `Family_Income`
- `Exam_Score`

All descriptive statistics, visualizations, probability calculations, inference tests, regression models, predictions, and report outputs are based on this dataset.

## Features

- Formal title page with team details and logo
- Dataset overview and automatic data cleaning
- Data type classification: qualitative, quantitative, discrete, continuous
- Descriptive statistics: mean, median, mode, range, variance, standard deviation, quartiles, percentiles, deciles, IQR
- EDA visualizations: bar chart, pie chart, histogram, box plot, scatter plot, correlation heatmap
- Probability analysis: simple probability, joint probability, conditional probability, Bayes theorem
- Probability distributions: binomial, Poisson, hypergeometric, normal distribution fit
- Inference: confidence interval, one-sample t-test, two-sample t-test, p-value conclusion
- Regression: OLS, simple linear regression, multiple linear regression
- Prediction form using the trained regression model
- Model evaluation: R-squared, MAE, MSE, RMSE
- Word report generation using `python-docx`
- Supplementary R syntax for selected statistics topics

## Project Structure

```text
student-performance-stats-project/
|-- .gitignore
|-- .streamlit/
|   |-- config.toml
|-- app.py
|-- DEPLOYMENT.md
|-- README.md
|-- render.yaml
|-- requirements.txt
|-- runtime.txt
|-- assets/
|   |-- logo_placeholder.png
|-- data/
|   |-- student_performance_factors.csv
|-- outputs/
|   |-- graphs/
|       |-- .gitkeep
|-- reports/
|   |-- .gitkeep
|-- src/
    |-- __init__.py
    |-- data_loader.py
    |-- preprocessing.py
    |-- eda.py
    |-- probability.py
    |-- distributions.py
    |-- inference.py
    |-- regression.py
    |-- report_generator.py
    |-- supplementary_r_syntax.py
```

## Files to Push to GitHub

Push these files and folders:

- `.gitignore`
- `.streamlit/config.toml`
- `app.py`
- `DEPLOYMENT.md`
- `README.md`
- `render.yaml`
- `requirements.txt`
- `runtime.txt`
- `assets/logo_placeholder.png`
- `data/student_performance_factors.csv`
- `outputs/graphs/.gitkeep`
- `reports/.gitkeep`
- `src/`

Do not manually push generated output files unless required by your instructor:

- `reports/*.docx`
- `outputs/graphs/*`
- `__pycache__/`
- `.venv/` or `venv/`
- `.env`

These generated/local files are already handled by `.gitignore`.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
streamlit run app.py
```

## Application Sections

1. Home / Title Page
2. Dataset Overview
3. Data Cleaning
4. Descriptive Statistics
5. EDA Visualizations
6. Probability Analysis
7. Probability Distributions
8. Confidence Intervals & Hypothesis Testing
9. Regression & Prediction
10. Final Results Summary
11. Report Generator

## Creating and Pushing a New GitHub Repository

### Option 1: GitHub Website + Terminal

1. Go to [GitHub](https://github.com).
2. Click **New repository**.
3. Repository name example:

```text
student-performance-stats-project
```

4. Keep it **Public** if you want to deploy it easily on free hosting.
5. Do not add a README, `.gitignore`, or license on GitHub if these files already exist locally.
6. Click **Create repository**.
7. Open a terminal inside the project folder.
8. Run these commands:

```bash
git init
git branch -M main
git add .
git commit -m "Initial project submission"
git remote add origin https://github.com/YOUR-USERNAME/student-performance-stats-project.git
git push -u origin main
```

If the local repository already has a commit, use:

```bash
git remote add origin https://github.com/YOUR-USERNAME/student-performance-stats-project.git
git push -u origin main
```

### Updating GitHub After Changes

After editing the project:

```bash
git status
git add .
git commit -m "Update project"
git push
```

## Free Deployment Options

### Recommended: Streamlit Community Cloud

Best option for this project.

Official guide: [Deploy your app on Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Why:

- Designed specifically for Streamlit apps
- Free for educational/personal apps
- Connects directly to GitHub
- Deploys from `app.py`
- Automatically updates when you push changes to GitHub

Deployment steps:

1. Push this repository to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io).
3. Sign in with GitHub.
4. Click **Create app**.
5. Select the GitHub repository.
6. Select branch:

```text
main
```

7. Select main file path:

```text
app.py
```

8. Click **Deploy**.

### Alternative: Render

Render can host Python web services and has a free instance option for suitable projects. Official guide: [Render free web services](https://render.com/docs/free).

Use these settings:

| Setting | Value |
| --- | --- |
| Service type | Web Service |
| Runtime | Python |
| Build command | `pip install -r requirements.txt` |
| Start command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| Plan | Free |

The included `render.yaml` also stores this configuration for Render Blueprint deployment.

### Alternative: Hugging Face Spaces

Hugging Face Spaces supports Streamlit apps. Official guide: [Streamlit Spaces](https://huggingface.co/docs/hub/en/spaces-sdks-streamlit).

Basic steps:

1. Create a new Space.
2. Select **Streamlit** as the SDK.
3. Upload or sync these files:
   - `app.py`
   - `requirements.txt`
   - `assets/`
   - `data/`
   - `src/`
4. Keep the default Streamlit port `8501`.

### Railway

Railway can deploy Streamlit apps, but the free plan is credit-based. Official pricing details: [Railway free trial](https://docs.railway.com/pricing/free-trial). It is useful for testing, but Streamlit Community Cloud is simpler for this project.

### Vercel

Vercel is not recommended for a direct Streamlit deployment. Streamlit runs as a long-running Python web server and uses WebSocket-style communication. Vercel Functions do not support acting as a WebSocket server according to the official [Vercel limits documentation](https://vercel.com/docs/limits/overview), so this project should be hosted on Streamlit Community Cloud, Render, Hugging Face Spaces, Railway, or another Python web hosting platform.

## Report Generation

The Report Generator page creates a `.docx` report containing:

- Project title page
- Team logo and group member table
- Problem statement
- Objectives
- Dataset description
- Statistical methods
- Results summary
- Important graphs
- Regression results
- Code references
- Supplementary R syntax appendix
- Conclusion

Generated reports are saved in:

```text
reports/
```

Generated graphs are saved in:

```text
outputs/graphs/
```

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- scikit-learn
- Statsmodels
- python-docx
- Pillow
