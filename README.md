# This READ.me is not fully updated w.r.t. the deployed app




# 🧠 Claravita

[![Hugging Face Spaces](https://img.shields.io/badge/HuggingFace-Deployed-yellow?logo=huggingface)](#)
[![Python](https://img.shields.io/badge/Built_with-Python_3.10-blue?logo=python)](#)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange?logo=scikit-learn)](#)
[![AI-Powered](https://img.shields.io/badge/AI-Powered-brightgreen?logo=openai)](#)
[![Status](https://img.shields.io/badge/Status-Active_Development-orange)](#)
[![Focus](https://img.shields.io/badge/Focus-Wellness_&_Burnout-ff69b4)](#)


---

## ✨ Overview

Your life leaves clues—your **sleep**, **stress**, **mood**, **activity**, and even **screen time** silently shape your long-term wellbeing. This AI-powered platform translates those hidden signals into powerful insights.

🧭 Whether you're a student, a working professional, or anyone seeking healthier routines — this app is your **personal wellness dashboard**, driven by machine learning, explainable AI (SHAP), and visual storytelling.

> 🌱 “Let your data whisper truths. Let your habits shape healing.”

---

Project Link : https://wellnesspredictor.streamlit.app/

Demo link : https://drive.google.com/file/d/1EQ9oInLqA_UwrnioxuD4o7bATXx5-ZnG/view?usp=sharing

## 📦 Project Structure

| Folder/File               | Purpose                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| `app.py`                 | 🚀 Main Streamlit app – includes prediction logic, visualizations, SHAP |
| `Models/`                | 📁 Trained ML models (not uploaded due to size; see note below)         |
| `data/`                  | 📊 Benchmark datasets for comparison, forecasts, recovery goals         |
| `styles/`                | 🎨 Custom CSS for styling and branding                                  |
| `utils/`                 | 🛠 Utility modules (recommendation engine)         |

---

## 🧰 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io) with custom HTML/CSS styling
- **Backend**: Python 3.10+
- **Machine Learning**: Scikit-Learn , Random Forest Regressors, NearestNeighbors , MultiOutputRegressor ,  SHAP (Explainable AI), Joblib models
- **Visualization**: Plotly, Altair, Matplotlib
- **Deployment**: Compatible with Streamlit Cloud 

---

## 🔍 Core Features

### 🔮 Burnout & Wellness Predictor
Enter lifestyle metrics like sleep, screen time, mood, etc., and get a personalized prediction of:

| Metrics Predicted         | Description                                                      |
|--------------------------|------------------------------------------------------------------|
| Burnout Risk Score       | How close you are to burnout                                     |
| Stress Level             | Current stress accumulation level                                |
| Stress per Study Hour    | Time-normalized stress indicator                                 |
| Wellbeing Score          | Overall life balance and harmony                                 |
| Sleep Quality Score      | Night-time recovery & sleep effectiveness                        |
| Mental Fatigue Score     | Cognitive and emotional exhaustion indicator                     |

---

### 🌿 What This Platform Does

- 🔍 **See the Unseen**  
  Translates your data into visual insights.

- 🧠 **Intelligent Wellness Mapping**  
  Detects burnout/stress risk via AI + lifestyle patterns.

- 🌐 **Know Where You Stand**  
  Compare yourself anonymously with national or peer group benchmarks.

- 💡 **Get Smarter, Not Busier**  
  Tailored micro-habit recommendations and nudges.

---

## 🎓 Explainable AI (SHAP)

Using SHAP (SHapley Additive exPlanations), the platform shows **which factors influence your burnout the most**, with detailed breakdowns and color-coded contributions:

> ✅ Positive = reduced burnout  
> ❌ Negative = increased burnout

Interactive bar charts let users explore how each input impacts their score — in real human terms.

---

## 📈 Forecasting Burnout

With the **Adaptive Forecasting Engine**, the app shows you how your current routine may affect you in:

- 3 days, 7 days, 30 days, and up to 1 year
- Animated line charts for visual time-based risk progression
- Warnings & messages when trends show danger zones

---

## 🎯 Recovery Goal Tracker

Track your burnout recovery goals with:

- Smart goal setting (custom burnout targets + deadlines)
- Auto-progress tracking with visual indicators
- Auto-reflection system with session history

---

## 🧠 Lifestyle Recommender Engine

An ML-based recommendation engine gives curated lifestyle tips, such as:

- 🧘 Breathing routines
- 🛌 Sleep rituals
- 💪 Exercise routines
- 📵 Digital detox tips

These are **fetched dynamically** using your personal metrics + burnout score.

---

## 🧾 Model & Data Note

> ⚠️ **Model files are not uploaded** to GitHub due to file size limitations.

To run the app locally, you will need the following files in the `Models/` directory:

Models/
├── LifestylePovertyIndex2.pkl
├── smart_recommender.pkl
├── days_trainer.pkl


These models are built using Random Forests and custom-trained regressors. Contact the repository maintainer if you wish to receive them privately or retrain from scratch.

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/yourusername/lifestyle-wellness-app.git
cd lifestyle-wellness-app
pip install -r requirements.txt
streamlit run app.py
```
---
## 🤝 Contributing
Contributions, feedback, and ideas are warmly welcome! Feel free to:

- Fork this repo 🍴

- Create pull requests 🔁

- Open issues 📬

- Suggest features 🌟

---

## 🧠 Made With Heart
This app isn’t just built—it’s crafted from late-night insights, lived experiences, and the dream that wellbeing should be personal, empowering, and kind. 💙

“This isn’t just an app. It’s a gentle nudge to pause, reflect, and grow.”

---

## 🙋‍♂️ Connect with Me

<p align="center">
  <a href="https://www.linkedin.com/in/mandeep-ray-b5535627b/">
    <img src="https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href="https://github.com/Mandy1200">
    <img src="https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="mailto:mandeepray05@gmail.com">
    <img src="https://img.shields.io/badge/-Gmail-red?style=flat-square&logo=gmail&logoColor=white" alt="Email">
  </a>
</p>

