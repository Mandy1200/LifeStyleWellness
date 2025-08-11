# app.py

# 📦 Standard & External Imports
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
import time
import joblib
import uuid
import os
import base64
from datetime import datetime

# 🔧 Internal Utility Imports
from Utils.recommender_engine import generate_suggestions

import lzma

with lzma.open('./Models/LifestylePovertyIndex2.pkl.xz', 'rb') as f:
    model = pickle.load(f)


# 🖥️ Streamlit UI Settings
st.set_page_config(layout="wide")

# 🎨 Load External Styling
with open("./Styles/styles.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 🧬 Lifestyle Mapping Setup
lifestyle_options = [
    "Athlete", "Early Riser", "Energy Drink Addict", "Fast Food Lover", "Gym Goer",
    "Healthy Eater", "Night Owl", "Non-Smoker", "Occasional Drinker",
    "Sedentary", "Smoker", "Yoga Enthusiast"
]
lifestyle_map = {val: i for i, val in enumerate(lifestyle_options)}

# 🧠 Core Scoring Logic
def get_suggestion(score):
    """
    Map numerical burnout score to a user-friendly burnout status label.
    """
    if score < 2:
        return "⚠️ High-risk burnout zone; urgent need for recovery and rest."
    elif score < 5:
        return "🌀 Turbulent Zone — Fatigue and emotional strain accumulating."
    elif score < 7:
        return "⚖️ Functionally stressed — Managing, but taxed."
    elif score < 9:
        return "🌿 Resilient & Adaptive — Healthy flow."
    else:
        return "✨ Peak Harmony — Optimal alignment."

# 🪄 Reusable UI Display Components

def display_card(label, value):
    """
    Render a value inside a styled hover card with a label.
    """
    formatted = f"{value:.2f}" if value is not None else "N/A"
    st.markdown(f"""
        <div class="hover-card">
            <strong>{label}</strong> {formatted}
        </div>
    """, unsafe_allow_html=True)

def display_suggestion_card(title, score):
    """
    Display burnout feedback message based on score in a compact card.
    """
    message = get_suggestion(score)
    st.markdown(f"""
        <div class="small-hover-card">
            <span class="suggestion-label">{title}</span>
            {message}
        </div>
    """, unsafe_allow_html=True)

def display_resource_card(title, url, icon="🔗"):
    """
    Generate an external link card for curated content.
    """
    return f"""
    <div class="resource-card">
        <a href="{url}" target="_blank">{icon} <strong>{title}</strong></a>
    </div>
    """

def display_about_card():
    """
    Render the enhanced 'About' section with rich formatting and dark theme.
    Introduces platform purpose, functionality, and wellness philosophy.
    """
    st.markdown("""
<div style="background-color: #2a2a2a; color: #ffffff; padding: 1.8rem 2rem; border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3); margin-top: 1.5rem;
            font-family: 'Segoe UI', sans-serif;">

  <h2 style="margin-bottom: 0.3rem;">📖 About This Wellness Intelligence Platform</h2>
  <h4 style="color: #cccccc; margin-top: 0;">Reclaim balance. Rebuild resilience. One insight at a time.</h4>

  <p style="margin-top: 1.2rem; line-height: 1.7; font-size: 15.8px;">
    ✨ Your life leaves clues—in your sleep rhythms, work stress, movement, mood, and even screen time.
    This platform decodes those hidden signals, helping you uncover how small daily habits shape your long-term wellbeing.
  </p>

  <h4 style="margin-top: 1.8rem;">🌿 What This Platform Does:</h4>

  <p style="margin-top: 1.4rem;"><strong>🔍 See the Unseen</strong><br>
  It doesn’t just collect data—it translates it. Understand how your energy dips, digital fatigue, or restless nights contribute to burnout or mental fog. Get crystal-clear insights into what’s helping or harming your wellness.</p>

  <p style="margin-top: 1.4rem;"><strong>🧠 Intelligent Wellness Mapping</strong><br>
  Using cutting-edge AI, we analyze your lifestyle patterns to spotlight burnout risk, stress imbalances, and emotional fatigue—with interactive visual storytelling that connects the dots for you.</p>

  <p style="margin-top: 1.4rem;"><strong>🌐 Know Where You Stand</strong><br>
  Compare your scores anonymously with national averages or peer groups—see if your stress patterns are unique or part of a larger trend. Context matters.</p>

  <p style="margin-top: 1.4rem;"><strong>💡 Get Smarter, Not Busier</strong><br>
  Receive curated strategies, personalized recommendations, and micro-habit nudges tailored to your routines—not generic advice. Whether it's breathing better, sleeping deeper, or disconnecting smarter—we help you work with your nature.</p>

  <h4 style="margin-top: 1.5rem;">🚀 Why It Matters</h4>
  <p style="line-height: 1.7;">
    Because resilience isn't built in a day—it’s crafted in the tiny decisions we make when no one's watching.
    This platform is your personal guide to:
  </p>

  <ul style="line-height: 1.8; font-size: 15.5px;">
    <li>🧭 Becoming more self-aware</li>
    <li>🛠 Making meaningful changes</li>
    <li>🌱 Living a healthier, more mindful life</li>
  </ul>

  <p style="margin-top: 1.4rem; font-style: italic; color: #dddddd;">
    Let your data whisper truths. Let your habits shape healing.<br>
    <strong>Start your wellness intelligence journey—today.</strong>
  </p>

</div>
    """, unsafe_allow_html=True)

def display_story_card():
    """
    Display the personal 'Story' section to convey emotional and human background.
    Emphasizes the mission and emotional intent behind building the platform.
    """
    st.markdown(f"""
        <div class="about-card">
            <h3>The Story</h3>
            <p>This began as more than an idea — it was a feeling. A quiet knowing that our daily habits shape more than our days; they shape our lives. That’s what sparked this journey.</p> <p>Every click, every feature, every word here carries a little heartbeat — born from late-night thoughts, real conversations, and the hope that we could make wellbeing more personal, more human, more kind.</p> <p>This isn’t just an app. It’s a gentle nudge. A reflection of the belief that small shifts can lead to brighter days — that understanding yourself is the first step toward taking care of yourself.</p> <p>What you see here is built with care, intention, and joy — to help you pause, reflect, and grow. It’s for the student pulling all-nighters, the dreamer chasing balance, the soul seeking calm in the chaos.</p> <p>We’re still learning, still building, still dreaming. But through it all, one thing stays true: this was made with heart — for every moment you choose to take care of yours. 💙</p>
            <p>Made With 💗 by Mandy</p>    
         </div>
    """, unsafe_allow_html=True)

def display_intro_card():
    """
    Render the top intro banner — compact, elegant, and thematic.
    """
    st.markdown("""
        <div id="top" class="card" style="display:flex; justify-content:space-between; align-items:center; padding: 6px 10px; margin-bottom: 10px;">
            <p style="margin: 0; font-size: 14px; color:#002f4b;">Reclaim balance. Rebuild resilience. One insight at a time</p>
        </div>
    """, unsafe_allow_html=True)

# 📌 Display intro banner on load
display_intro_card()


# -----------------------------------------------------------
# Title Section: Renders the app header with an emoji and title.
# -----------------------------------------------------------
st.markdown("""
<div class="card" style="text-align:center;">
    <h1>🧠 Lifestyle Wellness & Burnout Prediction</h1>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# User Input Section: Collects lifestyle and routine parameters 
# from the user through various input widgets.
# -----------------------------------------------------------

with st.container():
    st.markdown("""<div class="card"><h3>📥 Enter Your Lifestyle & Routine Data</h3></div>""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.3, 1.3])
    age = col1.number_input("Age", min_value=10.0, max_value=60.0, value=20.0, step=0.1)
    sleep_hours = col2.number_input("Sleep Hours", 0.0, 10.0, 6.5, 0.1)
    study_hours = col3.number_input("Study Hours/Work Hours", 0.0, 10.0, 4.0, 0.1)

    col4, col5 = st.columns(2)
    screen_time = col4.number_input("Screen Time Post 10PM", 0.0, 10.0, 2.0, 0.1)
    physical_activity = col5.number_input("Physical Activity Minutes", 0.0, 300.0, 30.0, 1.0)

    col6, col7 = st.columns(2)
    mood_score = col6.number_input("Mood Score (How's You Are Feeling Today Out Of 10)", 0.0, 10.0, 6.8, 0.1)
    sleep_debt = col7.number_input("Sleep Debt (rest you still owe your body)", 0.0, 10.0, 1.2, 0.1)

    selected_habits = st.multiselect("Lifestyle Type (select one or more)", options=lifestyle_options)
    lifestyle_encoded = lifestyle_map[selected_habits[0]] if selected_habits else 0

    predict_btn = st.button("🔮 Predict Wellness & Burnout Scores")

# -----------------------------------------------------------
# Prediction Trigger & Initialization: Prepares for prediction
# and handles the state for storing prediction results.
# -----------------------------------------------------------

if "predictions" not in st.session_state:
    st.session_state.predictions = None

if predict_btn:
    input_data = np.array([[age, sleep_hours, study_hours, screen_time,
                            physical_activity, mood_score, sleep_debt, lifestyle_encoded]])
    predictions = model.predict(input_data)[0]
    st.session_state.predictions = predictions

# Retrieve predictions (even after rerun)# -----------------------------------------------------------
# Prediction Output Section: Displays various scores from the 
# model after prediction is made.
# -----------------------------------------------------------

if st.session_state.predictions is not None:
    burnout, stress_level, stress_per_hr, well_score, sleep_quality, mental_fatigue = st.session_state.predictions
else:
    burnout = stress_level = stress_per_hr = well_score = sleep_quality = mental_fatigue = None

if st.session_state.predictions is not None:

# Output
    with st.container():
        st.markdown("""<div class="card"><h3>📤 Predictions</h3></div>""", unsafe_allow_html=True)
        if burnout is not None:
            display_card("Burnout Risk Score", burnout)
            display_card("Stress Level", stress_level)
            display_card("Stress per Study/Work Hour", stress_per_hr)
            display_card("Overall Wellbeing Score", well_score)
            display_card("Sleep Quality Score", sleep_quality)
            display_card("Mental Fatigue Score", mental_fatigue)

# -----------------------------------------------------------
# Peer Comparison Section: Allows the user to compare their
# scores against other groups using preloaded CSV data.
# -----------------------------------------------------------


if burnout is not None:
    st.markdown("""<div class="card"><h3>🌐 Compare Your Lifestyle</h3></div>""", unsafe_allow_html=True)

    compare_df = pd.read_csv("./Dataset/comparisons.csv")
    groups = compare_df['group'].unique()
    selected_group = st.selectbox("Choose a lifestyle group to compare with:", groups)

    # Fetch group data
    group_data = compare_df[compare_df['group'] == selected_group].iloc[0, 1:].values.reshape(1, -1)
    group_prediction = model.predict(group_data)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧍 Your Burnout Score")
        display_card("Burnout", burnout)
        display_card("Stress Level", stress_level)

    with col2:
        st.markdown(f"#### 👥 {selected_group}")
        display_card("Burnout", group_prediction[0])
        display_card("Stress Level", group_prediction[1])
  
# -----------------------------------------------------------
# Visualization Tabs Section: 
# 1. Radar chart for lifestyle scores.
# 2. Trendline comparison against peers and national average.
# -----------------------------------------------------------

if st.session_state.predictions is not None:
    tab1, tab2 = st.tabs(["🕸️ Lifestyle Radar", "📈 Trendline vs Peers"])

    # --- Tab 1: Radar Chart ---
    with tab1:
        st.markdown("""<div class="card"><h4>🕸️ Lifestyle Score Overview</h4></div>""", unsafe_allow_html=True)

        categories = ['Burnout', 'Stress', 'Stress/Hour', 'Wellbeing', 'Sleep Quality', 'Mental Fatigue']
        values = [burnout, stress_level, stress_per_hr, well_score, sleep_quality, mental_fatigue]

        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Your Scores',
            line=dict(color='royalblue')
        ))

        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=400
        )

        st.plotly_chart(radar_fig, use_container_width=True)

    # --- Tab 2: Trendline vs Peers ---
    with tab2:
        st.markdown("""<div class="card"><h4>📈 Trendline vs Peers</h4></div>""", unsafe_allow_html=True)

        peer_df = pd.read_csv("./data/comparisons.csv")
        national_avg = peer_df[peer_df["group"] == "National Average"].iloc[0, 1:].values
        selected_peer = peer_df[peer_df["group"] == selected_group].iloc[0, 1:].values
        user_scores = np.array(st.session_state.predictions[:6])

        score_labels = ["Burnout", "Stress Level", "Stress/Hour", "Wellbeing", "Sleep Quality", "Mental Fatigue"]

        df = pd.DataFrame({
            "Score Type": score_labels * 3,
            "Value": np.concatenate([user_scores, selected_peer[:6], national_avg[:6]]),
            "Source": ["You"] * 6 + [selected_group] * 6 + ["National Average"] * 6
        })

        trend_fig = px.line(df, x="Score Type", y="Value", color="Source", markers=True)
        trend_fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#D2ECFF"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(trend_fig, use_container_width=True)


# -----------------------------------------------------------
# Suggestions Section: Personalized feedback cards based on
# individual scores.
# -----------------------------------------------------------

RECOVERY_GOAL_FILE = "./data/recovery_goals.csv"
if st.session_state.predictions is not None:

# Suggestions
    with st.container():
        st.markdown("""<div class="card"><h3>📌 Suggestions Based on Your Scores</h3></div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if burnout is not None:
            display_suggestion_card("🔥 Burnout Risk Score", burnout)
            display_suggestion_card("💢 Stress Level", stress_level)
            display_suggestion_card("📚 Stress Per Study/Work Hour", stress_per_hr)
    
    with col2:
        if burnout is not None:
            display_suggestion_card("🧘 Overall Wellbeing Score", well_score)
            display_suggestion_card("😴 Sleep Quality Score", sleep_quality)
            display_suggestion_card("🧠 Mental Fatigue Score", mental_fatigue)
            
# -----------------------------------------------------------
# Explainability (SHAP) Section: Uses SHAP to visually explain 
# the impact of each input feature on the burnout score.
# -----------------------------------------------------------

if st.session_state.predictions is not None:

    if burnout is not None:
        st.markdown("""<div class="card"><h3>🔍 Why Burnout Score? (Explainable AI)</h3></div>""", unsafe_allow_html=True)
    
        # Define feature names (must match your model training order)
        features = [
        "Age", "Sleep_Hours", "Study_Hours/Work_Hours", "Screen_Time_Post_10PM",
        "Physical_Activity_Minutes", "Mood_Score", "Sleep_Debt", "Lifestyle_Encoded"
        ]
    
    # Form the same input vector used for prediction
        user_vector = np.array([[age, sleep_hours, study_hours, screen_time,
                             physical_activity, mood_score, sleep_debt, lifestyle_encoded]])
        # Select only burnout model (index 0)
        burnout_model = model.estimators_[0]
    
        # SHAP Explainer
        explainer = shap.TreeExplainer(burnout_model)
    
        shap_values = explainer(user_vector)
    
        shap_vals = shap_values.values[0]
        percent_contributions = np.abs(shap_vals) / np.sum(np.abs(shap_vals)) * 100
        sorted_idx = np.argsort(percent_contributions)[::-1]
        sorted_features = [features[i] for i in sorted_idx]
        sorted_percents = [percent_contributions[i] for i in sorted_idx]
        sorted_values = [shap_vals[i] for i in sorted_idx]
    
        # Plotly Bar Chart in Card
        with st.container():
            tip_map = {
        "Age": "🎯 Age may influence recovery rate and energy resilience.",
        "Sleep_Hours": "🛌 Increasing sleep hours can reduce mental fatigue and improve overall wellness.",
        "Study_Hours/Work_Hours": "📚 High study hours can increase stress—balance with breaks.",
        "Screen_Time_Post_10PM": "📱 Reducing late-night screen time improves sleep quality and reduces stress.",
        "Physical_Activity_Minutes": "💪 Regular activity boosts endorphins and reduces burnout risk.",
        "Mood_Score": "😊 A higher mood score often reflects emotional stability and mental resilience.",
        "Sleep_Debt": "😴 Reducing sleep debt can improve focus, mood, and reduce burnout.",
        "Lifestyle_Encoded": "🧬 Different lifestyle types can influence your physical and mental energy patterns."
    }
            fig = go.Figure(data=[
                  go.Bar(
                    x=sorted_percents,
                    y=sorted_features,
                    orientation='h',
                    text=[f"{v:+.2f}" for v in sorted_values],
                    textposition='outside',
                        textfont=dict( color='#0073ff', size=14,family='Arial'),
                    hovertext=[
                        f"{sorted_features[i]} → {tip_map.get(sorted_features[i], '')}"
                        for i in range(len(sorted_features))
                    ],
        hoverinfo='text',
        marker=dict(color='rgba(255,140,0,0.7)')
    )
    
            ])
    
            fig.update_layout(
                xaxis_title="Impact on Burnout Score (%)",
                yaxis_title="Feature",
                height=400,
                margin=dict(l=60, r=30, t=30, b=30),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#B1DDF1")
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
                        
        <div style="margin-top: 15px; background-color: #111; padding: 15px; border-radius: 8px;">
        <p style="color:#CCCCCC; font-size: 14px;">
            ✅ <strong>Positive (+)</strong> values mean the feature helped <span style="color:#90ee90;">reduce burnout</span>.<br>
            ❌ <strong>Negative (−)</strong> values mean the feature <span style="color:#ff726f;">increased burnout</span>.<br>
            📊 The number (like +0.35 or -0.16) shows how much it pushed your score.
        </p>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------
# Smart Recommendation System: Loads a separate ML model 
# for generating personalized lifestyle suggestions.
# -----------------------------------------------------------

if st.session_state.predictions is not None:
    st.markdown("""<div class="card"><h3>🎬 Smart Lifestyle Suggestions for You</h3></div>""", unsafe_allow_html=True)

    user_input = {
        "age": age,
        "sleep": sleep_hours,
        "study": study_hours,
        "screen": screen_time,
        "activity": physical_activity,
        "mood": mood_score,
        "debt": sleep_debt,
        "lifestyle": lifestyle_encoded
    }
    @st.cache_resource
    def load_recomm_model():
        return joblib.load("./Models/smart_recommender.pkl")

    recommender_model = load_recomm_model()

    suggestions = generate_suggestions(recommender_model, user_input)

    if suggestions:
        # Show first 9 in a 3x3 grid
        for row in range(3):
            cols = st.columns(3)
            for col_idx in range(3):
                idx = row * 3 + col_idx
                if idx < len(suggestions):
                    s = suggestions[idx]
                    card_html = display_resource_card(s["title"], s["url"], s["icon"])
                    cols[col_idx].markdown(card_html, unsafe_allow_html=True)


            
# -----------------------------------------------------------
# Forecast Model Loader: Loads an external model for 
# future trajectory forecasting of wellness scores.
# -----------------------------------------------------------

import lzma
import joblib

@st.cache_resource
def load_forecast_model():
    with lzma.open("./Models/days_trainer.pkl.xz", "rb") as f:
        return joblib.load(f)

forecast_model = load_forecast_model()


# -----------------------------------------------------------
# Adaptive Forecast Section: Predicts how scores will evolve
# over various future days if current lifestyle is maintained.
# -----------------------------------------------------------


if st.session_state.predictions is not None:
    st.markdown("""
        <div class="card">
            <h3>📈 Adaptive Burnout & Wellbeing Forecast</h3>
            <p style='font-size: 14px; margin-bottom:0;'>Projected scores if your current lifestyle continues.</p>
        </div>
    """, unsafe_allow_html=True)

    # 📥 Input rows for each day
    days = [3, 7, 14, 21, 30, 45, 90, 182, 365]
    forecast_df = pd.DataFrame([{
        "Age": age,
        "Sleep_Hours": sleep_hours,
        "Study_Hours": study_hours,
        "Screen_Time_Post_10PM": screen_time,
        "Physical_Activity_Minutes": physical_activity,
        "Mood_Score": mood_score,
        "Sleep_Debt": sleep_debt,
        "Day": d,
        "Lifestyle_Score": lifestyle_encoded
    } for d in days])

    # 🛑 Check for NaNs
    if forecast_df.isnull().any().any():
        st.error("🚫 Forecast data has missing values.")
        st.dataframe(forecast_df)
        st.stop()

    # 🔮 Predict
    try:
        predictions = forecast_model.predict(forecast_df)
    except Exception as e:
        st.error(f"⚠️ Forecast model failed: {str(e)}")
        st.stop()

    forecast_plot_df = pd.DataFrame(predictions, columns=[
        "Burnout_Risk_Score", "Stress_Level", "Stress_Per_Study_Hour/Work_Hour",
        "Overall_Wellbeing_Score", "Sleep_Quality_Score", "Mental_Fatigue_Score"
    ])
    forecast_plot_df["Day"] = days

    # 📊 Metric selection
    visible_metrics = st.multiselect(
        "📊 Select metrics to include in forecast:",
        forecast_plot_df.columns[:-1].tolist(),
        default=forecast_plot_df.columns[:-1].tolist()
    )

    # 🎞️ Animate toggle
    animate = st.checkbox("🎞️ Animate Forecast", value=True)

    chart_placeholder = st.empty()
    status_placeholder = st.empty()

    if animate:
        for i in range(1, len(forecast_plot_df) + 1):
            temp_df = forecast_plot_df.iloc[:i]
            long_df = temp_df.melt(id_vars="Day", var_name="Metric", value_name="Score")
            long_df = long_df[long_df["Metric"].isin(visible_metrics)]

            chart = alt.Chart(long_df).mark_line(point=True).encode(
                x=alt.X("Day:O", title="Day"),
                y=alt.Y("Score:Q", title="Score"),
                color=alt.Color("Metric:N", legend=alt.Legend(
                    orient="bottom",
                    direction="horizontal",
                    title=None,
                    labelFontSize=12,
                    symbolSize=100
                )),
                tooltip=["Day", "Metric", "Score"]
            ).properties(height=400)


            chart_placeholder.altair_chart(chart, use_container_width=True)

            current_day = int(temp_df["Day"].iloc[-1])
            current_burnout = float(temp_df["Burnout_Risk_Score"].iloc[-1])

            if current_burnout >= 85:
                status_placeholder.markdown(f"🛑 **Day {current_day}**: Burnout extremely high (**{current_burnout:.1f}**)! ⚠️")
            elif current_burnout >= 70:
                status_placeholder.markdown(f"⚠️ **Day {current_day}**: Burnout rising (**{current_burnout:.1f}**) — consider changes.")
            elif current_burnout >= 50:
                status_placeholder.markdown(f"🔄 **Day {current_day}**: Moderate burnout (**{current_burnout:.1f}**) — monitor balance.")
            else:
                status_placeholder.markdown(f"✅ **Day {current_day}**: Burnout healthy (**{current_burnout:.1f}**) — keep it up!")

            time.sleep(0.45)
    else:
        long_df = forecast_plot_df.melt(id_vars="Day", var_name="Metric", value_name="Score")
        long_df = long_df[long_df["Metric"].isin(visible_metrics)]

        chart = alt.Chart(long_df).mark_line(point=True).encode(
            x=alt.X("Day:O", title="Day"),
            y=alt.Y("Score:Q", title="Score"),
            color="Metric:N",
            tooltip=["Day", "Metric", "Score"]
        ).properties(height=400)

        chart_placeholder.altair_chart(chart, use_container_width=True)

# -----------------------------------------------------------
# Forecast Trend Summary: Provides analytical summary of the
# burnout score trend over time (increasing, stable, etc.).
# -----------------------------------------------------------

    burnout_scores = forecast_plot_df["Burnout_Risk_Score"].tolist()
    days = forecast_plot_df["Day"].tolist()
    
    min_score = min(burnout_scores)
    max_score = max(burnout_scores)
    start_score = burnout_scores[0]
    end_score = burnout_scores[-1]
    
    # Calculate net change
    delta = end_score - start_score
    
    # Get trend direction (very basic linearity check)
    rising = all(earlier <= later for earlier, later in zip(burnout_scores, burnout_scores[1:]))
    falling = all(earlier >= later for earlier, later in zip(burnout_scores, burnout_scores[1:]))
    
    if rising and delta > 5:
        st.error(f"🛑 Burnout is **steadily increasing** from {start_score:.1f} to {end_score:.1f} over the year — serious lifestyle adjustments are needed.")
    elif falling and delta < -5:
        st.success(f"✅ Burnout is **consistently decreasing** from {start_score:.1f} to {end_score:.1f} — great improvement ahead!")
    elif max_score - min_score < 8:
        st.info(f"📊 Burnout remains **relatively stable** between {min_score:.1f} and {max_score:.1f} throughout the year.")
    else:
        st.warning(f"⚠️ Burnout shows **fluctuating trend** (from {min_score:.1f} to {max_score:.1f}). Watch out for possible rebounds — try to maintain healthy consistency.")
    
    
# -----------------------------------------------------------
# Goal Tracker Setup: Handles user ID and persistence of
# recovery goals via Google Sheets integration.
# -----------------------------------------------------------
 
    
#     from utils.google_sheet_handler import (
#         save_goal_to_sheet,
#         load_latest_goal_from_sheet,
#         load_all_goals_for_user,
#         delete_latest_goal_for_user
#     )
    
#     # 🆔 User ID Setup
#     def get_or_create_user_id():
#         if "user_id" in st.session_state:
#             return st.session_state.user_id
    
#         query_params = st.query_params
#         cookie_id = query_params.get("user")
    
#         if cookie_id:
#             user_id = cookie_id
#         else:
#             user_id = "user_" + str(uuid.uuid4())[:8]
#             st.query_params["user"] = user_id
    
#         st.session_state.user_id = user_id
#         return user_id
    
#     user_id = get_or_create_user_id()

# # -----------------------------------------------------------
# # Goal Data Management: Includes goal save, load, and session 
# # tracking across app reruns.
# # -----------------------------------------------------------

    
#     def save_recovery_goal(user_id: str, goal_data: dict):
#         goal_data["user_id"] = user_id
#         goal_data["start_time"] = datetime.now().isoformat()
#         goal_data["timestamp"] = datetime.now().isoformat()
#         save_goal_to_sheet(user_id, goal_data)
    
#     def load_latest_recovery_goal(user_id: str):
#         return load_latest_goal_from_sheet(user_id)
    
#     # 🧠 Store session inputs
#     session_data = {
#         "timestamp": datetime.now().isoformat(),
#         "burnout": float(burnout),
#         "sleep": float(st.session_state.get("sleep", 0)),
#         "mood": float(st.session_state.get("mood", 0)),
#         "workload": float(st.session_state.get("workload", 0))
#     }
#     if "history" not in st.session_state:
#         st.session_state["history"] = []
#     st.session_state["history"].append(session_data)
    
# # -----------------------------------------------------------
# # Goal Tracker UI: Displays current progress, handles goal
# # creation, deletion, and guidance based on performance.
# # -----------------------------------------------------------

#     st.markdown("""<div class="card"><h3>🧭   </h3></div>""", unsafe_allow_html=True)
    
#     # Load goal
#     if "recovery_goal" not in st.session_state:
#         stored_goal = load_latest_recovery_goal(user_id)
#         if stored_goal:
#             st.session_state.recovery_goal = stored_goal
    
#     # --- Set or Track ---
#     if "recovery_goal" not in st.session_state:
#         with st.form("set_recovery_goal"):
#             st.markdown("🎯 Set a burnout recovery goal:")
#             target = st.slider("Target Burnout Score", 0.0, burnout, burnout - 1.0, step=0.1)
#             duration = st.number_input("Days to achieve", min_value=1, max_value=30, value=7)
#             submit = st.form_submit_button("Set Goal")
    
#             if submit:
#                 with st.spinner("🎯 Saving your goal..."):
#                     st.toast("🎯 Goal Saved!", icon="💾")
#                     st.session_state.recovery_goal = {
#                         "start": burnout,
#                         "target": target,
#                         "days": duration,
#                         "start_time": datetime.now().isoformat()
#                     }
#                     save_recovery_goal(user_id, st.session_state.recovery_goal)
#                     time.sleep(2)
#                     st.rerun()
#     else:
#         goal = st.session_state.recovery_goal
#         start = float(goal["start"])
#         target = float(goal["target"])
#         days = int(goal["days"])
#         start_time = pd.to_datetime(goal["start_time"])
#         days_passed = (datetime.now() - start_time).days
    
#         progress = 100 * max(0, start - burnout) / max(1e-5, start - target)
#         st.markdown(f"🧗 Goal: {start:.1f} → {target:.1f} in {days} days")
#         st.markdown(f"⏳ Day {days_passed}/{days}")
#         st.progress(min(progress / 100, 1.0))
    
#         if burnout <= target:
#             st.success("🎉 Goal Achieved!")
#         elif days_passed >= days:
#             st.error("⏱️ Time's up — consider resetting your goal.")
#         elif progress < 30:
#             st.warning("📉 Slow start — try small lifestyle shifts.")
#         else:
#             st.info("✅ You're on track — keep going!")
    
# # -----------------------------------------------------------
# # Goal Reset/Delete Feature: Allows the user to reset or delete
# # their recovery goal via UI confirmation.
# # -----------------------------------------------------------

#         if st.button("🗑️ Reset/Delete Goal"):
#             confirmed = st.radio("Are you sure?", ["No", "Yes"], index=0)
#             if confirmed == "Yes":
#                 success = delete_latest_goal_for_user(user_id)
#                 if success:
#                     st.session_state.pop("recovery_goal", None)
#                     st.success("✅ Goal deleted.")
#                     st.rerun()
#                 else:
#                     st.error("⚠️ No goal found to delete.")
    
# # -----------------------------------------------------------
# # Goal History Viewer: Shows all past goals for the user 
# # retrieved from persistent storage (Google Sheets).
# # -----------------------------------------------------------

#     with st.expander("📜 View All Past Goals", expanded=False):
#         all_goals_df = load_all_goals_for_user(user_id)
#         if not all_goals_df.empty:
#             st.dataframe(all_goals_df, height=250, use_container_width=True)
#         else:
#             st.info("No past goals found.")

        
         
# -----------------------------------------------------------
# About Section: Explains the platform’s philosophy, features, 
# and intended user impact.
# -----------------------------------------------------------

with st.container():
    st.markdown('<div id="about"></div>', unsafe_allow_html=True)
    display_about_card()

# -----------------------------------------------------------
# Story Section: Shares the backstory and motivation behind
# the creation of the platform in a warm, human tone.
# -----------------------------------------------------------

with st.container():
    st.markdown('<div id="story"></div>', unsafe_allow_html=True)
    display_story_card()

# st.markdown('<a id="top-btn" href="#top">🔝 Top</a>', unsafe_allow_html=True)






