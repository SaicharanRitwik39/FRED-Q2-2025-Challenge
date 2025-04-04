# FRED-Q2-2025-Challenge

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fred-q2-2025.streamlit.app/)

This Streamlit app provides **probabilistic forecasts** for selected questions from the **Federal Reserve Economic Data (FRED) Q2 2025 Challenge** using a hybrid of **Monte Carlo Simulations** and **Bootstrapping**.  

---

## 🔍 What This App Does
- Uses a **weighted combination of Monte Carlo Simulations (MCS)** and **bootstrapping** to generate probabilistic forecasts.
- Offers **interactive probability distributions** for selected FRED challenge questions.
- Provides an **intuitive interface** for uploading data, configuring simulations, and visualizing results.

---

## ⚙️ How It Works
1. **Data Upload**  
   Download the relevant data from the [FRED website](https://fred.stlouisfed.org/) and upload it through the app.
2. **Select Forecast Date**  
   Choose the future date for which you want to generate a forecast. This defines the horizon for the simulation.
3. **Choose Number of Simulations**  
   Specify how many Monte Carlo paths you want to simulate. More paths generally yield more stable distributions.
4. **Run Forecast & Visualize Results**  
   The app simulates thousands of potential futures based on historical patterns, generating a forecast distribution which you can explore visually.

---

## 🧠 Methodology: Monte Carlo + Bootstrapping

This app combines two forecasting techniques:
- **Monte Carlo Simulation (MCS)** assumes that **log returns follow a normal distribution**.
- **Bootstrapping** draws from **historical returns** to preserve empirical features like skewness and fat tails.

These are blended via a **weighted average** to strike a balance between theoretical structure and data-driven realism.

### ⚠️ Limitations
- **Normality Assumption**  
  Log returns are not truly normal; they often exhibit fat tails and volatility clustering.
- **Historical Dependence**  
  Bootstrapping assumes the future resembles the past, which may not hold in all cases.

### 🔜 Future Directions
- **Understand the underlying distributions** of returns (percent changes/log returns) through statistical analysis and empirical testing, and model accordingly.
- **Replace the normal distribution** with more appropriate alternatives such as **Student's t**, **skew-normal**, or other heavy-tailed/asymmetric distributions to better reflect market realities.
