import streamlit as st

def intro():
    st.title("🔮 'Right!' said FRED: Q2 2025 Forecasting Challenge")
    
    st.markdown("""
Welcome to the **FRED Q2 2025 Forecasting App**!  

📊 Built for the [GJ Open Q2 2025 FRED Challenge](https://www.gjopen.com/challenges/114-right-said-fred-q2-2025-finance-and-economics-challenge), this tool blends **Monte Carlo simulations** and **bootstrapping** techniques with a clean, visual interface.    
""")

    with st.expander("📘 About the Challenge"):
        st.markdown("""
The **Right! said FRED Challenge** invites forecasters to make probabilistic predictions on economic indicators like:

- CPI inflation
- GDP growth
- Exchange rates

Your goal is to estimate the probability that a given indicator will fall within certain ranges by the end of Q2 2025.

> This app helps you **simulate** future outcomes using past data, so you can make better informed probabilistic forecasts.
""")

    with st.expander("🛠️ How to Use This App"):
        st.markdown("""
1. **Upload a CSV** containing historical data for your chosen indicator (no editing needed).
2. **Select the respective forecasting question** from the sidebar.
3. **Tune your simulation settings** — pick number of runs, adjust method weights etc.
4. **Visualize the forecast** using charts and probability histograms.
5. **Interpret and use** the results to submit your predictions.

As of now this app supports two main techniques:
- **Monte Carlo simulation** (sampling assuming normality of log returns)
- **Bootstrapping** (resampling historical patterns)

Both can be combined to produce a range of future scenarios.
""")

    with st.expander("💡 Why I Built This"):
        st.markdown("""
As a forecasting enthusiast and participant in the GJ Open challenge, I wanted a tool that was:

- **Simple to use**
- **Flexible in its methods**
- **Visually transparent**
- **Helpful for forming confident judgments**
""")
        
    with st.expander("📬 About the Creator"):
        st.markdown("""
**Saicharan Ritwik**  
- Forecasting hobbyist, 
- [GJ Open Profile](https://www.gjopen.com/forecaster/sai_39)  

*Feel free to fork or improve the app on GitHub. Feedback is welcome!*
""")
        
    with st.expander("🚀 What next?"):
        st.markdown("""
- Include weights for the crowd forecast on GJ Open and also user intuition if possible.
- Study the empirical distribution of returns to identify suitable alternatives to the normal distribution for sampling. t-distribution is a possible alternative.
- Include appropriate forecasting methods for the remaining questions of the challenge.
    """)
    
    st.markdown("---")
    st.markdown("⬆️ Use the **sidebar** to get started by uploading your CSV!")