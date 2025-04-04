import io
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime
from Intro import intro

st.set_page_config(page_title="FRED Q2 2025 Forecasting")
page = st.sidebar.selectbox("Navigate to Page:", ["📘 Introduction", "📈Forecast Simulator"])

def preprocess_data(df, column_name):
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df[column_name] = pd.to_numeric(df[column_name])
    df = df.sort_values('observation_date').dropna()
    return df


def get_sim_params(df, column_name, target_date):
    target_date = datetime.combine(target_date, datetime.min.time())   # Why was this necessary? Why this error? ChatGPT fixed this line. Need to look up why my line of code did not work here.
    current_date = df['observation_date'].max()
    trading_days = np.busday_count(current_date.date(), target_date.date())
    if trading_days < 0:
        st.warning('Target Date cannot be lesser than the Current Date!')
    current_rate = df[column_name].iloc[-1]    
    return trading_days, current_rate


def MCS(df, column_name, target_date, num_simulations, bins):
    if uploaded_file is not None:
    
        df = preprocess_data(df, column_name)
        
        df['Log Return'] = np.log(df[column_name]/df[column_name].shift(1)).dropna()    # Calculating log returns.
        mean_returns = df['Log Return'].mean()
        std_returns = df['Log Return'].std()
        
        trading_days, current_rate = get_sim_params(df, column_name, target_date)
    
        simulated_rates = np.zeros((trading_days, num_simulations))
    
        for i in range(num_simulations):
            random_returns = np.random.normal(mean_returns, std_returns, trading_days)
            rate_path = current_rate * np.exp(np.cumsum(random_returns))
            simulated_rates[:, i] = rate_path
        
        predicted_rates = simulated_rates[-1:]     # Take the last row for the predictions. This corresponds to the value observed on the last day for all the runs computed.
        return predicted_rates.flatten()

    
def bootstrapping(df, column_name, target_date, num_simulations, bins):
    if uploaded_file is not None:
        
        df = preprocess_data(df, column_name)

        df['Returns'] = df[column_name].pct_change().dropna()
        
        trading_days, current_rate = get_sim_params(df, column_name, target_date)

        simulated_rates = np.zeros((trading_days, num_simulations))

        for i in range(num_simulations):
            sampled_returns = np.random.choice(df['Returns'].dropna(), size=trading_days, replace=True)
            price_path = np.zeros(trading_days)
            price_path[0] = current_rate  
            for j in range(1, trading_days):
                price_path[j] = price_path[j-1] * (1 + sampled_returns[j-1])  
            simulated_rates[:, i] = price_path  

        predicted_rates = simulated_rates[-1, :]  
        return predicted_rates.flatten()
        
    
def get_weight_inputs(bin_data):
    with st.expander("⚖️ Adjust Forecast Weights & Inputs", expanded=False):
        st.markdown("Distribute 100% weight across the forecast components:")

        col1, col2 = st.columns(2)
        with col1:
            mcs_weight = st.slider("🎲 Monte Carlo Simulation (%)", 0, 100, 50, 1)
        with col2:
            bootstrap_weight = 100-mcs_weight
            st.write(f"Bootstrap Weight: {bootstrap_weight}%")
                    
        return mcs_weight, bootstrap_weight
    
    
def agg_results(df1, df2, column_name, bin_data):
    if uploaded_file is not None:
        mcs_preds = MCS(df1, column_name, target_date, num_simulations, bin_data)
        boot_preds = bootstrapping(df2, column_name, target_date, num_simulations, bin_data)
        
        combined_preds = (mcs_weight / 100) * mcs_preds + (bootstrap_weight / 100) * boot_preds
        counts, bin_edges = np.histogram(combined_preds, bins=bin_data, density=True)
        labels = [f"{round(bin_edges[i], 2)} - {round(bin_edges[i + 1], 2)}" for i in range(len(bin_edges) - 1)]
        probabilities = counts / counts.sum()
        
        prob_df = pd.DataFrame({"Range": labels, "Probability": probabilities})
        styled_df = prob_df.style.format({"Probability": "{:.3%}"}).background_gradient(cmap='Blues')
        
        mean_predicted_rate = combined_preds.mean() 
        median_predicted_rate = np.median(combined_preds)
        std_predicted_rate = combined_preds.std()
        percentiles = np.percentile(combined_preds, [5, 25, 50, 75, 95])
        
        st.write('***')
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Mean Prediction", f"{mean_predicted_rate:.4f}")
        col2.metric("📌 Median Prediction", f"{median_predicted_rate:.4f}")
        col3.metric("📉 Std Deviation", f"{std_predicted_rate:.4f}")
        st.write('***')
        
        col4, col5 = st.columns([1,1])
        with col4:
             st.write('### Tabulated Probability Distribution')
             st.dataframe(styled_df)
             st.write('***')
                
        with col5:
             st.subheader("Percentile Summary:")
             st.write(f" 5th Percentile : {percentiles[0]:.2f}") 
             st.write(f" 25th Percentile : {percentiles[1]:.2f}")
             st.write(f" 50th Percentile (Median) : {percentiles[2]:.2f}")
             st.write(f" 75th Percentile : {percentiles[3]:.2f}")
             st.write(f" 95th Percentile : {percentiles[4]:.2f}")
                

        st.write("### Visual Overview of Simulation Results")
        col6, col7 = st.columns([1,1])
        with col6:
             fig1, ax1 = plt.subplots()
             ax1.hist(combined_preds, bins=bin_data, edgecolor='black', alpha=0.7, label='Predictions')
             ax1.axvline(mean_predicted_rate, color='red', linestyle='dashed', linewidth=2, label='Mean')
             ax1.axvline(median_predicted_rate, color='green', linestyle='dotted', linewidth=2, label='Median')
             lower = np.percentile(combined_preds, 10)
             upper = np.percentile(combined_preds, 90)
             ax1.axvspan(lower, upper, color='blue', alpha=0.1, label='80% Interval')
             ax1.set_title("Histogram of Combined Predictions")
             ax1.set_xlabel("Predicted Value")
             ax1.set_ylabel("Frequency")
             ax1.legend()
             st.pyplot(fig1)  
            
             
             fig3, ax3 = plt.subplots()
             sorted_preds = np.sort(combined_preds)
             cdf = np.arange(len(sorted_preds)) / float(len(sorted_preds))
             ax3.plot(sorted_preds, cdf, marker='.', linestyle='none')
             ax3.set_title("Cumulative Distribution Function (CDF)")
             ax3.set_xlabel("Predicted Value")
             ax3.set_ylabel("Cumulative Probability")
             st.pyplot(fig3)
            
        with col7:               
             fig2, ax2 = plt.subplots()
             ax2.bar(labels, probabilities, color='skyblue', edgecolor='black')
             ax2.set_title("Probability Distribution of Combined Predictions")
             ax2.set_xlabel("Prediction Range")
             ax2.set_ylabel("Probability")
             ax2.tick_params(axis='x', rotation=45)
             st.pyplot(fig2) 
   
                                                

if page == "📘 Introduction":
   intro()

if page == "📈Forecast Simulator":
   ####    SIDEBAR INFORMATION    ####        
   with st.sidebar:
       st.write('***') 
       ques_option = st.selectbox("Select your question from the FRED Q2 2025 Challenge:",
                                  ("What will be the yield for US 10-year Treasury securities on 30 June 2025?",
                                   "What will be the closing value of the S&P 500 Index on 30 June 2025?",
                                   "What will be the closing value of the NASDAQ Composite Index on 30 June 2025?",
                                   "What will be the closing value of the US dollar to Mexican peso exchange rate on 27 June 2025?",
                                   "What will be the price of bitcoin on 30 June 2025?",
                                   "What will be the spot price per million BTUs for natural gas on 30 June 2025?"
                                  )
       ) 
       num_simulations = st.slider("🎲 Number of Simulations", 10000, 50000, 10000, 100)
       uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
       if uploaded_file is not None:
           content = uploaded_file.read()
           df1 = pd.read_csv(io.BytesIO(content))
           df2 = pd.read_csv(io.BytesIO(content))
       target_date = st.sidebar.date_input("📅 Select Target Date", datetime(2025, 6, 26))

       st.write('***')        
    
 
   bins_data = {3:  [0,3.00,3.30,3.60,3.90,4.20,4.50,4.80,5.10,5.40,5.70,np.inf],
                6:  [0,5000,5250,5500,5750,6000,6250,6500,6750,7000,np.inf],
                7:  [0,15200,16000,16800,17600,18400,19200,20000,20800,21600,22400,23200,np.inf],
                8:  [0,17.8,18.4,19.0,19.6,20.2,20.8,21.4,22.0,22.6,23.2,np.inf],
                9:  [0,48000,56000,64000,72000,80000,88000,96000,104000,112000,120000,128000,136000,146000,158000,172000,np.inf],
                11: [0,2,2.6,3.2,3.8,4.4,5.0,5.6,6.2,6.8,np.inf]}


   if ques_option == "What will be the yield for US 10-year Treasury securities on 30 June 2025?":
       st.title("US 10-year Treasury Security Prediction")
       st.write('***') 
       mcs_weight, bootstrap_weight = get_weight_inputs(bins_data[3])
       if uploaded_file is not None:
           if df1.columns[1] != 'DGS10':
                st.warning("This doesn’t look like what we ordered. File a refund or re-upload?")
           else:     
                agg_results(df1, df2, 'DGS10', bins_data[3])

   if ques_option == "What will be the closing value of the S&P 500 Index on 30 June 2025?":
       st.title("S&P 500 Index Prediction")
       st.write('***')
       mcs_weight, bootstrap_weight = get_weight_inputs(bins_data[6])
       if uploaded_file is not None:
           if df1.columns[1] != 'SP500':
                st.warning("Plot twist: this might be the *wrong* file. Wanna give it another go?")
           else:     
                agg_results(df1, df2, 'SP500', bins_data[6])

   if ques_option == "What will be the closing value of the NASDAQ Composite Index on 30 June 2025?":
       st.title("NASDAQ Composite Index Prediction")
       st.write('***') 
       mcs_weight, bootstrap_weight = get_weight_inputs(bins_data[7])
       if uploaded_file is not None:
           if df1.columns[1] != 'NASDAQCOM':
                st.warning("Hmm... this file looks suspiciously off. Try again, maybe?")
           else:     
                agg_results(df1, df2, 'NASDAQCOM', bins_data[7])
    
   if ques_option == "What will be the closing value of the US dollar to Mexican peso exchange rate on 27 June 2025?":
       st.title("USD/MXN Exchange Rate Prediction")
       st.write('***') 
       mcs_weight, bootstrap_weight = get_weight_inputs(bins_data[8])
       if uploaded_file is not None:
           if df1.columns[1] != 'DEXMXUS':
                st.warning("This file smells funny. Not literally, but you get the point.")
           else:     
                agg_results(df1, df2, 'DEXMXUS', bins_data[8])
    
   if ques_option == "What will be the price of bitcoin on 30 June 2025?":
       st.title("Bitcoin Price Prediction")
       st.write('***') 
       mcs_weight, bootstrap_weight = get_weight_inputs(bins_data[9])
       if uploaded_file is not None:
           if df1.columns[1] != 'CBBTCUSD':
                st.warning("Yikes. Either you uploaded the wrong file, or we’re in for a surprise.")
           else:     
                agg_results(df1, df2, 'CBBTCUSD', bins_data[9]) 
    
   if ques_option == "What will be the spot price per million BTUs for natural gas on 30 June 2025?":
       st.title('Natural Gas Price Prediction')
       st.write('***') 
       mcs_weight, bootstrap_weight = get_weight_inputs(bins_data[11])
       if uploaded_file is not None:
           if df1.columns[1] != 'DHHNGSP':
                st.warning("Red flags. This file just raised a few. Better check again.")
           else:     
                agg_results(df1, df2, 'DHHNGSP', bins_data[11])