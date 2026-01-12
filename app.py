import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Alpha Architect | Institutional Portfolio Analytics", layout="wide", page_icon="📈")

# --- STYLING & HELPER FUNCTIONS ---
def local_css():
    st.markdown("""
    <style>
    .metric-card {
        background-color: #0E1117;
        border: 1px solid #262730;
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stMetric label { color: #888; }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- DATA ENGINE ---
@st.cache_data
def get_stock_data(tickers, start_date, end_date):
    """Fetches historical data for tickers."""
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

@st.cache_data
def get_ticker_info(tickers):
    """Fetches sector and name info. Slow operation, cached."""
    info_dict = {'Ticker': [], 'Sector': [], 'Industry': [], 'Name': []}
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            info = ticker.info
            info_dict['Ticker'].append(t)
            info_dict['Sector'].append(info.get('sector', 'Unknown'))
            info_dict['Industry'].append(info.get('industry', 'Unknown'))
            info_dict['Name'].append(info.get('shortName', t))
        except:
            info_dict['Ticker'].append(t)
            info_dict['Sector'].append('Unknown')
            info_dict['Industry'].append('Unknown')
            info_dict['Name'].append(t)
    return pd.DataFrame(info_dict)

# --- ANALYTICAL ENGINE ---
def calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe = (returns - risk_free_rate) / std
    return returns, std, sharpe

def calculate_max_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min(), drawdown

def calculate_var_cvar(returns, confidence_level=0.05):
    """Calculates Value at Risk (VaR) and Conditional VaR (CVaR)."""
    sorted_returns = np.sort(returns)
    index = int(confidence_level * len(sorted_returns))
    var = abs(sorted_returns[index])
    cvar = abs(sorted_returns[:index].mean())
    return var, cvar

# --- SIDEBAR INPUTS ---
st.sidebar.header("🚀 Portfolio Configuration")

input_method = st.sidebar.radio("Input Method", ["Manual Entry", "CSV Upload"])

tickers_input = []
weights_input = []

if input_method == "Manual Entry":
    st.sidebar.subheader("Asset Allocation")
    # Default example
    default_tickers = "AAPL, MSFT, GOOGL, JPM, GLD, TLT"
    default_weights = "20, 20, 15, 15, 15, 15"
    
    ticker_str = st.sidebar.text_area("Enter Tickers (comma separated)", default_tickers)
    weight_str = st.sidebar.text_area("Enter Weights % (comma separated)", default_weights)
    
    if ticker_str and weight_str:
        tickers_input = [x.strip().upper() for x in ticker_str.split(',')]
        weights_input = [float(x.strip()) for x in weight_str.split(',')]
        
elif input_method == "CSV Upload":
    uploaded_file = st.sidebar.file_uploader("Upload Portfolio CSV (Columns: Ticker, Weight)", type=['csv'])
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        tickers_input = df_upload['Ticker'].tolist()
        weights_input = df_upload['Weight'].tolist()

# Normalizing weights to 1 (100%)
if weights_input:
    total_weight = sum(weights_input)
    weights_input = [w / total_weight for w in weights_input]

benchmark_ticker = st.sidebar.text_input("Benchmark Asset", "SPY")
years_back = st.sidebar.slider("Analysis Period (Years)", 1, 10, 5)

start_date = datetime.now() - timedelta(days=years_back*365)
end_date = datetime.now()

# --- MAIN EXECUTION ---
if st.sidebar.button("Analyze Portfolio"):
    with st.spinner('Crunching numbers, fetching market data, and running simulations...'):
        
        # 1. FETCH DATA
        all_tickers = tickers_input + [benchmark_ticker]
        df_prices = get_stock_data(all_tickers, start_date, end_date)
        
        if df_prices.empty:
            st.error("No data found. Please check ticker symbols.")
            st.stop()
            
        # Separate Portfolio and Benchmark
        df_portfolio_prices = df_prices[tickers_input]
        df_benchmark_price = df_prices[benchmark_ticker]
        
        # Calculate Returns
        returns = df_portfolio_prices.pct_change().dropna()
        benchmark_returns = df_benchmark_price.pct_change().dropna()
        
        # Portfolio Weighted Returns
        portfolio_returns = returns.dot(np.array(weights_input))
        
        # 2. SECTOR ANALYSIS
        df_info = get_ticker_info(tickers_input)
        df_info['Weight'] = weights_input
        
        # 3. METRICS CALCULATION
        cum_ret = (1 + portfolio_returns).cumprod()
        bench_cum_ret = (1 + benchmark_returns).cumprod()
        
        total_return = cum_ret.iloc[-1] - 1
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.02) / annual_volatility
        sortino_ratio = (annual_return - 0.02) / (portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252))
        max_dd, dd_series = calculate_max_drawdown(cum_ret)
        var_95, cvar_95 = calculate_var_cvar(portfolio_returns, 0.05)
        
        # Correlation with Benchmark
        beta = np.cov(portfolio_returns, benchmark_returns)[0][1] / np.var(benchmark_returns)

        # --- DASHBOARD LAYOUT ---
        
        st.title(f"📊 Deep-Dive Portfolio Report")
        st.markdown(f"**Analysis Period:** {start_date.date()} to {end_date.date()} | **Benchmark:** {benchmark_ticker}")
        st.markdown("---")

        # Top Level Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Exp. Annual Return", f"{annual_return:.2%}", f"{annual_return - (benchmark_returns.mean()*252):.2%} vs Bench")
        col2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", help="Risk-adjusted return (>1 is good, >2 is excellent)")
        col3.metric("Volatility (Risk)", f"{annual_volatility:.2%}")
        col4.metric("Max Drawdown", f"{max_dd:.2%}", help="Maximum observed loss from a peak to a trough")
        col5.metric("Beta", f"{beta:.2f}", help="Volatility relative to benchmark. <1 is less volatile.")

        st.markdown("---")
        
        # ROW 2: Performance & Allocation
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("📈 Cumulative Performance")
            
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, mode='lines', name='Your Portfolio', line=dict(color='#00F0FF', width=2)))
            fig_perf.add_trace(go.Scatter(x=bench_cum_ret.index, y=bench_cum_ret, mode='lines', name=benchmark_ticker, line=dict(color='#888', dash='dash')))
            
            fig_perf.update_layout(
                template="plotly_dark", 
                height=400, 
                hovermode="x unified",
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            
        with c2:
            st.subheader("🏗 Sector Allocation")
            fig_pie = px.pie(df_info, values='Weight', names='Sector', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            fig_pie.update_layout(template="plotly_dark", height=400, showlegend=True, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True)

        # ROW 3: Risk Analysis
        st.markdown("### ⚠️ Advanced Risk Analytics")
        rc1, rc2, rc3 = st.columns(3)
        
        with rc1:
            st.markdown("**Value at Risk (VaR 95%)**")
            st.markdown(f"## {var_95:.2%}")
            st.caption("On any given day, there is a 95% chance your losses will not exceed this percentage.")
            
        with rc2:
            st.markdown("**Conditional VaR (CVaR)**")
            st.markdown(f"## {cvar_95:.2%}")
            st.caption("If things go really wrong (worst 5% of days), this is the average expected loss.")
            
        with rc3:
            st.markdown("**Sortino Ratio**")
            st.markdown(f"## {sortino_ratio:.2f}")
            st.caption("Like Sharpe, but only penalizes downside volatility. Higher is better.")

        # ROW 4: Correlation & Frontier
        st.markdown("---")
        m1, m2 = st.columns(2)
        
        with m1:
            st.subheader("🔗 Asset Correlation Matrix")
            corr_matrix = returns.corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            fig_corr.update_layout(template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_corr, use_container_width=True)

        with m2:
            st.subheader("🧪 Efficient Frontier Simulation")
            
            # Monte Carlo Simulation
            num_portfolios = 2000
            results = np.zeros((3, num_portfolios))
            mean_ret = returns.mean()
            cov_mat = returns.cov()
            
            for i in range(num_portfolios):
                weights = np.random.random(len(tickers_input))
                weights /= np.sum(weights)
                
                p_ret, p_std, p_sharpe = calculate_portfolio_performance(weights, mean_ret, cov_mat)
                results[0,i] = p_std
                results[1,i] = p_ret
                results[2,i] = p_sharpe
            
            # Current Portfolio Stats
            curr_ret, curr_std, curr_sharpe = calculate_portfolio_performance(np.array(weights_input), mean_ret, cov_mat)
            
            fig_ef = go.Figure()
            
            # Scatter of random portfolios
            fig_ef.add_trace(go.Scatter(
                x=results[0,:], 
                y=results[1,:], 
                mode='markers',
                marker=dict(
                    color=results[2,:], 
                    colorscale='Viridis', 
                    showscale=True, 
                    size=5,
                    opacity=0.5
                ),
                name='Simulations'
            ))
            
            # Current Portfolio Marker
            fig_ef.add_trace(go.Scatter(
                x=[curr_std], 
                y=[curr_ret], 
                mode='markers+text', 
                marker=dict(color='red', size=15, symbol='star'),
                text=["YOU"],
                textposition="top center",
                name='Current Portfolio'
            ))
            
            fig_ef.update_layout(
                template="plotly_dark", 
                xaxis_title="Volatility (Risk)", 
                yaxis_title="Expected Return",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_ef, use_container_width=True)
            
        st.success("Analysis Complete. Ready for next query.")

else:
    st.info("👈 Please configure your portfolio in the sidebar and click 'Analyze Portfolio' to generate the report.")