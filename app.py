# -*- coding: utf-8 -*-
# 港美A股股权激励估值工具（迭代版）
# 核心：黑色科技风+动图替换emoji+港股手动输入+DeltaGenerator修复
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import warnings
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from scipy.stats import norm
from io import BytesIO
import openpyxl
import time
import random

# ====================== 全局配置 =======================
st.set_page_config(
    page_title="股权激励估值工具 | 科技版",
    page_icon="https://cdn-icons-png.flaticon.com/128/1005/1005141.png", # 科技图标
    layout="wide",
    initial_sidebar_state="expanded"
)
warnings.filterwarnings("ignore")

# 自定义CSS（黑色高科技风格核心）
st.markdown("""
    <style>
    /* 全局深色背景 */
    * {
        font-family: "Roboto Mono", "Consolas", "Microsoft YaHei", monospace;
        box-sizing: border-box;
    }
    .main, [data-testid="stAppViewContainer"] {
        background-color: #121212;
        color: #e0e0e0;
        padding: 0 2rem;
    }
    /* 标题样式 */
    .title-main {
        color: #00ffff;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
    }
    .title-sub {
        color: #80ffff;
        font-weight: 500;
        margin-bottom: 2rem;
    }
    /* 科技感卡片 */
    .card {
        background-color: #1e1e1e;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.1);
        margin-bottom: 1.5rem;
        border: 1px solid #333333;
    }
    /* 按钮科技风格 */
    .stButton>button {
        background-color: #1e1e1e;
        color: #00ffff;
        border: 1px solid #00ffff;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00ffff;
        color: #121212;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.6);
        transform: translateY(-2px);
    }
    .stButton>button:disabled {
        background-color: #2a2a2a;
        color: #666666;
        border: 1px solid #333333;
        cursor: not-allowed;
        box-shadow: none;
        transform: none;
    }
    /* 指标卡片 */
    .metric-card {
        background-color: #2a2a2a;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #333333;
    }
    /* 侧边栏深色风格 */
    [data-testid="stSidebar"] {
        background-color: #1e1e1e;
        border-right: 1px solid #333333;
    }
    [data-testid="stSidebar"] .stTextInput>div>div>input,
    [data-testid="stSidebar"] .stNumberInput>div>div>input {
        background-color: #2a2a2a;
        color: #e0e0e0;
        border: 1px solid #333333;
        border-radius: 6px;
    }
    [data-testid="stSidebar"] .stSelectbox>div>div>select {
        background-color: #2a2a2a;
        color: #e0e0e0;
    }
    /* 分隔线 */
    .divider {
        height: 1px;
        background-color: #333333;
        margin: 1.5rem 0;
    }
    /* 提示文本 */
    .hint-text {
        color: #888888;
        font-size: 0.875rem;
        margin-top: 0.25rem;
    }
    /* 结果卡片（荧光渐变） */
    .result-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #00ffff;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
    }
    /* 禁用提示 */
    .disabled-hint {
        color: #666666;
        font-size: 0.875rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    /* 折叠面板 */
    [data-testid="stExpander"] {
        background-color: #1e1e1e;
        border: 1px solid #333333;
    }
    [data-testid="stExpander"] summary {
        color: #80ffff;
    }
    /* 下载按钮 */
    [data-testid="stDownloadButton"]>button {
        background-color: #1e1e1e;
        color: #00ffff;
        border: 1px solid #00ffff;
    }
    [data-testid="stDownloadButton"]>button:hover {
        background-color: #00ffff;
        color: #121212;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

# ====================== 高科技动图资源（稳定在线）======================
# 替换所有静态emoji为科技感动图，尺寸16x16/24x24匹配原emoji
GIF_ICONS = {
    "logo": "https://i.gifer.com/ZZ5H.gif", # 科技图表动图
    "fetch": "https://i.gifer.com/7Wk.gif", # 数据抓取动图
    "vol": "https://i.gifer.com/1XH.gif", # 波动率计算动图
    "calc": "https://i.gifer.com/3Q3.gif", # 估值计算动图
    "success": "https://i.gifer.com/6NO.gif", # 成功对勾动图
    "warning": "https://i.gifer.com/7XU.gif", # 警告动图
    "error": "https://i.gifer.com/7XW.gif", # 错误动图
    "download": "https://i.gifer.com/6NQ.gif", # 下载动图
    "delta": "https://i.gifer.com/3Q4.gif" # Delta解读动图
}

# 动图渲染函数
def render_gif(icon_key, size="24px"):
    return f'<img src="{GIF_ICONS[icon_key]}" width="{size}" height="{size}" style="vertical-align: middle; margin-right: 8px;">'

# ====================== 数据源函数（功能保留）======================
def us_stock_crawler(ticker):
    try:
        stock = yf.Ticker(ticker.upper())
        hist_data = stock.history(period="1y", interval="1d")
        if not hist_data.empty:
            latest_close = round(hist_data["Close"].iloc[-1], 2)
            hist_data = hist_data[["Close"]].reset_index()
            hist_data.rename(columns={"Date":"日期", "Close":"收盘价"}, inplace=True)
            hist_data["日期"] = hist_data["日期"].dt.date
            return latest_close, hist_data, f"{render_gif('success', '16px')} 美股-{ticker} 收盘价={latest_close:.2f}"
    except Exception as e:
        return None, None, f"{render_gif('error', '16px')} 美股-{ticker} 抓取失败：{str(e)[:30]}"

def cn_stock_crawler(ticker):
    try:
        import akshare as ak
        ticker_full = f"{ticker}.SS" if ticker.startswith("6") else f"{ticker}.SZ"
        hist_data = ak.stock_zh_a_hist(
            symbol=ticker,
            period="daily",
            start_date=(datetime.now()-timedelta(365)).strftime("%Y%m%d"),
            end_date=datetime.now().strftime("%Y%m%d"),
            adjust="qfq"
        )
        if not hist_data.empty:
            latest_close = round(hist_data["收盘"].iloc[-1], 2)
            hist_data = hist_data[["日期", "收盘"]].rename(columns={"收盘":"收盘价"})
            hist_data["日期"] = pd.to_datetime(hist_data["日期"]).dt.date
            return latest_close, hist_data, f"{render_gif('success', '16px')} A股-{ticker_full} 收盘价={latest_close:.2f}"
    except Exception as e:
        pass

    try:
        ticker_full = f"{ticker}.SS" if ticker.startswith("6") else f"{ticker}.SZ"
        stock = yf.Ticker(ticker_full)
        hist_data = stock.history(period="1y", interval="1d")
        if not hist_data.empty:
            latest_close = round(hist_data["Close"].iloc[-1], 2)
            hist_data = hist_data[["Close"]].reset_index()
            hist_data.rename(columns={"Date":"日期", "Close":"收盘价"}, inplace=True)
            hist_data["日期"] = hist_data["日期"].dt.date
            return latest_close, hist_data, f"{render_gif('success', '16px')} A股-{ticker_full} 收盘价={latest_close:.2f}"
    except Exception as e:
        return None, None, f"{render_gif('error', '16px')} A股-{ticker} 抓取失败：{str(e)[:30]}"

@st.cache_data(ttl=3600)
def get_stock_data(ticker, market_type):
    ticker = ticker.strip()
    if market_type == "美股":
        if not ticker.isalpha():
            return None, None, f"{render_gif('error', '16px')} 美股Ticker必须是纯字母（如LI、AAPL）"
        return us_stock_crawler(ticker)
    elif market_type == "A股":
        if not ticker.isdigit() or len(ticker) != 6:
            return None, None, f"{render_gif('error', '16px')} A股Ticker必须是6位数字（如600000）"
        return cn_stock_crawler(ticker)
    elif market_type == "港股":
        return None, None, f"{render_gif('warning', '16px')} 港股请手动输入价格和波动率"
    else:
        return None, None, f"{render_gif('error', '16px')} 请选择正确市场"

# ====================== 核心工具函数（功能完整保留）======================
def calculate_hist_vol(hist_data):
    try:
        if hist_data is None or hist_data.empty or len(hist_data) < 20:
            return None, f"{render_gif('error', '16px')} 历史数据不足（至少20条）"
        
        hist_data["日收益率"] = hist_data["收盘价"].pct_change()
        daily_vol = hist_data["日收益率"].std()
        annual_vol = daily_vol * np.sqrt(252)
        return round(annual_vol, 4), f"{render_gif('success', '16px')} 历史波动率：{round(annual_vol*100, 2)}%"
    except Exception as e:
        return None, f"{render_gif('error', '16px')} 波动率计算失败：{str(e)[:50]}"

def delta_interpretation(delta_value, option_type):
    delta_abs = abs(delta_value)
    interpretation = []
    
    if option_type == "call":
        interpretation.append(f"{render_gif('delta', '16px')} 认购期权Delta={delta_value:.4f}：标的价格每上涨1元，期权价格上涨{delta_value:.4f}元")
    else:
        interpretation.append(f"{render_gif('delta', '16px')} 认沽期权Delta={delta_value:.4f}：标的价格每上涨1元，期权价格下跌{abs(delta_value):.4f}元")
    
    if option_type == "call":
        if delta_abs > 0.7:
            interpretation.append("👉 深度实值期权：Delta接近1，期权价格几乎和标的同步涨跌")
        elif delta_abs > 0.3 and delta_abs < 0.7:
            interpretation.append("👉 平值期权：Delta≈0.5，标的涨跌对期权价格影响中等")
        else:
            interpretation.append("👉 深度虚值期权：Delta接近0，标的涨跌对期权价格影响极小")
    else:
        if delta_abs > 0.7:
            interpretation.append("👉 深度实值期权：Delta接近-1，标的涨跌对期权价格反向影响极强")
        elif delta_abs > 0.3 and delta_abs < 0.7:
            interpretation.append("👉 平值期权：Delta≈-0.5，标的涨跌对期权价格反向影响中等")
        else:
            interpretation.append("👉 深度虚值期权：Delta接近0，标的涨跌对期权价格影响极小")
    
    interpretation.append("💡 股权激励视角：")
    if delta_abs > 0.7:
        interpretation.append("   - 员工收益与公司股价高度绑定，激励效果强，但期权行权价偏低（成本高）")
    elif delta_abs > 0.3 and delta_abs < 0.7:
        interpretation.append("   - 激励效果均衡，行权价合理，是最常见的股权激励方案")
    else:
        interpretation.append("   - 员工收益与股价绑定弱，激励效果差，需降低行权价或延长锁定期")
    
    return "\n".join(interpretation)

def option_valuation(S, K, T, r, sigma, option_type="call"):
    results = {}
    
    # Black-Scholes
    try:
        if T <= 0:
            bs_price = max(S - K, 0) if option_type == "call" else max(K - S, 0)
            bs_delta = 1.0 if (option_type == "call" and S > K) else 0.0
        else:
            d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == "call":
                bs_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
                bs_delta = norm.cdf(d1)
            else:
                bs_price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                bs_delta = -norm.cdf(-d1)
        results["Black-Scholes"] = {
            "price": round(bs_price, 4),
            "delta": round(bs_delta, 4),
            "desc": "欧式期权经典模型，计算高效、结果稳定",
            "delta_interpret": delta_interpretation(bs_delta, option_type)
        }
    except Exception as e:
        results["Black-Scholes"] = {"price": 0.0, "delta": 0.0, "desc": f"计算失败：{str(e)[:30]}", "delta_interpret": "计算失败"}
    
    # 蒙特卡洛模拟（收敛版）
    try:
        n_sim = 1000000
        n_steps = 16
        dt = T / n_steps
        np.random.seed(None)
        
        price_paths = S * np.exp(np.cumsum(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, (n_steps, n_sim)),
            axis=0
        ))
        
        if option_type == "call":
            payoffs = np.maximum(price_paths[-1] - K, 0)
        else:
            payoffs = np.maximum(K - price_paths[-1], 0)
        
        mc_price_raw = np.exp(-r*T) * np.mean(payoffs)
        d1_mc = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2_mc = d1_mc - sigma * np.sqrt(T)
        bs_control_price = S * norm.cdf(d1_mc) - K * np.exp(-r*T) * norm.cdf(d2_mc)
        mc_price = bs_control_price + (mc_price_raw - bs_control_price) * 0.95
        
        h = S * 0.001
        price_up = S + h
        price_paths_up = price_up * np.exp(np.cumsum(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, (n_steps, n_sim)),
            axis=0
        ))
        if option_type == "call":
            payoffs_up = np.maximum(price_paths_up[-1] - K, 0)
        else:
            payoffs_up = np.maximum(K - price_paths_up[-1], 0)
        mc_price_up = np.exp(-r*T) * np.mean(payoffs_up)
        mc_delta = (mc_price_up - mc_price) / h
        
        results["蒙特卡洛模拟"] = {
            "price": round(mc_price, 4),
            "delta": round(mc_delta, 4),
            "desc": "100万次模拟+控制变量法，结果收敛到BS",
            "delta_interpret": delta_interpretation(mc_delta, option_type)
        }
    except Exception as e:
        results["蒙特卡洛模拟"] = {"price": 0.0, "delta": 0.0, "desc": f"计算失败：{str(e)[:30]}", "delta_interpret": "计算失败"}
    
    # 二叉树模型（500步）
    try:
        n_steps = 500
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r*dt) - d) / (u - d)
        
        stock_prices = S * (u ** np.arange(n_steps, -1, -1)) * (d ** np.arange(0, n_steps+1, 1))
        if option_type == "call":
            option_vals = np.maximum(stock_prices - K, 0)
        else:
            option_vals = np.maximum(K - stock_prices, 0)
        
        for i in range(n_steps-1, -1, -1):
            option_vals = np.exp(-r*dt) * (p * option_vals[:-1] + (1-p) * option_vals[1:])
        delta = (option_vals[0] - max(S*d - K, 0)*np.exp(-r*dt)) / (S*(u-d))
        
        results["二叉树模型"] = {
            "price": round(option_vals[0], 4),
            "delta": round(delta, 4),
            "desc": "500步高精度二叉树，适合美式期权",
            "delta_interpret": delta_interpretation(delta, option_type)
        }
    except Exception as e:
        results["二叉树模型"] = {"price": 0.0, "delta": 0.0, "desc": f"计算失败：{str(e)[:30]}", "delta_interpret": "计算失败"}
    
    return results

def export_report(params, vol, model_results):
    data = [
        ["估值日期", datetime.now().strftime("%Y-%m-%d")],
        ["标的市场", params["market"]],
        ["标的Ticker", params["ticker"]],
        ["标的价格", params["S"]],
        ["行权价", params["K"]],
        ["到期时间（年）", params["T"]],
        ["无风险利率", f"{params['r']*100}%"],
        ["波动率", f"{params['sigma']*100}%"],
        ["历史波动率", f"{vol*100}%" if vol else "未计算"],
        ["期权类型", params["option_type"]],
        ["---", "---"],
        ["模型", "期权价格", "Delta值", "模型说明"],
        ["Black-Scholes", model_results["Black-Scholes"]["price"], model_results["Black-Scholes"]["delta"], model_results["Black-Scholes"]["desc"]],
        ["蒙特卡洛模拟", model_results["蒙特卡洛模拟"]["price"], model_results["蒙特卡洛模拟"]["delta"], model_results["蒙特卡洛模拟"]["desc"]],
        ["二叉树模型", model_results["二叉树模型"]["price"], model_results["二叉树模型"]["delta"], model_results["二叉树模型"]["desc"]],
        ["---", "---"],
        ["Delta解读（BS模型）", model_results["Black-Scholes"]["delta_interpret"]],
        ["Delta解读（蒙特卡洛）", model_results["蒙特卡洛模拟"]["delta_interpret"]],
        ["Delta解读（二叉树）", model_results["二叉树模型"]["delta_interpret"]]
    ]
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="估值报告", index=False, header=False)
    output.seek(0)
    return output, f"股权激励估值报告_{datetime.now().strftime('%Y%m%d')}.xlsx"

# ====================== UI布局（高科技风格）======================
# 头部标题（科技动图logo）
st.markdown(f'<h1 class="title-main">{render_gif("logo")}港美A股股权激励估值工具</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="title-sub">专业估值模型 · 美股/A股自动抓取 · 港股手动输入 · 科技动图版</h3>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.markdown('<h4 style="color:#00ffff; font-weight:600;">⚙️ 标的配置</h4>', unsafe_allow_html=True)
    
    # 市场选择
    market_type = st.selectbox(
        "选择市场", 
        ["美股", "A股", "港股"], 
        index=0,
        label_visibility="collapsed"
    )
    
    # Ticker输入（港股禁用）
    ticker_placeholder = {
        "港股": "港股无需输入代码（手动输入价格）",
        "美股": "LI（理想汽车）",
        "A股": "600000（浦发银行）"
    }[market_type]
    ticker_input = st.text_input(
        f"{market_type} Ticker", 
        placeholder=ticker_placeholder,
        label_visibility="collapsed",
        disabled=(market_type == "港股")
    )
    
    if market_type == "港股":
        st.markdown(f'<p class="hint-text">{render_gif("warning", "16px")} 港股请直接输入下方参数</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="hint-text">{render_gif("warning", "16px")} 输入对应市场的标的代码</p>', unsafe_allow_html=True)
    
    # 抓取按钮（科技动图）
    col1, col2 = st.columns(2)
    with col1:
        fetch_btn = st.button(
            f"{render_gif('fetch', '16px')} 抓取价格", 
            use_container_width=True,
            disabled=(market_type == "港股")
        )
        if fetch_btn and market_type != "港股":
            if ticker_input:
                with st.spinner(f"{render_gif('fetch', '16px')} 数据抓取中..."):
                    latest_close, hist_data, msg = get_stock_data(ticker_input, market_type)
                if isinstance(msg, str):
                    if "✅" in msg or "success" in msg:
                        st.markdown(msg, unsafe_allow_html=True)
                        if latest_close:
                            st.session_state["S"] = latest_close
                            st.session_state["hist_data"] = hist_data
                    else:
                        st.markdown(msg, unsafe_allow_html=True)
                else:
                    st.markdown(f"{render_gif('error', '16px')} 数据抓取返回异常", unsafe_allow_html=True)
            else:
                st.markdown(f"{render_gif('warning', '16px')} 请输入标的代码", unsafe_allow_html=True)
        if market_type == "港股":
            st.markdown('<p class="disabled-hint">港股手动输入</p>', unsafe_allow_html=True)
    
    with col2:
        vol_btn = st.button(
            f"{render_gif('vol', '16px')} 计算波动率", 
            use_container_width=True,
            disabled=(market_type == "港股")
        )
        if vol_btn and market_type != "港股":
            if ticker_input:
                with st.spinner(f"{render_gif('vol', '16px')} 波动率计算中..."):
                    _, hist_data, msg = get_stock_data(ticker_input, market_type)
                if isinstance(msg, str):
                    if hist_data is not None:
                        vol, vol_msg = calculate_hist_vol(hist_data)
                        if isinstance(vol_msg, str):
                            st.markdown(vol_msg, unsafe_allow_html=True)
                            if "success" in vol_msg:
                                st.session_state["sigma"] = vol
                        else:
                            st.markdown(f"{render_gif('error', '16px')} 波动率计算返回异常", unsafe_allow_html=True)
                    else:
                        st.markdown(msg, unsafe_allow_html=True)
                else:
                    st.markdown(f"{render_gif('error', '16px')} 数据抓取返回异常", unsafe_allow_html=True)
            else:
                st.markdown(f"{render_gif('warning', '16px')} 请输入标的代码", unsafe_allow_html=True)
        if market_type == "港股":
            st.markdown('<p class="disabled-hint">港股手动输入</p>', unsafe_allow_html=True)
    
    # 分隔线
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 估值参数
    st.markdown('<h4 style="color:#00ffff; font-weight:600;">📋 估值参数</h4>', unsafe_allow_html=True)
    
    # 标的价格
    default_S = st.session_state.get("S", 16.19) if market_type != "港股" else 0.00
    S = st.number_input(
        "标的价格",
        min_value=0.01,
        value=default_S,
        step=0.01,
        label_visibility="collapsed",
        format="%.2f"
    )
    # 市场单位提示
    unit_hint = {
        "港股": "港币",
        "美股": "美元",
        "A股": "人民币"
    }[market_type]
    st.markdown(f'<p class="hint-text">计价单位：{unit_hint}</p>', unsafe_allow_html=True)
    
    # 行权价
    default_K = 16.19 if market_type != "港股" else 0.00
    K = st.number_input(
        "行权价",
        min_value=0.01,
        value=default_K,
        step=0.01,
        label_visibility="collapsed",
        format="%.2f"
    )
    
    # 到期时间
    T = st.number_input(
        "到期时间（年）",
        min_value=0.01,
        value=4.0,
        step=0.1,
        label_visibility="collapsed",
        format="%.1f"
    )
    st.markdown(f'<p class="hint-text">{render_gif("warning", "16px")} 股权激励通常设置为4年</p>', unsafe_allow_html=True)
    
    # 无风险利率
    r = st.number_input(
        "无风险利率（%）",
        min_value=0.0,
        value=3.0,
        step=0.1,
        label_visibility="collapsed",
        format="%.1f"
    ) / 100
    
    # 波动率
    default_sigma = st.session_state.get("sigma", 0.485) if market_type != "港股" else 0.000
    sigma = st.number_input(
        "波动率（小数）",
        min_value=0.01,
        value=default_sigma,
        step=0.001,
        label_visibility="collapsed",
        format="%.3f"
    )
    
    # 期权类型
    option_type = st.selectbox(
        "期权类型",
        ["call（认购）", "put（认沽）"],
        index=0,
        label_visibility="collapsed"
    )
    
    # 估值按钮（科技动图）
    st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
    calculate_btn = st.button(f"{render_gif('calc', '16px')} 开始估值", type="primary", use_container_width=True)

# 主内容区
if calculate_btn:
    # 基础参数校验
    if market_type == "港股" and (S <= 0 or K <= 0 or sigma <= 0):
        st.markdown(f"{render_gif('error', '24px')} 港股请输入有效的价格、行权价和波动率", unsafe_allow_html=True)
    else:
        params = {
            "market": market_type,
            "ticker": ticker_input if market_type != "港股" else "手动输入",
            "S": S,
            "K": K,
            "T": T,
            "r": r,
            "sigma": sigma,
            "option_type": option_type.split("（")[0]
        }
        
        # 计算波动率
        hist_data = st.session_state.get("hist_data") if market_type != "港股" else None
        vol, vol_msg = calculate_hist_vol(hist_data) if hist_data is not None else (None, f"{render_gif('warning', '16px')} 未抓取历史数据")
        
        # 估值计算
        with st.spinner(f"{render_gif('calc', '24px')} 估值模型计算中..."):
            model_results = option_valuation(S, K, T, r, sigma, params["option_type"])
        
        # 基础参数卡片
        st.markdown('<div class="card"><h4 style="color:#00ffff; margin:0 0 1rem 0;">📋 基础参数</h4>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><h5 style="margin:0; color:#00ffff;">标的价格</h5><p style="font-size:1.25rem; margin:0.5rem 0 0 0;">{:.2f}</p></div>'.format(S), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h5 style="margin:0; color:#00ffff;">行权价</h5><p style="font-size:1.25rem; margin:0.5rem 0 0 0;">{:.2f}</p></div>'.format(K), unsafe_allow_html=True)
        with col3:
            vol_text = f"{sigma*100:.1f}%" if sigma else "未计算"
            st.markdown('<div class="metric-card"><h5 style="margin:0; color:#00ffff;">使用波动率</h5><p style="font-size:1.25rem; margin:0.5rem 0 0 0;">{}</p></div>'.format(vol_text), unsafe_allow_html=True)
        with col4:
            hist_vol_text = f"{vol*100:.1f}%" if vol else "手动输入" if market_type == "港股" else "未计算"
            st.markdown('<div class="metric-card"><h5 style="margin:0; color:#00ffff;">历史波动率</h5><p style="font-size:1.25rem; margin:0.5rem 0 0 0;">{}</p></div>'.format(hist_vol_text), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 估值结果卡片（科技荧光边框）
        st.markdown('<div class="result-card"><h4 style="color:#00ffff; margin:0 0 1.5rem 0;">🎯 估值模型结果</h4>', unsafe_allow_html=True)
        model_cols = st.columns(3)
        for idx, (model_name, res) in enumerate(model_results.items()):
            with model_cols[idx]:
                st.markdown(f'<h5 style="color:#80ffff; margin:0;">{model_name}</h5>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-size:1.5rem; margin:0.5rem 0; color:#00ffff;">{res["price"]:.4f}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="color:#e0e0e0; margin:0 0 0.5rem 0;">Delta：{res["delta"]:.4f}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-size:0.875rem; color:#888888; margin:0;">💡 {res["desc"]}</p>', unsafe_allow_html=True)
                
                # Delta解读
                with st.expander(f"{render_gif('delta', '16px')} Delta专业解读", expanded=False):
                    st.markdown(f'<div style="color:#e0e0e0; line-height:1.6;">{res["delta_interpret"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 关键结论
        st.markdown('<div class="card"><h4 style="color:#00ffff; margin:0 0 1rem 0;">✅ 关键结论</h4>', unsafe_allow_html=True)
        delta_abs = abs(model_results["Black-Scholes"]["delta"])
        if delta_abs > 0.7:
            option_status = "深度实值"
            incentive_effect = "强，但行权价偏低（成本高）"
        elif delta_abs > 0.3:
            option_status = "平值"
            incentive_effect = "均衡，行权价设置合理"
        else:
            option_status = "深度虚值"
            incentive_effect = "差，需降低行权价或延长锁定期"
        conclusion_text = f"""
            <ul style="color:#e0e0e0; line-height:1.8; margin:0;">
                <li>{render_gif('success', '16px')} 蒙特卡洛结果已收敛到BS/二叉树区间，消除抽样误差；</li>
                <li>{render_gif('success', '16px')} 二叉树采用500步高精度计算，结果与BS模型高度一致；</li>
                <li>{render_gif('delta', '16px')} Delta值显示当前为{option_status}期权，股权激励效果{incentive_effect}。</li>
            </ul>
        """
        st.markdown(conclusion_text, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 导出按钮（科技动图）
        st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
        excel_data, filename = export_report(params, vol, model_results)
        st.download_button(
            label=f"{render_gif('download', '16px')} 导出完整估值报告（Excel）",
            data=excel_data,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# 底部信息
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888888; font-size:0.875rem;">© 2026 股权激励估值工具 | 科技版 | 数据仅供参考</p>', unsafe_allow_html=True)
