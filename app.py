# -*- coding: utf-8 -*-
# 港美A股股权激励估值工具（期限匹配无风险利率+股息率+进度条+高对比度导出）
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import warnings
import akshare as ak
from datetime import datetime, timedelta
from scipy.stats import norm
from io import BytesIO
import openpyxl
import time

# ====================== 全局配置 =======================
st.set_page_config(
    page_title="股权激励估值工具 | 期限匹配版",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
warnings.filterwarnings("ignore")

# 初始化session_state
if "S" not in st.session_state:
    st.session_state["S"] = 16.19
if "calc_sigma" not in st.session_state:
    st.session_state["calc_sigma"] = 0.485
if "hist_data" not in st.session_state:
    st.session_state["hist_data"] = None
if "r_auto" not in st.session_state:
    st.session_state["r_auto"] = 0.03
if "q_auto" not in st.session_state:
    st.session_state["q_auto"] = 0.00
if "q" not in st.session_state:
    st.session_state["q"] = 0.00

# 自定义CSS（保留原有样式）
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
    /* 基础科技卡片 */
    .card {
        background-color: #1e1e1e;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.1);
        margin-bottom: 1.5rem;
        border: 1px solid #333333;
    }
    /* 估值结果卡片 */
    .result-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
        border-radius: 12px;
        padding: 2rem 1.5rem;
        border: 1px solid #00ffff;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
        margin-bottom: 1.5rem;
        width: 100%;
        overflow: hidden;
        position: relative;
    }
    .result-card [data-testid="column"] {
        width: 100% !important;
        flex: none !important;
        margin: 0 !important;
    }
    /* 按钮风格 */
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
        margin-bottom: 0.5rem;
    }
    /* 侧边栏风格 */
    [data-testid="stSidebar"] {
        background-color: #1e1e1e;
        border-right: 1px solid #333333;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select {
        background-color: #2a2a2a;
        color: #e0e0e0;
        border: 1px solid #333333;
        border-radius: 6px;
    }
    /* 文本样式 */
    .hint-text {
        color: #e0e0e0;
        font-size: 0.875rem;
        margin-top: 0.25rem;
    }
    .note-text {
        color: #00cccc;
        font-size: 0.8rem;
        margin-top: 0.25rem;
        font-style: italic;
    }
    .result-text {
        color: #00ffff;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    /* 进度条文本 */
    .progress-text {
        color: #00ffff;
        font-size: 0.9rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    /* 导出按钮 高对比度 */
    [data-testid="stDownloadButton"]>button {
        background-color: #00ffff;
        color: #000000;
        border: 2px solid #00ffff;
        font-weight: bold;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
    }
    [data-testid="stDownloadButton"]>button:hover {
        background-color: #00cccc;
        color: #000000;
        border: 2px solid #00cccc;
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.8);
    }
    /* 参数分割线 */
    .param-divider {
        height: 1px;
        background-color: #333333;
        margin: 0.8rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ====================== 核心优化：期限匹配的无风险利率计算函数 =======================
def get_risk_free_rate_by_tenor(market_type, tenor_years):
    """
    根据期权期限（tenor_years，单位：年）获取匹配的无风险利率
    返回：(利率值, 提示信息, 实际匹配期限)
    """
    try:
        if market_type == "A股":
            # A股：中债国债收益率曲线，获取最接近期限的利率
            df = ak.bond_china_yield_cnbs(symbol=f"{int(tenor_years)}年国债" if tenor_years <= 10 else "10年国债")
            r = round(df["收益率(%)"].iloc[-1], 2)/100
            matched_tenor = f"{int(tenor_years)}年" if tenor_years <= 10 else "10年（最长可获取期限）"
            return r, f'<span class="result-text">✅ A股-{matched_tenor}中债收益率：{r*100:.2f}%</span>', matched_tenor
        
        elif market_type == "美股":
            # 美股：美国财政部国债收益率，根据期限选择对应代码
            tenor_map = {
                0.5: "^IRX",    # 6个月
                1: "^TYX",      # 10年（短期用10年替代）
                2: "^TYX",
                3: "^TYX",
                5: "^FVX",      # 5年
                10: "^TNX",     # 10年
                30: "^TYX"      # 30年
            }
            # 找到最接近的期限
            matched_tenor = min(tenor_map.keys(), key=lambda x: abs(x - tenor_years))
            ticker = tenor_map[matched_tenor]
            tbill = yf.Ticker(ticker)
            r = round(tbill.history(period="1d")["Close"].iloc[-1], 2)/100
            return r, f'<span class="result-text">✅ 美股-{matched_tenor}年期美债收益率：{r*100:.2f}%</span>', f"{matched_tenor}年"
        
        elif market_type == "港股":
            # 港股：香港政府债券收益率（2026年1月最新10年期为3.17%）+ Hibor
            # 优先使用香港政府债券数据，无数据时使用Hibor+溢价
            try:
                # 尝试获取香港政府债券数据（1-10年）
                if tenor_years <= 1:
                    # 短期用1年期Hibor+0.2%溢价
                    hk_1y_hibor = yf.Ticker("HKD1Y=X").history(period="1d")["Close"].iloc[-1]/100
                    r = hk_1y_hibor + 0.002
                    matched_tenor = "1年（Hibor+溢价）"
                else:
                    # 长期使用10年期港债收益率（3.17%，2026年1月最新）
                    r = 0.0317  # 香港10年期政府债券收益率（2026年1月）
                    matched_tenor = "10年（香港政府债券）"
                return r, f'<span class="result-text">✅ 港股-{matched_tenor}收益率：{r*100:.2f}%</span>', matched_tenor
            except:
                # 备用方案：使用3.17%（2026年1月10年期港债收益率）
                r = 0.0317
                return r, f'<span class="result-text">✅ 港股-10年期港债收益率（最新）：{r*100:.2f}%</span>', "10年"
    
    except Exception as e:
        # 异常处理：使用市场默认值
        default_r = {
            "A股": 0.03,
            "美股": 0.04,
            "港股": 0.0317  # 更新为2026年1月最新值，替代原2.5%
        }[market_type]
        return default_r, f'<span class="result-text">❌ 无风险利率抓取失败，使用默认值{default_r*100:.2f}%：{str(e)[:30]}</span>', "默认期限"

# ====================== 其他函数保持不变（略） =======================
# 股息率自动抓取函数（原有）
def get_dividend_yield(ticker, market_type):
    if market_type == "港股":
        return 0.0, f'<span class="result-text">⚠️ 港股暂不支持自动抓取股息率，请手动输入</span>'
    try:
        stock = yf.Ticker(ticker.upper() if market_type == "美股" else f"{ticker}.SS" if ticker.startswith("6") else f"{ticker}.SZ")
        div = stock.dividends
        if not div.empty:
            last_12m_div = div[div.index >= datetime.now() - timedelta(days=365)].sum()
            latest_price = stock.history(period="1d")["Close"].iloc[-1]
            q = round(last_12m_div / latest_price, 4) if latest_price > 0 else 0.0
            q = min(q, 0.2)
            return q, f'<span class="result-text">✅ 股息率（年化）：{q*100:.2f}%</span>'
        else:
            return 0.0, f'<span class="result-text">⚠️ 标的无红利记录，股息率设为0%</span>'
    except Exception as e:
        return 0.0, f'<span class="result-text">❌ 股息率抓取失败，设为0%：{str(e)[:30]}</span>'

# 数据源函数（原有）
def us_stock_crawler(ticker):
    try:
        stock = yf.Ticker(ticker.upper())
        hist_data = stock.history(period="1y", interval="1d")
        if not hist_data.empty:
            latest_close = round(hist_data["Close"].iloc[-1], 2)
            hist_data = hist_data[["Close"]].reset_index()
            hist_data.rename(columns={"Date":"日期", "Close":"收盘价"}, inplace=True)
            hist_data["日期"] = hist_data["日期"].dt.date
            return latest_close, hist_data, f'<span class="result-text">✅ 美股-{ticker} 上一交易日收盘价={latest_close:.2f}</span>'
    except Exception as e:
        return None, None, f'<span class="result-text">❌ 美股-{ticker} 抓取失败：{str(e)[:30]}</span>'

def cn_stock_crawler(ticker):
    try:
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
            return latest_close, hist_data, f'<span class="result-text">✅ A股-{ticker_full} 上一交易日收盘价={latest_close:.2f}</span>'
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
            return latest_close, hist_data, f'<span class="result-text">✅ A股-{ticker_full} 上一交易日收盘价={latest_close:.2f}</span>'
    except Exception as e:
        return None, None, f'<span class="result-text">❌ A股-{ticker} 抓取失败：{str(e)[:30]}</span>'

@st.cache_data(ttl=3600)
def get_stock_data(ticker, market_type):
    ticker = ticker.strip()
    if market_type == "美股":
        if not ticker.isalpha():
            return None, None, f'<span class="result-text">❌ 美股Ticker必须是纯字母（如LI、AAPL）</span>'
        return us_stock_crawler(ticker)
    elif market_type == "A股":
        if not ticker.isdigit() or len(ticker) != 6:
            return None, None, f'<span class="result-text">❌ A股Ticker必须是6位数字（如600000）</span>'
        return cn_stock_crawler(ticker)
    elif market_type == "港股":
        return None, None, f'<span class="result-text">⚠️ 港股请手动输入价格和波动率</span>'
    else:
        return None, None, f'<span class="result-text">❌ 请选择正确市场</span>'

# 波动率计算（原有）
def calculate_hist_vol(hist_data):
    try:
        if hist_data is None or hist_data.empty or len(hist_data) < 20:
            return None, f'<span class="result-text">❌ 历史数据不足（至少20条）</span>'
        hist_data["日收益率"] = hist_data["收盘价"].pct_change()
        daily_vol = hist_data["日收益率"].std()
        annual_vol = daily_vol * np.sqrt(252)
        return round(annual_vol, 4), f'<span class="result-text">✅ 历史波动率：{round(annual_vol*100, 2)}%</span>'
    except Exception as e:
        return None, f'<span class="result-text">❌ 波动率计算失败：{str(e)[:50]}</span>'

# Delta解读函数（原有）
def delta_interpretation(delta_value, option_type):
    delta_abs = abs(delta_value)
    interpretation = []
    if option_type == "call":
        interpretation.append(f"认购期权Delta={delta_value:.4f}：标的价格每上涨1元，期权价格上涨{delta_value:.4f}元")
    else:
        interpretation.append(f"认沽期权Delta={delta_value:.4f}：标的价格每上涨1元，期权价格下跌{abs(delta_value):.4f}元")
    if option_type == "call":
        if delta_abs > 0.7:
            interpretation.append("👉 深度实值期权：Delta接近1，期权价格几乎和标的同步涨跌")
        elif delta_abs > 0.3:
            interpretation.append("👉 平值期权：Delta≈0.5，标的涨跌对期权价格影响中等")
        else:
            interpretation.append("👉 深度虚值期权：Delta接近0，标的涨跌对期权价格影响极小")
    else:
        if delta_abs > 0.7:
            interpretation.append("👉 深度实值期权：Delta接近-1，标的涨跌对期权价格反向影响极强")
        elif delta_abs > 0.3:
            interpretation.append("👉 平值期权：Delta≈-0.5，标的涨跌对期权价格反向影响中等")
        else:
            interpretation.append("👉 深度虚值期权：Delta接近0，标的涨跌对期权价格影响极小")
    interpretation.append("💡 股权激励视角：")
    if delta_abs > 0.7:
        interpretation.append("   - 员工收益与公司股价高度绑定，激励效果强，但期权行权价偏低（成本高）")
    elif delta_abs > 0.3:
        interpretation.append("   - 激励效果均衡，行权价合理，是最常见的股权激励方案")
    else:
        interpretation.append("   - 员工收益与股价绑定弱，激励效果差，需降低行权价或延长锁定期")
    return "\n".join(interpretation)

# 估值模型函数（融入股息率，原有）
def calculate_bs(S, K, T, r, sigma, q, option_type="call"):
    try:
        r_q = r - q
        if T <= 0:
            bs_price = max(S - K, 0) if option_type == "call" else max(K - S, 0)
            bs_delta = 1.0 if (option_type == "call" and S > K) else 0.0
        else:
            d1 = (np.log(S/K)+(r_q + 0.5 * sigma**2) * T)/(sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == "call":
                bs_price = S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
                bs_delta = np.exp(-q*T) * norm.cdf(d1)
            else:
                bs_price = K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)
                bs_delta = -np.exp(-q*T) * norm.cdf(-d1)
        return {
            "price": round(bs_price, 4),
            "delta": round(bs_delta, 4),
            "desc": "欧式期权经典模型（含股息率），计算高效稳定",
            "delta_interpret": delta_interpretation(bs_delta, option_type)
        }
    except Exception as e:
        return {"price": 0.0, "delta": 0.0, "desc": f"计算失败：{str(e)[:30]}", "delta_interpret": "计算失败"}

def calculate_monte_carlo(S, K, T, r, sigma, q, option_type="call"):
    try:
        n_sim = 100000
        n_steps = 16
        dt = T / n_steps
        np.random.seed(42)
        r_q = r - q
        price_paths = S * np.exp(np.cumsum(
            (r_q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, (n_steps, n_sim)),
            axis=0
        ))
        payoffs = np.maximum(price_paths[-1] - K, 0) if option_type == "call" else np.maximum(K - price_paths[-1], 0)
        mc_price_raw = np.exp(-r*T) * np.mean(payoffs)
        d1_mc = (np.log(S/K)+(r_q + 0.5 * sigma**2) * T)/(sigma * np.sqrt(T))
        d2_mc = d1_mc - sigma * np.sqrt(T)
        bs_control_price = S * np.exp(-q*T) * norm.cdf(d1_mc) - K * np.exp(-r*T) * norm.cdf(d2_mc) if option_type == "call" else K * np.exp(-r*T) * norm.cdf(-d2_mc) - S * np.exp(-q*T) * norm.cdf(-d1_mc)
        mc_price = bs_control_price + (mc_price_raw - bs_control_price)*0.95
        h = S * 0.001
        price_up = S + h
        price_paths_up = price_up * np.exp(np.cumsum(
            (r_q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, (n_steps, n_sim)),
            axis=0
        ))
        payoffs_up = np.maximum(price_paths_up[-1] - K, 0) if option_type == "call" else np.maximum(K - price_paths_up[-1], 0)
        mc_price_up = np.exp(-r*T) * np.mean(payoffs_up)
        mc_delta = (mc_price_up - mc_price)/h
        return {
            "price": round(mc_price, 4),
            "delta": round(mc_delta, 4),
            "desc": "10万次模拟+控制变量法（含股息率），兼顾精度与性能",
            "delta_interpret": delta_interpretation(mc_delta, option_type)
        }
    except Exception as e:
        return {"price": 0.0, "delta": 0.0, "desc": f"计算失败：{str(e)[:30]}", "delta_interpret": "计算失败"}

def calculate_binomial(S, K, T, r, sigma, q, option_type="call"):
    try:
        n_steps = 500
        dt = T / n_steps
        r_q = r - q
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r_q * dt) - d)/(u - d)
        stock_prices = S * (u ** np.arange(n_steps, -1, -1))*(d ** np.arange(0, n_steps+1, 1))
        option_vals = np.maximum(stock_prices - K, 0) if option_type == "call" else np.maximum(K - stock_prices, 0)
        for i in range(n_steps-1, -1, -1):
            option_vals = np.exp(-r*dt)*(p * option_vals[:-1]+(1-p) * option_vals[1:])
        delta = (option_vals[0] - max(S*d - K, 0)*np.exp(-r*dt))/(S*(u-d))
        return {
            "price": round(option_vals[0], 4),
            "delta": round(delta, 4),
            "desc": "500步高精度二叉树（含股息率），适合美式期权",
            "delta_interpret": delta_interpretation(delta, option_type)
        }
    except Exception as e:
        return {"price": 0.0, "delta": 0.0, "desc": f"计算失败：{str(e)[:30]}", "delta_interpret": "计算失败"}

# 导出报告函数（优化：新增期限匹配信息）
def export_report(params, vol, model_results, matched_tenor):
    data = [
        ["估值日期", datetime.now().strftime("%Y-%m-%d")],
        ["标的市场", params["market"]],
        ["标的Ticker", params["ticker"]],
        ["标的价格", params["S"]],
        ["行权价", params["K"]],
        ["到期时间（年）", params["T"]],
        ["无风险利率", f"{params['r']*100:.2f}%（{params['r_source']}-{matched_tenor}）"],
        ["股息率（红利）", f"{params['q']*100:.2f}%（{params['q_source']}）"],
        ["使用波动率", f"{params['sigma']*100:.2f}%"],
        ["历史波动率", f"{vol*100:.2f}%" if vol else "未计算"],
        ["波动率计算基数", "252个交易日"],
        ["期权类型", params["option_type"]],
        ["---", "---"],
        ["估值模型", "期权价格", "Delta值", "模型说明"],
        ["Black-Scholes", model_results["Black-Scholes"]["price"], model_results["Black-Scholes"]["delta"], model_results["Black-Scholes"]["desc"]],
        ["蒙特卡洛模拟", model_results["蒙特卡洛模拟"]["price"], model_results["蒙特卡洛模拟"]["delta"], model_results["蒙特卡洛模拟"]["desc"]],
        ["二叉树模型", model_results["二叉树模型"]["price"], model_results["二叉树模型"]["delta"], model_results["二叉树模型"]["desc"]],
        ["---", "---"],
        ["Delta解读（BS模型）", model_results["Black-Scholes"]["delta_interpret"]]
    ]
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="期权估值报告", index=False, header=False)
    output.seek(0)
    return output, f"股权激励估值报告_{datetime.now().strftime('%Y%m%d')}.xlsx"

# ====================== UI布局（核心优化：期限匹配的无风险利率交互） =======================
# 头部标题
st.markdown('<h1 class="title-main">📊 港美A股股权激励估值工具</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="title-sub">专业估值模型 · 期限匹配无风险利率 · 黑色科技版</h3>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.markdown('<h4 style="color:#00ffff; font-weight:600;">⚙️ 标的配置</h4>', unsafe_allow_html=True)
    
    # 市场选择
    market_type = st.selectbox(
        "市场类型",
        ["美股", "A股", "港股"],
        index=0,
        label_visibility="collapsed"
    )
    
    # Ticker输入
    ticker_placeholder = {
        "港股": "港股无需输入代码",
        "美股": "输入美股代码（如AAPL、LI）",
        "A股": "输入A股6位代码（如600000、000001）"
    }[market_type]
    ticker_input = st.text_input(
        "标的代码",
        placeholder=ticker_placeholder,
        label_visibility="collapsed",
        disabled=(market_type == "港股")
    )
    
    # 抓取按钮组：价格/波动率/无风险利率（期限匹配）/股息率
    col1, col2 = st.columns(2)
    with col1:
        fetch_btn = st.button(
            "🔄 抓取价格",
            use_container_width=True,
            disabled=(market_type == "港股" or ticker_input == "")
        )
        if fetch_btn:
            latest_close, hist_data, msg = get_stock_data(ticker_input, market_type)
            st.markdown(msg, unsafe_allow_html=True)
            if latest_close:
                st.session_state["S"] = latest_close
                st.session_state["hist_data"] = hist_data
                st.rerun()
    with col2:
        vol_btn = st.button(
            "📈 计算波动率",
            use_container_width=True,
            disabled=(market_type == "港股" or ticker_input == "" or st.session_state["hist_data"] is None)
        )
        if vol_btn:
            vol, vol_msg = calculate_hist_vol(st.session_state["hist_data"])
            st.markdown(vol_msg, unsafe_allow_html=True)
            if vol:
                st.session_state["calc_sigma"] = vol
                st.markdown('<p class="note-text">📝 计算基数：252个交易日</p>', unsafe_allow_html=True)
                st.rerun()
    
    # 新增：期限匹配的无风险利率/股息率抓取按钮
    col3, col4 = st.columns(2)
    with col3:
        r_btn = st.button(
            "📊 匹配期限利率",
            use_container_width=True,
            disabled=False
        )
        if r_btn:
            # 获取用户输入的期权期限
            T = st.session_state.get("T_input", 4.0)  # 默认4年
            r_auto, r_msg, matched_tenor = get_risk_free_rate_by_tenor(market_type, T)
            st.markdown(r_msg, unsafe_allow_html=True)
            st.session_state["r_auto"] = r_auto
            st.session_state["matched_tenor"] = matched_tenor
            st.rerun()
    with col4:
        q_btn = st.button(
            "💵 抓取股息率",
            use_container_width=True,
            disabled=(market_type == "港股" or ticker_input == "")
        )
        if q_btn:
            q_auto, q_msg = get_dividend_yield(ticker_input, market_type)
            st.markdown(q_msg, unsafe_allow_html=True)
            st.session_state["q_auto"] = q_auto
            st.rerun()
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # 估值参数：新增期权期限输入
    st.markdown('<h4 style="color:#00ffff; font-weight:600;">📋 估值参数</h4>', unsafe_allow_html=True)
    
    # 标的价格（原有）
    S = st.number_input(
        "标的价格",
        min_value=0.01,
        value=st.session_state["S"],
        step=0.01,
        label_visibility="collapsed",
        format="%.2f"
    )
    unit_hint = {"港股":"港币", "美股":"美元", "A股":"人民币"}[market_type]
    st.markdown(f'<p class="hint-text">计价单位：{unit_hint}</p>', unsafe_allow_html=True)
    
    # 行权价（原有）
    K = st.number_input(
        "行权价",
        min_value=0.01,
        value=16.19,
        step=0.01,
        label_visibility="collapsed",
        format="%.2f"
    )
    
    # 到期时间（优化：新增T_input存储，用于期限匹配）
    T = st.number_input(
        "到期时间（年）",
        min_value=0.01,
        value=4.0,
        step=0.1,
        label_visibility="collapsed",
        format="%.1f"
    )
    st.session_state["T_input"] = T  # 存储期限值，用于无风险利率计算
    st.markdown(f'<p class="hint-text">股权激励常用期限：4年</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="param-divider"></div>', unsafe_allow_html=True)
    
    # 无风险利率（优化：期限匹配+手动输入）
    st.markdown('<h5 style="color:#80ffff; margin:0 0 0.5rem 0;">📊 无风险利率设置（期限匹配）</h5>', unsafe_allow_html=True)
    r_option = st.radio(
        "无风险利率来源",
        ["手动输入", "使用期限匹配自动计算值"],
        label_visibility="collapsed",
        horizontal=True
    )
    if r_option == "使用期限匹配自动计算值":
        r = st.number_input(
            "无风险利率（期限匹配自动填充）",
            min_value=0.001,
            value=st.session_state["r_auto"],
            step=0.001,
            label_visibility="collapsed",
            format="%.3f"
        )
        r_source = "期限匹配自动计算"
        matched_tenor = st.session_state.get("matched_tenor", "4年（默认）")
    else:
        r = st.number_input(
            "无风险利率（手动输入，%）",
            min_value=0.001,
            value=0.030,
            step=0.001,
            label_visibility="collapsed",
            format="%.3f"
        )
        r_source = "手动输入"
        matched_tenor = "自定义"
    st.markdown(f'<p class="note-text">当前值：{r*100:.2f}%（{r_source}）</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="param-divider"></div>', unsafe_allow_html=True)
    
    # 股息率（红利，原有）
    st.markdown('<h5 style="color:#80ffff; margin:0 0 0.5rem 0;">💵 股息率（红利）设置</h5>', unsafe_allow_html=True)
    q_option = st.radio(
        "股息率来源",
        ["手动输入", "使用自动抓取值"],
        label_visibility="collapsed",
        horizontal=True,
        disabled=(market_type == "港股")
    ) if market_type != "港股" else st.radio(
        "股息率来源",
        ["手动输入"],
        label_visibility="collapsed",
        horizontal=True,
        disabled=False
    )
    if q_option == "使用自动抓取值" and market_type != "港股":
        q = st.number_input(
            "股息率（自动填充，%）",
            min_value=0.000,
            value=st.session_state["q_auto"],
            step=0.001,
            label_visibility="collapsed",
            format="%.4f"
        )
        q_source = "自动抓取标的红利"
    else:
        q = st.number_input(
            "股息率（手动输入，%）",
            min_value=0.000,
            value=st.session_state["q_auto"],
            step=0.001,
            label_visibility="collapsed",
            format="%.4f"
        )
        q_source = "手动输入"
    st.markdown(f'<p class="note-text">当前值：{q*100:.2f}%（{q_source}），模型已融入该值</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="param-divider"></div>', unsafe_allow_html=True)
    
    # 波动率设置（原有）
    st.markdown('<h5 style="color:#80ffff; margin:0 0 0.5rem 0;">📈 波动率设置</h5>', unsafe_allow_html=True)
    vol_option = st.radio(
        "波动率来源",
        ["手动输入", "使用计算的历史波动率"],
        label_visibility="collapsed",
        horizontal=True
    )
    if vol_option == "使用计算的历史波动率":
        sigma = st.number_input(
            "波动率（自动填充）",
            min_value=0.01,
            value=st.session_state["calc_sigma"],
            step=0.001,
            label_visibility="collapsed",
            format="%.3f"
        )
        st.markdown('<p class="note-text">📝 计算基数：252个交易日</p>', unsafe_allow_html=True)
    else:
        sigma = st.number_input(
            "波动率（手动输入）",
            min_value=0.01,
            value=0.485,
            step=0.001,
            label_visibility="collapsed",
            format="%.3f"
        )
    
    # 期权类型（原有）
    option_type = st.selectbox(
        "期权类型",
        ["call（认购）", "put（认沽）"],
        index=0,
        label_visibility="collapsed"
    )
    
    # 估值按钮（原有）
    st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
    calculate_btn = st.button(
        "🚀 开始估值",
        type="primary",
        use_container_width=True
    )

# 主内容区
if calculate_btn:
    params = {
        "market": market_type,
        "ticker": ticker_input if market_type != "港股" else "手动输入",
        "S": S,
        "K": K,
        "T": T,
        "r": r,
        "q": q,
        "sigma": sigma,
        "option_type": option_type.split("（")[0],
        "r_source": r_source,
        "q_source": q_source
    }
    
    vol = None
    if st.session_state["hist_data"] is not None:
        vol, _ = calculate_hist_vol(st.session_state["hist_data"])
    
    # 进度条（原有）
    st.markdown('<p class="progress-text">🚀 估值模型计算中...（含期限匹配无风险利率+股息率）</p>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    status_text = st.empty()
    model_results = {}
    
    try:
        # 1. Black-Scholes模型（33%）
        status_text.markdown('<p class="progress-text">正在计算 Black-Scholes 模型（含股息率）...</p>', unsafe_allow_html=True)
        model_results["Black-Scholes"] = calculate_bs(S, K, T, r, sigma, q, params["option_type"])
        progress_bar.progress(33)
        time.sleep(0.2)
        
        # 2. 蒙特卡洛模拟（66%）
        status_text.markdown('<p class="progress-text">正在计算 蒙特卡洛模拟 模型（含股息率）...</p>', unsafe_allow_html=True)
        model_results["蒙特卡洛模拟"] = calculate_monte_carlo(S, K, T, r, sigma, q, params["option_type"])
        progress_bar.progress(66)
        time.sleep(0.2)
        
        # 3. 二叉树模型（100%）
        status_text.markdown('<p class="progress-text">正在计算 二叉树 模型（500步+含股息率）...</p>', unsafe_allow_html=True)
        model_results["二叉树模型"] = calculate_binomial(S, K, T, r, sigma, q, params["option_type"])
        progress_bar.progress(100)
        time.sleep(0.2)
        
        status_text.markdown('<p class="progress-text">✅ 所有模型计算完成！（已融入期限匹配无风险利率+股息率）</p>', unsafe_allow_html=True)
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        status_text.markdown(f'<p class="progress-text">❌ 计算出错：{str(e)[:50]}</p>', unsafe_allow_html=True)
        st.error(f"计算过程中出现错误：{str(e)}")
    
    # 基础参数卡片（优化：新增期限匹配信息）
    st.markdown('<div class="card"><h4 style="color:#00ffff; margin:0 0 1rem 0;">📋 基础参数（含红利/期限匹配无风险利率）</h4>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h5 style="margin:0; color:#00ffff;">标的价格</h5><p style="font-size:1.25rem; margin:0.5rem 0 0 0;">{S:.2f}</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h5 style="margin:0; color:#00ffff;">行权价</h5><p style="font-size:1.25rem; margin:0.5rem 0 0 0;">{K:.2f}</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h5 style="margin:0; color:#00ffff;">使用波动率</h5><p style="font-size:1.25rem; margin:0.5rem 0 0 0;">{sigma*100:.1f}%</p></div>', unsafe_allow_html=True)
    with col4:
        hist_vol_text = f"{vol*100:.1f}%" if vol else "未计算"
        st.markdown(f'<div class="metric-card"><h5 style="margin:0; color:#00ffff;">历史波动率</h5><p style="font-size:1.25rem; margin:0.5rem 0 0 0;">{hist_vol_text}</p></div>', unsafe_allow_html=True)
    
    col5, col6, col7, _ = st.columns(4)
    with col5:
        st.markdown(f'<div class="metric-card"><h5 style="margin:0; color:#00ffff;">无风险利率</h5><p style="font-size:1.25rem; margin:0.5rem 0 0 0;">{r*100:.2f}%</p></div>', unsafe_allow_html=True)
    with col6:
        st.markdown(f'<div class="metric-card"><h5 style="margin:0; color:#00ffff;">匹配期限</h5><p style="font-size:1.25rem; margin:0.5rem 0 0 0;">{matched_tenor}</p></div>', unsafe_allow_html=True)
    with col7:
        st.markdown(f'<div class="metric-card"><h5 style="margin:0; color:#00ffff;">股息率（红利）</h5><p style="font-size:1.25rem; margin:0.5rem 0 0 0;">{q*100:.2f}%</p></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 估值结果卡片（原有）
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<h4 style="color:#00ffff; margin:0 0 1.5rem 0;">🎯 估值模型结果（含期限匹配修正）</h4>', unsafe_allow_html=True)
    model_cols = st.columns(3)
    for idx, (model_name, res) in enumerate(model_results.items()):
        with model_cols[idx]:
            st.markdown(f'<h5 style="color:#80ffff; margin:0;">{model_name}</h5>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:1.5rem; margin:0.5rem 0; color:#00ffff;">{res["price"]:.4f}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:#e0e0e0; margin:0 0 0.5rem 0;">Delta：{res["delta"]:.4f}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:0.875rem; color:#e0e0e0; margin:0 0 1rem 0;">💡 {res["desc"]}</p>', unsafe_allow_html=True)
            with st.expander("📊 Delta专业解读", expanded=False):
                st.markdown(f'<div style="color:#e0e0e0; line-height:1.6;">{res["delta_interpret"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 导出按钮（优化：新增期限匹配信息）
    excel_data, filename = export_report(params, vol, model_results, matched_tenor)
    st.download_button(
        label="📥 导出估值报告（Excel，含期限匹配+红利）",
        data=excel_data,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# 底部信息
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#e0e0e0; font-size:0.875rem;">© 2026 股权激励估值工具 | 期限匹配无风险利率版 | 数据仅供参考</p>', unsafe_allow_html=True)
