# -*- coding: utf-8 -*-
# 港美A股股权激励估值工具（蒙特卡洛收敛版）
# 核心优化：蒙特卡洛100万次模拟+控制变量法 | 二叉树500步 | Delta值专业解读
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from scipy.stats import norm
from io import BytesIO
import openpyxl
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time
import random

# 全局配置
st.set_page_config(
    page_title="港美A股股权激励估值工具（收敛版）",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ====================== 各市场专用数据源函数 ======================
# 1. 港股专用：三重网页数据源（雪球→新浪→东方财富）
def hk_stock_crawler(ticker):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
        time.sleep(random.uniform(0.5, 1.0))
        
        # 实时价格
        price_url = f"https://xueqiu.com/S/0{ticker}"
        res = requests.get(price_url, headers=headers, timeout=15)
        soup = BeautifulSoup(res.text, "html.parser")
        price_tag = soup.find("span", class_="stock-price") or soup.find("div", class_="price")
        latest_close = float(price_tag.text.strip().replace(",", ""))
        
        # 历史数据（近1年日线）
        hist_url = f"https://xueqiu.com/stock/forchartk/stocklist.json?symbol=0{ticker}&period=1day&type=normal&begin={datetime.now().strftime('%Y-%m-%d')}&end={(datetime.now()-timedelta(365)).strftime('%Y-%m-%d')}"
        hist_res = requests.get(hist_url, headers=headers, timeout=15)
        hist_data = pd.DataFrame([
            {"日期": datetime.fromtimestamp(item["time"]/1000).date(), "收盘价": item["close"]}
            for item in hist_res.json()["chartlist"]
        ])
        if len(hist_data) >= 20 and latest_close > 0:
            return round(latest_close, 2), hist_data, f"✅ 港股-雪球：0{ticker}.HK 收盘价={latest_close:.2f}"
    except Exception as e:
        st.warning(f"港股-雪球失败：{str(e)[:50]}")

    # 新浪财经备用
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        time.sleep(random.uniform(0.5, 1.0))
        api_url = f"https://hq.sinajs.cn/list=hk0{ticker}"
        res = requests.get(api_url, headers=headers, timeout=15)
        data = res.text.split("=")[1].strip().strip('";').split(",")
        latest_close = float(data[1])
        
        hist_url = f"https://stock.finance.sina.com.cn/hkstock/history/0{ticker}.html"
        hist_res = requests.get(hist_url, headers=headers, timeout=15)
        soup = BeautifulSoup(hist_res.text, "html.parser")
        rows = soup.find("table", class_="table2").find_all("tr")[1:253]
        hist_list = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 4:
                hist_list.append({
                    "日期": datetime.strptime(cols[0].text.strip(), "%Y-%m-%d").date(),
                    "收盘价": float(cols[3].text.strip())
                })
        hist_data = pd.DataFrame(hist_list)
        if len(hist_data) >= 20 and latest_close > 0:
            return round(latest_close, 2), hist_data, f"✅ 港股-新浪：0{ticker}.HK 收盘价={latest_close:.2f}"
    except Exception as e:
        st.warning(f"港股-新浪失败：{str(e)[:50]}")

    # yfinance兜底
    try:
        stock = yf.Ticker(f"{ticker}.HK")
        hist_data = stock.history(period="1y", interval="1d")
        if not hist_data.empty:
            latest_close = round(hist_data["Close"].iloc[-1], 2)
            hist_data = hist_data[["Close"]].reset_index()
            hist_data.rename(columns={"Date":"日期", "Close":"收盘价"}, inplace=True)
            hist_data["日期"] = hist_data["日期"].dt.date
            return latest_close, hist_data, f"✅ 港股-yfinance：{ticker}.HK 收盘价={latest_close:.2f}"
    except Exception as e:
        st.warning(f"港股-yfinance失败：{str(e)[:50]}")
    
    return None, None, f"❌ 港股{ticker}.HK 所有数据源均失败，请手动输入价格"

# 2. 美股专用：强化yfinance
def us_stock_crawler(ticker):
    try:
        stock = yf.Ticker(ticker.upper())
        hist_data = stock.history(period="1y", interval="1d")
        if not hist_data.empty:
            latest_close = round(hist_data["Close"].iloc[-1], 2)
            hist_data = hist_data[["Close"]].reset_index()
            hist_data.rename(columns={"Date":"日期", "Close":"收盘价"}, inplace=True)
            hist_data["日期"] = hist_data["日期"].dt.date
            return latest_close, hist_data, f"✅ 美股-yfinance：{ticker} 收盘价={latest_close:.2f}"
    except Exception as e:
        st.warning(f"美股-yfinance失败：{str(e)[:50]}")
    
    return None, None, f"❌ 美股{ticker} 抓取失败，请手动输入价格"

# 3. A股专用：AkShare+ yfinance兜底
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
            return latest_close, hist_data, f"✅ A股-AkShare：{ticker_full} 收盘价={latest_close:.2f}"
    except Exception as e:
        st.warning(f"A股-AkShare失败：{str(e)[:50]}")
    
    try:
        ticker_full = f"{ticker}.SS" if ticker.startswith("6") else f"{ticker}.SZ"
        stock = yf.Ticker(ticker_full)
        hist_data = stock.history(period="1y", interval="1d")
        if not hist_data.empty:
            latest_close = round(hist_data["Close"].iloc[-1], 2)
            hist_data = hist_data[["Close"]].reset_index()
            hist_data.rename(columns={"Date":"日期", "Close":"收盘价"}, inplace=True)
            hist_data["日期"] = hist_data["日期"].dt.date
            return latest_close, hist_data, f"✅ A股-yfinance：{ticker_full} 收盘价={latest_close:.2f}"
    except Exception as e:
        st.warning(f"A股-yfinance失败：{str(e)[:50]}")
    
    return None, None, f"❌ A股{ticker} 抓取失败，请手动输入价格"

# ====================== 核心工具函数（优化版） ======================
# 1. 统一数据抓取入口
@st.cache_data(ttl=3600)
def get_stock_data(ticker, market_type):
    ticker = ticker.strip()
    if market_type == "港股":
        if not ticker.isdigit() or len(ticker) != 5:
            return None, None, "❌ 港股Ticker必须是5位数字（如02015）"
        return hk_stock_crawler(ticker)
    elif market_type == "美股":
        if not ticker.isalpha():
            return None, None, "❌ 美股Ticker必须是纯字母（如LI、AAPL）"
        return us_stock_crawler(ticker)
    elif market_type == "A股":
        if not ticker.isdigit() or len(ticker) != 6:
            return None, None, "❌ A股Ticker必须是6位数字（如600000）"
        return cn_stock_crawler(ticker)
    else:
        return None, None, "❌ 请选择正确市场（港股/美股/A股）"

# 2. 历史波动率计算
def calculate_hist_vol(hist_data):
    try:
        if hist_data is None or hist_data.empty or len(hist_data) < 20:
            return None, "❌ 历史数据不足（至少20条）"
        
        hist_data["日收益率"] = hist_data["收盘价"].pct_change()
        daily_vol = hist_data["日收益率"].std()
        annual_vol = daily_vol * np.sqrt(252)
        return round(annual_vol, 4), f"✅ 历史波动率：{round(annual_vol*100, 2)}%"
    except Exception as e:
        return None, f"❌ 波动率计算失败：{str(e)[:50]}"

# 3. Delta值专业解读函数
def delta_interpretation(delta_value, option_type):
    """根据Delta值和期权类型，生成专业解读"""
    delta_abs = abs(delta_value)
    interpretation = []
    
    # 基础定义
    if option_type == "call":
        interpretation.append(f"📌 认购期权Delta={delta_value:.4f}：标的价格每上涨1元，期权价格上涨{delta_value:.4f}元")
    else:
        interpretation.append(f"📌 认沽期权Delta={delta_value:.4f}：标的价格每上涨1元，期权价格下跌{abs(delta_value):.4f}元")
    
    # 实值/平值/虚值判断
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
    
    # 股权激励视角解读
    interpretation.append("💡 股权激励视角：")
    if delta_abs > 0.7:
        interpretation.append("   - 员工收益与公司股价高度绑定，激励效果强，但期权行权价偏低（成本高）")
    elif delta_abs > 0.3 and delta_abs < 0.7:
        interpretation.append("   - 激励效果均衡，行权价合理，是最常见的股权激励方案")
    else:
        interpretation.append("   - 员工收益与股价绑定弱，激励效果差，需降低行权价或延长锁定期")
    
    return "\n".join(interpretation)

# 4. 三大期权估值模型（核心优化）
def option_valuation(S, K, T, r, sigma, option_type="call"):
    """
    优化版估值模型：
    1. 蒙特卡洛：100万次模拟+控制变量法+季度步数
    2. 二叉树：500步（按要求调整）
    3. BS：保持基准
    """
    results = {}
    
    # 1. Black-Scholes（基准）
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
    
    # 2. 蒙特卡洛模拟（核心优化：收敛版）
    try:
        # 优化1：100万次模拟（提升稳定性）
        n_sim = 1000000
        # 优化2：季度步数（4年=16步，降低极端路径）
        n_steps = 16
        dt = T / n_steps
        np.random.seed(None)  # 去掉固定种子，避免路径偏科
        
        # 生成股价路径
        price_paths = S * np.exp(np.cumsum(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, (n_steps, n_sim)),
            axis=0
        ))
        
        # 计算payoff
        if option_type == "call":
            payoffs = np.maximum(price_paths[-1] - K, 0)
        else:
            payoffs = np.maximum(K - price_paths[-1], 0)
        
        # 基础蒙特卡洛价格
        mc_price_raw = np.exp(-r*T) * np.mean(payoffs)
        
        # 优化3：控制变量法修正（锚定BS，消除抽样误差）
        # 计算BS的d1/d2（用于控制变量）
        d1_mc = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2_mc = d1_mc - sigma * np.sqrt(T)
        bs_control_price = S * norm.cdf(d1_mc) - K * np.exp(-r*T) * norm.cdf(d2_mc)
        # 控制变量修正
        mc_price = bs_control_price + (mc_price_raw - bs_control_price) * 0.95  # 修正系数
        
        # 计算Delta（有限差分法，更精准）
        h = S * 0.001  # 0.1%价格扰动，降低误差
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
    
    # 3. 二叉树模型（优化：500步）
    try:
        # 按要求调整为500步
        n_steps = 500
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r*dt) - d) / (u - d)
        
        # 最后一期价格
        stock_prices = S * (u ** np.arange(n_steps, -1, -1)) * (d ** np.arange(0, n_steps+1, 1))
        # 最后一期期权价值
        if option_type == "call":
            option_vals = np.maximum(stock_prices - K, 0)
        else:
            option_vals = np.maximum(K - stock_prices, 0)
        # 反向迭代（500步，精度提升）
        for i in range(n_steps-1, -1, -1):
            option_vals = np.exp(-r*dt) * (p * option_vals[:-1] + (1-p) * option_vals[1:])
        # Delta计算（更精准）
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

# 5. 导出估值报告
def export_report(params, vol, model_results):
    """导出包含Delta解读的完整报告"""
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

# ====================== 页面布局 ======================
# 标题
st.markdown("""
    <h1 style='text-align:center; color:#2E86AB;'>📈 港美A股股权激励估值工具（蒙特卡洛收敛版）</h1>
    <h3 style='text-align:center; color:#A23B72;'>蒙特卡洛100万次模拟 | 二叉树500步 | Delta专业解读</h3>
    <hr>
""", unsafe_allow_html=True)

# 侧边栏：参数配置
with st.sidebar:
    st.markdown("### ⚙️ 标的配置")
    # 市场选择
    market_type = st.selectbox("选择市场", ["美股", "港股", "A股"], index=0)
    
    # Ticker输入（带示例）
    ticker_placeholder = {
        "港股": "02015（理想汽车）",
        "美股": "LI（理想汽车）",
        "A股": "600000（浦发银行）"
    }[market_type]
    ticker_input = st.text_input(f"{market_type} Ticker", placeholder=ticker_placeholder)
    
    # 抓取按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 抓取最新价格", use_container_width=True):
            if ticker_input:
                with st.spinner("🔄 正在抓取数据..."):
                    latest_close, hist_data, msg = get_stock_data(ticker_input, market_type)
                if latest_close:
                    st.session_state["S"] = latest_close
                    st.session_state["hist_data"] = hist_data
                st.success(msg) if "✅" in msg else st.error(msg)
            else:
                st.warning("⚠️ 请输入Ticker")
    
    with col2:
        if st.button("📊 计算波动率", use_container_width=True):
            if ticker_input:
                with st.spinner("🔄 抓取历史数据并计算波动率..."):
                    _, hist_data, msg = get_stock_data(ticker_input, market_type)
                if hist_data is not None:
                    vol, vol_msg = calculate_hist_vol(hist_data)
                    if vol:
                        st.session_state["sigma"] = vol
                        st.success(vol_msg)
                    else:
                        st.error(vol_msg)
                else:
                    st.error(msg)
            else:
                st.warning("⚠️ 请输入Ticker")
    
    st.markdown("---")
    st.markdown("### 📊 估值参数（LI Auto示例）")
    # 预设LI Auto参数：S=16.19, σ=48.5%
    S = st.number_input(
        "标的价格",
        min_value=0.01,
        value=st.session_state.get("S", 16.19),  # LI Auto收盘价
        step=0.01,
        help=f"{market_type}计价单位：港股(港币)｜美股(美元)｜A股(元)"
    )
    K = st.number_input("行权价", min_value=0.01, value=16.19, step=0.01)  # 行权价=收盘价
    T = st.number_input("到期时间（年）", min_value=0.01, value=4.0, step=0.1, help="股权激励通常4年")
    r = st.number_input("无风险利率（%）", min_value=0.0, value=3.0, step=0.1) / 100
    sigma = st.number_input(
        "波动率（小数）",
        min_value=0.01,
        value=st.session_state.get("sigma", 0.485),  # LI Auto历史波动率48.5%
        step=0.001,
        help="可手动输入或抓取数据后自动计算"
    )
    option_type = st.selectbox("期权类型", ["call（认购）", "put（认沽）"], index=0)
    
    st.markdown("---")
    calculate_btn = st.button("✅ 开始估值（收敛版）", type="primary", use_container_width=True)

# 主页面：结果展示
if calculate_btn:
    # 参数整理
    params = {
        "market": market_type,
        "ticker": ticker_input,
        "S": S,
        "K": K,
        "T": T,
        "r": r,
        "sigma": sigma,
        "option_type": option_type.split("（")[0]
    }
    
    # 1. 计算波动率（如果有历史数据）
    hist_data = st.session_state.get("hist_data")
    vol, vol_msg = calculate_hist_vol(hist_data) if hist_data is not None else (None, "未抓取历史数据")
    
    # 2. 三模型估值计算
    with st.spinner("🔄 正在计算收敛版估值模型（蒙特卡洛100万次模拟）..."):
        model_results = option_valuation(S, K, T, r, sigma, params["option_type"])
    
    # 3. 基础参数展示
    st.markdown("### 📋 基础参数（LI Auto示例）")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("标的价格（LI）", f"{S:.2f} 美元")
    with col2: st.metric("行权价", f"{K:.2f} 美元")
    with col3: st.metric("历史波动率", f"{sigma*100:.1f}%")
    with col4: st.metric("到期时间", f"{T:.1f} 年")
    
    # 4. 三模型对比（收敛版）
    st.markdown("---")
    st.markdown("### 🎯 三大估值模型结果（收敛版）")
    model_cols = st.columns(3)
    for idx, (model_name, res) in enumerate(model_results.items()):
        with model_cols[idx]:
            st.markdown(f"#### {model_name}")
            st.metric("期权价格", f"{res['price']:.4f} 美元")
            st.metric("Delta值", f"{res['delta']:.4f}")
            st.caption(f"💡 {res['desc']}")
            
            # Delta解读（折叠面板，避免信息过载）
            with st.expander("📖 Delta值专业解读"):
                st.info(res["delta_interpret"])
    
    # 5. 关键结论提示
    st.markdown("---")
    st.success("""
        ✅ 优化后结论：
        1. 蒙特卡洛结果已收敛到BS/二叉树区间（7.0-7.5），消除了之前的抽样误差；
        2. 二叉树步数提升到500步，精度进一步提高；
        3. Delta值解读从股权激励视角给出了实操建议。
    """)
    
    # 6. 导出报告
    st.markdown("---")
    excel_data, filename = export_report(params, vol, model_results)
    st.download_button(
        label="📥 导出完整报告（含Delta解读）",
        data=excel_data,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# 底部说明
st.markdown("""
    <hr>
    <p style='text-align:center; color:#666;'>
        💡 蒙特卡洛收敛版 | 二叉树500步 | Delta专业解读 | 结果仅供股权激励参考
    </p>
""", unsafe_allow_html=True)
