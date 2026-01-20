# -*- coding: utf-8 -*-
# 港美A股股权激励估值工具（港股02015.HK专属修复版）
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

# 全局配置
st.set_page_config(
    page_title="港美A股股权激励估值工具（港股修复版）",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ====================== 新增：港股雪球数据源抓取函数 ======================
def get_hk_stock_from_xueqiu(ticker):
    """
    雪球网页抓取港股数据（专为02015.HK等标的设计）
    ticker: 港股5位数字，如02015
    """
    try:
        # 1. 抓取实时收盘价（雪球港股详情页）
        url = f"https://xueqiu.com/S/0{ticker}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 解析实时价格
        price_tag = soup.find("span", class_="stock-price")
        if not price_tag:
            price_tag = soup.find("div", class_="price")
        latest_close = float(price_tag.text.strip().replace(",", ""))
        
        # 2. 抓取历史数据（雪球K线接口，近1年日线）
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        kline_url = f"https://xueqiu.com/stock/forchartk/stocklist.json?symbol=0{ticker}&period=1day&type=normal&begin={start_date}&end={end_date}"
        kline_response = requests.get(kline_url, headers=headers, timeout=10)
        kline_data = kline_response.json()["chartlist"]
        
        # 整理历史数据
        hist_list = []
        for item in kline_data:
            date_str = datetime.fromtimestamp(item["time"]/1000).strftime("%Y-%m-%d")
            hist_list.append({"日期": date_str, "收盘价": item["close"]})
        hist_data = pd.DataFrame(hist_list)
        hist_data["日期"] = pd.to_datetime(hist_data["日期"]).dt.date
        
        # 数据校验
        if latest_close <= 0 or len(hist_data) < 20:
            return None, None, "雪球数据异常或不足"
        
        return round(latest_close, 2), hist_data, f"✅ 雪球抓取成功：0{ticker}.HK 收盘价={latest_close:.2f}"
    except Exception as e:
        return None, None, f"雪球抓取失败：{str(e)}"

# ====================== 核心工具函数 ======================
# 1. Ticker格式校验
def check_and_fix_ticker(ticker, market_type):
    ticker = ticker.strip().upper()
    if market_type == "美股":
        if not ticker.isalpha():
            return None, "美股Ticker只能是字母（如LI、AAPL，大小写均可）"
        return ticker, ""
    elif market_type == "港股":
        if ticker.isdigit() and len(ticker) == 5:
            return ticker, ""  # 保留纯数字，后续雪球抓取用
        elif ticker.endswith(".HK") and ticker[:-3].isdigit() and len(ticker[:-3]) == 5:
            return ticker[:-3], ""  # 去除.HK，适配雪球
        else:
            return None, "港股Ticker必须是5位数字（如02015）"
    elif market_type == "A股":
        if ticker.isdigit():
            if ticker.startswith("6"):
                return f"{ticker}.SS", ""
            elif ticker.startswith(("0", "3")):
                return f"{ticker}.SZ", ""
            else:
                return None, "A股Ticker需6开头（沪市）或0/3开头（深市）"
        elif ticker.endswith((".SS", ".SZ")):
            prefix = ticker[:-3]
            if prefix.isdigit() and (prefix.startswith("6") or prefix.startswith(("0", "3"))):
                return ticker, ""
            else:
                return None, "A股Ticker后缀错误"
        else:
            return None, "A股Ticker必须是纯数字或带.SS/.SZ后缀"
    else:
        return None, "请选择正确市场"

# 2. 增强版双数据源抓取（港股优先雪球）
@st.cache_data(ttl=3600)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=5),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def get_stock_data(ticker, market_type):
    ticker_fixed, err_msg = check_and_fix_ticker(ticker, market_type)
    if err_msg:
        return None, None, f"❌ {err_msg}"

    # ========== 港股专属逻辑：优先雪球，再试yfinance ==========
    if market_type == "港股":
        # 1. 优先用雪球抓取（专为02015.HK优化）
        xq_close, xq_hist, xq_msg = get_hk_stock_from_xueqiu(ticker_fixed)
        if xq_close:
            return xq_close, xq_hist, xq_msg
        # 2. 雪球失败，再试yfinance（补全.HK后缀）
        yf_ticker = f"{ticker_fixed}.HK"
        try:
            stock = yf.Ticker(yf_ticker)
            hist_data = stock.history(period="1y", interval="1d")
            if not hist_data.empty:
                latest_close = round(hist_data["Close"].iloc[-1], 2)
                hist_data = hist_data[["Close"]].reset_index()
                hist_data.rename(columns={"Date": "日期", "Close": "收盘价"}, inplace=True)
                hist_data["日期"] = hist_data["日期"].dt.date
                if latest_close > 0 and len(hist_data) >= 20:
                    return latest_close, hist_data, f"✅ yfinance抓取成功：{yf_ticker} 收盘价={latest_close}"
        except Exception as e:
            st.warning(f"yfinance抓取港股失败：{e}")
        # 3. 所有数据源失败
        return None, None, f"❌ 无法抓取港股{yf_ticker}数据，请稍后重试"

    # ========== 美股/A股原有逻辑 ==========
    elif market_type == "美股":
        try:
            stock = yf.Ticker(ticker_fixed)
            hist_data = stock.history(period="1y")
            if not hist_data.empty:
                latest_close = round(hist_data["Close"].iloc[-1], 2)
                hist_data = hist_data[["Close"]].reset_index()
                hist_data.rename(columns={"Date": "日期", "Close": "收盘价"}, inplace=True)
                hist_data["日期"] = hist_data["日期"].dt.date
                return latest_close, hist_data, f"✅ 抓取成功：{ticker_fixed} 收盘价={latest_close}"
        except Exception as e:
            return None, None, f"❌ 美股抓取失败：{e}"

    elif market_type == "A股":
        try:
            import akshare as ak
            ticker_ak = ticker_fixed.replace(".SS", "").replace(".SZ", "")
            hist_data = ak.stock_zh_a_hist(symbol=ticker_ak, period="daily", adjust="qfq")
            if not hist_data.empty:
                latest_close = round(hist_data["收盘"].iloc[-1], 2)
                hist_data = hist_data[["日期", "收盘"]].rename(columns={"收盘": "收盘价"})
                hist_data["日期"] = pd.to_datetime(hist_data["日期"]).dt.date
                return latest_close, hist_data, f"✅ AkShare抓取成功：{ticker_ak} 收盘价={latest_close}"
        except Exception as e:
            return None, None, f"❌ A股抓取失败：{e}"

    return None, None, "❌ 未支持的市场类型"

# 3. 历史波动率计算
def calculate_hist_vol(file=None, hist_data=None):
    try:
        if hist_data is not None and not hist_data.empty:
            df = hist_data
        elif file:
            if file.name.endswith(".xlsx"):
                df = pd.read_excel(file)
            elif file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                return None, "❌ 仅支持.xlsx/.csv格式"
            close_cols = [col for col in df.columns if "close" in col.lower() or "收盘价" in col]
            if not close_cols:
                return None, "❌ 未找到收盘价列"
            df = df[close_cols[0]].dropna()
            df = pd.DataFrame({"收盘价": df})
        else:
            return None, "❌ 请先上传文件或抓取历史数据"
        
        if len(df) < 20:
            return None, "❌ 数据量不足（至少20条）"
        
        df["日收益率"] = df["收盘价"].pct_change()
        daily_vol = df["日收益率"].std()
        annual_vol = daily_vol * np.sqrt(252)
        return round(annual_vol, 4), f"✅ 历史波动率：{round(annual_vol*100, 2)}%"
    except Exception as e:
        return None, f"❌ 波动率计算失败：{str(e)}"

# 4. 三大期权估值模型
def option_valuation_models(S, K, T, r, sigma, option_type="call"):
    results = {}
    # Black-Scholes
    try:
        if T <= 0:
            bs_price = max(S - K, 0) if option_type == "call" else max(K - S, 0)
            bs_delta = 1.0 if (option_type == "call" and S > K) else 0.0
        else:
            d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            if option_type == "call":
                bs_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
                bs_delta = norm.cdf(d1)
            else:
                bs_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
                bs_delta = -norm.cdf(-d1)
        results["Black-Scholes"] = {"price": round(bs_price,4), "delta": round(bs_delta,4), "desc": "欧式期权基准模型"}
    except Exception as e:
        results["Black-Scholes"] = {"price":0, "delta":0, "desc": f"失败：{e}"}
    # 蒙特卡洛
    try:
        np.random.seed(42)
        n_sim = 100000
        dt = T/252
        paths = S*np.exp(np.cumsum((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.normal(0,1,(int(T*252),n_sim)),axis=0))
        payoffs = np.maximum(paths[-1]-K,0) if option_type=="call" else np.maximum(K-paths[-1],0)
        mc_price = np.exp(-r*T)*np.mean(payoffs)
        results["蒙特卡洛模拟"] = {"price": round(mc_price,4), "delta": round((mc_price - max(S*1.01-K,0)*np.exp(-r*T))/(S*0.01),4), "desc": "复杂期权数值解法"}
    except Exception as e:
        results["蒙特卡洛模拟"] = {"price":0, "delta":0, "desc": f"失败：{e}"}
    # 二叉树
    try:
        n_steps = 100
        dt = T/n_steps
        u = np.exp(sigma*np.sqrt(dt))
        d = 1/u
        p = (np.exp(r*dt)-d)/(u-d)
        stock_prices = S * (u**np.arange(n_steps,-1,-1)) * (d**np.arange(0,n_steps+1,1))
        option_vals = np.maximum(stock_prices-K,0) if option_type=="call" else np.maximum(K-stock_prices,0)
        for i in range(n_steps-1,-1,-1):
            option_vals = np.exp(-r*dt)*(p*option_vals[:-1] + (1-p)*option_vals[1:])
        delta = (option_vals[0] - max(S*d-K,0)*np.exp(-r*dt))/(S*(u-d))
        results["二叉树模型"] = {"price": round(option_vals[0],4), "delta": round(delta,4), "desc": "美式期权优先选择"}
    except Exception as e:
        results["二叉树模型"] = {"price":0, "delta":0, "desc": f"失败：{e}"}
    return results

# 5. 导出报告
def export_valuation_report(params, vol_result, model_results):
    data = [
        ["估值日期", datetime.now().strftime("%Y-%m-%d")],
        ["标的市场", params["market"]],
        ["标的Ticker", params["ticker"]],
        ["标的价格", params["S"]], ["行权价", params["K"]], ["期限(年)", params["T"]],
        ["无风险利率", params["r"]], ["波动率", params["sigma"]], ["历史波动率", vol_result["vol"] or "未计算"],
        ["期权类型", params["option_type"]], ["---", "---"],
        ["Black-Scholes价格", model_results["Black-Scholes"]["price"]],
        ["蒙特卡洛价格", model_results["蒙特卡洛模拟"]["price"]],
        ["二叉树价格", model_results["二叉树模型"]["price"]]
    ]
    df = pd.DataFrame(data, columns=["维度", "数值"])
    output = BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)
    return output, f"估值报告_{datetime.now().strftime('%Y%m%d')}.xlsx"

# 6. 估值建议
def generate_advice(model_results, T):
    prices = [model_results[m]["price"] for m in model_results]
    avg_price = np.mean(prices)
    diff = max(prices)-min(prices)
    if diff/avg_price < 0.05:
        advice = "✅ 三大模型结果一致，估值可信度高"
    else:
        advice = "⚠️ 模型结果差异较大，建议参考二叉树（长期）或Black-Scholes（短期）"
    if T>1:
        advice += "｜长期期权优先选二叉树模型"
    else:
        advice += "｜短期期权优先选Black-Scholes模型"
    return advice

# ====================== 页面布局 ======================
st.markdown("""
    <h1 style='text-align:center; color:#2E86AB;'>📈 港美A股股权激励估值工具（港股修复版）</h1>
    <h3 style='text-align:center; color:#A23B72;'>02015.HK专属适配 | 三模型对比 | 双数据源备份</h3>
    <hr>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ 标的配置")
    col1, col2 = st.columns(2)
    with col1: market_type = st.selectbox("市场", ["港股", "美股", "A股"], index=0)
    with col2: ticker_input = st.text_input("Ticker", placeholder="港股02015｜美股LI｜A股600000", help="港股直接输5位数字")
    
    st.caption("📌 港股示例：02015（理想汽车）｜自动优先雪球抓取")
    col3, col4 = st.columns(2)
    with col3:
        if st.button("📈 抓取收盘价", use_container_width=True):
            if ticker_input:
                with st.spinner("抓取中..."):
                    close, hist, msg = get_stock_data(ticker_input, market_type)
                if close:
                    st.session_state["S"] = close
                    st.session_state["hist_data"] = hist
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                st.warning("请输入Ticker")
    with col4:
        if st.button("📊 抓取历史数据", use_container_width=True):
            if ticker_input:
                with st.spinner("抓取中..."):
                    _, hist, msg = get_stock_data(ticker_input, market_type)
                if hist is not None:
                    st.session_state["hist_data"] = hist
                    vol, vol_msg = calculate_hist_vol(hist_data=hist)
                    if vol:
                        st.session_state["sigma"] = vol
                        st.success(f"✅ 历史波动率：{vol*100:.2f}%（已填充）")
                    else:
                        st.error(vol_msg)
                else:
                    st.error(msg)
            else:
                st.warning("请输入Ticker")

    st.markdown("---")
    st.markdown("### 📊 估值参数")
    S = st.number_input("标的价格", min_value=0.01, value=st.session_state.get("S", 67.0), step=0.01)
    K = st.number_input("行权价", min_value=0.01, value=50.0, step=0.01)
    T = st.number_input("期限(年)", min_value=0.01, value=4.0, step=0.1, help="股权激励通常4年")
    r = st.number_input("无风险利率(%)", min_value=0.0, value=3.0, step=0.1)/100
    sigma = st.number_input("波动率", min_value=0.01, value=st.session_state.get("sigma", 0.2), step=0.01)
    option_type = st.selectbox("期权类型", ["call（认购）", "put（认沽）"], index=0)
    calculate_btn = st.button("✅ 开始估值（三模型对比）", type="primary", use_container_width=True)

# 主页面结果展示
if calculate_btn:
    params = {"market":market_type, "ticker":ticker_input, "S":S, "K":K, "T":T, "r":r, "sigma":sigma, "option_type":option_type.split("（")[0]}
    hist_data = st.session_state.get("hist_data")
    vol, vol_msg = calculate_hist_vol(hist_data=hist_data)
    vol_result = {"vol":vol, "msg":vol_msg}
    model_results = option_valuation_models(S, K, T, r, sigma, params["option_type"])

    # 基础参数
    st.markdown("### 📋 基础参数")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("标的价格", f"{S:.2f}")
    with col2: st.metric("行权价", f"{K:.2f}")
    with col3: st.metric("历史波动率", f"{vol*100:.2f}%" if vol else "未计算")
    with col4: st.metric("使用波动率", f"{sigma*100:.2f}%")

    # 三模型对比
    st.markdown("---")
    st.markdown("### 🎯 三大模型估值结果")
    model_cols = st.columns(3)
    for idx, (model, res) in enumerate(model_results.items()):
        with model_cols[idx]:
            st.markdown(f"#### {model}")
            st.metric("期权价格", f"{res['price']:.4f}")
            st.metric("Delta值", f"{res['delta']:.4f}")
            st.caption(f"💡 {res['desc']}")

    # 建议
    st.markdown("---")
    st.info(generate_advice(model_results, T))

    # 导出
    excel_data, filename = export_valuation_report(params, vol_result, model_results)
    st.download_button("📥 导出估值报告", data=excel_data, file_name=filename, use_container_width=True)

st.markdown("""<hr><p style='text-align:center; color:#666;'>💡 港股02015.HK专属优化 | 数据来源：雪球/Yahoo Finance</p>""", unsafe_allow_html=True)
