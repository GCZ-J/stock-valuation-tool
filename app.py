# -*- coding: utf-8 -*-
# 港美A股股权激励期权估值工具（稳定版+提示词优化+无冗余输出）
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import norm
from io import BytesIO
import openpyxl
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 全局配置
st.set_page_config(
    page_title="港美A股股权激励期权估值工具（稳定版）",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ====================== 核心工具函数 ======================
# 1. Ticker格式校验（自动转大写，兼容大小写输入）
def check_and_fix_ticker(ticker, market_type):
    """
    严格校验Ticker格式，自动转大写（兼容小写输入如li→LI）
    """
    ticker = ticker.strip().upper()  # 强制转大写，解决Li/LI输入差异问题
    if market_type == "美股":
        if not ticker.isalpha():
            return None, "美股Ticker只能是字母（如AAPL、MSFT，大小写均可）"
        return ticker, ""
    elif market_type == "港股":
        if ticker.isdigit() and len(ticker) == 5:
            return f"{ticker}.HK", ""
        elif ticker.endswith(".HK") and ticker[:-3].isdigit() and len(ticker[:-3]) == 5:
            return ticker, ""
        else:
            return None, "港股Ticker必须是5位数字（如00700）或带.HK后缀（如00700.HK）"
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
                return None, "A股Ticker后缀错误（沪市.SS/深市.SZ）"
        else:
            return None, "A股Ticker必须是纯数字或带.SS/.SZ后缀"
    else:
        return None, "请选择正确的市场类型（美股/港股/A股）"

# 2. 双数据源股价抓取（带自动重试+数据校验）
@st.cache_data(ttl=3600)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=3),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def get_stock_data(ticker, market_type):
    """双数据源抓取：优先yfinance，A股失败切AkShare"""
    ticker_full, err_msg = check_and_fix_ticker(ticker, market_type)
    if err_msg:
        return None, None, f"❌ Ticker格式错误：{err_msg}"

    # 优先yfinance
    try:
        stock = yf.Ticker(ticker_full)
        hist_data = stock.history(period="1y")
        if not hist_data.empty:
            latest_close = round(hist_data["Close"].iloc[-1], 2)
            hist_data = hist_data[["Close"]].reset_index()
            hist_data.rename(columns={"Date": "日期", "Close": "收盘价"}, inplace=True)
            hist_data["日期"] = hist_data["日期"].dt.date
            
            # 数据校验
            if latest_close <= 0:
                return None, None, f"❌ 数据异常：{ticker_full} 收盘价={latest_close}（需大于0）"
            if len(hist_data) < 20:
                return None, None, f"❌ 历史数据不足：仅{len(hist_data)}条（至少需要20条）"
            
            return latest_close, hist_data, f"✅ 抓取成功：{ticker_full} 收盘价={latest_close}"
    except Exception as e:
        st.warning(f"ℹ️ yfinance抓取失败，尝试备用数据源...")

    # A股备用AkShare
    if market_type == "A股":
        try:
            import akshare as ak
            ticker_ak = ticker_full.replace(".SS", "").replace(".SZ", "")
            hist_data = ak.stock_zh_a_hist(
                symbol=ticker_ak,
                period="daily",
                start_date=(datetime.now() - timedelta(days=365)).strftime("%Y%m%d"),
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq"
            )
            if not hist_data.empty:
                hist_data = hist_data[["日期", "收盘"]].rename(columns={"收盘": "收盘价"})
                hist_data["日期"] = pd.to_datetime(hist_data["日期"]).dt.date
                latest_close = round(hist_data["收盘价"].iloc[-1], 2)
                
                if latest_close <= 0:
                    return None, None, f"❌ 数据异常：{ticker_ak} 收盘价={latest_close}"
                if len(hist_data) < 20:
                    return None, None, f"❌ 历史数据不足：仅{len(hist_data)}条"
                
                return latest_close, hist_data, f"✅ 抓取成功：{ticker_ak} 收盘价={latest_close}"
        except ImportError:
            return None, None, "❌ AkShare未安装（需添加akshare>=1.10.0到requirements.txt）"
        except Exception as e:
            return None, None, f"❌ AkShare抓取失败：{str(e)}"

    return None, None, f"❌ 所有数据源均失败：未获取到{ticker_full}数据"

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

# 4. Black-Scholes估值模型
def black_scholes(S, K, T, r, sigma, option_type="call"):
    try:
        if T <= 0:
            return max(S - K, 0) if option_type == "call" else max(K - S, 0), 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
        return round(option_price, 4), round(delta, 4)
    except Exception as e:
        st.error(f"❌ 估值失败：{str(e)}")
        return 0.0, 0.0

# 5. 导出估值报告
def export_valuation_report(params, vol_result, bs_result):
    data = [
        ["估值日期", datetime.now().strftime("%Y-%m-%d")],
        ["标的市场", params["market"]],
        ["标的Ticker", params["ticker"]],
        ["标的当前价格", params["S"]],
        ["行权价(K)", params["K"]],
        ["到期时间(T,年)", params["T"]],
        ["无风险利率(r)", params["r"]],
        ["波动率(σ)", params["sigma"]],
        ["历史年化波动率", vol_result["vol"] if vol_result["vol"] else "未计算"],
        ["期权公允价值", bs_result["price"]],
        ["Delta值", bs_result["delta"]],
        ["期权类型", params["option_type"]]
    ]
    df = pd.DataFrame(data, columns=["估值维度", "结果"])
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="估值报告", index=False)
    output.seek(0)
    
    filename = f"股权激励期权估值报告_{datetime.now().strftime('%Y%m%d')}.xlsx"
    return output, filename

# ====================== 页面布局 ======================
# 标题
st.markdown("""
    <h1 style='text-align: center; color: #2E86AB;'>📈 港美A股股权激励期权估值工具（稳定版）</h1>
    <h3 style='text-align: center; color: #A23B72;'>Ticker自动抓取 | 双数据源备份 | 历史波动率计算</h3>
    <hr>
""", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.markdown("### ⚙️ 标的信息配置")
    
    # 优化后的Ticker输入（明确大小写无关）
    col1, col2 = st.columns(2)
    with col1:
        market_type = st.selectbox("标的市场", ["美股", "港股", "A股"], index=0)
    with col2:
        ticker_input = st.text_input(
            "标的Ticker", 
            placeholder="AAPL/00700/600000", 
            help="✅ 美股：AAPL（大小写均可，如li→LI）｜港股：00700｜A股：600000"
        )
    # 更清晰的提示词
    st.caption("📌 格式说明：港股自动补.HK | A股沪市补.SS/深市补.SZ | 大小写无关")
    
    # 抓取按钮（修复DeltaGenerator输出问题：改用标准if-else）
    col3, col4 = st.columns(2)
    with col3:
        if st.button("📈 抓取最新收盘价", use_container_width=True):
            if ticker_input:
                with st.spinner("🔄 正在抓取数据...（最多重试3次）"):
                    latest_close, hist_data, msg = get_stock_data(ticker_input, market_type)
                # 修复：不用三元表达式，改用标准if-else（避免返回DeltaGenerator对象）
                if "✅" in msg:
                    st.session_state["S"] = latest_close
                    st.session_state["hist_data"] = hist_data
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                st.warning("⚠️ 请先输入标的Ticker")
    
    with col4:
        if st.button("📊 抓取历史数据（算波动率）", use_container_width=True):
            if ticker_input:
                with st.spinner("🔄 正在抓取历史数据..."):
                    latest_close, hist_data, msg = get_stock_data(ticker_input, market_type)
                # 修复：标准if-else
                if hist_data is not None and not hist_data.empty:
                    st.session_state["hist_data"] = hist_data
                    vol, vol_msg = calculate_hist_vol(hist_data=hist_data)
                    if vol:
                        st.session_state["sigma"] = vol
                        st.success(f"{vol_msg}（已填充到波动率）")
                    else:
                        st.error(vol_msg)
                else:
                    st.error(msg)
            else:
                st.warning("⚠️ 请先输入标的Ticker")
    
    st.markdown("---")
    st.markdown("### 📊 估值核心参数")
    S = st.number_input(
        "标的当前价格",
        min_value=0.01, max_value=10000.0,
        value=st.session_state.get("S", 67.0),
        step=0.01,
        help="A股(元)｜港股(港币)｜美股(美元)（可手动修改）"
    )
    K = st.number_input("行权价", min_value=0.01, max_value=10000.0, value=50.0, step=0.01)
    T = st.number_input("到期时间（年）", min_value=0.01, max_value=10.0, value=4.0, step=0.1, help="股权激励通常4年解锁")
    r = st.number_input("无风险利率（%）", min_value=0.0, max_value=20.0, value=3.0, step=0.1) / 100
    sigma = st.number_input(
        "波动率（小数）",
        min_value=0.01, max_value=2.0,
        value=st.session_state.get("sigma", 0.2),
        step=0.01,
        help="可手动输入或通过历史数据计算（自动填充）"
    )
    option_type = st.selectbox("期权类型", ["call（认购）", "put（认沽）"], index=0)
    
    st.markdown("---")
    st.markdown("### 📁 手动上传历史数据（可选）")
    uploaded_file = st.file_uploader("上传Excel/CSV文件（含收盘价）", type=["xlsx", "csv"])
    if st.button("🧮 计算历史波动率", use_container_width=True):
        hist_data = st.session_state.get("hist_data")
        vol, vol_msg = calculate_hist_vol(file=uploaded_file, hist_data=hist_data)
        if vol:
            st.session_state["sigma"] = vol
            st.success(vol_msg)
        else:
            st.error(vol_msg)
    
    st.markdown("---")
    calculate_btn = st.button("✅ 开始估值计算", type="primary", use_container_width=True)

# 主页面：估值结果
if calculate_btn:
    params = {
        "market": market_type,
        "ticker": ticker_input,
        "S": S,
        "K": K,
        "T": T,
        "r": r,
        "sigma": sigma,
        "option_type": "call" if "call" in option_type else "put"
    }
    
    hist_data = st.session_state.get("hist_data")
    vol, vol_msg = calculate_hist_vol(hist_data=hist_data)
    vol_result = {"vol": vol, "msg": vol_msg}
    
    option_price, delta = black_scholes(S, K, T, r, sigma, params["option_type"])
    bs_result = {"price": option_price, "delta": delta}
    
    # 结果展示
    st.markdown("### 📋 基础参数与波动率结果")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("标的当前价格", f"{S:.2f}")
    with col6:
        st.metric("行权价", f"{K:.2f}")
    with col7:
        st.metric("历史波动率", f"{vol*100:.2f}%" if vol else "未计算")
    with col8:
        st.metric("使用的波动率", f"{sigma*100:.2f}%")
    
    st.markdown("---")
    st.markdown("### 🎯 Black-Scholes期权估值结果")
    col9, col10 = st.columns(2)
    with col9:
        st.metric("期权公允价值", f"{option_price:.4f}", help="股权激励核心估值结果")
    with col10:
        st.metric("Delta值", f"{delta:.4f}", help="期权价格对标的价格的敏感度")
    
    # 导出报告
    excel_data, filename = export_valuation_report(params, vol_result, bs_result)
    st.download_button(
        label="📥 导出估值报告（Excel）",
        data=excel_data,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# 底部说明
st.markdown("""
    <hr>
    <p style='text-align: center; color: #666;'>
        💡 估值结果仅供参考 | 数据来源：Yahoo Finance/AkShare | 无风险利率建议使用对应市场国债收益率
    </p>
""", unsafe_allow_html=True)
