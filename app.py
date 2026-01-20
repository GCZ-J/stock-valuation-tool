# -*- coding: utf-8 -*-
# 港美A股股权激励期权估值工具（稳定抓取版+字体配置修复）
# 核心优化：Ticker格式校验+双数据源+自动重试+数据有效性校验+可视化反馈
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import warnings
import matplotlib.pyplot as plt  # 新增：导入matplotlib用于字体配置
from datetime import datetime, timedelta
from scipy.stats import norm
from io import BytesIO
import openpyxl
# 新增：重试机制+A股备用数据源
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 全局配置：页面设置+matplotlib中文字体配置（修复核心错误）
st.set_page_config(
    page_title="港美A股股权激励期权估值工具（稳定版）",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
warnings.filterwarnings("ignore")  # 屏蔽无关警告
# 修正：用matplotlib的plt.rcParams配置字体，而非st.rcParams
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# ====================== 核心工具函数 ======================
# 1. Ticker格式校验与自动补全（杜绝格式错误）
def check_and_fix_ticker(ticker, market_type):
    """
    严格校验不同市场的Ticker格式，并自动补全后缀
    :param ticker: 用户输入的原始Ticker
    :param market_type: 美股/港股/A股
    :return: 修正后的Ticker, 错误信息（空字符串表示无错误）
    """
    ticker = ticker.strip().upper()  # 统一转大写+去除首尾空格
    if market_type == "美股":
        # 美股Ticker仅允许字母（如AAPL、MSFT）
        if not ticker.isalpha():
            return None, "美股Ticker只能是纯字母（如AAPL、MSFT）"
        return ticker, ""
    elif market_type == "港股":
        # 港股Ticker必须是5位数字，自动补.HK后缀
        if ticker.isdigit() and len(ticker) == 5:
            return f"{ticker}.HK", ""
        elif ticker.endswith(".HK") and ticker[:-3].isdigit() and len(ticker[:-3]) == 5:
            return ticker, ""
        else:
            return None, "港股Ticker必须是5位数字（如00700）或带.HK后缀（如00700.HK）"
    elif market_type == "A股":
        # A股：沪市6开头(.SS)、深市0/3开头(.SZ)
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

# 2. 双数据源股价抓取（带自动重试+数据有效性校验）
@st.cache_data(ttl=3600)  # 缓存1小时，避免重复请求
@retry(
    stop=stop_after_attempt(3),  # 最多重试3次
    wait=wait_exponential(multiplier=1, min=1, max=3),  # 重试间隔：1s→2s→3s
    retry=retry_if_exception_type((Exception,)),  # 任何异常都触发重试
    reraise=True  # 重试失败后抛出原异常
)
def get_stock_data(ticker, market_type):
    """
    双数据源抓取股价：优先yfinance，A股失败自动切AkShare
    :param ticker: 用户输入的原始Ticker
    :param market_type: 美股/港股/A股
    :return: latest_close(最新收盘价), hist_data(历史数据DF), msg(提示信息)
    """
    # 第一步：先校验并修正Ticker格式
    ticker_full, err_msg = check_and_fix_ticker(ticker, market_type)
    if err_msg:
        return None, None, f"❌ Ticker格式错误：{err_msg}"

    # 第二步：优先使用yfinance抓取（适配美股/港股/A股）
    try:
        stock = yf.Ticker(ticker_full)
        hist_data = stock.history(period="1y")  # 近1年日线数据
        if not hist_data.empty:
            # 提取最新收盘价（复权后，更贴近实际）
            latest_close = round(hist_data["Close"].iloc[-1], 2)
            # 整理历史数据格式
            hist_data = hist_data[["Close"]].reset_index()
            hist_data.rename(columns={"Date": "日期", "Close": "收盘价"}, inplace=True)
            hist_data["日期"] = hist_data["日期"].dt.date  # 格式化日期
            
            # 第三步：数据有效性校验（过滤异常数据）
            if latest_close <= 0:
                return None, None, f"❌ 数据异常：{ticker_full} 收盘价={latest_close}（需大于0）"
            if len(hist_data) < 20:
                return None, None, f"❌ 历史数据不足：仅{len(hist_data)}条（至少需要20条）"
            
            return latest_close, hist_data, f"✅ yfinance抓取成功：{ticker_full} 收盘价={latest_close}"
    except Exception as e:
        st.warning(f"ℹ️ yfinance抓取{market_type}失败：{str(e)}，尝试备用数据源...")

    # 第四步：A股专属备用方案（AkShare，A股数据更稳定）
    if market_type == "A股":
        try:
            import akshare as ak
            # 转换AkShare所需的Ticker格式（去除.SS/.SZ后缀）
            ticker_ak = ticker_full.replace(".SS", "").replace(".SZ", "")
            # 抓取A股复权日线数据（前复权）
            hist_data = ak.stock_zh_a_hist(
                symbol=ticker_ak,
                period="daily",
                start_date=(datetime.now() - timedelta(days=365)).strftime("%Y%m%d"),
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq"  # 前复权
            )
            if not hist_data.empty:
                # 整理数据格式
                hist_data = hist_data[["日期", "收盘"]].rename(columns={"收盘": "收盘价"})
                hist_data["日期"] = pd.to_datetime(hist_data["日期"]).dt.date
                latest_close = round(hist_data["收盘价"].iloc[-1], 2)
                
                # 数据有效性校验
                if latest_close <= 0:
                    return None, None, f"❌ 数据异常：{ticker_ak} 收盘价={latest_close}（需大于0）"
                if len(hist_data) < 20:
                    return None, None, f"❌ 历史数据不足：仅{len(hist_data)}条（至少需要20条）"
                
                return latest_close, hist_data, f"✅ AkShare抓取成功：{ticker_ak} 收盘价={latest_close}"
        except ImportError:
            return None, None, "❌ AkShare未安装（需在requirements.txt添加akshare>=1.10.0）"
        except Exception as e:
            return None, None, f"❌ AkShare抓取失败：{str(e)}"

    # 其他市场无备用数据源，返回失败
    return None, None, f"❌ 所有数据源均失败：未获取到{ticker_full}的有效数据，请检查Ticker或稍后重试"

# 3. 历史波动率计算（支持上传文件/自动抓取数据）
def calculate_hist_vol(file=None, hist_data=None):
    """
    计算年化历史波动率（252个交易日）
    :param file: 上传的Excel/CSV文件
    :param hist_data: 自动抓取的历史数据DF
    :return: annual_vol(年化波动率), msg(提示信息)
    """
    try:
        # 优先使用自动抓取的历史数据
        if hist_data is not None and not hist_data.empty:
            df = hist_data
        elif file:
            # 读取上传文件
            if file.name.endswith(".xlsx"):
                df = pd.read_excel(file)
            elif file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                return None, "❌ 仅支持.xlsx/.csv格式文件"
            
            # 自动识别收盘价列
            close_cols = [col for col in df.columns if "close" in col.lower() or "收盘价" in col]
            if not close_cols:
                return None, "❌ 未找到收盘价列（列名含close/收盘价）"
            df = df[close_cols[0]].dropna()
            df = pd.DataFrame({"收盘价": df})
        else:
            return None, "❌ 请先上传数据文件或抓取历史数据"
        
        # 检查数据量
        if len(df) < 20:
            return None, "❌ 数据量不足（至少需要20个交易日收盘价）"
        
        # 计算日收益率和年化波动率
        df["日收益率"] = df["收盘价"].pct_change()
        daily_vol = df["日收益率"].std()
        annual_vol = daily_vol * np.sqrt(252)  # 年化（252个交易日）
        
        return round(annual_vol, 4), f"✅ 历史波动率计算成功：{round(annual_vol*100, 2)}%"
    except Exception as e:
        return None, f"❌ 波动率计算失败：{str(e)}"

# 4. Black-Scholes期权定价模型（股权激励核心估值）
def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes模型计算期权公允价值
    :param S: 标的当前价格
    :param K: 行权价
    :param T: 到期时间（年）
    :param r: 无风险利率
    :param sigma: 波动率
    :param option_type: 期权类型（call=认购/put=认沽）
    :return: option_price(期权价格), delta(对冲值)
    """
    try:
        if T <= 0:
            return max(S - K, 0) if option_type == "call" else max(K - S, 0), 0.0
        
        # 计算d1和d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma **2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # 计算期权价格
        if option_type == "call":
            option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
        return round(option_price, 4), round(delta, 4)
    except Exception as e:
        st.error(f"❌ 期权估值失败：{str(e)}")
        return 0.0, 0.0

# 5. 导出估值报告（Excel）
def export_valuation_report(params, vol_result, bs_result):
    """
    导出完整的估值报告
    :param params: 输入参数
    :param vol_result: 波动率结果
    :param bs_result: 期权估值结果
    :return: BytesIO对象, 文件名
    """
    data = [
        # 基础参数
        ["估值日期", datetime.now().strftime("%Y-%m-%d")],
        ["标的市场", params["market"]],
        ["标的Ticker", params["ticker"]],
        ["标的当前价格", params["S"]],
        ["行权价(K)", params["K"]],
        ["到期时间(T,年)", params["T"]],
        ["无风险利率(r)", params["r"]],
        ["波动率(σ)", params["sigma"]],
        # 波动率结果
        ["历史年化波动率", vol_result["vol"] if vol_result["vol"] else "未计算"],
        # 期权估值结果
        ["期权公允价值", bs_result["price"]],
        ["Delta值", bs_result["delta"]],
        ["期权类型", params["option_type"]]
    ]
    df = pd.DataFrame(data, columns=["估值维度", "结果"])
    
    # 写入Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="估值报告", index=False)
    output.seek(0)
    
    filename = f"股权激励期权估值报告_{datetime.now().strftime('%Y%m%d')}.xlsx"
    return output, filename

# ====================== 页面布局（带可视化加载反馈） ======================
# 标题区域
st.markdown("""
    <h1 style='text-align: center; color: #2E86AB;'>📈 港美A股股权激励期权估值工具（稳定版）</h1>
    <h3 style='text-align: center; color: #A23B72;'>Ticker自动抓取 | 双数据源备份 | 历史波动率计算 | Black-Scholes估值</h3>
    <hr>
""", unsafe_allow_html=True)

# 侧边栏：参数配置（带清晰的Ticker提示）
with st.sidebar:
    st.markdown("### ⚙️ 标的信息配置")
    # 市场+Ticker配置（带示例提示）
    col1, col2 = st.columns(2)
    with col1:
        market_type = st.selectbox("标的市场", ["美股", "港股", "A股"], index=0)
    with col2:
        ticker_input = st.text_input(
            "标的Ticker", 
            placeholder="AAPL/00700/600000", 
            help="✅ 美股：AAPL｜港股：00700｜A股：600000（沪市）/000001（深市）"
        )
    # Ticker格式提示（降低用户输入错误率）
    st.caption("📌 格式自动补全：港股补.HK | A股沪市补.SS | 深市补.SZ")
    
    # 抓取按钮（带加载动画）
    col3, col4 = st.columns(2)
    with col3:
        if st.button("📈 抓取最新收盘价", use_container_width=True):
            if ticker_input:
                with st.spinner("🔄 正在抓取数据...（最多重试3次）"):  # 加载动画
                    latest_close, hist_data, msg = get_stock_data(ticker_input, market_type)
                if latest_close:
                    st.session_state["S"] = latest_close
                    st.session_state["hist_data"] = hist_data
                st.success(msg) if "✅" in msg else st.error(msg)
            else:
                st.warning("⚠️ 请先输入标的Ticker")
    with col4:
        if st.button("📊 抓取历史数据（算波动率）", use_container_width=True):
            if ticker_input:
                with st.spinner("🔄 正在抓取历史数据...（最多重试3次）"):
                    latest_close, hist_data, msg = get_stock_data(ticker_input, market_type)
                if hist_data is not None and not hist_data.empty:
                    st.session_state["hist_data"] = hist_data
                    # 自动计算波动率并填充
                    vol, vol_msg = calculate_hist_vol(hist_data=hist_data)
                    if vol:
                        st.session_state["sigma"] = vol
                        st.success(f"{vol_msg}（已填充到波动率输入框）")
                    else:
                        st.error(vol_msg)
                else:
                    st.error(msg)
            else:
                st.warning("⚠️ 请先输入标的Ticker")
    
    st.markdown("---")
    st.markdown("### 📊 估值核心参数")
    # 标的价格（优先使用抓取的收盘价）
    S = st.number_input(
        "标的当前价格",
        min_value=0.01, max_value=10000.0,
        value=st.session_state.get("S", 67.0),
        step=0.01,
        help="A股(元)｜港股(港币)｜美股(美元)（可手动修改）"
    )
    K = st.number_input("行权价", min_value=0.01, max_value=10000.0, value=50.0, step=0.01)
    T = st.number_input("到期时间（年）", min_value=0.01, max_value=10.0, value=4.0, step=0.1, help="股权激励通常4年解锁")
    r = st.number_input("无风险利率（%）", min_value=0.0, max_value=20.0, value=3.0, step=0.1) / 100  # 转为小数
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
        # 优先使用自动抓取的历史数据，其次是上传的文件
        hist_data = st.session_state.get("hist_data")
        vol, vol_msg = calculate_hist_vol(file=uploaded_file, hist_data=hist_data)
        if vol:
            st.session_state["sigma"] = vol
            st.success(vol_msg)
        else:
            st.error(vol_msg)
    
    st.markdown("---")
    calculate_btn = st.button("✅ 开始估值计算", type="primary", use_container_width=True)

# 主页面：估值结果展示
if calculate_btn:
    # 整理参数
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
    
    # 1. 计算波动率（备用）
    hist_data = st.session_state.get("hist_data")
    vol, vol_msg = calculate_hist_vol(hist_data=hist_data)
    vol_result = {"vol": vol, "msg": vol_msg}
    
    # 2. Black-Scholes估值
    option_price, delta = black_scholes(S, K, T, r, sigma, params["option_type"])
    bs_result = {"price": option_price, "delta": delta}
    
    # 3. 展示结果（清晰的分栏）
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
        st.metric("Delta值", f"{delta:.4f}", help="期权价格对标的价格的敏感度（越大越敏感）")
    
    # 4. 导出报告（一键下载）
    excel_data, filename = export_valuation_report(params, vol_result, bs_result)
    st.download_button(
        label="📥 导出估值报告（Excel）",
        data=excel_data,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# 底部说明（含数据来源提示）
st.markdown("""
    <hr>
    <p style='text-align: center; color: #666;'>
        💡 估值结果仅供股权激励方案设计参考 | 数据来源：Yahoo Finance/AkShare | 无风险利率建议使用对应市场国债收益率
    </p>
""", unsafe_allow_html=True)
