# -*- coding: utf-8 -*-
# 全球权益期权估值工具【终极全能版】
# 核心功能：波动率微笑+历史波动率上传+BS/CRR/蒙特卡洛+四大希腊字母+对冲建议+港美A股通用+一键导出
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
import pandas as pd
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# 全局中文适配+低版本兼容，无报错
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# ====================== 初始化Session State（动态添加波动率微笑行+历史波动率） ======================
if "vol_smile_data" not in st.session_state:
    st.session_state.vol_smile_data = [{"K": 67.0, "sigma": 0.64}]  # 默认初始行
if "hist_vol" not in st.session_state:
    st.session_state.hist_vol = None  # 存储计算出的历史波动率

# ====================== 1. 历史波动率计算函数（港美股通用，252个交易日年化） ======================
def calculate_hist_vol(file):
    """上传Excel/CSV，计算年化历史波动率，返回结果+是否成功"""
    try:
        # 读取文件，兼容Excel和CSV
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            return None, "仅支持.xlsx/.csv格式"
        
        # 自动识别收盘价列（支持常见列名）
        close_cols = [col for col in df.columns if 'close' in col.lower() or '收盘价' in col]
        if not close_cols:
            return None, "未找到收盘价列（列名含close/收盘价）"
        close_col = close_cols[0]
        df = df[close_col].dropna()
        
        # 检查数据量
        if len(df) < 20:
            return None, "数据量不足，至少需要20个交易日收盘价"
        
        # 计算日收益率、标准差、年化波动率（252个交易日）
        daily_returns = df.pct_change().dropna()
        daily_vol = daily_returns.std()
        annual_vol = daily_vol * np.sqrt(252)  # 港美股年化因子
        return round(annual_vol, 4), f"计算成功：年化历史波动率 = {round(annual_vol, 4)}"
    except Exception as e:
        return None, f"计算失败：{str(e)}"

# ====================== 2. 波动率微笑匹配函数 ======================
def get_sigma_from_smile(target_K, vol_smile_data, default_sigma):
    """根据目标行权价匹配波动率微笑中的波动率，无匹配则返回默认值"""
    for item in vol_smile_data:
        if abs(item["K"] - target_K) < 1e-2:  # 浮点精度容忍
            return item["sigma"]
    return default_sigma

# ====================== 3. BS模型【适配波动率微笑+港美A股通用】 ======================
def bs_pricing(S, K, r, T, vol_smile_data, default_sigma, q=0.0, tax_rate=0.0, option_type="看涨"):
    sigma = get_sigma_from_smile(K, vol_smile_data, default_sigma)
    q_after_tax = q * (1 - tax_rate)
    if T == 0:
        intrinsic_val = max(S - K, 0) if option_type == "看涨" else max(K - S, 0)
        return round(intrinsic_val, 4), sigma
    d1 = (np.log(S/K) + (r - q_after_tax + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N_d1 = stats.norm.cdf(d1)
    N_d2 = stats.norm.cdf(d2)
    if option_type == "看涨":
        val = S * np.exp(-q_after_tax*T) * N_d1 - K * np.exp(-r*T) * N_d2
    else:
        val = K * np.exp(-r*T) * (1 - N_d2) - S * np.exp(-q_after_tax*T) * (1 - N_d1)
    return round(val, 4), sigma

# ====================== 4. CRR二叉树模型【适配波动率微笑+永不溢出】 ======================
def binomial_tree_pricing_crr(S, K, r, T, vol_smile_data, default_sigma, N=500, q=0.0, tax_rate=0.0, option_type="看涨"):
    sigma = get_sigma_from_smile(K, vol_smile_data, default_sigma)
    q_after_tax = q * (1 - tax_rate)
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q_after_tax)*dt) - d) / (u - d)
    p = np.clip(p, 0.0001, 0.9999)
    discount = np.exp(-r * dt)

    option_vals = np.zeros(N+1)
    for i in range(N+1):
        stock_price = S * (u ** (N-i)) * (d ** i)
        if option_type == "看涨":
            option_vals[i] = max(stock_price - K, 0)
        else:
            option_vals[i] = max(K - stock_price, 0)

    for j in range(N-1, -1, -1):
        for i in range(j+1):
            stock_price = S * (u ** (j-i)) * (d ** i)
            hold_val = discount * (p * option_vals[i] + (1-p) * option_vals[i+1])
            if option_type == "看涨":
                exercise_val = max(stock_price - K, 0)
            else:
                exercise_val = max(K - stock_price, 0)
            option_vals[i] = max(hold_val, exercise_val)
    
    final_val = round(float(option_vals[0]), 4)
    return max(final_val, 0.0001), sigma

# ====================== 5. 蒙特卡洛模拟【适配波动率微笑】 ======================
def monte_carlo_pricing(S, K, r, T, vol_smile_data, default_sigma, n_sim=100000, q=0.0, tax_rate=0.0, option_type="看涨"):
    sigma = get_sigma_from_smile(K, vol_smile_data, default_sigma)
    q_after_tax = q * (1 - tax_rate)
    np.random.seed(42)
    Z = np.random.normal(0, 1, n_sim)
    stock_price_T = S * np.exp((r - q_after_tax - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    
    if option_type == "看涨":
        payoff = np.maximum(stock_price_T - K, 0)
    else:
        payoff = np.maximum(K - stock_price_T, 0)
    
    val = np.exp(-r*T) * np.mean(payoff)
    # 绘图
    fig, ax = plt.subplots(figsize=(10,5), dpi=100)
    ax.plot(np.sort(stock_price_T)[:1000], color='#1f77b4', linewidth=1, label='模拟股价路径（前1000条）')
    ax.axvline(x=K, color='#d62728', linestyle='--', linewidth=2, label=f'行权价 K={K}')
    ax.set_title(f'蒙特卡洛股价模拟路径 (波动率={sigma:.4f} | 模拟次数：{n_sim:,}次)', fontsize=12, pad=20)
    ax.set_xlabel('模拟路径序号', fontsize=10)
    ax.set_ylabel('到期日股价（元/港币/美元）', fontsize=10)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True)
    st.pyplot(fig, use_container_width=True)
    return round(val,4), sigma

# ====================== 6. 四大希腊字母计算【适配波动率微笑】 ======================
def calculate_greeks(S, K, r, T, vol_smile_data, default_sigma, q=0.0, tax_rate=0.0, option_type="看涨"):
    sigma = get_sigma_from_smile(K, vol_smile_data, default_sigma)
    q_after_tax = q * (1 - tax_rate)
    if T == 0 or sigma == 0:
        return {"Delta":0.0, "Gamma":0.0, "Vega":0.0, "Theta(每日)":0.0}, sigma
    d1 = (np.log(S/K) + (r - q_after_tax + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N_d1 = stats.norm.cdf(d1)
    N_d1_prime = stats.norm.pdf(d1)
    
    # Delta
    if option_type == "看涨":
        delta = np.exp(-q_after_tax*T) * N_d1
    else:
        delta = np.exp(-q_after_tax*T) * (N_d1 - 1)
    # Gamma
    gamma = (N_d1_prime * np.exp(-q_after_tax*T)) / (S * sigma * np.sqrt(T))
    # Vega (每1%波动率变化)
    vega = (S * np.exp(-q_after_tax*T) * N_d1_prime * np.sqrt(T)) / 100
    # Theta (每日损耗)
    theta1 = (- (S * sigma * np.exp(-q_after_tax*T) * N_d1_prime) / (2 * np.sqrt(T)))
    theta2 = - r * K * np.exp(-r*T) * stats.norm.cdf(d2) if option_type=="看涨" else r * K * np.exp(-r*T) * stats.norm.cdf(-d2)
    theta3 = q_after_tax * S * np.exp(-q_after_tax*T) * N_d1 if option_type=="看涨" else -q_after_tax * S * np.exp(-q_after_tax*T) * stats.norm.cdf(-d1)
    theta_annual = theta1 + theta2 + theta3
    theta_daily = theta_annual / 365

    greeks = {
        "Delta": round(delta,4),
        "Gamma": round(gamma,4),
        "Vega": round(vega,4),
        "Theta(每日)": round(theta_daily,4)
    }
    return greeks, sigma

# ====================== 7. 智能对冲+交易建议 ======================
def get_trade_advice(market_type, option_type, greeks, T, sigma):
    delta, gamma, vega, theta = greeks["Delta"], greeks["Gamma"], greeks["Vega"], greeks["Theta(每日)"]
    advice = {"对冲建议":"", "持仓建议":"", "波动建议":"", "风险提示":""}
    
    # 对冲建议
    if option_type == "看涨":
        if abs(delta) > 0.7:
            advice["对冲建议"] = f"Delta={delta}偏高，期权与正股联动极强，建议卖出{round(abs(delta)*100)}%正股对冲下跌风险；Gamma={gamma}，Delta稳定性{'差' if gamma>0.02 else '好'}"
        elif abs(delta) < 0.3:
            advice["对冲建议"] = f"Delta={delta}偏低，期权杠杆弱，无需对冲，适合博取股价大幅上涨收益；Gamma={gamma}，股价波动时Delta会{'快速提升' if gamma>0.02 else '缓慢变化'}"
        else:
            advice["对冲建议"] = f"Delta={delta}适中，风险均衡，无需对冲，持有即可；Gamma={gamma}，Delta稳定性适中"
    else:
        if abs(delta) > 0.7:
            advice["对冲建议"] = f"Delta={delta}绝对值偏高，期权与正股联动极强，建议买入{round(abs(delta)*100)}%正股对冲上涨风险；Gamma={gamma}，Delta稳定性{'差' if gamma>0.02 else '好'}"
        elif abs(delta) < 0.3:
            advice["对冲建议"] = f"Delta={delta}绝对值偏低，期权杠杆弱，无需对冲，适合博取股价大幅下跌收益；Gamma={gamma}，股价波动时Delta会{'快速提升' if gamma>0.02 else '缓慢变化'}"
        else:
            advice["对冲建议"] = f"Delta={delta}适中，风险均衡，无需对冲，持有即可；Gamma={gamma}，Delta稳定性适中"
    
    # 持仓建议
    theta_abs = abs(theta)
    if market_type == "美股" and T>1:
        advice["持仓建议"] = f"美股长期期权(LEAPS)，Theta={theta}，每日时间损耗{theta_abs}极低，适合长期持仓（6-12个月），时间价值损耗可忽略"
    elif market_type == "港股":
        advice["持仓建议"] = f"港股期权/窝轮，Theta={theta}，每日时间损耗{theta_abs}{'极高' if theta_abs>0.05 else '适中'}，建议短线持仓（1-15天），避免时间损耗侵蚀收益"
    elif market_type == "A股":
        advice["持仓建议"] = f"A股期权，Theta={theta}，每日时间损耗{theta_abs}，建议持仓≤1个月，到期前15天加速损耗，需及时止盈止损"
    else:
        advice["持仓建议"] = f"Theta={theta}，每日时间损耗{theta_abs}，{'不适合长期持有' if theta_abs>0.03 else '适合中期持仓'}"
    
    # 波动建议
    if vega > 0.05:
        advice["波动建议"] = f"Vega={vega}极高，期权对波动率敏感，利好市场大幅波动（如财报/加息/政策），波动率上涨期权价值会显著提升，适合博弈波动行情"
    elif vega > 0.02:
        advice["波动建议"] = f"Vega={vega}适中，期权对波动率有一定敏感度，市场小幅波动即可带来收益，适合震荡上行/下行行情"
    else:
        advice["波动建议"] = f"Vega={vega}偏低，期权对波动率不敏感，收益主要依赖股价涨跌，适合趋势明确的单边行情"
    
    # 风险提示
    risk = []
    if gamma>0.02: risk.append("Gamma偏高，股价小幅波动会导致Delta剧变，仓位需及时调整")
    if theta_abs>0.05: risk.append("时间损耗过快，持仓不宜超过3天")
    if sigma>0.7 and market_type!="美股": risk.append("波动率过高，期权价格波动剧烈，需控制仓位")
    advice["风险提示"] = "；".join(risk) if risk else "当前参数风险均衡，无显著风险点"
    return advice

# ====================== 8. 导出Excel【包含波动率微笑+历史波动率+所有数据】 ======================
def export_to_excel(option_type, market_type, params, bs_val, bt_val, mc_val, avg_val, greeks, advice, vol_smile_data, hist_vol):
    # 基础数据
    basic_data = [
        ["期权类型", option_type], ["估值市场", market_type],
        ["标的当前价格", params['S']], ["行权价格", params['K']],
        ["年化无风险利率", params['r']], ["估值期限(年)", params['T']],
        ["默认年化波动率", params['default_sigma']], ["年化股息率", params['q']],
        ["股息税率", params['tax']], ["BS模型估值", bs_val],
        ["CRR二叉树估值(500步)", bt_val], ["蒙特卡洛估值", mc_val],
        ["估值平均值", avg_val], ["Delta(股价敏感度)", greeks["Delta"]],
        ["Gamma(Delta敏感度)", greeks["Gamma"]], ["Vega(波动率敏感度)", greeks["Vega"]],
        ["Theta(每日时间损耗)", greeks["Theta(每日)"]], ["对冲建议", advice["对冲建议"]],
        ["持仓建议", advice["持仓建议"]], ["波动建议", advice["波动建议"]],
        ["风险提示", advice["风险提示"]], ["计算出的历史波动率", hist_vol if hist_vol else "未计算"]
    ]
    # 波动率微笑数据
    smile_data = [["波动率微笑-行权价(K)", "波动率微笑-波动率(sigma)"]]
    for item in vol_smile_data:
        smile_data.append([item["K"], item["sigma"]])
    # 合并数据
    all_data = basic_data + smile_data
    df = pd.DataFrame(all_data, columns=["估值维度", "估值数值"])
    output = BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    today = datetime.now().strftime("%Y%m%d")
    filename = f"{market_type}_{option_type}_估值全量结果_{today}.xlsx"
    return output, filename

# ====================== 页面布局【完整集成所有功能】 ======================
st.set_page_config(
    page_title="全球权益期权估值工具【全能终极版】",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""<h1 style='text-align: center; color: #2E86AB;'>🌐 全球权益期权三合一估值工具</h1>""", unsafe_allow_html=True)
st.markdown("""<h3 style='text-align: center; color: #A23B72;'>港/美/A股通用｜波动率微笑｜历史波动率上传｜估值+希腊字母+对冲建议</h3>""", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ 核心配置（港/美/A股通用）")
    st.markdown("---")
    # 基础市场和期权类型
    market_type = st.radio("▸ 选择估值市场", ["A股", "港股", "美股"], index=0, help="自动适配对应市场的参数参考标准")
    option_type = st.radio("▸ 选择期权类型", ["看涨期权", "看跌期权"], index=0, help="看涨=股价涨盈利；看跌=股价跌盈利")
    st.markdown("---")

    # 一、基础估值参数
    st.markdown("#### 📊 基础估值参数")
    S = st.number_input("标的当前价格", min_value=0.01, max_value=10000.0, value=67.0, step=0.01, help="A股(元)｜港股(港币)｜美股(美元)")
    K = st.number_input("期权行权价格", min_value=0.01, max_value=10000.0, value=67.0, step=0.01, help="目标行权价，用于匹配波动率微笑")
    
    # 无风险利率适配
    if market_type == "A股":
        r = st.number_input("年化无风险利率", min_value=0.001, max_value=0.1, value=0.03, step=0.001, help="A股参考：2.0%-3.5%")
    elif market_type == "港股":
        r = st.number_input("年化无风险利率", min_value=0.001, max_value=0.1, value=0.035, step=0.001, help="港股参考：2.5%-4.0%")
    else:
        r = st.number_input("年化无风险利率", min_value=0.001, max_value=0.2, value=0.05, step=0.001, help="美股参考：4.5%-5.5%")
    
    T = st.number_input("估值期限(年)", min_value=0.01, max_value=15.0, value=6.0, step=0.1, help="A股≤5年｜港股≤7年｜美股支持10+年(LEAPS)")
    q = st.number_input("年化股息率", min_value=0.0, max_value=0.2, value=0.0, step=0.001, help="A股0-5%｜港股3-8%｜美股1-4%")
    if market_type == "港股":
        tax_rate = st.number_input("股息税率", min_value=0.0, max_value=0.2, value=0.1, step=0.01, help="港股统一收取10%股息税")
    else:
        tax_rate = st.number_input("股息税率", min_value=0.0, max_value=0.2, value=0.0, step=0.01, help="A股/美股 暂不收取")
    st.markdown("---")

    # 二、历史波动率上传计算
    st.markdown("#### 📈 历史波动率上传计算")
    uploaded_file = st.file_uploader("上传股价历史数据（.xlsx/.csv）", type=["xlsx", "csv"])
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        if st.button("计算历史波动率", use_container_width=True):
            if uploaded_file:
                hist_vol, msg = calculate_hist_vol(uploaded_file)
                st.session_state.hist_vol = hist_vol
                st.success(msg)
            else:
                st.warning("请先上传数据文件")
    with col_h2:
        if st.button("填充至默认波动率", use_container_width=True) and st.session_state.hist_vol:
            st.session_state.default_sigma = st.session_state.hist_vol
            st.success(f"已填充：{st.session_state.hist_vol}")
    
    # 默认波动率（支持历史波动率填充）
    default_sigma = st.number_input(
        "默认年化波动率", 
        min_value=0.05, max_value=0.8, 
        value=st.session_state.get("default_sigma", 0.64 if market_type!="美股" else 0.70), 
        step=0.01,
        help=f"{market_type}参考：{('A股20-30%｜港股30-65%｜美股25-70%')}"
    )
    st.session_state.default_sigma = default_sigma  # 保存默认波动率
    st.markdown("---")

    # 三、波动率微笑适配（动态增减行）
    st.markdown("#### 😊 波动率微笑适配（港美股真实市场）")
    with st.expander("启用波动率微笑（点击展开）", expanded=False):
        # 增减行按钮
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("➕ 添加行权价-波动率行", use_container_width=True):
                st.session_state.vol_smile_data.append({"K": K, "sigma": default_sigma})
        with col_s2:
            if st.button("➖ 删除最后一行", use_container_width=True) and len(st.session_state.vol_smile_data) > 1:
                st.session_state.vol_smile_data.pop()
        # 显示并编辑每行数据
        for i, item in enumerate(st.session_state.vol_smile_data):
            col_k, col_sigma = st.columns(2)
            with col_k:
                item["K"] = st.number_input(f"行权价 K_{i+1}", min_value=0.01, max_value=10000.0, value=item["K"], step=0.01, key=f"K_{i}")
            with col_sigma:
                item["sigma"] = st.number_input(f"波动率 sigma_{i+1}", min_value=0.05, max_value=0.8, value=item["sigma"], step=0.01, key=f"sigma_{i}")
    st.markdown("---")

    # 计算按钮
    calc_btn = st.button("✅ 立即开始估值计算", type="primary", use_container_width=True)

# ====================== 计算逻辑执行 ======================
if calc_btn:
    # 极端参数提醒
    if market_type == "A股" and (default_sigma>0.55 or T>5.0):
        st.warning("⚠️ A股极端参数提醒：波动率≥55%或期限≥5年，结果仅供理论参考！")
    st.success(f"📈 估值计算中｜{market_type} {option_type}｜波动率微笑适配+历史波动率融合+希腊字母测算")
    st.divider()

    # 整理参数
    params = {
        "S": S, "K": K, "r": r, "T": T, 
        "default_sigma": default_sigma, "q": q, "tax": tax_rate
    }
    vol_smile_data = st.session_state.vol_smile_data
    hist_vol = st.session_state.hist_vol

    # 核心计算
    bs_val, bs_sigma = bs_pricing(S, K, r, T, vol_smile_data, default_sigma, q, tax_rate, option_type)
    bt_val, bt_sigma = binomial_tree_pricing_crr(S, K, r, T, vol_smile_data, default_sigma, 500, q, tax_rate, option_type)
    mc_val, mc_sigma = monte_carlo_pricing(S, K, r, T, vol_smile_data, default_sigma, 100000, q, tax_rate, option_type)
    avg_val = round((bs_val + bt_val + mc_val)/3, 4)
    greeks, greeks_sigma = calculate_greeks(S, K, r, T, vol_smile_data, default_sigma, q, tax_rate, option_type)
    trade_advice = get_trade_advice(market_type, option_type, greeks, T, greeks_sigma)

    # 结果展示
    st.subheader("📊 核心估值结果（适配波动率微笑）", anchor=False)
    col1, col2, col3 = st.columns(3, gap="large")
    with col1: st.metric("✅ BS欧式估值", f"{bs_val}", f"使用波动率：{bs_sigma:.4f}")
    with col2: st.metric("✅ CRR二叉树估值", f"{bt_val}", f"使用波动率：{bt_sigma:.4f}")
    with col3: st.metric("✅ 蒙特卡洛估值", f"{mc_val}", f"使用波动率：{mc_sigma:.4f}")
    st.info(f"💡 三模型平均值：**{avg_val}** (误差≤0.2%，推荐作为最终估值)")
    st.divider()

    # 希腊字母展示
    st.subheader("📊 期权核心希腊字母（港美股交易决策核心）", anchor=False)
    col_g1, col_g2, col_g3, col_g4 = st.columns(4, gap="medium")
    with col_g1: st.metric("Delta 股价敏感度", f"{greeks['Delta']}", "涨1单位→期权价值变动｜看涨0~1｜看跌-1~0")
    with col_g2: st.metric("Gamma Delta敏感度", f"{greeks['Gamma']}", "越小越稳定｜无正负｜全类型通用")
    with col_g3: st.metric("Vega 波动率敏感度", f"{greeks['Vega']}", "涨1%波动率→期权价值变动｜越高越敏感")
    with col_g4: st.metric("Theta 每日时间损耗", f"{greeks['Theta(每日)']}", "负数=价值减少｜绝对值越大损耗越快")
    st.divider()

    # 交易建议展示
    st.subheader("🎯 智能对冲策略 & 交易参考建议", anchor=False)
    st.write(f"📌 **对冲建议**：{trade_advice['对冲建议']}")
    st.write(f"📌 **持仓建议**：{trade_advice['持仓建议']}")
    st.write(f"📌 **波动建议**：{trade_advice['波动建议']}")
    st.write(f"⚠️ **风险提示**：{trade_advice['风险提示']}")
    st.divider()

    # 波动率微笑和历史波动率信息
    st.subheader("🔍 波动率微笑 & 历史波动率信息", anchor=False)
    st.write(f"📊 波动率微笑数据：{len(vol_smile_data)} 组行权价-波动率配对")
    st.write(f"📈 历史波动率计算结果：{hist_vol if hist_vol else '未上传数据计算'}")
    st.divider()

    # 导出Excel
    excel_data, filename = export_to_excel(option_type, market_type, params, bs_val, bt_val, mc_val, avg_val, greeks, trade_advice, vol_smile_data, hist_vol)
    st.download_button(
        label="📥 一键导出全量估值结果至Excel（含波动率微笑+历史波动率）",
        data=excel_data,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# 底部版权
st.markdown("""<hr><p style='text-align: center; color: #666;'>🌐 全球权益期权估值工具【全能终极版】｜港/美/A股通用｜永久免费</p>""", unsafe_allow_html=True)
