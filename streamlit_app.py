import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize, linprog

# ---------------- Assumptions ----------------
MRP = 0.055            # market risk premium (annual). Common long-run estimates run ~4.5-6%
                       # (e.g. Damodaran / Duff & Phelps / Fernandez surveys). Edit to taste.
MARKET = "SPY"         # market proxy (S&P 500 ETF) used as the "market" in the beta regression
MIN_WEIGHT = 0.01      # smallest position allowed
MAX_WEIGHT = 0.20      # largest position allowed
PERIODS_PER_YEAR = 52  # weekly returns
DEFAULT_RF = 0.04      # fallback risk-free rate if the T-bill download fails

UNIVERSE = {
    # Utilities / staples / telecom (usually low beta)
    "DUK": "Duke Energy", "SO": "Southern Co", "NEE": "NextEra Energy", "AEP": "American Electric Power",
    "ED": "Consolidated Edison", "XEL": "Xcel Energy", "KO": "Coca-Cola", "PEP": "PepsiCo",
    "PG": "Procter & Gamble", "CL": "Colgate-Palmolive", "GIS": "General Mills", "KMB": "Kimberly-Clark",
    "WMT": "Walmart", "COST": "Costco", "MCD": "McDonald's", "VZ": "Verizon", "T": "AT&T",
    # Health care
    "JNJ": "Johnson & Johnson", "MRK": "Merck", "PFE": "Pfizer", "ABBV": "AbbVie", "UNH": "UnitedHealth",
    "LLY": "Eli Lilly", "TMO": "Thermo Fisher", "ABT": "Abbott Labs",
    # Financials
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "GS": "Goldman Sachs", "MS": "Morgan Stanley",
    "V": "Visa", "MA": "Mastercard", "BRK-B": "Berkshire Hathaway", "AXP": "American Express",
    # Industrials / energy / materials
    "CAT": "Caterpillar", "DE": "Deere", "HON": "Honeywell", "UPS": "UPS", "GE": "GE Aerospace",
    "LMT": "Lockheed Martin", "XOM": "ExxonMobil", "CVX": "Chevron", "LIN": "Linde",
    # Consumer discretionary
    "HD": "Home Depot", "NKE": "Nike", "SBUX": "Starbucks", "AMZN": "Amazon", "TSLA": "Tesla",
}

st.set_page_config(page_title="Low-Beta Portfolio Optimizer", layout="wide")
st.title("Low-Beta Portfolio Optimizer")
st.caption(
    "Finds the stock mix with the lowest market beta that still clears a minimum Sharpe ratio, "
    "then compares it to what you hold today."
)

# ---------------- How it works ----------------
with st.expander("How this app works (read me)", expanded=False):
    st.markdown(
        f"""
**The pipeline, step by step**

1. **Download prices.** Weekly adjusted closes from Yahoo Finance (`yfinance`) for every stock in the
   universe, any stocks you hold, and the market proxy (**{MARKET}**).
2. **Compute returns.** Simple weekly percentage returns.
3. **Estimate beta.** For each stock, a linear regression (`scikit-learn`) of the stock's weekly returns on
   {MARKET}'s weekly returns. The slope is the stock's **beta**: how much it tends to move per 1% market move.
4. **Expected return.** Two options in the sidebar:
   - **CAPM:** `E[r] = risk-free + beta x MRP`. This is a forward-looking, model-based estimate.
   - **Historical:** the average past return, annualized. Noisy, but reflects what actually happened.
5. **Risk.** Annualized covariance matrix of weekly returns (x {PERIODS_PER_YEAR}). Portfolio volatility is
   `sqrt(w' Σ w)`.
6. **Sharpe ratio.** `(portfolio expected return - risk-free) / portfolio volatility`.
7. **Optimize.** Minimize portfolio beta (`Σ wᵢ·βᵢ`) subject to:
   - weights sum to 100%, no shorting
   - no position above the max weight
   - Sharpe ratio >= the slider value
   - any position that ends up below the minimum weight is dropped and the optimizer re-runs on the rest,
     so every surviving position is between the min and max.
   The solver is `scipy.optimize.minimize` (SLSQP). A `linprog` solve of the pure minimum-beta problem
   (no Sharpe constraint) is used as the starting point.
8. **Compare to your holdings.** Your current portfolio's beta, volatility and Sharpe are computed the same
   way, and a trade list shows what to buy/sell to reach the target weights.

**Why the Sharpe slider matters:** the lowest-beta stocks are not always the best risk-adjusted performers.
Raising the minimum Sharpe forces the optimizer to give up some beta reduction in exchange for better
return per unit of risk. If the slider is set higher than anything achievable under your position limits,
the app tells you the maximum Sharpe it *can* reach.

**Where the assumptions come from**

| Assumption | Value | Source / reasoning |
|---|---|---|
| Market risk premium | {MRP:.1%} (sidebar) | Judgment call within the typical 4.5-6% range found in academic and practitioner surveys. Not estimated from data here. |
| Market proxy | {MARKET} | Liquid ETF tracking the S&P 500; standard stand-in for "the market". |
| Risk-free rate | 13-week T-bill (`^IRX`) | Pulled live from Yahoo Finance; falls back to {DEFAULT_RF:.0%} if unavailable. |
| Position limits | {MIN_WEIGHT:.0%} to {MAX_WEIGHT:.0%} (sidebar) | Diversification guardrails so the optimizer can't put everything in one name. |
| Betas, vol, returns | Computed | From the weekly price history window you choose. |
| Universe | Hand-picked ~50 large caps | Skewed to defensive sectors, which tend to have low beta. |

**Caveats:** betas and returns are estimated from the past and will change. CAPM returns assume the market
prices risk the way the model says. This is an educational tool, not financial advice.
"""
    )

# ---------------- Sidebar inputs ----------------
st.sidebar.header("Your portfolio")
st.sidebar.caption("One holding per line: `TICKER shares` (comma or space separated). Example: `KO 25`")
holdings_text = st.sidebar.text_area(
    "Stocks you currently hold",
    value="",
    height=150,
    placeholder="KO 25\nAAPL 10\nJNJ 15",
)
include_holdings = st.sidebar.checkbox(
    "Let the optimizer also pick my holdings that aren't in the universe",
    value=True,
    help="If checked, stocks you hold are added to the candidate pool.",
)
default_value = st.sidebar.number_input(
    "Portfolio value if you hold nothing ($)", min_value=1000, value=10000, step=1000,
    help="Used to size dollar allocations when no holdings are entered.",
)

st.sidebar.header("Optimization settings")
target_sharpe = st.sidebar.slider(
    "Minimum Sharpe ratio", min_value=0.0, max_value=2.0, value=0.4, step=0.05,
    help="The optimized portfolio must have at least this Sharpe ratio. Higher = better risk-adjusted "
         "return, but usually a higher beta.",
)
return_model = st.sidebar.radio(
    "Expected return model", ["CAPM (rf + beta x MRP)", "Historical average"],
    help="CAPM is forward-looking and model-based. Historical uses past average returns.",
)
mrp = st.sidebar.slider("Market risk premium", 0.03, 0.08, MRP, 0.005, format="%.3f",
                        help="Used by the CAPM return model only.")
years = st.sidebar.slider("History window (years)", 1, 10, 5)
min_w = st.sidebar.slider("Min position size", 0.0, 0.10, MIN_WEIGHT, 0.005, format="%.3f")
max_w = st.sidebar.slider("Max position size", 0.05, 0.50, MAX_WEIGHT, 0.01, format="%.2f")
run = st.sidebar.button("Run optimization", type="primary")


# ---------------- Helpers ----------------
def parse_holdings(text):
    out = {}
    for line in text.strip().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            out[parts[0].upper().replace(".", "-")] = out.get(parts[0].upper().replace(".", "-"), 0.0) + float(parts[1])
        except ValueError:
            continue
    return out


@st.cache_data(show_spinner=False, ttl=3600)
def load_prices(tickers, years):
    data = yf.download(list(tickers), period=f"{years}y", interval="1wk",
                       auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data


@st.cache_data(show_spinner=False, ttl=3600)
def load_rf():
    try:
        irx = yf.download("^IRX", period="5d", progress=False, auto_adjust=True)["Close"].dropna()
        return float(np.squeeze(irx.iloc[-1])) / 100.0
    except Exception:
        return DEFAULT_RF


def estimate_betas(returns, market_col):
    x = returns[market_col].values.reshape(-1, 1)
    betas = {}
    for t in returns.columns:
        if t == market_col:
            continue
        betas[t] = LinearRegression().fit(x, returns[t].values).coef_[0]
    return pd.Series(betas)


def port_stats(w, mu, cov, betas, rf):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    return {"beta": float(w @ betas), "ret": ret, "vol": vol,
            "sharpe": (ret - rf) / vol if vol > 0 else np.nan}


def max_sharpe_weights(mu, cov, rf, hi):
    n = len(mu)
    res = minimize(lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w), np.ones(n) / n,
                   bounds=[(0, hi)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP")
    return res.x


def optimize_min_beta(mu, cov, betas, rf, target, lo, hi):
    """Minimize portfolio beta s.t. sum(w)=1, Sharpe >= target, lo <= w <= hi (dropping tiny positions)."""
    n = len(mu)
    if n * hi < 1:
        return None, "Max position size is too small for the number of candidate stocks."

    def sharpe_con(w):
        return (w @ mu - rf) / np.sqrt(w @ cov @ w) - target

    def solve(idx, lower):
        m = len(idx)
        b, mu_s, cov_s = betas[idx], mu[idx], cov[np.ix_(idx, idx)]
        lp = linprog(b, A_eq=np.ones((1, m)), b_eq=[1], bounds=[(lower, hi)] * m, method="highs")
        x0 = lp.x if lp.success else np.ones(m) / m
        res = minimize(lambda w: w @ b, x0, bounds=[(lower, hi)] * m, method="SLSQP",
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1},
                                    {"type": "ineq",
                                     "fun": lambda w: (w @ mu_s - rf) / np.sqrt(w @ cov_s @ w) - target}])
        return res

    all_idx = np.arange(n)
    res = solve(all_idx, 0.0)
    if not res.success:
        return None, "Target Sharpe not reachable under these position limits."
    w = np.zeros(n)
    w[all_idx] = res.x

    # Drop positions below the minimum weight and re-solve with a true [lo, hi] band
    keep = np.where(w >= max(lo, 1e-4))[0]
    if lo > 0 and len(keep) * hi >= 1 and len(keep) * lo <= 1:
        res2 = solve(keep, lo)
        if res2.success:
            w = np.zeros(n)
            w[keep] = res2.x
    w[w < 1e-6] = 0
    return w / w.sum(), None


# ---------------- Main ----------------
holdings = parse_holdings(holdings_text)
if not run:
    st.info("Set your holdings (optional), pick a minimum Sharpe ratio, then click **Run optimization** in the sidebar.")
    st.stop()

candidates = list(UNIVERSE.keys())
if include_holdings:
    candidates += [t for t in holdings if t not in candidates]
download_list = sorted(set(candidates) | set(holdings) | {MARKET})

with st.spinner("Downloading prices..."):
    prices = load_prices(tuple(download_list), years)
    rf = load_rf()

prices = prices.dropna(axis=1, thresh=int(0.8 * len(prices))).dropna()
if MARKET not in prices.columns or len(prices) < 30:
    st.error("Not enough price data came back. Try again or shorten the history window.")
    st.stop()

missing = [t for t in holdings if t not in prices.columns]
if missing:
    st.warning(f"No usable price data for: {', '.join(missing)}. They were ignored.")
holdings = {t: s for t, s in holdings.items() if t in prices.columns}

returns = prices.pct_change().dropna()
betas_all = estimate_betas(returns, MARKET)
stock_cols = list(betas_all.index)
cov_all = returns[stock_cols].cov() * PERIODS_PER_YEAR

if return_model.startswith("CAPM"):
    mu_all = rf + betas_all * mrp
else:
    mu_all = returns[stock_cols].mean() * PERIODS_PER_YEAR

pool = [t for t in candidates if t in stock_cols]
mu = mu_all[pool].values
cov = cov_all.loc[pool, pool].values
betas = betas_all[pool].values

c1, c2, c3 = st.columns(3)
c1.metric("Risk-free rate (13wk T-bill)", f"{rf:.2%}")
c2.metric("Candidate stocks", len(pool))
c3.metric("Weeks of data", len(returns))

w_opt, err = optimize_min_beta(mu, cov, betas, rf, target_sharpe, min_w, max_w)
if w_opt is None:
    w_ms = max_sharpe_weights(mu, cov, rf, max_w)
    best = port_stats(w_ms, mu, cov, betas, rf)["sharpe"]
    st.error(f"{err} The highest Sharpe achievable with a {max_w:.0%} cap is about {best:.2f}. "
             f"Lower the slider or raise the max position size.")
    st.stop()

opt = port_stats(w_opt, mu, cov, betas, rf)

# Current portfolio
cur = None
if holdings:
    last_px = prices.iloc[-1]
    values = pd.Series({t: holdings[t] * last_px[t] for t in holdings})
    total_value = values.sum()
    w_cur_s = values / total_value
    idx = list(w_cur_s.index)
    w_cur = w_cur_s.values
    cur = port_stats(w_cur, mu_all[idx].values, cov_all.loc[idx, idx].values, betas_all[idx].values, rf)
else:
    total_value = float(default_value)

# ---------------- Results ----------------
st.subheader("Results")
st.caption("All figures are annualized estimates from the history window and return model you selected.")
rows = [("Optimized", opt)] + ([("Your current", cur)] if cur else [])
cols = st.columns(len(rows))
for col, (name, s) in zip(cols, rows):
    col.markdown(f"**{name} portfolio**")
    col.metric("Beta", f"{s['beta']:.2f}")
    col.metric("Expected return", f"{s['ret']:.2%}")
    col.metric("Volatility", f"{s['vol']:.2%}")
    col.metric("Sharpe ratio", f"{s['sharpe']:.2f}")

weights = pd.Series(w_opt, index=pool)
weights = weights[weights > 0].sort_values(ascending=False)
table = pd.DataFrame({
    "Company": [UNIVERSE.get(t, "(your holding)") for t in weights.index],
    "Target weight": weights.values,
    "Beta": betas_all[weights.index].values,
    "Expected return": mu_all[weights.index].values,
    "Target $": weights.values * total_value,
}, index=weights.index)

left, right = st.columns([1, 1])
with left:
    st.markdown("**Target allocation**")
    st.dataframe(table.style.format({"Target weight": "{:.1%}", "Beta": "{:.2f}",
                                     "Expected return": "{:.1%}", "Target $": "${:,.0f}"}),
                 use_container_width=True)
with right:
    fig = px.pie(table.reset_index(), names="index", values="Target weight", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

if holdings:
    st.subheader("Rebalance plan")
    st.caption("Dollar changes needed to move from your current holdings to the target weights "
               "(using the latest weekly close; ignores taxes and fees).")
    all_t = sorted(set(weights.index) | set(holdings))
    cur_val = pd.Series({t: values.get(t, 0.0) for t in all_t})
    tgt_val = pd.Series({t: weights.get(t, 0.0) * total_value for t in all_t})
    plan = pd.DataFrame({
        "Current $": cur_val, "Target $": tgt_val, "Trade $": tgt_val - cur_val,
        "Current weight": cur_val / total_value, "Target weight": tgt_val / total_value,
    })
    plan["Action"] = np.where(plan["Trade $"] > 1, "Buy", np.where(plan["Trade $"] < -1, "Sell", "Hold"))
    plan["Approx. shares"] = (plan["Trade $"] / last_px.reindex(plan.index)).round(1)
    st.dataframe(plan.sort_values("Trade $").style.format({
        "Current $": "${:,.0f}", "Target $": "${:,.0f}", "Trade $": "${:+,.0f}",
        "Current weight": "{:.1%}", "Target weight": "{:.1%}"}), use_container_width=True)

st.subheader("Betas in the candidate pool")
beta_df = pd.DataFrame({"Beta": betas_all[pool]}).sort_values("Beta")
beta_df["Held"] = ["Yes" if t in holdings else "" for t in beta_df.index]
st.plotly_chart(px.bar(beta_df.reset_index(), x="index", y="Beta", color="Held",
                       labels={"index": "Ticker"}), use_container_width=True)
st.caption("Beta < 1 means the stock has historically moved less than the market. Estimated by regressing "
           f"weekly stock returns on {MARKET}.")
