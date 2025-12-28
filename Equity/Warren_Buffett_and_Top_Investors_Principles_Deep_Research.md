# Warren Buffett and Top Investors: Deep Principles for US Equity Selection (ML-Ready)

Date: 2025-12-27

Purpose: Compile the core investing principles of Warren Buffett and other top investors, translate them into measurable features and screening rules, and outline risk/validation practices suitable for training a model to identify superior US equities.

---

## 1. Core Philosophy Overview
- Value investing: buy quality businesses below intrinsic value; require a margin of safety; be patient and long-term oriented. [Sources: 2]
- Quality first: durable competitive advantages (“economic moats”), high returns on capital, strong free cash flow, prudent leverage, and capable, shareholder-aligned management. [Sources: 2, 3]
- Circle of competence: invest only where you understand the economics; avoid the herd and emotion-driven decisions. [Sources: 3]
- Diversify intelligently and manage risk across regimes; avoid over-optimization; let balance and compounding work over time. [Sources: 4]

References: see section 9.

---

## 2. Warren Buffett (Value, Quality, Intrinsic Value)
- Focus: High-quality businesses with durable moats, predictable cash flows, high ROIC, and competent, honest management. Buy below intrinsic value and hold long-term.
- Practice: Uses simple but disciplined valuation (owner earnings/FCF, conservative assumptions), patience for price, and aversion to excessive debt.
- ML feature translation:
  - Profitability: ROIC ≥ 12%; ROA ≥ 6%; Gross margin stable or rising 3y.
  - Cash generation: FCF margin ≥ 8%; FCF growth positive in ≥ 3 of last 5 years.
  - Capital discipline: Net debt/EBITDA ≤ 2.5; Interest coverage ≥ 6x.
  - Quality: Low accruals; stable margins; low earnings volatility; low share count dilution.
  - Valuation guardrails: FCF yield ≥ 5%; EV/EBIT ≤ sector median; P/E vs normalized earnings ≤ 18 when quality metrics are strong.
- Rationale: Emphasizes margin of safety, cash economics over accounting, and durable compounding. [Sources: 1, 2]

---

## 3. Benjamin Graham (Margin of Safety, Mr. Market)
- Focus: Buy equities materially below conservative intrinsic estimates; exploit market overreactions; prefer simple, defensive criteria for non-experts.
- Classic thresholds: Historically favored deep discounts (e.g., ≤ 2/3 liquidation value), low P/B, and ample safety margins. [Sources: 2]
- ML feature translation:
  - Valuation: P/B ≤ 1.2 (non-financials) with tangible equity positive; P/E ≤ 12 on normalized earnings.
  - Safety: Net current asset value (NCAV) positive; Debt/equity ≤ 0.5; Current ratio ≥ 1.5.
  - Quality filters: No recent recurring “extraordinary” losses; footnotes clarity (proxy via 10-K readability and restatement flags).
- Rationale: Systematizes margin of safety to reduce downside risk amid uncertainty. [Sources: 2]

---

## 4. Charlie Munger (Mental Models, Quality + Patience)
- Focus: Latticework of mental models; circle of competence; incentives; second-order thinking; prefer outstanding businesses at fair prices vs fair businesses at outstanding prices.
- ML feature translation:
  - Moat proxies: Persistent ROIC ≥ WACC + 8%; stable market share; high switching costs (proxy via low churn/contract terms if available).
  - Incentives: Insider ownership ≥ 5%; insider buying patterns positive; incentive comp tied to ROIC/FCF (proxy via text features in proxy statements).
  - Simplicity: Avoid overly complex, multi-segment conglomerates where segment data is opaque (proxy via segment reporting consistency).
- Rationale: Better outcomes from quality, aligned incentives, and staying within competence. [Sources: 3]

---

## 5. Peter Lynch (GARP, Practical Growth at Reasonable Price)
- Focus: Invest in what you know; classify companies (stalwarts, fast growers, turnarounds); prefer growth at reasonable price (PEG ≈ 1.0). [Sources: 5]
- ML feature translation:
  - Growth quality: Revenue CAGR 3–5y ≥ 10%; EPS CAGR 3–5y ≥ 12% with minimal dilution.
  - Valuation-growth balance: PEG (forward) ≤ 1.0 preferred; 1.0–1.5 acceptable if moat/quality strong. [Sources: 5]
  - Category tags: Stalwarts (large, steady ROIC), fast growers (smaller, high growth but positive FCF), cyclicals (use cycle-aware features), turnarounds (limit exposure, require improving FCF and leverage downtrend).
- Rationale: Avoid overpaying for growth; seek sensible price relative to growth sustainability. [Sources: 5]

---

## 6. Ray Dalio (Regime Balance, Risk Parity, Macro Drivers)
- Focus: Portfolios should be resilient across growth/inflation regimes; balance risks, not just dollars; separate beta from alpha; diversify across uncorrelated exposures. [Sources: 4]
- ML implications (risk-aware selection overlays):
  - Regime mapping: Tag exposures (cyclical vs defensive; inflation-sensitive vs disinflation beneficiaries).
  - Position sizing: Risk-adjust position weights by volatility; cap exposure to single macro regime.
  - Drawdown controls: Reject equities with severe, unexplained drawdowns and deteriorating fundamentals.
- Rationale: Stocks’ performance ties to regime surprises; balanced exposure reduces tail risks. [Sources: 4]

---

## 7. Additional Voices (Risk, Global, Scuttlebutt)
- Howard Marks (Oaktree memos): Cycles, risk, second-level thinking; avoid overconfidence; seek favorable risk/reward. [Sources: 6]
- Seth Klarman: Margin of safety discipline; liquidity and downside protection. [Sources: 7]
- Philip Fisher: Scuttlebutt method; management quality; innovation; durable growth (apply qualitative features via text/NLP and proxy metrics). [Context]
- John Templeton: Global diversification; contrarian buying at maximum pessimism; valuation discipline (e.g., forward P/E bands). [Sources: 8]
- ML implications:
  - Risk cycle features: Credit spreads, volatility regime tags; avoid crowded trades.
  - Qualitative-to-quant proxies: Innovation intensity (R&D/sales), employee reviews sentiment, customer concentration risk.
  - Contrarian filter: Identify deep value in quality names after broad selloffs with fundamentals intact.

---

## 8. Unified Screening Rules (US Equities)
- Entry screen (quality + value):
  - ROIC ≥ 12%; FCF margin ≥ 8%; 3–5y FCF CAGR ≥ 5%.
  - Net debt/EBITDA ≤ 2.5; Interest coverage ≥ 6x; no persistent GAAP/Non-GAAP divergence.
  - Valuation: FCF yield ≥ 5% or EV/EBIT ≤ sector median; PEG (forward) ≤ 1.2 (≤ 1 preferred for GARP).
  - Governance: Insider ownership ≥ 3%; no major related-party red flags.
- Moat confirmation:
  - Gross margin stability; retention/churn proxies; pricing power (rising gross margin w/o cost-cutting-only drivers).
- Risk overlays:
  - Sector and regime diversification limits; cyclicals sized conservatively; exclude names with opaque footnotes or frequent restatements.
- Hold/exit rules:
  - Hold while core quality and cash generation persist and valuation remains within reason.
  - Exit on structural moat impairment, leverage creep beyond thresholds, or accounting quality deterioration.

---

## 9. ML Feature Map (Training-Ready)
- Profitability & Quality
  - `roic_pct`: trailing 12m and 3y average; threshold 12+.
  - `fcf_margin_pct`: trailing 12m; threshold 8+.
  - `gross_margin_trend`: 3y slope; non-negative preferred.
  - `accruals_ratio`: low values preferred (quality earnings).
- Cash Flow & Leverage
  - `fcf_cagr_5y`: threshold ≥ 5%.
  - `net_debt_to_ebitda`: ≤ 2.5.
  - `interest_coverage`: ≥ 6.
- Valuation
  - `fcf_yield_pct`: ≥ 5%.
  - `ev_to_ebit`: ≤ sector median.
  - `peg_forward`: ≤ 1.0 preferred; ≤ 1.2 acceptable. [Sources: 5]
- Governance & Incentives
  - `insider_ownership_pct`: ≥ 3%.
  - `insider_buying_score`: positive net buying 12m.
  - `comp_alignment_score`: mentions of ROIC/FCF in proxy (NLP).
- Moat Proxies
  - `roic_minus_wacc`: ≥ 8 pts.
  - `market_share_stability`: low variance; proxy via revenue rank consistency.
  - `switching_cost_proxy`: contract terms, churn, or customer retention proxies.
- Risk & Regime
  - `volatility_regime_tag`: macro/sector regime classification.
  - `drawdown_flag`: severe drawdown with deteriorating fundamentals = exclude.
  - `accounting_opacity_flag`: footnote complexity/restatement history.

Labels/Targets:
- Binary classification: “Investable Quality Value” (meets unified screen) vs not.
- Regression auxiliary: Expected excess return over sector median (12–36m) conditional on quality/valuation.

---

## 10. Data and Validation Strategy
- Data hygiene: Use SEC 10-K/10-Q, standardized fundamentals, restatement flags, insider transactions (Form 4), and consensus forward EPS for PEG.
- Train/test splits: Time-based (rolling windows) to avoid look-ahead bias; sector-aware stratification.
- Regime robustness: Validate across distinct macro regimes (e.g., high inflation vs disinflation) to ensure generalization.
- Sanity checks: Outlier handling, accruals quality, GAAP vs non-GAAP reconciliation, text-based governance signals.

---

## 11. Risk Management Integration
- Position sizing by risk: Volatility-normalized weights; cap regime concentration.
- Drawdown guardrails: Hard stops on structural fundamental deterioration; soft stops on price-only moves.
- Portfolio balance: Blend stalwarts and select fast growers; diversify cyclicals defensively.

---

## 12. Practical Notes and Limitations
- PEG sensitivity: Depends on forward EPS estimates—validate with multiple sources; treat cyclical and hyper-growth with caution. [Sources: 5]
- Footnote complexity: Use NLP heuristics, but human review is beneficial for edge cases.
- Moat inference: Proxy-based; combine multiple signals (ROIC spreads, margin trends, retention proxies).

---

## 13. Cross-Links to Existing Guide
- [Quick Reference Card](Warren_Buffett_Investment_Principles_Comprehensive_Guide.md#quick-reference-card)
- [Warren Buffett's Core Investment Philosophy](Warren_Buffett_Investment_Principles_Comprehensive_Guide.md#warren-buffetts-core-investment-philosophy)
- [Benjamin Graham - The Father of Value Investing](Warren_Buffett_Investment_Principles_Comprehensive_Guide.md#benjamin-graham)
- [Charlie Munger - Mental Models & Quality Focus](Warren_Buffett_Investment_Principles_Comprehensive_Guide.md#charlie-munger)
- [Peter Lynch - Growth at Reasonable Price](Warren_Buffett_Investment_Principles_Comprehensive_Guide.md#peter-lynch)
- [Ray Dalio - Principles & Diversification](Warren_Buffett_Investment_Principles_Comprehensive_Guide.md#ray-dalio)
- [Unified Quantitative Screening Model](Warren_Buffett_Investment_Principles_Comprehensive_Guide.md#unified-quantitative-screening-model)
- [Quantitative Metrics for Stock Analysis](Warren_Buffett_Investment_Principles_Comprehensive_Guide.md#quantitative-metrics)
- [Qualitative Factors & Business Moats](Warren_Buffett_Investment_Principles_Comprehensive_Guide.md#qualitative-factors)
- [Risk Management Principles](Warren_Buffett_Investment_Principles_Comprehensive_Guide.md#risk-management)
- [ML Model Training Considerations](Warren_Buffett_Investment_Principles_Comprehensive_Guide.md#ml-model-considerations)
- [Financial Glossary](Warren_Buffett_Investment_Principles_Comprehensive_Guide.md#financial-glossary)

---

## 14. Sources and References
1. Berkshire Hathaway Shareholder Letters Index (principles; intrinsic value, quality focus): https://www.berkshirehathaway.com/letters/letters.html
2. Investopedia – Value Investing overview (intrinsic value, margin of safety, contrarian stance): https://www.investopedia.com/terms/v/valueinvesting.asp
3. Farnam Street – Mental Models (circle of competence, margin of safety, incentives, second-order thinking): https://fs.blog/mental-models/
4. Bridgewater Associates – The All Weather Story (risk parity; regime balance): https://www.bridgewater.com/research-and-insights/the-all-weather-story
5. Investopedia – PEG Ratio (GARP heuristics; PEG ≈ 1.0): https://www.investopedia.com/terms/p/pegratio.asp
6. Oaktree – Howard Marks memos (cycles, risk, second-level thinking): https://www.oaktreecapital.com/insights/memo
7. GuruFocus – Seth Klarman “Margin of Safety” summary (discipline, downside focus): https://www.gurufocus.com/news/602576/seth-klarman-margin-of-safety-book-summary
8. Wikipedia – John Templeton (global diversification, contrarian rules, valuation discipline): https://en.wikipedia.org/wiki/John_Templeton

Notes: This document summarizes public sources and canonical ideas; it paraphrases concepts and links to originals, avoiding reproduction of copyrighted text.
