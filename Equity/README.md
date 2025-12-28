# Equity README

Date: 2025-12-27

This folder contains two complementary documents designed to support feature engineering and model training for identifying high-quality US equities.

## Files
- Warren_Buffett_Investment_Principles_Comprehensive_Guide.md
  - A structured, comprehensive guide that consolidates principles, quantitative/qualitative frameworks, risk management, and ML considerations.
- Warren_Buffett_and_Top_Investors_Principles_Deep_Research.md
  - A concise, ML-ready research synthesis translating core investor principles (Buffett, Graham, Munger, Lynch, Dalio, plus Marks/Klarman/Templeton) into specific screening rules, thresholds, and feature definitions, with citations.

## How to Use for Feature Engineering
1. Define feature set
   - Use Section 9 (ML Feature Map) in the deep research document as the canonical list (e.g., `roic_pct`, `fcf_margin_pct`, `net_debt_to_ebitda`, `peg_forward`, `insider_ownership_pct`).
   - Map each feature to data sources: SEC 10-K/10-Q, fundamentals providers, Form 4 (insider), consensus EPS for PEG.
2. Implement thresholds
   - Apply unified screening rules (Section 8) as labels or filters: e.g., ROIC ≥ 12, FCF yield ≥ 5%, net debt/EBITDA ≤ 2.5, PEG ≤ 1.2.
   - Use sector-relative valuation (EV/EBIT ≤ sector median) where appropriate.
3. Add qualitative proxies
   - Governance and incentives via proxy/NLP (e.g., comp alignment mentions, insider buying patterns).
   - Moat proxies via margin stability, ROIC-WACC spread, retention/churn signals.
4. Risk overlays
   - Tag regime sensitivity and diversify across growth/inflation exposures (see All Weather concepts).
   - Exclude names with accounting opacity or severe drawdown + deteriorating fundamentals.

## Training & Validation Tips
- Splits: Use time-based rolling windows; avoid look-ahead bias; stratify by sector.
- Targets: Binary label for “Investable Quality Value”; optional regression on excess return vs sector median.
- Robustness: Validate across macro regimes (inflation/disinflation, expansions/recessions).
- Sanity checks: Accruals quality, GAAP vs non-GAAP reconciliation, restatement flags.

## Cross-Navigation
- The deep research file includes direct links back to matching sections in the comprehensive guide for quick reference.
- The comprehensive guide’s Table of Contents links to the deep research file under “Deep Research (ML-Ready)”.

## Notes
- Citations point to canonical public sources; concepts are paraphrased to avoid reproducing copyrighted text.
- Adjust thresholds by sector/industry where warranted (banks, utilities, cyclicals).