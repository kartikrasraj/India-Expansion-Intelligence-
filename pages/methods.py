import pandas as pd
import streamlit as st

from etl.common import REP, ROOT
from scoring.data import registry, spec
from ui import theme

S = spec()
st.title("How the index works")
theme.lead("What the model can and cannot say, the exact weights used, and where the numbers come from. "
           "Last refresh 3 Nov 2026 · scoring weights v4.")

st.subheader("Forecast models and targets")
reg = registry()
rows = []
for ind, prof in S["industries"].items():
    key = prof.get("model_target") or ind
    r = reg.get(key, {})
    rows.append({"Industry": prof["label"], "Status": r.get("status", "not trained").replace("_", " "),
                 "Target": r.get("target", "none (ranking only)"), "Reason": r.get("reason", prof.get("target_note", ""))})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.subheader("Default weights (v4)")
st.caption("Design defaults from the blueprint, tilted by industry. NOT learned optima. Indicators are scored as their "
           "percentile among 30 states (cost and risk reversed); missing indicators are dropped and weights "
           "renormalised, never filled; a state missing a mandatory dimension is excluded.")
w = pd.DataFrame({p["label"]: p["weights"] for p in S["industries"].values()}).T[list(S["dimensions"])]
w.columns = [S["dimensions"][c]["label"] for c in w.columns]
st.dataframe(w.astype(object).where(w.notna(), "—"), use_container_width=True)
st.caption("— = the industry has no such dimension: no auditable indicator exists for it, so its weight is spread over "
           "the others.")

st.subheader("Investment capital: scale bands")
st.caption("The investment you enter selects a band; the band multiplies some dimension weights before all weights are "
           "renormalised. Design defaults, not statutory MSME limits or state incentive thresholds.")
st.dataframe(pd.DataFrame([{"Band": b["label"].replace("Rs ", "₹"),
                            "Weight tilt": ", ".join(f"{S['dimensions'][d]['label']} x{m}" for d, m in b["tilt"].items())
                            or "none", "Rationale": b["fit"]} for b in S["capital"]["bands"]]),
             hide_index=True, use_container_width=True)
st.caption("For manufacturing, the results page also compares your investment with each state's average fixed capital "
           "per registered factory and its factory GVA per ₹100 of fixed capital (ASI 2023-24).")

st.subheader("How strengths and watch-outs are chosen")
st.caption("Each indicator's lever = (its percentile score minus the median of the other compared states) x its "
           "weight in the index. Strengths are the largest positive levers, watch-outs the largest negative ones; "
           "indicators in your priorities and operational needs are listed first. Each bullet also gives the state's "
           "distance from the 30-state national average in standard deviations.")

st.subheader("Indicators used by each industry, and why")
st.caption("Only indicators with a stated business link are scored; "
           "operational needs can add indicators (e.g. digital-first adds internet subscriptions). Strengths and "
           "watch-outs on the results page compare each state with the other states you chose.")
for prof_ in S["industries"].values():
    with st.expander(prof_["label"]):
        st.dataframe(pd.DataFrame([{"Dimension": S["dimensions"][d]["label"], "Indicator": S["indicators"][c]["label"],
                                    "Weight": wt, "Why it is in this profile": prof_.get("why", {}).get(c, "")}
                                   for d, m in prof_["indicators"].items() for c, wt in m.items()]),
                     hide_index=True, use_container_width=True)
with st.expander("Indicator definitions"):
    ind = pd.DataFrame([{"code": k, "indicator": v["label"], "unit": v["unit"],
                         "direction": "higher is better" if v["direction"] > 0 else "lower is better",
                         "caveat": v.get("proxy", "")} for k, v in S["indicators"].items()])
    st.dataframe(ind, hide_index=True, use_container_width=True)

st.subheader("Data sources")
reg_df = pd.read_csv(ROOT / "data/source_register.csv", dtype=str).fillna("")
st.dataframe(reg_df[["dataset_id", "publisher", "url", "local_path", "sha256", "acquisition_date", "status"]],
             hide_index=True, use_container_width=True,
             column_config={"url": st.column_config.LinkColumn("url"), "sha256": st.column_config.TextColumn(width="small")})
un = REP / "unavailable_sources.csv"
if un.exists():
    st.markdown("**Not available (never estimated)**")
    for r in pd.read_csv(un).itertuples():
        st.markdown(f"- **{r.source.replace('_', ' ')}**: {r.reason}")
theme.note("Secondary pages (GST collections, minimum wages, private salary surveys) are shown only where the number "
           "was found in a saved copy of the cited page, and are labelled; they never enter a score. Ookla speed data "
           "is CC BY-NC-SA 4.0 (non-commercial): a commercial deployment needs Ookla's permission.")

theme.footer()
