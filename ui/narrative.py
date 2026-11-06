"""Deterministic text built only from scored numbers (no LLM, so no summary can contradict the figures)."""
from __future__ import annotations

import math

import pandas as pd

from etl.states import name_of
from scoring.data import registry, spec
from scoring.score import reasons as score_reasons
from ui.state import NEED_LABELS


def indian(n: float, decimals: int = 0) -> str:
    """12345678 -> '1,23,45,678' (Indian digit grouping)."""
    neg = n < 0
    s = f"{abs(n):.{decimals}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = ",".join([head[max(0, i - 2):i] for i in range(len(head), 0, -2)][::-1])
        whole = f"{head},{tail}"
    return ("-" if neg else "") + whole + (f".{frac}" if frac else "")


def fmt(v, unit: str = "") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    u = unit.lower()
    if u.startswith("category:"):  # ordinal government category, e.g. 'category: 1=Aspirer, 2=Fast Mover, ...'
        labels = dict(x.strip().split("=", 1) for x in unit.split(":", 1)[1].split(","))
        lo, hi = math.floor(v), math.ceil(v)
        if lo != hi:  # a median between two categories
            return f"between {labels.get(str(lo), lo)} and {labels.get(str(hi), hi)}"
        return labels.get(str(lo), f"{v:.0f}")
    if u.startswith("of 25"):
        return (f"{v:.0f}" if v == int(v) else f"{v:.1f}") + " of 25"
    if u.startswith("score 0-100"):
        return f"{v:.1f} / 100"
    if u.startswith("rs crore"):
        return f"₹{indian(v)} cr"
    if u.startswith("rs") and "day" in u:
        return f"₹{indian(v)}/day"
    if u.startswith("rs"):
        return f"₹{indian(v)}"
    if "%" in u:
        return f"{v:.1f}%" if abs(v) >= 1 or v == 0 else f"{v:.2f}%"
    if u in {"persons", "number"}:
        return indian(v)
    if "usd million" in u:
        return f"US${indian(v)} mn"
    if abs(v) >= 1000:
        return indian(v)
    return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"


SOURCE_TAGS = {"dpiit_brap_2024": "DPIIT BRAP 2024", "leads_2025": "LEADS 2025", "niti_epi_2024": "NITI EPI 2024",
               "startup_ranking_5": "DPIIT Startup Ranking 5.0", "aishe_2023_24": "AISHE 2023-24",
               "plfs_annual_2023_24": "PLFS 2023-24 T8"}


def source_tag(row) -> str:
    f = str(row.source_file).split(";")[0]
    tbl = ""
    if "rbi_hsis" in f:
        tbl = "RBI " + f.split("/")[-1].split("_")[0]
    elif "hces" in f:
        tbl = "HCES 2023-24"
    elif "trai" in f:
        tbl = "TRAI PIR"
    elif "registry" in f:
        tbl = "Forecast model"
    else:
        tbl = next((v for k, v in SOURCE_TAGS.items() if k in f), "")
    if row.quality == "derived":
        tbl = (tbl + ", derived") if tbl else "derived"
    return f"{tbl} · {row.period}".strip(" ·")


def lower_first(label: str) -> str:
    """'Urban household spending (MPCE)' -> 'urban household spending (MPCE)'; keeps acronyms such as 'GSDP'."""
    return label if label[:2].isupper() else label[:1].lower() + label[1:]


def national_note(x: dict, positive: bool) -> str:
    """How far the state sits from the 30-state national average, in standard deviations (signed: + = favourable)."""
    z = x.get("z_national")
    if z is None or z != z:
        return ""
    if abs(z) < 0.05:
        return "at the national average"
    s = f"{z:+.1f} SD vs national average"
    if positive and z < 0:
        s += "; an edge over these peers, not nationally"
    elif not positive and z > 0:
        s += "; still above the national average"
    return s


def headline(x: dict) -> str:
    h = x.get("headline") or x.get("meaning") or x["label"]
    return h[:1].upper() + h[1:]


def reason_line(x: dict, positive: bool = True) -> str:
    """One strength / watch-out: an executive headline, then the evidence (vs the other compared states and vs the
    national benchmark) and why it matters for this industry."""
    tags = []
    if x["priority"]:
        tags.append(f"your priority: {x['dimension_label'].lower()}")
    tags += [f"your need: {NEED_LABELS.get(n, n).lower()}" for n in x["needs"]]
    nat = national_note(x, positive)
    return (f"{headline(x)} — {lower_first(x['label'])} {fmt(x['value'], x['unit'])} vs "
            f"{fmt(x['peer_median'], x['unit'])} peer median"
            + (f", {nat}" if nat else "") + f" ({source_tag(_Row(x))}). {x['why']}"
            + (f" [{'; '.join(tags)}]" if tags else "")).replace(" . ", " ").strip()


class _Row:
    def __init__(self, x):
        self.source_file, self.quality, self.period = x["source_file"], x["quality"], x["period"]


NO_POS = "No indicator in this profile is materially stronger than the other compared states."
NO_NEG = "No indicator in this profile is materially weaker than the other compared states."


def reasons(res, sid: str, k: int = 3) -> tuple[list[str], list[str]]:
    pos, neg = score_reasons(res, sid, k)
    return [reason_line(x, True) for x in pos] or [NO_POS], [reason_line(x, False) for x in neg] or [NO_NEG]


def managerial(res) -> str:
    """Plain-language reading of the ranking, built from the same relative reasons as the state cards."""
    rk = res.ranking
    parts = []
    for i, r in enumerate(rk.head(3).itertuples()):
        pos, neg = score_reasons(res, r.state_id, 1)
        lead = ["leads", "follows", "is third"][i]
        txt = f"{r.state} {lead} ({r.score:.1f})"
        if pos:
            txt += f". Biggest edge: {lower_first(headline(pos[0]))} ({pos[0]['dimension_label'].lower()})"
        if neg:
            txt += f"{';' if pos else '.'} main watch-out: {lower_first(headline(neg[0]))} ({neg[0]['dimension_label'].lower()})"
        parts.append(txt + ".")
    gap = rk.score.iloc[0] - rk.score.iloc[1] if len(rk) > 1 else float("nan")
    if gap == gap and gap < 2:
        parts.append(f"The top two are {gap:.1f} points apart, which is within what a modest change of weights "
                     "can reverse: treat them as a tie and compare on site visits.")
    parts.append("State averages hide corridor- and district-level differences; use this list to decide where to "
                 "look next, not where to sign.")
    return " ".join(parts)


def rupees_crore(v: float) -> str:
    return f"₹{indian(v, 1 if v < 100 else 0)} cr"


def capital_summary(fit: dict) -> str:
    """What the investment entered does to the ranking (band + design-default tilt)."""
    b = fit["band"]
    return (f"₹{fit['capital']:g} crore puts this in the {b['label'].replace('Rs ', '₹')} band. {b['fit']} "
            "Band edges and tilts are design defaults, not statutory or incentive thresholds.")


def capital_line(row) -> str:
    """Per-state capital fit from ASI: scale vs the state's average registered factory and capital productivity."""
    r = row.scale_ratio
    scale = ("larger than" if r >= 1.25 else "close to" if r >= 0.75 else "smaller than")
    return (f"Your ₹{row.plant_cr * row.scale_ratio:g} cr is {r:.1f}x the average registered factory here "
            f"({rupees_crore(row.plant_cr)}), {scale} the typical plant. Factories here add ₹{row.gva_per_100:.0f} "
            f"of GVA a year per ₹100 of fixed capital: about {rupees_crore(row.implied_gva_cr)} a year at that "
            f"average (ASI {row.period}; GVA is not profit).")


def model_explanation(industry: str) -> tuple[str, bool]:
    s = spec()["industries"][industry]
    target = s.get("model_target")
    reg = registry().get(target or industry, {})
    if not target or reg.get("status") != "validated":
        why = reg.get("reason") or s.get("target_note", "")
        return (f"Ranking only. {why} The outlook dimension therefore uses historical growth momentum, labelled as "
                "such, and no forecast is shown."), False
    vm, hm = reg["validation_mae"], reg["holdout_mae"]
    return (f"Next-year forecast of {reg['target']}: {reg['selected_spec'].split('|')[0]} on lagged growth, trained "
            f"with expanding-window CV ({reg['trained_on']}). Validation MAE {vm['model']:.1f} pp vs "
            f"{vm[reg['best_baseline']]:.1f} pp for the {reg['best_baseline'].replace('mean3', '3-year mean')} "
            f"baseline; untouched FY 2024-25 hold-out MAE {hm['model']:.1f} pp vs "
            f"{min(hm['persistence'], hm['mean3']):.1f} pp for the best baseline. The bootstrap interval for the gain "
            "crosses zero and the 80% band is about ±12 pp, so read the forecast as a direction, not a promise. "
            "Drivers (last year's growth, 3-year mean, national growth) are model attributions, not causes."), True


def counter(res) -> str | None:
    return res.counter


def disclosure_lines(res) -> list[str]:
    out = []
    for r in res.ranking.itertuples():
        if r.disclosure:
            out.append(f"{r.state}: {r.disclosure}")
    for r in res.excluded.itertuples():
        out.append(f"{r.state} excluded: {r.reason}")
    return out


def state_name(sid: str) -> str:
    return name_of(sid)
