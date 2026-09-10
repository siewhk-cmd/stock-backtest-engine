from __future__ import annotations
import operator
import pandas as pd

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq}


def compare(series: pd.Series, op: str, value):
    return OPS[op](series, value).fillna(False)


def crossed_above(a: pd.Series, b):
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
    return ((a > b) & (a.shift(1) <= b.shift(1))).fillna(False)


def crossed_below(a: pd.Series, b):
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
    return ((a < b) & (a.shift(1) >= b.shift(1))).fillna(False)


def combine_conditions(conditions: list[pd.Series], logic="ALL", index=None):
    if not conditions:
        return pd.Series(True, index=index)
    result = conditions[0].astype(bool).copy()
    for cond in conditions[1:]:
        result = (result | cond.astype(bool)) if logic == "ANY" else (result & cond.astype(bool))
    return result.fillna(False)


def describe_rules(config: dict) -> list[str]:
    lines = []
    t = config.get("technical", {})
    if t.get("stoch_enabled"):
        lines.append(f"Stoch({t['stoch_lookback']},{t['stoch_k']},{t['stoch_d']}) Slow %K {t['stoch_op']} {t['stoch_threshold']}")
    if t.get("rsi_enabled"):
        lines.append(f"RSI({t['rsi_period']}) {t['rsi_op']} {t['rsi_threshold']}")
    if t.get("atr_enabled"):
        lines.append(f"ATR({t['atr_period']})/Close % {t['atr_op']} {t['atr_threshold']}")
    pats = config.get("candlesticks", {}).get("patterns", [])
    if pats:
        lines.append(f"Candlesticks ({config.get('candlesticks',{}).get('logic','ANY')}): " + ", ".join(pats))
    return lines
