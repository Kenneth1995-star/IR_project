# src/Smart.py
"""
Contains functions to use with the different
smart schemes in query processor.
"""
import math
from typing import Callable, Dict


LOG = math.log10


# ---------- TF ----------
def tf_n(tf, **_):
    return tf


def tf_l(tf, **_):
    return 1.0 + LOG(tf)


def tf_a(tf, var=None, **_):
    # var is max tf
    return 0.5 + 0.5 * tf / var


def tf_b(tf, **_):
    return 1.0 if tf > 0 else 0.0


def tf_L(tf, var=None, **_):
    # var = average tf
    return (1.0 + LOG(tf)) / (1.0 + LOG(var))


TF_FUNCS: Dict[str, Callable] = {
    "n": tf_n,
    "l": tf_l,
    "a": tf_a,
    "b": tf_b,
    "L": tf_L,
}


# ---------- DF ----------
def df_n(df=None, N=None, **_):
    return 1.0


def df_t(df, N, **_):
    return LOG(N / df)


def df_p(df, N, **_):
    return max(0.0, LOG((N - df) / df))


DF_FUNCS: Dict[str, Callable] = {"n": df_n, "t": df_t, "p": df_p}


# ---------- Normalization variants ----------
def norm_n(first, **_):
    return 1.0


def norm_c(first, **_):
    denom = math.sqrt(sum(w * w for w in first)) or 1.0
    return 1.0 / denom


def norm_u(first, second=None, third=None, slope=0.2, **_):
    """
    second is unique term count, third is average unique count
    """
    # Pivoted-unique: queries and documents share avg_unique
    if second in (None, 0) or not third:
        return 1.0
    g = (1.0 - slope) + slope * (second / third)
    return 1.0 / g


def norm_b(first, second=None, alpha=0.5, **_):
    """
    second = byte length
    """
    if not second:
        return 1.0
    return 1.0 / (second**alpha)


NORM_FUNCS: Dict[str, Callable] = {
    "n": norm_n,
    "c": norm_c,
    "u": norm_u,
    "b": norm_b,
}
