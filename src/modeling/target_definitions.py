# -*- coding: utf-8 -*-
"""
Hedef (target) tanımları — tez için karşılaştırmalı deneyler.

Her mod ikili sınıflandırma üretir: 0 = aşağı / 1 = yukarı.
"""

# GUI ve kod için anahtarlar
TARGET_MODES = {
    "next_bar": "Hedef A: Sonraki mum (klasik)",
    "threshold_next": "Hedef B: Sonraki mum (eşikli hareket)",
    "horizon_k": "Hedef C: k mum sonra yön",
}


def describe_target(mode: str, threshold_pct: float, horizon: int) -> str:
    """Kısa açıklama (rapor / arayüz)."""
    if mode == "next_bar":
        return "close[t+1] > close[t]"
    if mode == "threshold_next":
        thr = threshold_pct * 100
        return (
            f"Sonraki getiri |r| >= %{thr:.2f}; "
            f"yukarı: r>0, aşağı: r<0 (küçük hareketler elenir)"
        )
    if mode == "horizon_k":
        return f"close[t+{horizon}] > close[t] (son {horizon} satır elenir)"
    return mode


def build_target(df, mode: str, *, threshold_pct: float = 0.002, horizon: int = 3):
    """
    df: OHLCV + türetilmiş sütunlar; 'close' zorunlu.

    Dönüş: (df_with_target, meta_dict)
    meta: satır sayısı, elenen örnek, açıklama
    """
    df = df.copy()
    n0 = len(df)
    c = df["close"].astype(float)

    if mode == "next_bar":
        df["target"] = (c.shift(-1) > c).astype(int)
        df = df.iloc[:-1].reset_index(drop=True)

    elif mode == "threshold_next":
        # Bir sonraki mum getirisi (oran)
        future_ret = (c.shift(-1) - c) / c
        mask = future_ret.notna() & (future_ret.abs() >= threshold_pct)
        df["target"] = (future_ret > 0).astype(int)
        df = df.loc[mask].reset_index(drop=True)

    elif mode == "horizon_k":
        h = int(horizon)
        if h < 1:
            raise ValueError("horizon en az 1 olmalı")
        df["target"] = (c.shift(-h) > c).astype(int)
        df = df.iloc[:-h].reset_index(drop=True)

    else:
        raise ValueError(f"Bilinmeyen mode: {mode!r}. İzin verilenler: {list(TARGET_MODES)}")

    n1 = len(df)
    meta = {
        "mode": mode,
        "rows_before": n0,
        "rows_after": n1,
        "rows_dropped": n0 - n1,
        "description": describe_target(mode, threshold_pct, horizon),
        "threshold_pct": threshold_pct,
        "horizon": int(horizon) if mode == "horizon_k" else None,
    }
    return df, meta
