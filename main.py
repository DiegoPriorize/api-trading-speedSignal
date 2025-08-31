from typing import List, Optional, Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field
from datetime import datetime
import math

app = FastAPI(
    title="Trade Signal API",
    version="1.0.0",
    description="Generates BUY/SELL/HOLD signals from OHLCV candles using EMA cross + RSI filter and suggests SL/TP via ATR."
)

# --------- Data models ---------
class StrategyParams(BaseModel):
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    rsi_buy: float = 30.0
    rsi_sell: float = 70.0
    atr_period: int = 14
    atr_tp_mult: float = 2.0
    atr_sl_mult: float = 1.0
    min_volume: float = 0.0
    require_rsi_filter: bool = False  # if True, apply RSI thresholds as a filter

class SignalResponse(BaseModel):
    signal: Literal["BUY", "SELL", "HOLD"]
    timestamp: int
    iso_time: str
    price: float
    reason: str
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    precision: Optional[float] = None

class SignalRequest(BaseModel):
    T: List[str]  # Timestamps
    O: List[str]  # Preço de Abertura (como strings)
    C: List[str]  # Preço de Fechamento (como strings)
    H: List[str]  # Máxima do Período (como strings)
    I: List[str]  # Mínimo do Período (como strings)
    V: List[str]  # Volume (como strings)
    params: Optional[StrategyParams] = None


# --------- Indicator helpers (pure Python, no pandas) ---------
def ema(values: List[float], period: int) -> List[Optional[float]]:
    if period <= 0:
        raise ValueError("period must be > 0")
    k = 2 / (period + 1)
    out: List[Optional[float]] = [None] * len(values)
    if len(values) == 0:
        return out
    if len(values) >= period:
        seed = sum(values[:period]) / period
        out[period-1] = seed
        prev = seed
        for i in range(period, len(values)):
            prev = values[i] * k + prev * (1 - k)
            out[i] = prev
    else:
        prev = values[0]
        out[0] = prev
        for i in range(1, len(values)):
            prev = values[i] * k + prev * (1 - k)
            out[i] = prev
    return out

def rsi(values: List[float], period: int) -> List[Optional[float]]:
    if period <= 0:
        raise ValueError("period must be > 0")
    rsi_vals: List[Optional[float]] = [None] * len(values)
    gains = [0.0] * len(values)
    losses = [0.0] * len(values)
    for i in range(1, len(values)):
        chg = values[i] - values[i-1]
        gains[i] = max(chg, 0.0)
        losses[i] = max(-chg, 0.0)
    if len(values) > period:
        avg_gain = sum(gains[1:period+1]) / period
        avg_loss = sum(losses[1:period+1]) / period
        rs = (avg_gain / avg_loss) if avg_loss != 0 else math.inf
        rsi_vals[period] = 100 - (100 / (1 + rs))
    for i in range(period+1, len(values)):
        avg_gain = (avg_gain * (period-1) + gains[i]) / period
        avg_loss = (avg_loss * (period-1) + losses[i]) / period
        rs = (avg_gain / avg_loss) if avg_loss != 0 else math.inf
        rsi_vals[i] = 100 - (100 / (1 + rs))
    return rsi_vals

def atr(highs: List[float], lows: List[float], closes: List[float], period: int) -> List[Optional[float]]:
    trs: List[float] = [0.0] * len(highs)
    for i in range(len(highs)):
        if i == 0:
            trs[i] = highs[i] - lows[i]
        else:
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            trs[i] = max(tr1, tr2, tr3)
    out: List[Optional[float]] = [None] * len(highs)
    if len(trs) >= period:
        seed = sum(trs[1:period+1]) / period
        out[period] = seed
        prev = seed
        for i in range(period+1, len(trs)):
            prev = (prev * (period - 1) + trs[i]) / period
            out[i] = prev
    return out

def calcular_precision(candles: SignalRequest, signal: str, price: float, forecast_time: int) -> float:
    # Certifique-se de que os timestamps são inteiros
    timestamps_futuros = [int(c) for c in candles.T if int(c) > forecast_time]
    
    precisao = 0.0
    if signal == "BUY":
        for ts in timestamps_futuros:
            future_price = next((float(candles.C[i]) for i, t in enumerate(candles.T) if int(candles.T[i]) == ts), None)
            if future_price and future_price > price:
                precisao += 1
    elif signal == "SELL":
        for ts in timestamps_futuros:
            future_price = next((float(candles.C[i]) for i, t in enumerate(candles.T) if int(candles.T[i]) == ts), None)
            if future_price and future_price < price:
                precisao += 1
    return precisao / len(timestamps_futuros) * 100 if timestamps_futuros else 0.0

def generate_signal(candles: SignalRequest, params: StrategyParams) -> SignalResponse:
    # Convertendo os valores de strings para float
    closes = [float(c) for c in candles.C]
    highs = [float(h) for h in candles.H]
    lows = [float(i) for i in candles.I]
    volumes = [float(v) for v in candles.V]

    # Convertendo os timestamps de strings para inteiros
    timestamps = [int(t) for t in candles.T]
    
    ema_fast = ema(closes, params.ema_fast)
    ema_slow = ema(closes, params.ema_slow)
    rsi_vals = rsi(closes, params.rsi_period)
    atr_vals = atr(highs, lows, closes, params.atr_period)
    
    i = len(timestamps) - 1  # Último candle
    if i < 1 or ema_fast[i] is None or ema_slow[i] is None:
        return SignalResponse(
            signal="HOLD",
            timestamp=timestamps[i],
            iso_time=datetime.utcfromtimestamp(timestamps[i]).isoformat() + "Z",
            price=closes[i],
            reason="Insufficient data for EMA crossover."
        )
    
    if volumes[i] < params.min_volume:
        return SignalResponse(
            signal="HOLD",
            timestamp=timestamps[i],
            iso_time=datetime.utcfromtimestamp(timestamps[i]).isoformat() + "Z",
            price=closes[i],
            reason=f"Volume {volumes[i]:.2f} < min_volume ({params.min_volume})."
        )
    
    diff_now = ema_fast[i] - ema_slow[i]
    diff_prev = (ema_fast[i-1] - ema_slow[i-1]) if ema_fast[i-1] is not None and ema_slow[i-1] is not None else None
    
    signal: Literal["BUY", "SELL", "HOLD"] = "HOLD"
    reason_parts = []
    
    if diff_prev is not None:
        if diff_prev <= 0 and diff_now > 0:
            signal = "BUY"
            reason_parts.append(f"EMA{params.ema_fast} cruzou acima da EMA{params.ema_slow}.")
        elif diff_prev >= 0 and diff_now < 0:
            signal = "SELL"
            reason_parts.append(f"EMA{params.ema_fast} cruzou abaixo da EMA{params.ema_slow}.")
        else:
            reason_parts.append("Sem crossover no candle atual.")
    
    if params.require_rsi_filter and rsi_vals[i] is not None:
        r = rsi_vals[i]
        if signal == "BUY" and r > params.rsi_sell:
            signal = "HOLD"
            reason_parts.append(f"RSI {r:.1f} acima de {params.rsi_sell} (filtro bloqueou compra).")
        elif signal == "SELL" and r < params.rsi_buy:
            signal = "HOLD"
            reason_parts.append(f"RSI {r:.1f} abaixo de {params.rsi_buy} (filtro bloqueou venda).")
        else:
            reason_parts.append(f"RSI {r:.1f} aceito no filtro.")
    
    tp = sl = None
    if atr_vals[i] is not None and atr_vals[i] > 0:
        a = atr_vals[i]
        price = closes[i]
        if signal == "BUY":
            sl = price - params.atr_sl_mult * a
            tp = price + params.atr_tp_mult * a
        elif signal == "SELL":
            sl = price + params.atr_sl_mult * a
            tp = price - params.atr_tp_mult * a
    
    reason = " ".join(reason_parts) if reason_parts else "Nenhum motivo específico."
    
    # Calcular precisão
    precision = calcular_precision(candles, signal, closes[i], timestamps[i])
    
    return SignalResponse(
        signal=signal,
        timestamp=timestamps[i],
        iso_time=datetime.utcfromtimestamp(timestamps[i]).isoformat() + "Z",
        price=closes[i],
        reason=reason,
        take_profit=tp,
        stop_loss=sl,
        precision=precision
    )

# --------- API endpoints ---------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/signal", response_model=SignalResponse)
def signal(req: SignalRequest):
    if not req.T or len(req.T) < 5:
        return SignalResponse(
            signal="HOLD",
            timestamp=req.T[-1],
            iso_time=datetime.utcfromtimestamp(req.T[-1]).isoformat(),
            price=float(req.C[-1]) if req.C else 0.0,
            reason="Envie pelo menos 5 candles para estabilidade."
        )
    params = req.params or StrategyParams()
    return generate_signal(req, params)
