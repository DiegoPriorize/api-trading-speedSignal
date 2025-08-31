from typing import List, Optional, Literal, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import math
import statistics

app = FastAPI(
    title="Trade Signal API",
    version="1.2.0",
    description=(
        "Generates BUY/SELL/HOLD signals from OHLCV candles using EMA cross + optional RSI filter, "
        "adds MACD, Bollinger Bands, Fibonacci levels, ATR-based SL/TP, 5 short-term forecasts, "
        "and provides a fallback suggestion (signal1) when the main signal is HOLD."
    ),
)

# --------- Data models ---------
class StrategyParams(BaseModel):
    # Núcleo (compatível com versão anterior)
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    rsi_buy: float = 30.0
    rsi_sell: float = 70.0
    atr_period: int = 14
    atr_tp_mult: float = 2.0
    atr_sl_mult: float = 1.0
    min_volume: float = 0.0
    require_rsi_filter: bool = False  # se True, aplica filtro de RSI

    # Novos indicadores
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    bb_period: int = 20
    bb_k: float = 2.0

    fib_lookback: int = 80  # janela para localizar swing high/low mais recente


class ForecastItem(BaseModel):
    operacao: Literal["COMPRA", "VENDA"]
    timestamp: int
    iso_time: str
    precisao: float
    motivo: str


class SignalResponse(BaseModel):
    # Sinal principal (pode ser HOLD)
    signal: Literal["BUY", "SELL", "HOLD"]
    timestamp: int
    iso_time: str
    price: float
    reason: str
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    precision: Optional[float] = None

    # NOVOS (fallback quando signal == HOLD)
    signal1: Optional[Literal["BUY", "SELL"]] = None
    signal1_precision: Optional[float] = None
    signal1_reason: Optional[str] = None

    # Indicadores e adicionais
    indicators: Optional[Dict[str, float]] = None
    fibonacci: Optional[Dict[str, float]] = None

    # 5 previsões curtas (<= 30 min)
    previsoes: Optional[List[ForecastItem]] = None


class SignalRequest(BaseModel):
    T: List[str]  # Timestamps (strings)
    O: List[str]  # Open (strings)
    C: List[str]  # Close (strings)
    H: List[str]  # High (strings)
    I: List[str]  # Low (strings)  OBS: campo "I" mantém compatibilidade
    V: List[str]  # Volume (strings)
    params: Optional[StrategyParams] = None


# --------- Indicator helpers (pure Python, no pandas) ---------
def ema(values: List[float], period: int) -> List[Optional[float]]:
    if period <= 0:
        raise ValueError("period must be > 0")
    out: List[Optional[float]] = [None] * len(values)
    if not values:
        return out
    k = 2 / (period + 1)
    if len(values) >= period:
        seed = sum(values[:period]) / period
        out[period - 1] = seed
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


def sma(values: List[float], period: int) -> List[Optional[float]]:
    if period <= 0:
        raise ValueError("period must be > 0")
    out: List[Optional[float]] = [None] * len(values)
    if len(values) < period:
        return out
    s = sum(values[:period])
    out[period - 1] = s / period
    for i in range(period, len(values)):
        s += values[i] - values[i - period]
        out[i] = s / period
    return out


def rolling_std(values: List[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if len(values) < period or period <= 1:
        return out
    window = values[:period]
    out[period - 1] = statistics.pstdev(window)
    for i in range(period, len(values)):
        window = values[i - period + 1 : i + 1]
        out[i] = statistics.pstdev(window)
    return out


def rsi(values: List[float], period: int) -> List[Optional[float]]:
    if period <= 0:
        raise ValueError("period must be > 0")
    rsi_vals: List[Optional[float]] = [None] * len(values)
    if len(values) <= period:
        return rsi_vals
    gains = [0.0] * len(values)
    losses = [0.0] * len(values)
    for i in range(1, len(values)):
        chg = values[i] - values[i - 1]
        gains[i] = max(chg, 0.0)
        losses[i] = max(-chg, 0.0)
    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period
    rs = (avg_gain / avg_loss) if avg_loss != 0 else math.inf
    rsi_vals[period] = 100 - (100 / (1 + rs))
    for i in range(period + 1, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
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
            tr2 = abs(highs[i] - closes[i - 1])
            tr3 = abs(lows[i] - closes[i - 1])
            trs[i] = max(tr1, tr2, tr3)
    out: List[Optional[float]] = [None] * len(highs)
    if len(trs) > period:
        seed = sum(trs[1 : period + 1]) / period
        out[period] = seed
        prev = seed
        for i in range(period + 1, len(trs)):
            prev = (prev * (period - 1) + trs[i]) / period
            out[i] = prev
    return out


def macd(values: List[float], fast: int, slow: int, signal_p: int):
    ema_fast_vals = ema(values, fast)
    ema_slow_vals = ema(values, slow)
    macd_line: List[Optional[float]] = [None] * len(values)
    for i in range(len(values)):
        if ema_fast_vals[i] is not None and ema_slow_vals[i] is not None:
            macd_line[i] = ema_fast_vals[i] - ema_slow_vals[i]
    signal_line = ema([v if v is not None else 0.0 for v in macd_line], signal_p)
    hist: List[Optional[float]] = [None] * len(values)
    for i in range(len(values)):
        if macd_line[i] is not None and signal_line[i] is not None:
            hist[i] = macd_line[i] - signal_line[i]
    return macd_line, signal_line, hist


def bollinger(values: List[float], period: int, k: float):
    mid = sma(values, period)
    sd = rolling_std(values, period)
    upper: List[Optional[float]] = [None] * len(values)
    lower: List[Optional[float]] = [None] * len(values)
    for i in range(len(values)):
        if mid[i] is not None and sd[i] is not None:
            upper[i] = mid[i] + k * sd[i]
            lower[i] = mid[i] - k * sd[i]
    return mid, upper, lower


def find_swing_high_low(highs: List[float], lows: List[float], lookback: int):
    """Encontra o último swing (máximo e mínimo) dentro da janela lookback mais recente."""
    n = len(highs)
    if n == 0:
        return None, None
    start = max(0, n - lookback)
    window_high = max(highs[start:])
    window_low = min(lows[start:])
    return window_high, window_low


def fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """Gera níveis de Fibonacci (retracement + algumas extensões)."""
    if high is None or low is None or high == low:
        return {}
    diff = high - low
    levels = {
        "0.236": high - 0.236 * diff,
        "0.382": high - 0.382 * diff,
        "0.5": high - 0.5 * diff,
        "0.618": high - 0.618 * diff,
        "0.786": high - 0.786 * diff,
        "1.272": high + 0.272 * diff,
        "1.618": high + 0.618 * diff,
    }
    # espelho low->high
    levels_alt = {
        "alt_0.236": low + 0.236 * diff,
        "alt_0.382": low + 0.382 * diff,
        "alt_0.5": low + 0.5 * diff,
        "alt_0.618": low + 0.618 * diff,
        "alt_0.786": low + 0.786 * diff,
        "alt_1.272": low - 0.272 * diff,
        "alt_1.618": low - 0.618 * diff,
    }
    levels.update(levels_alt)
    return levels


def calcular_precision(candles: SignalRequest, signal: str, price: float, forecast_time: int) -> float:
    timestamps_futuros = [int(t) for t in candles.T if int(t) > forecast_time]
    precisao = 0.0
    if signal == "BUY":
        for ts in timestamps_futuros:
            future_price = next((float(candles.C[i]) for i, _t in enumerate(candles.T) if int(_t) == ts), None)
            if future_price and future_price > price:
                precisao += 1
    elif signal == "SELL":
        for ts in timestamps_futuros:
            future_price = next((float(candles.C[i]) for i, _t in enumerate(candles.T) if int(_t) == ts), None)
            if future_price and future_price < price:
                precisao += 1
    return precisao / len(timestamps_futuros) * 100 if timestamps_futuros else 0.0


def confluence_score(
    close_now: float,
    ema_fast_now: Optional[float],
    ema_slow_now: Optional[float],
    macd_hist_now: Optional[float],
    rsi_now: Optional[float],
    bb_upper_now: Optional[float],
    bb_lower_now: Optional[float],
) -> float:
    """Score heurístico 0–1: >0.5 sugere COMPRA, <0.5 sugere VENDA."""
    score = 0.5
    # EMA relação
    if ema_fast_now is not None and ema_slow_now is not None:
        if ema_fast_now > ema_slow_now:
            score += 0.15
        elif ema_fast_now < ema_slow_now:
            score -= 0.15
    # MACD hist
    if macd_hist_now is not None:
        if macd_hist_now > 0:
            score += 0.15
        elif macd_hist_now < 0:
            score -= 0.15
    # RSI
    if rsi_now is not None:
        if rsi_now < 30:
            score += 0.10  # sobrevenda -> possível alta
        elif rsi_now > 70:
            score -= 0.10  # sobrecompra -> possível queda
    # Bollinger
    if bb_lower_now is not None and bb_upper_now is not None and bb_upper_now != bb_lower_now:
        if close_now <= bb_lower_now:
            score += 0.10
        elif close_now >= bb_upper_now:
            score -= 0.10
    return max(0.0, min(1.0, score))


def pick_direction_by_confluence(
    close_now: float,
    ema_fast_now: Optional[float],
    ema_slow_now: Optional[float],
    macd_hist_now: Optional[float],
    rsi_now: Optional[float],
    bb_upper_now: Optional[float],
    bb_lower_now: Optional[float],
):
    score = confluence_score(
        close_now, ema_fast_now, ema_slow_now, macd_hist_now, rsi_now, bb_upper_now, bb_lower_now
    )
    if score >= 0.5:
        operacao = "BUY"
        precisao = (score - 0.5) * 200  # 50%..100%
        motivo = "Confluência altista (EMAfast>EMAslow/MACD hist>0/RSI baixo ou toque banda inferior)."
    else:
        operacao = "SELL"
        precisao = (0.5 - score) * 200  # 50%..100%
        motivo = "Confluência baixista (EMAfast<EMAslow/MACD hist<0/RSI alto ou toque banda superior)."
    return operacao, float(round(max(50.0, min(100.0, precisao)), 2)), motivo


def gerar_previsoes(
    timestamps: List[int],
    closes: List[float],
    ema_fast_vals: List[Optional[float]],
    ema_slow_vals: List[Optional[float]],
    macd_hist: List[Optional[float]],
    rsi_vals: List[Optional[float]],
    bb_upper: List[Optional[float]],
    bb_lower: List[Optional[float]],
    max_minutes: int = 30,
    n_preds: int = 5,
) -> List[ForecastItem]:
    """Gera n previsões espaçadas uniformemente dentro de max_minutes a partir do último timestamp."""
    preds: List[ForecastItem] = []
    if not timestamps:
        return preds

    i = len(timestamps) - 1
    last_ts = timestamps[i]
    last_close = closes[i]

    # Estima delta mediano de timestamp (segundos) para respeitar ≤ 30 min
    if len(timestamps) >= 3:
        deltas = [timestamps[k] - timestamps[k - 1] for k in range(1, len(timestamps))]
        median_dt = int(statistics.median(deltas))
        median_dt = max(1, median_dt)
    else:
        median_dt = 60  # fallback: 1m

    total_window = max_minutes * 60
    step = max(1, min(median_dt, total_window // n_preds))

    for k in range(1, n_preds + 1):
        ts_k = last_ts + k * step
        # Usa a confluência "agora" (nowcast simples) para classificar
        score = confluence_score(
            close_now=last_close,
            ema_fast_now=ema_fast_vals[i],
            ema_slow_now=ema_slow_vals[i],
            macd_hist_now=macd_hist[i],
            rsi_now=rsi_vals[i],
            bb_upper_now=bb_upper[i] if i < len(bb_upper) else None,
            bb_lower_now=bb_lower[i] if i < len(bb_lower) else None,
        )
        if score >= 0.5:
            operacao = "COMPRA"
            precisao = (score - 0.5) * 200
            motivo = "Confluência altista (EMAfast>EMAslow/MACD hist>0/RSI mais baixo ou toque banda inferior)."
        else:
            operacao = "VENDA"
            precisao = (0.5 - score) * 200
            motivo = "Confluência baixista (EMAfast<EMAslow/MACD hist<0/RSI mais alto ou toque banda superior)."

        preds.append(
            ForecastItem(
                operacao=operacao,
                timestamp=ts_k,
                iso_time=datetime.utcfromtimestamp(ts_k).isoformat() + "Z",
                precisao=float(round(max(0.0, min(100.0, precisao)), 2)),
                motivo=motivo,
            )
        )
    return preds


# --------- Core logic ---------
def generate_signal(candles: SignalRequest, params: StrategyParams) -> SignalResponse:
    # Converte strings -> numéricos
    closes = [float(c) for c in candles.C]
    highs = [float(h) for h in candles.H]
    lows = [float(i) for i in candles.I]
    volumes = [float(v) for v in candles.V]
    timestamps = [int(t) for t in candles.T]

    ema_fast_vals = ema(closes, params.ema_fast)
    ema_slow_vals = ema(closes, params.ema_slow)
    rsi_vals = rsi(closes, params.rsi_period)
    atr_vals = atr(highs, lows, closes, params.atr_period)
    macd_line, macd_signal, macd_hist = macd(closes, params.macd_fast, params.macd_slow, params.macd_signal)
    bb_mid, bb_upper, bb_lower = bollinger(closes, params.bb_period, params.bb_k)

    i = len(timestamps) - 1

    # Guardas básicas
    if i < 1 or ema_fast_vals[i] is None or ema_slow_vals[i] is None:
        # Mesmo com pouco dado, geramos previsões conservadoras (agora) e preenchemos signal1 padrão
        previsoes = gerar_previsoes(
            timestamps, closes, ema_fast_vals, ema_slow_vals, macd_hist, rsi_vals, bb_upper, bb_lower
        )
        last_ts = timestamps[i]
        last_price = closes[i]
        # Fallback sugerido (sem confluência robusta ainda)
        return SignalResponse(
            signal="HOLD",
            timestamp=last_ts,
            iso_time=datetime.utcfromtimestamp(last_ts).isoformat() + "Z",
            price=last_price,
            reason="Insufficient data for EMA crossover.",
            precision=0.0,
            indicators=None,
            fibonacci=None,
            previsoes=previsoes,
            signal1="BUY",
            signal1_precision=50.0,
            signal1_reason="Poucos dados; sugestão padrão até haver mais histórico."
        )

    if volumes[i] < params.min_volume:
        previsoes = gerar_previsoes(
            timestamps, closes, ema_fast_vals, ema_slow_vals, macd_hist, rsi_vals, bb_upper, bb_lower
        )
        return SignalResponse(
            signal="HOLD",
            timestamp=timestamps[i],
            iso_time=datetime.utcfromtimestamp(timestamps[i]).isoformat() + "Z",
            price=closes[i],
            reason=f"Volume {volumes[i]:.2f} < min_volume ({params.min_volume}).",
            precision=0.0,
            previsoes=previsoes,
            signal1=None,
            signal1_precision=None,
            signal1_reason=None,
        )

    # Núcleo: Crossover de EMAs
    diff_now = ema_fast_vals[i] - ema_slow_vals[i]
    diff_prev = (
        ema_fast_vals[i - 1] - ema_slow_vals[i - 1]
        if (ema_fast_vals[i - 1] is not None and ema_slow_vals[i - 1] is not None)
        else None
    )

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

    # Filtro RSI (opcional)
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

    # Comentários auxiliares com MACD/Bollinger
    if macd_hist[i] is not None:
        if signal == "BUY" and macd_hist[i] <= 0:
            reason_parts.append("MACD histograma ≤ 0 (compra com menor convicção).")
        elif signal == "SELL" and macd_hist[i] >= 0:
            reason_parts.append("MACD histograma ≥ 0 (venda com menor convicção).")
    if bb_upper[i] is not None and bb_lower[i] is not None:
        if signal == "BUY" and closes[i] >= bb_upper[i]:
            reason_parts.append("Preço junto/acima da banda superior (risco de exaustão).")
        if signal == "SELL" and closes[i] <= bb_lower[i]:
            reason_parts.append("Preço junto/abaixo da banda inferior (risco de repique).")

    # Fallback sugerido (signal1) quando o principal ficou HOLD
    signal1 = None
    signal1_precision = None
    signal1_reason = None
    if signal == "HOLD":
        pick_sig, pick_prec, pick_msg = pick_direction_by_confluence(
            close_now=closes[i],
            ema_fast_now=ema_fast_vals[i],
            ema_slow_now=ema_slow_vals[i],
            macd_hist_now=macd_hist[i],
            rsi_now=rsi_vals[i] if i < len(rsi_vals) else None,
            bb_upper_now=bb_upper[i] if i < len(bb_upper) else None,
            bb_lower_now=bb_lower[i] if i < len(bb_lower) else None,
        )
        signal1 = pick_sig  # "BUY" ou "SELL"
        signal1_precision = pick_prec  # 50..100
        signal1_reason = f"Sinal principal: HOLD. Sugestão por confluência: {pick_msg}"

    # TP/SL via ATR (apenas para signal BUY/SELL principal)
    tp = sl = None
    if signal in ("BUY", "SELL") and atr_vals[i] is not None and atr_vals[i] > 0:
        a = atr_vals[i]
        price = closes[i]
        if signal == "BUY":
            sl = price - params.atr_sl_mult * a
            tp = price + params.atr_tp_mult * a
        elif signal == "SELL":
            sl = price + params.atr_sl_mult * a
            tp = price - params.atr_tp_mult * a

    reason = " ".join(reason_parts) if reason_parts else "Nenhum motivo específico."

    # Precisão (retro) do sinal principal
    precision = calcular_precision(candles, signal, closes[i], timestamps[i])

    # Fibonacci (com base no swing mais recente)
    swing_high, swing_low = find_swing_high_low(highs, lows, params.fib_lookback)
    fib_levels = fibonacci_levels(swing_high, swing_low)

    # Indicadores atuais úteis
    indicators_out: Dict[str, float] = {}
    def put_if(name: str, val: Optional[float]):
        if val is not None and isinstance(val, (float, int)):
            indicators_out[name] = float(val)

    put_if("ema_fast", ema_fast_vals[i])
    put_if("ema_slow", ema_slow_vals[i])
    put_if("rsi", rsi_vals[i] if i < len(rsi_vals) else None)
    put_if("atr", atr_vals[i] if i < len(atr_vals) else None)
    put_if("macd_line", macd_line[i] if i < len(macd_line) else None)
    put_if("macd_signal", macd_signal[i] if i < len(macd_signal) else None)
    put_if("macd_hist", macd_hist[i] if i < len(macd_hist) else None)
    put_if("bb_mid", bb_mid[i] if i < len(bb_mid) else None)
    put_if("bb_upper", bb_upper[i] if i < len(bb_upper) else None)
    put_if("bb_lower", bb_lower[i] if i < len(bb_lower) else None)

    # 5 previsões dentro de 30 minutos
    previsoes = gerar_previsoes(
        timestamps, closes, ema_fast_vals, ema_slow_vals, macd_hist, rsi_vals, bb_upper, bb_lower,
        max_minutes=30, n_preds=5
    )

    return SignalResponse(
        signal=signal,
        timestamp=timestamps[i],
        iso_time=datetime.utcfromtimestamp(timestamps[i]).isoformat() + "Z",
        price=closes[i],
        reason=reason,
        take_profit=tp,
        stop_loss=sl,
        precision=precision,
        indicators=indicators_out if indicators_out else None,
        fibonacci=fib_levels if fib_levels else None,
        previsoes=previsoes,
        signal1=signal1,
        signal1_precision=signal1_precision,
        signal1_reason=signal1_reason,
    )


# --------- API endpoints ---------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/signal", response_model=SignalResponse)
def signal(req: SignalRequest):
    if not req.T or len(req.T) < 5:
        # Early-return com tipos corretos e fallback sugerido
        last_ts = int(req.T[-1]) if req.T else int(datetime.utcnow().timestamp())
        last_close = float(req.C[-1]) if req.C else 0.0
        return SignalResponse(
            signal="HOLD",
            timestamp=last_ts,
            iso_time=datetime.utcfromtimestamp(last_ts).isoformat() + "Z",
            price=last_close,
            reason="Envie pelo menos 5 candles para estabilidade.",
            precision=0.0,
            previsoes=[],
            signal1="BUY",
            signal1_precision=50.0,
            signal1_reason="Dados insuficientes; sugestão padrão até haver mais histórico."
        )
    params = req.params or StrategyParams()
    return generate_signal(req, params)
