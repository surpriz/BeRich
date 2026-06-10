"""Generate the daily ready-to-read video script (French) from the paper book's FACTS.

Same honesty contract as the rest of the platform: the script narrates only what the robot
actually DID (executions, realized P&L net of costs) plus the live portfolio state — never
the continuously-recomputed forecast. Losses are stated as plainly as gains, the forward-test
status is spelled out, and the mandatory disclaimer closes every script. The "why" behind
each trade comes from the served model's SHAP-style contributions, translated to plain
French so a lay audience can follow.

Pure read + shape (like ``digest.py``): nothing here opens, closes or sizes a trade.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import pandas as pd

from berich.signals.paper import (
    get_equity_curve,
    get_open_positions,
    get_paper_metrics,
    recent_executions,
)

if TYPE_CHECKING:
    from berich.config import Config
    from berich.data.store import OhlcvStore

logger = logging.getLogger(__name__)

# Plain-French translations of the model features, for the "pourquoi" of each trade.
# Keys match FEATURE_COLUMNS (+ the optional earnings/news/microstructure add-ons).
_FEATURE_FR: dict[str, str] = {
    "ret_1": "la variation de la dernière séance",
    "ret_5": "la tendance des 5 derniers jours",
    "mom_10": "la dynamique sur 10 jours",
    "mom_20": "la dynamique sur 1 mois",
    "mom_60": "la dynamique sur 3 mois",
    "mom_120": "la dynamique sur 6 mois",
    "rsi_14": "le RSI (zone de surachat/survente)",
    "macd": "le MACD (croisement de tendances)",
    "macd_signal": "la ligne de signal du MACD",
    "macd_hist": "l'élan du MACD",
    "atr_pct": "la volatilité (amplitude moyenne des bougies)",
    "rvol_20": "la volatilité réalisée sur 20 jours",
    "close_sma20_ratio": "la position du prix face à sa moyenne 20 jours",
    "close_sma50_ratio": "la position du prix face à sa moyenne 50 jours",
    "volume_z20": "un volume inhabituel face aux 20 derniers jours",
    "dist_high_60": "la distance au plus haut de 60 jours",
    "dist_low_60": "la distance au plus bas de 60 jours",
    "month_sin": "l'effet calendaire (période de l'année)",
    "month_cos": "l'effet calendaire (période de l'année)",
    "days_to_month_end": "la proximité de la fin de mois",
    "spy_ret_20": "le régime du marché global (tendance)",
    "spy_rvol_20": "le régime du marché global (nervosité)",
    "clv": "où le prix a clôturé dans la bougie du jour",
    "gap_open": "le gap d'ouverture",
    "amihud_20": "la liquidité de l'actif",
    "roll_spread_20": "le coût de friction implicite",
    "parkinson_20": "la volatilité intra-séance",
}

_EXIT_FR = {
    "closed_target": "objectif atteint",
    "closed_stop": "stop touché — perte coupée comme prévu",
    "closed_time": "sortie à l'échéance (le scénario ne s'est pas joué à temps)",
    "closed_trail": "stop suiveur déclenché — gains sécurisés",
}

# One discipline lesson per day, rotated deterministically by date (no randomness: a re-run
# of the same day produces the same script).
_LESSONS = (
    "Le stop n'est pas une option, c'est le prix d'entrée du métier : on connaît sa perte "
    "maximale AVANT d'acheter, jamais après.",
    "Une perte coupée à -1 % n'est pas un échec, c'est le système qui fonctionne. L'échec, "
    "c'est la perte qu'on laisse grossir.",
    "Le robot ne se venge jamais : après une perte, il ne double pas la mise — la taille de "
    "position reste calculée, pas émotionnelle.",
    "On ne juge pas une stratégie sur un trade, ni sur cinq. On la juge sur des dizaines de "
    "trades, nets de frais.",
    "L'inaction est une décision : quand aucun signal ne franchit la barre, le robot reste à "
    "plat — et c'est souvent le meilleur trade de la journée.",
    "Les gains se laissent courir, les pertes se coupent court. C'est contre-intuitif, c'est "
    "pour ça qu'un système le fait mieux qu'un humain.",
    "La diversification n'est pas d'avoir 10 positions, c'est d'avoir 10 paris DIFFÉRENTS — "
    "trois paires de devises qui partagent la même jambe ne font qu'un seul pari.",
)

DISCLAIMER = (
    "⚠️ Rappel obligatoire : ceci n'est PAS un conseil en investissement. Je documente ce que "
    "mon robot fait en simulation (paper trading) — aucun capital réel n'est engagé, le "
    "système est en phase de test et ses modèles ne sont PAS encore validés. Les performances "
    "simulées ne préjugent pas des performances futures. Investir comporte un risque de perte "
    "en capital. Faites vos propres recherches."
)


def _fmt_px(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".") if abs(value) < 10 else f"{value:.2f}"  # noqa: PLR2004


def _fmt_pct(value: float) -> str:
    return f"{'+' if value >= 0 else ''}{value * 100:.1f}%"


def _humanize_factors(factors: list[dict[str, object]], *, short: bool) -> str:
    """Top SHAP contributions -> one plain-French clause ("parce que ...")."""
    parts: list[str] = []
    for f in factors[:3]:
        name = str(f.get("feature", ""))
        label = _FEATURE_FR.get(name, name.replace("_", " "))
        contrib = cast("float", f.get("contribution", 0.0) or 0.0)
        # A positive contribution pushes P(win) of the modeled side up.
        pushes_for = contrib > 0
        verdict = "joue pour" if pushes_for else "joue contre"
        parts.append(f"{label} ({verdict})")
    if not parts:
        return ""
    side = "la baisse" if short else "la hausse"
    return f"Les facteurs principaux derrière ce pari sur {side} : " + " ; ".join(parts) + "."


def _why_for_ticker(config: Config, store: OhlcvStore, ticker: str, *, short: bool) -> str:
    """Best-effort plain-French 'why' from the served model's contributions."""
    try:
        from berich.signals.service import explain_signal  # noqa: PLC0415 — heavy import path

        payload = explain_signal(ticker, config, store, top_k=3)
    except Exception:  # noqa: BLE001 — the script must build even when explain fails
        logger.warning("video_script: explain failed for %s", ticker, exc_info=True)
        return ""
    if not payload:
        return ""
    factors = cast("list[dict[str, object]]", payload.get("top_features") or [])
    return _humanize_factors(factors, short=short)


def _portfolio_section(config: Config, store: OhlcvStore) -> list[str]:
    metrics = get_paper_metrics(config, store)
    equity = get_equity_curve(config, store)
    capital = float(metrics["capital"])
    value = capital * (1.0 + float(metrics["total_return_paper"]))
    dd = 0.0
    if not equity.empty:
        series = pd.Series(equity["equity_paper"].to_numpy(), dtype=float).dropna()
        if not series.empty:
            peak = float(series.cummax().iloc[-1])
            dd = max(0.0, 1.0 - float(series.iloc[-1]) / peak) if peak else 0.0
    spy = float(metrics["total_return_spy"])
    spy_txt = "" if pd.isna(spy) else f" Sur la même période, le marché (SPY) fait {_fmt_pct(spy)}."
    return [
        f"LE PORTEFEUILLE (simulation, capital fictif de {capital:,.0f} €)".replace(",", " "),
        f"Valeur actuelle : {value:,.0f} € ({_fmt_pct(float(metrics['total_return_paper']))} "
        f"depuis le départ).{spy_txt}".replace(",", " "),
        f"Repli depuis le plus haut : {_fmt_pct(-dd)}. "
        f"{int(metrics['n_open'])} position(s) en cours, "
        f"{int(metrics['n_closed'])} clôturée(s) au total"
        + (
            f", {_fmt_pct(float(metrics['win_rate']))[1:]} de trades gagnants."
            if int(metrics["n_closed"])
            else "."
        ),
    ]


def build_video_script(config: Config, store: OhlcvStore) -> dict[str, object]:
    """Assemble the daily ready-to-read script. Returns ``{date, title, script}``.

    Robust to an empty book and to any failing sub-read: every section degrades to an
    honest "rien à signaler" rather than aborting — the daily video must never be blocked
    by a data hiccup.
    """
    today = pd.Timestamp.today().normalize()
    date_iso = str(today.date())
    executions = recent_executions(config, store)

    lines: list[str] = [
        f"LE JOURNAL DU ROBOT — {date_iso}",
        "",
        "Bonjour à tous ! Comme chaque jour, voici ce que mon robot de trading a réellement "
        "fait — pas des prévisions, pas des promesses : des faits, gains ET pertes, nets de "
        "frais. Le système est en phase de test, sans argent réel.",
        "",
    ]

    lines.extend(_portfolio_section(config, store))
    lines.append("")

    # --- last run's executions (facts only) ---
    lines.append("CE QUE LE ROBOT A FAIT AU DERNIER POINTAGE")
    opened = executions["open"]
    closed = executions["close"]
    adjust = executions["adjust"]
    if not opened and not closed:
        lines.append(
            "Aucune ouverture ni clôture : aucun signal n'a franchi la barre de validation. "
            "Le robot reste discipliné — ne rien faire est une décision."
        )
    for o in opened:
        short = o["direction"] == "short"
        sens = "VENTE à découvert (pari sur la baisse)" if short else "ACHAT (pari sur la hausse)"
        entry, stop, target = (cast("float", o[k]) for k in ("entry", "stop", "target"))
        notional = cast("float", o["notional"])
        lines.append(
            f"• {o['ticker']} — {sens} : entrée {_fmt_px(entry)}, "
            f"stop {_fmt_px(stop)}, objectif {_fmt_px(target)}, "
            f"taille {o['size_shares']} (≈{notional:,.0f} €).".replace(",", " ")
        )
        why = _why_for_ticker(config, store, str(o["ticker"]), short=short)
        if why:
            lines.append(f"  {why}")
    for c in closed:
        pnl = c.get("pnl_pct")
        pnl_txt = _fmt_pct(cast("float", pnl)) if pnl is not None else "—"
        reason = _EXIT_FR.get(str(c.get("status")), "sortie")
        sens_close = "vendeuse" if c["direction"] == "short" else "acheteuse"
        lines.append(
            f"• {c['ticker']} — position {sens_close} clôturée : {pnl_txt} net de frais ({reason})."
        )
    if adjust:
        moves = ", ".join(
            f"{a['ticker']} → stop {_fmt_px(cast('float', a['effective_stop']))}" for a in adjust
        )
        lines.append(f"Stops suiveurs remontés (gains progressivement sécurisés) : {moves}.")
    lines.append("")

    # --- open positions (held, not advice) ---
    positions = get_open_positions(config, store)
    lines.append("LES POSITIONS EN COURS (on ne touche à rien : le stop et l'objectif décident)")
    if not positions:
        lines.append("Aucune — le robot est entièrement à plat.")
    for p in positions:
        mtm = f" — latent {_fmt_pct(p.mtm_pct)}" if p.mtm_pct is not None else ""
        lines.append(
            f"• {p.ticker} ({'baisse' if p.direction == 'short' else 'hausse'}), "
            f"jour {p.days_held}{mtm}, stop effectif "
            f"{_fmt_px(float(p.trail_stop if p.trail_stop is not None else p.stop))}."
        )
    lines.append("")

    lesson = _LESSONS[int(today.toordinal()) % len(_LESSONS)]
    lines.extend(["LA LEÇON DU JOUR", lesson, "", DISCLAIMER])

    script = "\n".join(lines)
    return {"date": date_iso, "title": f"Le journal du robot — {date_iso}", "script": script}


__all__ = ["DISCLAIMER", "build_video_script"]
