#!/usr/bin/env python3
"""Render figures from results/*.csv.

    python scripts/make_figures.py

Stdlib csv + matplotlib only, so the figure pipeline shares no dependency with
the result builder.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def load(name):
    with (RES / name).open() as fh:
        return list(csv.DictReader(fh))


def f(x):
    return float(x) if x not in (None, "", "None") else 0.0


def fig_mtp():
    rows = load("mtp_ab.csv")
    arms = [r["arm"] for r in rows]
    pp = [f(r["pp_mean"]) for r in rows]
    tg = [f(r["tg_mean"]) for r in rows]
    pp_sem = [f(r["pp_sem"]) for r in rows]
    tg_sem = [f(r["tg_sem"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].bar(arms, pp, yerr=pp_sem, capsize=5, color="#4c72b0")
    ax[0].set_title("Prefill (PP) at 64k"); ax[0].set_ylabel("t/s")
    ax[1].bar(arms, tg, yerr=tg_sem, capsize=5, color="#dd8452")
    ax[1].set_title("Deep decode (TG) at 64k"); ax[1].set_ylabel("t/s")
    for a, v in zip(ax[1].patches, tg):
        ax[1].annotate(f"{v:.1f}", (a.get_x() + a.get_width() / 2, v),
                       ha="center", va="bottom", fontsize=10)
    fig.suptitle("MTP speculative decoding on/off — live 29B MoE, 64k context")
    fig.tight_layout()
    fig.savefig(FIG / "fig_mtp_ab.png", dpi=150); plt.close(fig)


def fig_prefill_vs_ctx():
    rows = load("prefill_vs_ctx.csv")
    xs = [f(r["ctx"]) / 1024 for r in rows]
    ys = [f(r["pp_t_s"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(xs, ys, "o-", color="#4c72b0")
    ax.set_xscale("log", base=2); ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(x)}k" for x in xs])
    ax.set_xlabel("context (tokens)"); ax.set_ylabel("prefill PP (t/s)")
    ax.set_title("Prefill throughput vs context (expert cache on)")
    fig.tight_layout(); fig.savefig(FIG / "fig_prefill_vs_ctx.png", dpi=150)
    plt.close(fig)


def fig_moe_cache():
    rows = load("moe_cache.csv")
    labels = [r["config"] for r in rows]
    pp = [f(r["pp_t_s"]) for r in rows]
    colors = ["#c44e52" if "no_moe" in l or l.endswith("auto") or "soft" in l
              else "#55a868" for l in labels]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(labels, pp, color=colors)
    ax.set_xlabel("prefill PP (t/s)")
    ax.set_title("Expert-cache effect on prefill")
    fig.tight_layout(); fig.savefig(FIG / "fig_moe_cache.png", dpi=150)
    plt.close(fig)


def fig_deep_vs_depth0():
    rows = load("deep_vs_depth0.csv")
    labels = [r["mode"] for r in rows]
    tg = [f(r["tg_t_s"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(labels, tg, color=["#c44e52", "#55a868", "#55a868"])
    for i, v in enumerate(tg):
        ax.annotate(f"{v:.1f}", (i, v), ha="center", va="bottom")
    ax.set_ylabel("decode TG (t/s)")
    ax.set_title("Depth-0 benchmark vs measured deep-context decode")
    fig.tight_layout(); fig.savefig(FIG / "fig_deep_vs_depth0.png", dpi=150)
    plt.close(fig)


def fig_coding():
    rows = load("coding_summary.csv")
    labels = [f"{r['class_name']}" for r in rows]
    rate = [f(r["passes"]) / f(r["runs"]) for r in rows]
    colors = ["#55a868" if x == 1.0 else "#c44e52" if x == 0.0 else "#dd8452"
              for x in rate]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.barh(labels, rate, color=colors)
    ax.set_xlim(0, 1.05); ax.set_xlabel("runs passed (of 3)")
    ov = load("coding_overall.csv")[0]
    ax.axvline(f(ov["pass_rate"]), color="k", ls="--", lw=1,
               label=f"overall {f(ov['pass_rate'])*100:.1f}%")
    ax.legend(loc="lower right")
    ax.set_title("Coding completion by task (live 29B MoE, shipped sampling)")
    fig.tight_layout(); fig.savefig(FIG / "fig_coding_passrate.png", dpi=150)
    plt.close(fig)


def main():
    fig_mtp(); fig_prefill_vs_ctx(); fig_moe_cache()
    fig_deep_vs_depth0(); fig_coding()
    for p in sorted(FIG.glob("*.png")):
        print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
