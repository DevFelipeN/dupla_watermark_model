#!/usr/bin/env python3
"""
gerar_figura_arquitetura.py

Desenha a arquitetura da versão 3 (marcação + ataque + extração), com o
formato dos dados e o número de parâmetros de cada bloco, e salva em
my_tests/arquitetura/arquitetura_v3.png e .pdf (vetorial, para a dissertação).

    python gerar_figura_arquitetura.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_tests", "arquitetura")

# cores: transformada fixa | rede de marcação | rede de extração | mensagem | operação fixa | ataque
C_FIXED = "#E6E6E6"
C_EMB = "#CFE3F7"
C_EXT = "#D5EFD5"
C_MSG = "#E9DDF5"
C_OP = "#FFF1CC"
C_ATK = "#F8D3D3"
C_IMG = "#FFFFFF"
EDGE = "#333333"


def box(ax, x, y, w, h, title, sub="", color=C_FIXED, bold=True, size=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.6",
                                fc=color, ec=EDGE, lw=1.1))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=size,
                weight="bold" if bold else "normal")
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=size - 2, color="#333333",
                linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=size,
                weight="bold" if bold else "normal")
    return (x, y, w, h)


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def top(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


def arrow(ax, p, q, color=EDGE, style="-|>", lw=1.2, ls="-", rad=0.0):
    ax.annotate("", xy=q, xytext=p,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls,
                                shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def path(ax, points, color=EDGE, lw=1.2, ls="-"):
    """Seta em L: segmentos retos, ponta só no último."""
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls)
    arrow(ax, points[-2], points[-1], color=color, lw=lw, ls=ls)


def main():
    fig, ax = plt.subplots(figsize=(17, 11.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    # =============================================================
    # (a) MARCAÇÃO
    # =============================================================
    ax.text(0.5, 64.6, "(a) Marcação — rede de 0,22 M parâmetros", fontsize=13, weight="bold", va="center")

    H = 5.6
    y1 = 54  # linha da imagem
    img = box(ax, 0.5, y1, 9, H, "Imagem RGB", "256×256", C_IMG)
    yl = box(ax, 11.5, y1, 9.5, H, "Luminância Y", "BT.601\n(como no JPEG)")
    dwt = box(ax, 23, y1, 9, H, "DWT Haar", "LL, LH, HL, HH\n4 × 128×128")
    dct = box(ax, 34, y1, 8.5, H, "DCT 8×8", "em cada sub-banda")
    senc = box(ax, 44.5, y1, 12.5, H, "Encoder de sub-bandas", "2 conv → 64 × 128×128\n19,9 mil parâm.", C_EMB, size=8.5)
    for a, b in ((img, yl), (yl, dwt), (dwt, dct), (dct, senc)):
        arrow(ax, right(a), left(b))

    y2 = 44  # linha da mensagem
    msg = box(ax, 0.5, y2, 9, H, "Mensagem", "30 bits", C_MSG)
    lin = box(ax, 11.5, y2, 10, H, "Linear", "30 → mapa 16×16\n7,9 mil parâm.", C_EMB)
    wenc = box(ax, 23.5, y2, 13, H, "Encoder da marca", "ConvTranspose ×8 + conv\n64 × 128×128 · 20,8 mil", C_EMB)
    arrow(ax, right(msg), left(lin))
    arrow(ax, right(lin), left(wenc))

    # soma das características
    sx, sy = 59.5, (y1 + y2 + H) / 2
    ax.add_patch(Circle((sx, sy), 1.3, fc="white", ec=EDGE, lw=1.2))
    ax.text(sx, sy, "+", ha="center", va="center", fontsize=14, weight="bold")
    path(ax, [right(senc), (sx, right(senc)[1]), (sx, sy + 1.3)])
    path(ax, [right(wenc), (sx, right(wenc)[1]), (sx, sy - 1.3)])

    fus = box(ax, 62.5, sy - H / 2, 10.5, H, "Fusão", "3 conv → 32 × 128×128\n166,6 mil parâm.", C_EMB)
    arrow(ax, (sx + 1.3, sy), left(fus))
    tanh = box(ax, 74.5, sy - H / 2, 10.5, H, "Conv + tanh", "modificação ∈ [−1, 1]\n4 × 128×128 · 1,2 mil", C_EMB)
    arrow(ax, right(fus), left(tanh))
    mask = box(ax, 86.5, sy - H / 2, 13, H, "Máscara DCT mid-band", "só 1 ≤ u+v ≤ 7\n(35 de 64 coef. por bloco)", C_OP, size=8.5)
    arrow(ax, right(tanh), left(mask))

    y3 = 32.5  # reconstrução (da direita para a esquerda)
    wts = box(ax, 86.5, y3, 13, H, "× pesos e × α", "LL 0,30 · LH 0,35 · HL 0,35\nHH 0,001 · α = 5", C_OP)
    arrow(ax, bottom(mask), top(wts))
    add = box(ax, 72.5, y3, 12, H, "+ coef. originais", "coef. DCT das sub-bandas", C_OP)
    arrow(ax, left(wts), right(add))
    idct = box(ax, 62.5, y3, 8, H, "IDCT 8×8")
    arrow(ax, left(add), right(idct))
    idwt = box(ax, 52.5, y3, 8, H, "IDWT Haar")
    arrow(ax, left(idct), right(idwt))
    rgb = box(ax, 39.5, y3, 11, H, "ΔY em R, G e B", "as cores não mudam")
    arrow(ax, left(idwt), right(rgb))
    out = box(ax, 26.5, y3, 11, H, "Imagem marcada", "256×256\nPSNR ≈ 47,7 dB", C_IMG)
    arrow(ax, left(rgb), right(out))
    # coeficientes originais: da DCT até a soma, por fora do encoder da marca
    cx = dct[0] + dct[2] - 1.5
    path(ax, [(cx, y1), (cx, 41.2), (top(add)[0], 41.2), top(add)], color="#888888", ls="--")
    ax.text(cx + 1, 41.9, "coeficientes originais", fontsize=7.5, color="#666666", va="center")

    # =============================================================
    # ATAQUE
    # =============================================================
    atk = box(ax, 0.5, y3, 23.5, H, "Ataque", "JPEG, blur, median, ruído, salt & pepper,\ndropout, resize ou deepfake", C_ATK)
    arrow(ax, left(out), right(atk))

    # =============================================================
    # (b) EXTRAÇÃO
    # =============================================================
    ax.text(8, 27.6, "(b) Extração — rede de 0,79 M parâmetros", fontsize=13, weight="bold", va="center")
    y4 = 16.5
    yx = box(ax, 0.5, y4, 10.5, H, "Y → DWT → DCT", "mesmas transformadas\n(usa só LL, LH, HL)", size=9)
    arrow(ax, (top(yx)[0], y3), top(yx))
    se = box(ax, 12.5, y4, 10, H, "Squeeze-and-\nExcitation", "peso por sub-banda\n3 × 128×128 · 10 parâm.", C_EXT, size=8.5)
    arrow(ax, right(yx), left(se))

    # U-Net
    ux, uw = 24.5, 50
    ax.add_patch(FancyBboxPatch((ux, 6.6), uw, 19.4, boxstyle="round,pad=0.02,rounding_size=0.8",
                                fc="#F2FAF2", ec="#5A9A5A", lw=1.1, ls="--"))
    ax.text(ux + uw / 2, 24.9, "U-Net — 457 mil parâmetros", ha="center", fontsize=10, weight="bold", color="#2F6B2F")
    bw, bh = 9, 4.6
    e1 = box(ax, ux + 1.5, 18.0, bw, bh, "enc1", "32 × 128×128", C_EXT, size=9)
    e2 = box(ax, ux + 9.5, 12.6, bw, bh, "enc2", "64 × 64×64", C_EXT, size=9)
    bn = box(ax, ux + 20.5, 7.6, bw, bh, "bottleneck", "128 × 32×32", C_EXT, size=9)
    d2 = box(ax, ux + 31.5, 12.6, bw, bh, "dec2", "64 × 64×64", C_EXT, size=9)
    d1 = box(ax, ux + 39.5, 18.0, bw, bh, "dec1", "32 × 128×128", C_EXT, size=9)
    arrow(ax, right(se), left(e1))
    # descida (max-pool) e subida (up-conv) em L, sem cruzar as caixas
    path(ax, [bottom(e1), (bottom(e1)[0], e2[1] + bh / 2), left(e2)])
    path(ax, [bottom(e2), (bottom(e2)[0], bn[1] + bh / 2), left(bn)])
    path(ax, [right(bn), (bottom(d2)[0], bn[1] + bh / 2), bottom(d2)])
    path(ax, [right(d2), (bottom(d1)[0], d2[1] + bh / 2), bottom(d1)])
    ax.text(bottom(e1)[0] - 0.6, 16.3, "max-pool", fontsize=7, color="#555555", ha="right")
    ax.text(bottom(e2)[0] - 0.6, 10.9, "max-pool", fontsize=7, color="#555555", ha="right")
    ax.text(bottom(d2)[0] + 0.6, 10.9, "up-conv", fontsize=7, color="#555555")
    ax.text(bottom(d1)[0] + 0.6, 16.3, "up-conv", fontsize=7, color="#555555")
    # conexões laterais (skip)
    arrow(ax, right(e1), left(d1), color="#5A9A5A", ls="--")
    arrow(ax, right(e2), left(d2), color="#5A9A5A", ls="--")
    ax.text(ux + uw / 2, 21.2, "conexões laterais (concatenação)", ha="center", fontsize=7.5, color="#2F6B2F")

    ref = box(ax, 76.5, y4, 11, H, "Refino + redução", "conv · conv 8×8 / 8\n32 × 16×16 · 74,9 mil", C_EXT, size=9)
    arrow(ax, right(d1), left(ref))
    fc = box(ax, 76.5, 7.5, 11, H, "Linear", "8.192 → 30\n245,8 mil parâm.", C_EXT, size=9)
    arrow(ax, bottom(ref), top(fc))
    bits = box(ax, 89, 7.5, 10.5, H, "Mensagem lida", "30 bits (bit = 1 se > 0)\nopcional: BCH → 15 bits", C_MSG, size=9)
    arrow(ax, right(fc), left(bits))

    # legenda
    items = [(C_FIXED, "transformada fixa"), (C_EMB, "rede de marcação"), (C_EXT, "rede de extração"),
             (C_OP, "operação fixa"), (C_MSG, "mensagem"), (C_ATK, "ataque")]
    lx = 0.5
    for color, label in items:
        ax.add_patch(FancyBboxPatch((lx, 1.9), 2.2, 1.6, boxstyle="round,pad=0.02,rounding_size=0.3",
                                    fc=color, ec=EDGE, lw=0.8))
        ax.text(lx + 2.8, 2.7, label, fontsize=8.5, va="center")
        lx += 16.5
    ax.text(0.5, -0.4, "Treino conjunto: perda = BCE(bits) + MSE + 0,005·LPIPS(imagem); "
            "o ataque é sorteado a cada lote. Formatos indicados para uma imagem 256×256.",
            fontsize=8.5, va="center", color="#333333", style="italic")
    ax.set_ylim(-1.8, 66)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"arquitetura_v3.{ext}"), dpi=200, bbox_inches="tight")
    print("Figura:", os.path.join(OUT, "arquitetura_v3.png"))


if __name__ == "__main__":
    main()
