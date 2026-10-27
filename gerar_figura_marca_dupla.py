#!/usr/bin/env python3
"""
gerar_figura_marca_dupla.py

Desenha a arquitetura da marca dupla (versão 4): marcação (robusta v3 +
semi-frágil no rosto), verificação (origem + detecção de alteração) e o
treino da semi-frágil. Por dentro, cada rede tem a mesma arquitetura da v3
(figura do gerar_figura_arquitetura.py). Salva em
my_tests/arquitetura/arquitetura_marca_dupla.png e .pdf.

    python gerar_figura_marca_dupla.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

from gerar_figura_arquitetura import (box, right, left, top, bottom, arrow, path,
                                      C_OP, C_MSG, C_ATK, C_IMG, EDGE, OUT)

C_ROB = "#CFE3F7"     # marca robusta (congelada)
C_SEMI = "#FAD9B5"    # marca semi-frágil (treinada)
C_OK = "#D5EFD5"      # resultado da verificação


def main():
    fig, ax = plt.subplots(figsize=(17, 12.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(-4, 70)
    ax.axis("off")
    H = 6

    # =============================================================
    # (a) MARCAÇÃO
    # =============================================================
    ax.text(0.5, 69, "(a) Marcação — duas marcas na mesma imagem", fontsize=13, weight="bold", va="center")

    y1 = 57
    img = box(ax, 0.5, y1, 9.5, H, "Imagem RGB", "256×256", C_IMG)
    rob = box(ax, 12.5, y1, 15, H, "Marcação robusta v3", "congelada · α = 5\nimagem toda · 0,22 M parâm.", C_ROB)
    so_r = box(ax, 30, y1, 10, H, "Só robusta", "PSNR ≈ 47,7 dB", C_IMG)
    semi = box(ax, 42.5, y1, 16, H, "Marcação semi-frágil", "mesma arquitetura da v3, treinada\nα = 3 · 0,22 M parâm.", C_SEMI)
    msk = box(ax, 61, y1, 13, H, "× máscara do rosto", "elipse suave aplicada\nà diferença ΔY", C_OP)
    sx, sy = 77.5, y1 + H / 2
    ax.add_patch(Circle((sx, sy), 1.3, fc="white", ec=EDGE, lw=1.2))
    ax.text(sx, sy, "+", ha="center", va="center", fontsize=14, weight="bold")
    dup = box(ax, 81, y1, 18.5, H, "Imagem com as duas marcas", "robusta (imagem toda)\n+ semi-frágil (só no rosto)", C_IMG)
    arrow(ax, right(img), left(rob))
    arrow(ax, right(rob), left(so_r))
    arrow(ax, right(so_r), left(semi))
    arrow(ax, right(semi), left(msk))
    ax.text((right(semi)[0] + left(msk)[0]) / 2, sy + 0.9, "ΔY", ha="center", fontsize=8, color="#555555")
    arrow(ax, right(msk), (sx - 1.3, sy))
    arrow(ax, (sx + 1.3, sy), left(dup))
    # a imagem só com a robusta também vai direto para a soma
    path(ax, [top(so_r), (top(so_r)[0], 65.2), (sx, 65.2), (sx, sy + 1.3)])

    y2 = 46
    msg = box(ax, 0.5, y2, 15, H, "Mensagem de origem", "30 bits", C_MSG)
    path(ax, [top(msg), (top(msg)[0], 54.5), (bottom(rob)[0], 54.5), bottom(rob)])
    xor = box(ax, 42.5, y2, 16, H, "Mensagem semi-frágil", "origem ⊕ chave secreta\n30 bits", C_MSG)
    arrow(ax, right(msg), left(xor))
    arrow(ax, top(xor), bottom(semi))

    # =============================================================
    # (b) VERIFICAÇÃO
    # =============================================================
    ax.text(0.5, 40, "(b) Verificação — não precisa da imagem original", fontsize=13, weight="bold", va="center")

    ya, yb = 30.5, 17
    sus = box(ax, 0.5, (ya + yb) / 2, 11.8, H, "Imagem a verificar", "pode ter sido editada\nou ser um deepfake", C_IMG)
    exr = box(ax, 15, ya, 15, H, "Extração robusta v3", "congelada · 0,79 M parâm.", C_ROB)
    exs = box(ax, 15, yb, 15, H, "Extração semi-frágil", "treinada · 0,79 M parâm.", C_SEMI)
    jx = 13.6
    path(ax, [right(sus), (jx, right(sus)[1]), (jx, left(exr)[1]), left(exr)])
    path(ax, [(jx, right(sus)[1]), (jx, left(exs)[1]), left(exs)])

    org = box(ax, 82, ya, 17.5, H, "ORIGEM", "qual é a imagem original\n(sobrevive aos deepfakes)", C_OK)
    arrow(ax, right(exr), left(org))
    ax.text(56, right(exr)[1] + 0.8, "30 bits lidos", ha="center", fontsize=8, color="#555555")

    esp = box(ax, 47, (ya + yb) / 2 + 0.3, 14, H - 0.6, "Esperada", "origem ⊕ chave secreta", C_MSG)
    path(ax, [(38, right(exr)[1]), (38, left(esp)[1]), left(esp)])

    con = box(ax, 64, yb, 15, H, "Concordância", "% de bits iguais entre a\nsemi-frágil lida e a esperada", C_OP)
    arrow(ax, right(exs), left(con))
    ax.text(40, left(exs)[1] + 0.8, "30 bits lidos", ha="center", fontsize=8, color="#555555")
    path(ax, [right(esp), (top(con)[0], right(esp)[1]), top(con)])

    dec = box(ax, 82, yb, 17.5, H, "INTEGRIDADE", "≥ 80%: autêntica\n< 80%: alterada (deepfake)", C_OK)
    arrow(ax, right(con), left(dec))

    # =============================================================
    # (c) TREINO DA SEMI-FRÁGIL
    # =============================================================
    ax.text(0.5, 10.5, "(c) Treino da semi-frágil — a robusta fica congelada; a imagem com as duas marcas passa por:",
            fontsize=13, weight="bold", va="center")
    yc, w, gap = 1.5, 18.6, 1.6
    itens = [
        ("Ataque comum", "JPEG, blur, ruído, resize…\n→ deve LER (BCE)", C_SEMI),
        ("Manipulação do rosto", "troca, regeneração, deformação\nou mistura → NÃO ler (saída ≈ 0)", C_ATK),
        ("Só a robusta", "sem a semi-frágil\n→ NÃO ler (saída ≈ 0)", C_IMG),
        ("Extração robusta", "congelada: continua lendo\na origem (BCE)", C_ROB),
        ("Qualidade", "MSE + 0,005·LPIPS\n(dupla × só robusta)", C_OP),
    ]
    for i, (t, s, c) in enumerate(itens):
        box(ax, 0.5 + i * (w + gap), yc, w, H, t, s, c, size=9.5)
    ax.text(0.5, -1.6, "Nenhum dos 7 deepfakes reais é usado no treino: eles só aparecem na avaliação "
            "(mostra se a detecção generaliza). Formatos para uma imagem 256×256.",
            fontsize=8.5, color="#333333", style="italic", va="center")

    # legenda
    itens_leg = [(C_ROB, "marca robusta (congelada)"), (C_SEMI, "marca semi-frágil (treinada)"),
                 (C_MSG, "mensagem"), (C_OP, "operação fixa"), (C_ATK, "manipulação"), (C_OK, "resultado")]
    lx = 0.5
    for color, label in itens_leg:
        ax.add_patch(FancyBboxPatch((lx, -3.9), 2.2, 1.4, boxstyle="round,pad=0.02,rounding_size=0.3",
                                    fc=color, ec=EDGE, lw=0.8))
        ax.text(lx + 2.8, -3.2, label, fontsize=8.5, va="center")
        lx += 16.5

    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"arquitetura_marca_dupla.{ext}"), dpi=200, bbox_inches="tight")
    print("Figura:", os.path.join(OUT, "arquitetura_marca_dupla.png"))


if __name__ == "__main__":
    main()
