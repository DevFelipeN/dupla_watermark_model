#!/usr/bin/env python3
"""
analise_fourier.py

Análise de Fourier das duas marcas (robusta e semi-frágil) e um ataque de
remoção no domínio da frequência.

Parte 1 — onde as marcas ficam no espectro
    Para cada imagem: resíduo da robusta (só robusta − original), da
    semi-frágil (dupla − só robusta) e total (dupla − original), na
    luminância Y. FFT 2D de cada resíduo, potência média em todas as imagens.
    Figuras:
      espectro_medio.png   espectro médio (log) da imagem original e dos 3 resíduos
      perfil_radial.png    potência média × frequência radial (0 = DC, 0,5 = Nyquist)
      sobrevivencia.png    fração da potência da semi-frágil que sobra depois de
                           JPEG 75/50/30, resize 0,5/0,3 e blur, por frequência
    "Picos": razão entre o maior valor do espectro e a mediana (sem o DC) e a
    fração da energia nos 1% de frequências mais fortes. Uma marca com picos
    fixos (como uma grade) dá razões altas e é fácil de filtrar.

Parte 2 — ataque de remoção espectral (atacante forte)
    O atacante conhece o espectro médio do resíduo total, estimado na
    PRIMEIRA metade das imagens (ele teria acesso a pares original/marcada:
    é o pior caso para a marca). Ele zera, em R, G e B, a fração q das
    frequências com maior razão marca/imagem, e o ataque é aplicado na
    SEGUNDA metade. Para cada q: robusta (bits certos, marca exata),
    semi-frágil (concordância, % alterada com --limiar) e o estrago na
    imagem (PSNR atacada × dupla).

Saídas em --output: as figuras, resumo_espectro.csv e ataque_espectral.csv.

Ambiente: watermark-v8.

    python analise_fourier.py --num_images 500
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from attacks import AttackLayer
from watermark_embedding import WatermarkEmbedding, watermark_embedding_tensor
from watermark_extraction import WatermarkExtraction
from marca_fragil import mascara_rosto, inserir_fragil, chave_fragil, mensagem_fragil
from pontos_rosto import carregar_pontos, pontos_do_lote, mascara_por_pontos
from train_fragil import carregar_robusta, ROBUSTA_PADRAO
from avaliar_marca_dupla import find_images, limit_by_gender, quantize, SEMIFRAGIL_PADRAO

IMAGE_SIZE = 256
FRACOES = [0.0, 0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90]
ATAQUES_FILTRO = [("JPEG 75", "jpeg_test", {"quality": 75}), ("JPEG 50", "jpeg_test", {"quality": 50}),
                  ("JPEG 30", "jpeg_test", {"quality": 30}), ("resize 0,5", "resize", {"scale": 0.5}),
                  ("resize 0,3", "resize", {"scale": 0.3}), ("blur σ=2", "blur", {"sigma": 2.0})]


def luminancia(x):
    """(B, 3, H, W) em [0, 1] -> Y (B, H, W) em [0, 255]"""
    x = x * 255
    return 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]


def potencia(y):
    """|FFT2|² de (B, H, W), sem shift."""
    return torch.fft.fft2(y.double()).abs() ** 2


def psnr(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-12)
    return 10 * torch.log10(1.0 / mse)


def raio_normalizado(n, device):
    f = torch.fft.fftfreq(n, device=device)
    return torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)


def perfil_radial(p, r, bins):
    """média de p em anéis de frequência radial"""
    idx = torch.bucketize(r.flatten(), bins) - 1
    out = torch.zeros(len(bins) - 1, dtype=p.dtype, device=p.device)
    cnt = torch.zeros_like(out)
    ok = (idx >= 0) & (idx < len(bins) - 1)
    out.index_add_(0, idx[ok], p.flatten()[ok])
    cnt.index_add_(0, idx[ok], torch.ones_like(p.flatten()[ok]))
    return (out / cnt.clamp_min(1)).cpu().numpy()


def picos(p):
    """razão máximo/mediana (sem o DC) e fração da energia no 1% mais forte"""
    v = p.flatten()[1:]
    k = max(1, int(0.01 * v.numel()))
    return (v.max() / v.median()).item(), (v.topk(k).values.sum() / v.sum()).item()


def simetrica(m):
    """garante que a máscara de frequências trate f e −f igual (sinal real)"""
    espelho = torch.roll(torch.flip(m, [0, 1]), (1, 1), (0, 1))
    return m * espelho


def filtrar(x, mascara):
    """multiplica o espectro de cada canal RGB pela máscara (0 = frequência removida)"""
    f = torch.fft.fft2(x.double())
    return torch.fft.ifft2(f * mascara).real.float().clamp(0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fragil", default=SEMIFRAGIL_PADRAO)
    p.add_argument("--robusta", default=ROBUSTA_PADRAO)
    p.add_argument("--celeba_val_dir", default="Dataset/celeba_hq/val")
    p.add_argument("--pontos", default="pontos_scrfd/val.npz")
    p.add_argument("--num_images", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=25)
    p.add_argument("--limiar", type=float, default=0.75)
    p.add_argument("--output", default="results/analise_fourier")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    # -------------------------
    # Modelos
    # -------------------------
    ck_r, emb_r, ext_r = carregar_robusta(args.robusta, IMAGE_SIZE, device)
    nb, grid = ck_r["num_bits"], ck_r["watermark_size"]
    ck_f = torch.load(args.fragil, map_location=device)
    emb_f = WatermarkEmbedding(grid, IMAGE_SIZE, num_bits=nb).to(device).eval()
    emb_f.load_state_dict(ck_f["embedding_model"])
    emb_f.alpha = ck_f["alpha"]
    ext_f = WatermarkExtraction(grid, IMAGE_SIZE, use_hh=False, num_bits=nb).to(device).eval()
    ext_f.load_state_dict(ck_f["extraction_model"])
    piso, elipse = ck_f.get("mask_floor", 0.0), ck_f.get("elipse", "fixa")
    chave = chave_fragil(nb).to(device)
    if elipse in ("scrfd", "posicao"):
        pontos_val, _ = carregar_pontos(args.pontos)
        molde = torch.from_numpy(np.array(ck_f["molde"], dtype=np.float32))
    print(f"Robusta: {args.robusta} | semi-frágil: {args.fragil} (elipse {elipse}) | limiar {args.limiar}")

    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    caminhos = limit_by_gender(find_images(args.celeba_val_dir), args.num_images)
    attack_layer = AttackLayer().to(device).eval()

    def marcar(caminhos_lote):
        x = quantize(torch.stack([transform(Image.open(c).convert("RGB")) for c in caminhos_lote]).to(device))
        m_r = torch.randint(0, 2, (x.shape[0], nb), device=device).float()
        x_r = quantize(watermark_embedding_tensor(x, m_r, emb_r)[0])
        if elipse in ("scrfd", "posicao"):
            pts = pontos_do_lote([os.path.basename(c) for c in caminhos_lote], pontos_val, molde.numpy()).to(device)
            mascara = mascara_por_pontos(pts, molde, IMAGE_SIZE, so_posicao=elipse == "posicao")[0]
        else:
            mascara = mascara_rosto(IMAGE_SIZE).to(device)
        x_d = quantize(inserir_fragil(x_r, mensagem_fragil(m_r, chave), emb_f, mascara, piso)[0])
        return x, x_r, x_d, m_r

    # ==========================================================
    # Parte 1: espectros (todas as imagens)
    # ==========================================================
    N = IMAGE_SIZE
    soma = {k: torch.zeros(N, N, dtype=torch.float64, device=device)
            for k in ["imagem", "robusta", "semi-frágil", "total"]}
    sobra = {nome: torch.zeros(N, N, dtype=torch.float64, device=device) for nome, _, _ in ATAQUES_FILTRO}
    n_total = 0
    metade = len(caminhos) // 2
    soma_estimativa = torch.zeros(N, N, dtype=torch.float64, device=device)
    soma_imagem_est = torch.zeros_like(soma_estimativa)
    lotes_ataque = []

    for i in tqdm(range(0, len(caminhos), args.batch_size), desc="Espectros"):
        lote = caminhos[i:i + args.batch_size]
        x, x_r, x_d, m_r = marcar(lote)
        Y, Yr, Yd = luminancia(x), luminancia(x_r), luminancia(x_d)
        pot = {"imagem": potencia(Y - Y.mean((1, 2), keepdim=True)), "robusta": potencia(Yr - Y),
               "semi-frágil": potencia(Yd - Yr), "total": potencia(Yd - Y)}
        for k, v in pot.items():
            soma[k] += v.sum(0)
        n_total += x.shape[0]

        # quanto da semi-frágil sobra depois de cada ataque, por frequência
        for nome, atk, kw in ATAQUES_FILTRO:
            ad = quantize(attack_layer(x_d, attack=atk, **kw)[0])
            ar = quantize(attack_layer(x_r, attack=atk, **kw)[0])
            sobra[nome] += potencia(luminancia(ad) - luminancia(ar)).sum(0)

        # metade 1: o atacante estima o espectro da marca; metade 2: é atacada
        sel_est = [j for j in range(len(lote)) if i + j < metade]
        sel_atk = [j for j in range(len(lote)) if i + j >= metade]
        if sel_est:
            soma_estimativa += pot["total"][sel_est].sum(0)
            soma_imagem_est += pot["imagem"][sel_est].sum(0)
        if sel_atk:
            lotes_ataque.append((x[sel_atk].cpu(), x_d[sel_atk].cpu(), m_r[sel_atk].cpu()))

    media = {k: v / n_total for k, v in soma.items()}
    r = raio_normalizado(N, device)
    bins = torch.linspace(0, 0.5 * np.sqrt(2), 46, device=device)
    centros = ((bins[1:] + bins[:-1]) / 2).cpu().numpy()

    # --- figura 1: espectro médio
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, k in zip(axs, ["imagem", "robusta", "semi-frágil", "total"]):
        im = ax.imshow(torch.log10(torch.fft.fftshift(media[k]) + 1e-6).cpu().numpy(), cmap="magma",
                       extent=[-0.5, 0.5, 0.5, -0.5])
        ax.set_title({"imagem": "Imagem original", "robusta": "Resíduo da robusta",
                      "semi-frágil": "Resíduo da semi-frágil", "total": "Resíduo das duas marcas"}[k])
        ax.set_xlabel("frequência horizontal (ciclos/px)")
        fig.colorbar(im, ax=ax, fraction=0.046, label="log₁₀ potência")
    axs[0].set_ylabel("frequência vertical (ciclos/px)")
    fig.suptitle(f"Espectro médio de Fourier (luminância Y, {n_total} imagens; centro = frequência 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "espectro_medio.png"), dpi=150)
    plt.close(fig)

    # --- figura 2: perfil radial
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, cor in [("imagem", "gray"), ("robusta", "tab:blue"), ("semi-frágil", "tab:red"), ("total", "tab:purple")]:
        ax.semilogy(centros, perfil_radial(media[k], r, bins), label=k, color=cor)
    ax.axvline(0.5, color="k", lw=0.5, ls=":")
    ax.set_xlabel("frequência radial (ciclos/px; 0,5 = Nyquist)")
    ax.set_ylabel("potência média")
    ax.set_title("Potência por frequência: imagem × resíduos das marcas")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "perfil_radial.png"), dpi=150)
    plt.close(fig)

    # --- figura 3: quanto da semi-frágil sobrevive a cada ataque
    fig, ax = plt.subplots(figsize=(8, 5))
    base = perfil_radial(media["semi-frágil"], r, bins)
    for nome, _, _ in ATAQUES_FILTRO:
        ax.plot(centros, perfil_radial(sobra[nome] / n_total, r, bins) / np.maximum(base, 1e-12), label=nome)
    ax.axhline(1, color="k", lw=0.5, ls=":")
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("frequência radial (ciclos/px; 0,5 = Nyquist)")
    ax.set_ylabel("potência da semi-frágil depois / antes do ataque")
    ax.set_title("Quanto da semi-frágil sobrevive a cada ataque, por frequência")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "sobrevivencia.png"), dpi=150)
    plt.close(fig)

    # --- resumo: picos e faixa de cada resíduo
    linhas = []
    for k in ["imagem", "robusta", "semi-frágil", "total"]:
        pm = media[k]
        razao, frac = picos(pm)
        ordem_r = torch.argsort(r.flatten())
        acum = torch.cumsum(pm.flatten()[ordem_r], 0)
        mediana = r.flatten()[ordem_r][torch.searchsorted(acum, acum[-1] / 2)].item()
        linhas.append({"sinal": k, "pico/mediana": razao, "energia no 1% mais forte (%)": frac * 100,
                       "freq. radial mediana da energia": mediana})
    resumo = pd.DataFrame(linhas)
    resumo.round(4).to_csv(os.path.join(args.output, "resumo_espectro.csv"), index=False)
    print("\nPicos e faixa de frequência (ciclos/px):")
    print(resumo.round(3).to_string(index=False))

    # ==========================================================
    # Parte 2: ataque de remoção espectral
    # ==========================================================
    razao_marca = soma_estimativa / soma_imagem_est.clamp_min(1e-12)
    razao_marca[0, 0] = 0   # o DC (brilho médio) não é tocado
    ordem = torch.argsort(razao_marca.flatten(), descending=True)

    resultados = []
    for q in FRACOES:
        m = torch.ones(N * N, dtype=torch.float64, device=device)
        m[ordem[:int(q * N * N)]] = 0
        m = simetrica(m.view(N, N))
        removidas = 1 - m.mean().item()
        acc_r, exata, concord, alterada, ps_d, ps_o = [], [], [], [], [], []
        for x, x_d, m_r in lotes_ataque:
            x, x_d, m_r = x.to(device), x_d.to(device), m_r.to(device)
            a = quantize(filtrar(x_d, m) if q > 0 else x_d)
            lida_r = (ext_r(a) >= 0).float()
            lida_f = (ext_f(a) >= 0).float()
            c = (lida_f == mensagem_fragil(lida_r, chave)).float().mean(1)
            acc_r.append((lida_r == m_r).float().mean(1))
            exata.append((lida_r == m_r).all(1).float())
            concord.append(c)
            alterada.append((c < args.limiar).float())
            ps_d.append(psnr(a, x_d))
            ps_o.append(psnr(a, x))
        cat = lambda v: torch.cat(v).mean().item()
        resultados.append({"frequências removidas (%)": removidas * 100,
                           "robusta: bits certos (%)": cat(acc_r) * 100,
                           "robusta: marca exata (%)": cat(exata) * 100,
                           "semi-frágil: concordância (%)": cat(concord) * 100,
                           "alterada (%)": cat(alterada) * 100,
                           "PSNR atacada × dupla (dB)": cat(ps_d),
                           "PSNR atacada × original (dB)": cat(ps_o)})
    tab = pd.DataFrame(resultados)
    tab.round(2).to_csv(os.path.join(args.output, "ataque_espectral.csv"), index=False)
    print(f"\nAtaque espectral: o atacante zera as frequências com maior razão marca/imagem "
          f"(estimadas em {metade} imagens, aplicado em {len(caminhos) - metade}):")
    print(tab.round(2).to_string(index=False))
    print(f"\nSalvo em {args.output}")


if __name__ == "__main__":
    main()
