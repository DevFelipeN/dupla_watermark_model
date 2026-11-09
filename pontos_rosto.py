"""
pontos_rosto.py

Posiciona a elipse da marca semi-frágil pelo rosto detectado (SCRFD), em vez
de deixá-la fixa no centro da imagem.

Os 5 pontos do rosto (olhos, nariz e cantos da boca) de cada imagem vêm de
preparar_pontos_scrfd.py (calculados uma vez e salvos em .npz). A elipse do
treino (marca_fragil.mascara_rosto) foi desenhada para o rosto do CelebA-HQ
alinhado; o "molde" é a posição média dos 5 pontos nesse rosto alinhado. Para
cada imagem, uma transformação de semelhança (rotação, escala uniforme e
translação) leva o molde aos pontos detectados, e a mesma transformação leva a
elipse até o rosto.

Também tem o aumento geométrico do treino: reduz, gira e desloca a imagem (e os
pontos junto), para a semi-frágil aprender a ler a marca com o rosto em
qualquer posição.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from marca_fragil import mascara_rosto


# ==========================================================
# Pontos salvos
# ==========================================================

def carregar_pontos(caminho):
    """Devolve ({nome do arquivo: pontos (5, 2) em px de 256x256}, molde (5, 2)).
    Imagens sem rosto detectado ficam de fora (quem usa cai no molde)."""
    d = np.load(caminho, allow_pickle=False)
    pontos = {str(n): p for n, p, ok in zip(d["nomes"], d["pontos"], d["detectado"]) if ok}
    return pontos, d["molde"].astype(np.float32)


def pontos_do_lote(nomes, pontos, molde):
    """(B, 5, 2) para uma lista de nomes de arquivo; sem detecção = molde."""
    return torch.from_numpy(np.stack([pontos.get(n, molde) for n in nomes]).astype(np.float32))


# ==========================================================
# Geometria (matrizes 3x3 em pixels)
# ==========================================================

def semelhanca(src, dst):
    """Umeyama em lote: (B, K, 2) -> (B, K, 2). Devolve (B, 3, 3) que leva src a dst."""
    ms, md = src.mean(1, keepdim=True), dst.mean(1, keepdim=True)
    s, d = src - ms, dst - md
    cov = d.transpose(1, 2) @ s / src.shape[1]
    u, sig, vt = torch.linalg.svd(cov)
    sinal = torch.sign(torch.linalg.det(u @ vt))
    D = torch.diag_embed(torch.stack([torch.ones_like(sinal), sinal], 1))
    r = u @ D @ vt
    escala = (sig * torch.diagonal(D, dim1=1, dim2=2)).sum(1) / (s ** 2).sum(2).mean(1)
    t = md.squeeze(1) - escala[:, None] * (r @ ms.transpose(1, 2)).squeeze(2)
    m = torch.zeros(src.shape[0], 3, 3, device=src.device, dtype=src.dtype)
    m[:, :2, :2] = escala[:, None, None] * r
    m[:, :2, 2] = t
    m[:, 2, 2] = 1
    return m


def _normalizar(size, device, dtype):
    """px -> coordenadas de affine_grid (align_corners=False)."""
    return torch.tensor([[2 / size, 0, 1 / size - 1], [0, 2 / size, 1 / size - 1], [0, 0, 1]],
                        device=device, dtype=dtype)


def theta_amostragem(m, size):
    """Para grid_sample: a matriz m leva a imagem de ORIGEM à de DESTINO (px).
    Devolve o theta (B, 2, 3) que, para cada pixel do destino, aponta onde ler na origem."""
    n = _normalizar(size, m.device, m.dtype)
    return (n @ torch.linalg.inv(m) @ torch.linalg.inv(n))[:, :2]


def aplicar_pontos(m, pontos):
    """(B, 3, 3) x (B, K, 2) -> (B, K, 2)"""
    return pontos @ m[:, :2, :2].transpose(1, 2) + m[:, None, :2, 2]


# ==========================================================
# Elipse pelo rosto
# ==========================================================

def posicionar(mascara_canonica, geo):
    """Leva uma máscara desenhada no rosto alinhado (1 ou B, 1, H, W) até o rosto
    de cada imagem. geo = theta de mascara_por_pontos."""
    B, H = geo.shape[0], mascara_canonica.shape[-1]
    grid = F.affine_grid(geo, (B, 1, H, H), align_corners=False)
    return F.grid_sample(mascara_canonica.expand(B, -1, -1, -1), grid, align_corners=False)


def translacao(src, dst):
    """(B, 3, 3) que só desloca: leva o centro dos pontos src ao centro dos dst."""
    m = torch.eye(3, device=src.device, dtype=src.dtype).repeat(src.shape[0], 1, 1)
    m[:, :2, 2] = dst.mean(1) - src.mean(1)
    return m


def mascara_por_pontos(pontos, molde, size, so_posicao=False):
    """Elipse do treino levada até o rosto de cada imagem pelos 5 pontos.
    so_posicao=True: a elipse só é deslocada até o rosto (mesmo tamanho e
    inclinação da fixa); senão, também acompanha a escala e a rotação do rosto.
    Devolve a máscara (B, 1, H, W) e geo (B, 2, 3), que posiciona qualquer outra
    máscara desenhada no rosto alinhado (as regiões das manipulações)."""
    B = pontos.shape[0]
    alvo = molde.to(pontos.device, pontos.dtype).expand(B, -1, -1)
    m = translacao(alvo, pontos) if so_posicao else semelhanca(alvo, pontos)
    geo = theta_amostragem(m, size)
    return posicionar(mascara_rosto(size).to(pontos.device, pontos.dtype), geo), geo


# ==========================================================
# Aumento geométrico (só no treino)
# ==========================================================

def aumento_geometrico(x, pontos, outras, prob=0.7, escala=(0.5, 1.0), rotacao=15.0, deslocamento=0.0):
    """Reduz (escala), gira (±rotacao graus) e desloca uma fração prob das imagens
    do lote, com o mesmo movimento nos pontos e nas imagens "outras" (de onde as
    manipulações tiram o rosto colado, que assim cai em cima do rosto do alvo).
    O deslocamento vai até a folga deixada pela redução mais ±deslocamento px.
    O que sobra em volta é preenchido com outra imagem do lote (um fundo de foto).
    Devolve x, pontos e outras transformados."""
    B, _, H, W = x.shape
    dev, dt = x.device, x.dtype
    usa = (torch.rand(B, device=dev) < prob).to(dt)
    s = 1 + usa * (torch.empty(B, device=dev).uniform_(*escala) - 1)
    ang = usa * torch.empty(B, device=dev).uniform_(-rotacao, rotacao) * math.pi / 180
    folga = (1 - s) * W / 2 + usa * deslocamento
    t = (torch.rand(B, 2, device=dev) * 2 - 1) * folga[:, None]
    c = W / 2
    cos, sin = torch.cos(ang) * s, torch.sin(ang) * s
    m = torch.zeros(B, 3, 3, device=dev, dtype=dt)
    m[:, 0, 0], m[:, 0, 1], m[:, 1, 0], m[:, 1, 1] = cos, -sin, sin, cos
    m[:, 0, 2] = c - cos * c + sin * c + t[:, 0]
    m[:, 1, 2] = c - sin * c - cos * c + t[:, 1]
    m[:, 2, 2] = 1

    grid = F.affine_grid(theta_amostragem(m, W), (B, 1, H, W), align_corners=False)
    dentro = F.grid_sample(torch.ones(B, 1, H, W, device=dev, dtype=dt), grid, align_corners=False)
    fundo = x.roll(2, 0)

    def mover(img):
        return F.grid_sample(img, grid, align_corners=False) * dentro + fundo * (1 - dentro)

    return mover(x), aplicar_pontos(m, pontos), mover(outras)
