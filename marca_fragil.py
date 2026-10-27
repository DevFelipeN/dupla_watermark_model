"""
marca_fragil.py

Peças da marca SEMI-FRÁGIL (versão 4, marca dupla):

    - mascara_rosto: elipse suave centrada; no CelebA-HQ os rostos estão
      alinhados, então uma elipse fixa cobre o rosto sem precisar de detector;
    - inserir_fragil: mesma rede da marca robusta (Y -> DWT -> DCT -> CNN),
      mas a diferença em Y é multiplicada pela máscara do rosto, ou seja, a
      marca semi-frágil fica concentrada no rosto;
    - MANIPULACOES: manipulações baratas e diferenciáveis usadas no treino
      como "deepfake de mentira" (4 grosseiras: troca, regeneração,
      deformação e mistura do rosto; 3 sutis: troca do miolo do rosto,
      troca dos detalhes finos e edição de uma região pequena). Nenhum dos
      7 deepfakes reais é usado no treino, só na avaliação (mostra se a
      detecção generaliza);
    - chave_fragil: na avaliação a mensagem semi-frágil é a robusta XOR uma chave
      fixa; quem verifica lê a robusta, calcula a semi-frágil esperada e compara.
"""

import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from watermark_embedding import watermark_embedding_tensor


# ==========================================================
# Máscara do rosto
# ==========================================================

def mascara_rosto(size, cx=0.50, cy=0.48, ax=0.34, ay=0.44, nitidez=9.0):
    """Elipse suave (1, 1, size, size): 1 no centro do rosto, 0 fora, transição suave.
    Mesmos parâmetros do experimento semi-frágil do DeepGuard (CelebA alinhado)."""
    ys = torch.linspace(0, 1, size).view(size, 1).expand(size, size)
    xs = torch.linspace(0, 1, size).view(1, size).expand(size, size)
    d = torch.sqrt(((xs - cx) / ax) ** 2 + ((ys - cy) / ay) ** 2)   # 1 na borda da elipse
    return torch.sigmoid((1.0 - d) * nitidez).view(1, 1, size, size)


# ==========================================================
# Inserção da marca semi-frágil
# ==========================================================

def inserir_fragil(imagem, bits, modelo, mascara, piso=0.0):
    """
    imagem: (B, 3, H, W) em [0, 1] (já com a marca robusta)
    bits:   (B, num_bits) em {0, 1}
    mascara: (1, 1, H, W); piso > 0 deixa um pouco de marca fora do rosto.

    Devolve a imagem com as duas marcas e a modificação (para o log).
    """
    marcada, _, modificacao = watermark_embedding_tensor(imagem, bits, modelo)
    peso = piso + (1.0 - piso) * mascara
    # a diferença é a mesma em R, G e B (só Y muda), então multiplicar a
    # diferença RGB pela máscara equivale a multiplicar o ΔY
    dupla = imagem + (marcada - imagem) * peso
    return dupla.clamp(0, 1), modificacao


# ==========================================================
# Manipulações de treino ("deepfakes de mentira")
# ==========================================================

def _mascara_variada(mascara, B):
    """Elipse do rosto com escala e deslocamento sorteados por imagem."""
    _, _, H, W = mascara.shape
    dev = mascara.device
    escala = 1 + (torch.rand(B, device=dev) * 2 - 1) * 0.15
    desloc = (torch.rand(B, 2, device=dev) * 2 - 1) * 0.06
    zero = torch.zeros(B, device=dev)
    theta = torch.stack([
        torch.stack([1 / escala, zero, desloc[:, 0]], 1),
        torch.stack([zero, 1 / escala, desloc[:, 1]], 1),
    ], 1)
    grid = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
    return F.grid_sample(mascara.expand(B, -1, -1, -1), grid, align_corners=False)


def _casar_cor(fonte, alvo, m):
    """Ajusta média e desvio de cada canal da fonte aos do alvo dentro da máscara
    (como o face swap faz para o rosto novo combinar com o resto da imagem)."""
    peso = m.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    def stats(x):
        mu = (x * m).sum(dim=(2, 3), keepdim=True) / peso
        sd = (((x - mu) ** 2) * m).sum(dim=(2, 3), keepdim=True).div(peso).sqrt().clamp_min(1e-3)
        return mu, sd
    mu_f, sd_f = stats(fonte)
    mu_a, sd_a = stats(alvo)
    return ((fonte - mu_f) / sd_f * sd_a + mu_a).clamp(0, 1)


def troca_rosto(imagens, mascara, outras):
    """Face swap de mentira: cola o rosto de outra imagem do batch (que também
    tem as suas marcas, com outras mensagens), com a cor ajustada e borda suave."""
    B = imagens.shape[0]
    m = _mascara_variada(mascara, B)
    fonte = _casar_cor(outras, imagens, m)
    return imagens * (1 - m) + fonte * m


def regenera_rosto(imagens, mascara, outras=None):
    """Rosto "regenerado": a região do rosto perde os detalhes finos (reduz
    para 1/4 e volta) e ganha ruído novo, como a saída de um gerador."""
    B, _, H, W = imagens.shape
    m = _mascara_variada(mascara, B)
    fator = random.choice([0.2, 0.25, 0.33])
    baixa = F.interpolate(imagens, scale_factor=fator, mode="bilinear", align_corners=False)
    regen = F.interpolate(baixa, size=(H, W), mode="bilinear", align_corners=False)
    regen = (regen + torch.randn_like(regen) * random.uniform(0.01, 0.03)).clamp(0, 1)
    return imagens * (1 - m) + regen * m


def deforma_rosto(imagens, mascara, outras=None):
    """Reencenação de mentira: desloca os pixels do rosto com um campo de
    deformação suave e aleatório (muda expressão/pose sem trocar o rosto)."""
    B, _, H, W = imagens.shape
    dev = imagens.device
    m = _mascara_variada(mascara, B)
    campo = torch.randn(B, 2, 4, 4, device=dev) * random.uniform(0.03, 0.06)
    campo = F.interpolate(campo, size=(H, W), mode="bicubic", align_corners=False)
    base = F.affine_grid(torch.eye(2, 3, device=dev).unsqueeze(0).expand(B, -1, -1),
                         (B, 3, H, W), align_corners=False)
    grid = base + campo.permute(0, 2, 3, 1) * m.permute(0, 2, 3, 1)
    return F.grid_sample(imagens, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def mistura_rosto(imagens, mascara, outras):
    """Troca parcial: mistura os dois rostos (como um swap mal feito ou morphing)."""
    B = imagens.shape[0]
    m = _mascara_variada(mascara, B) * random.uniform(0.5, 0.9)
    return imagens * (1 - m) + _casar_cor(outras, imagens, m) * m


def _elipse(size, cx, cy, ax, ay, nitidez, device):
    return mascara_rosto(size, cx, cy, ax, ay, nitidez).to(device)


def troca_miolo(imagens, mascara, outras):
    """Como o BlendFace: troca só o miolo do rosto (olhos, nariz e boca), com
    borda bem suave, mantendo contorno, testa e queixo da imagem original."""
    B, _, H, W = imagens.shape
    miolo = _elipse(H, 0.50, 0.53, random.uniform(0.18, 0.24), random.uniform(0.22, 0.28),
                    random.uniform(4.0, 6.0), imagens.device)
    m = _mascara_variada(miolo, B)
    return imagens * (1 - m) + _casar_cor(outras, imagens, m) * m


def _passa_baixa(x, sigma):
    k = int(2 * round(3 * sigma) + 1)
    return TF.gaussian_blur(x, kernel_size=k, sigma=sigma)


def troca_detalhes(imagens, mascara, outras):
    """Como o DiffAE/StarGAN: o rosto mantém cor e forma (frequências baixas),
    mas os detalhes finos (frequências altas, onde a marca fica) são refeitos;
    aqui vêm de outra imagem. Visualmente quase não muda."""
    B = imagens.shape[0]
    m = _mascara_variada(mascara, B)
    sigma = random.uniform(0.8, 1.5)
    baixa = _passa_baixa(imagens, sigma)
    detalhes = outras - _passa_baixa(outras, sigma)
    novo = (baixa + detalhes * random.uniform(0.6, 1.0)).clamp(0, 1)
    return imagens * (1 - m) + novo * m


# regiões pequenas do rosto alinhado (cx, cy, ax, ay)
_REGIOES = {
    "olhos": (0.50, 0.43, 0.26, 0.08),
    "olho_esq": (0.37, 0.43, 0.10, 0.07),
    "olho_dir": (0.63, 0.43, 0.10, 0.07),
    "boca": (0.50, 0.70, 0.15, 0.07),
    "nariz": (0.50, 0.56, 0.07, 0.09),
}


def edita_regiao(imagens, mascara, outras):
    """Como o HiSD (óculos) ou edição de atributo: só uma região pequena
    (olhos, boca ou nariz) vem de outra imagem."""
    B, _, H, W = imagens.shape
    cx, cy, ax, ay = random.choice(list(_REGIOES.values()))
    regiao = _elipse(H, cx, cy, ax, ay, random.uniform(5.0, 8.0), imagens.device)
    m = _mascara_variada(regiao, B)
    return imagens * (1 - m) + _casar_cor(outras, imagens, m) * m


MANIPULACOES = {
    "troca_rosto": troca_rosto,
    "regenera_rosto": regenera_rosto,
    "deforma_rosto": deforma_rosto,
    "mistura_rosto": mistura_rosto,
    "troca_miolo": troca_miolo,
    "troca_detalhes": troca_detalhes,
    "edita_regiao": edita_regiao,
}


# ==========================================================
# Chave: mensagem semi-frágil derivada da robusta
# ==========================================================

def chave_fragil(num_bits, seed=12345):
    """Chave fixa de num_bits (0/1). Na prática ela é o segredo de quem marca."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (num_bits,), generator=g).float()


def mensagem_fragil(bits_robustos, chave):
    """Mensagem semi-frágil = mensagem robusta XOR chave."""
    return (bits_robustos + chave.to(bits_robustos.device)) % 2
