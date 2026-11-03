#!/usr/bin/env python3
"""
comparar_deslocado.py

Elipse FIXA x elipse pelo SCRFD, com o rosto fora do lugar do CelebA-HQ.

As mesmas imagens de validação passam por quatro cenários ANTES de serem
marcadas (como se fossem outras fotos):

    original    o CelebA-HQ como é (rosto centralizado)
    deslocado   mesmo tamanho, rosto deslocado até ±40 px
    rosto 75%   imagem reduzida a 75%, girada (±15°) e deslocada
    rosto 50%   imagem reduzida a 50%, girada (±15°) e deslocada

Em cada cenário a imagem recebe a robusta (igual para todos) e a semi-frágil
de cada modelo: o fixo marca a elipse no centro da imagem; o SCRFD marca a
elipse no rosto (pontos do SCRFD levados junto com a imagem); o "posicao"
(opcional, --posicao) desloca a elipse até o rosto, sem mudar o tamanho. Depois:

    ataques benignos   não deveriam acusar "alterada" (alarme falso)
    manipulações       deveriam acusar "alterada" (detecção); elas mexem no
                       rosto VERDADEIRO, onde quer que ele esteja, para os dois
                       modelos

Protocolo igual ao avaliar_marca_dupla.py: lê a robusta, semi-frágil esperada =
robusta XOR chave, concordância < --limiar = alterada. Cada ataque usa o mesmo
sorteio nos dois modelos.

Os pontos do rosto são os do SCRFD na imagem original, transformados junto com
a imagem (não há nova detecção depois de reduzir/girar).

Saídas em --output: resultados.csv (cenário, modelo, ataque, % alterada,
concordância, robusta) e a tabela de % alterada no terminal.

Ambiente: watermark-v8.

    python comparar_deslocado.py --num_images 500
    python comparar_deslocado.py --num_images 500 --posicao output_fragil/fragil_posicao/checkpoints/<nome>_melhor.pth
"""

import os
import random
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from attacks import AttackLayer
from watermark_embedding import WatermarkEmbedding, watermark_embedding_tensor
from watermark_extraction import WatermarkExtraction
from marca_fragil import mascara_rosto, inserir_fragil, chave_fragil, mensagem_fragil, MANIPULACOES
from pontos_rosto import carregar_pontos, pontos_do_lote, mascara_por_pontos, aumento_geometrico
from train_fragil import carregar_robusta, ROBUSTA_PADRAO
from avaliar_marca_dupla import find_images, limit_by_gender, quantize

IMAGE_SIZE = 256

# (nome, escala, rotação máxima em graus, deslocamento extra máximo em px)
CENARIOS = [("original", None, 0, 0), ("deslocado", 1.0, 0, 40), ("rosto 75%", 0.75, 15, 0), ("rosto 50%", 0.5, 15, 0)]

BENIGNOS = [
    ("sem ataque", "none",        {}),
    ("JPEG 75",    "jpeg_test",   {"quality": 75}),
    ("JPEG 50",    "jpeg_test",   {"quality": 50}),
    ("blur",       "blur",        {"sigma": 2.0}),
    ("resize 0,5", "resize",      {"scale": 0.5}),
]


def carregar_fragil(caminho, ck_r, device):
    ck = torch.load(caminho, map_location=device)
    emb = WatermarkEmbedding(ck_r["watermark_size"], IMAGE_SIZE, num_bits=ck_r["num_bits"]).to(device).eval()
    emb.load_state_dict(ck["embedding_model"])
    emb.alpha = ck["alpha"]
    ext = WatermarkExtraction(ck_r["watermark_size"], IMAGE_SIZE, use_hh=False, num_bits=ck_r["num_bits"]).to(device).eval()
    ext.load_state_dict(ck["extraction_model"])
    return emb, ext, ck.get("mask_floor", 0.0), ck.get("elipse", "fixa")


def semente(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixa", default="output/checkpoints/semifragil_final.pth", help="Semi-frágil com elipse fixa")
    p.add_argument("--scrfd", default="output_fragil/fragil_scrfd/checkpoints/fragil_28102026_15:34:08_melhor.pth",
                   help="Semi-frágil com elipse pelo SCRFD")
    p.add_argument("--posicao", default=None, help="Semi-frágil com elipse só deslocada até o rosto (opcional)")
    p.add_argument("--robusta", default=ROBUSTA_PADRAO)
    p.add_argument("--celeba_val_dir", default="Dataset/celeba_hq/val")
    p.add_argument("--pontos", default="pontos_scrfd/val.npz")
    p.add_argument("--num_images", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--limiar", type=float, default=0.8)
    p.add_argument("--output", default="results/comparar_deslocado")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    ck_r, emb_r, ext_r = carregar_robusta(args.robusta, IMAGE_SIZE, device)
    num_bits = ck_r["num_bits"]
    modelos = {"fixa": carregar_fragil(args.fixa, ck_r, device), "SCRFD": carregar_fragil(args.scrfd, ck_r, device)}
    if args.posicao:
        modelos["posicao"] = carregar_fragil(args.posicao, ck_r, device)
    pontos, molde = carregar_pontos(args.pontos)
    molde_t = torch.from_numpy(molde)
    chave = chave_fragil(num_bits).to(device)
    fixa_centro = mascara_rosto(IMAGE_SIZE).to(device)
    attack_layer = AttackLayer().to(device).eval()

    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    caminhos = limit_by_gender(find_images(args.celeba_val_dir), args.num_images)
    nomes = [os.path.basename(c) for c in caminhos]

    linhas = []
    for cenario, escala, rotacao, deslocamento in CENARIOS:
        for i in tqdm(range(0, len(caminhos), args.batch_size), desc=cenario):
            x = torch.stack([transform(Image.open(c).convert("RGB")) for c in caminhos[i:i + args.batch_size]]).to(device)
            pts = pontos_do_lote(nomes[i:i + args.batch_size], pontos, molde).to(device)
            outras = x.roll(1, 0)
            if escala is not None:
                semente(args.seed + i)
                x, pts, outras = aumento_geometrico(x, pts, outras, prob=1.0, escala=(escala, escala),
                                                    rotacao=rotacao, deslocamento=deslocamento)
            x = quantize(x)
            rosto, geo = mascara_por_pontos(pts, molde_t, IMAGE_SIZE)
            so_posicao = mascara_por_pontos(pts, molde_t, IMAGE_SIZE, so_posicao=True)[0]

            B = x.shape[0]
            semente(args.seed + i)
            m_r = torch.randint(0, 2, (B, num_bits), device=device).float()
            x_r = quantize(watermark_embedding_tensor(x, m_r, emb_r)[0])

            for modelo, (emb_f, ext_f, piso, elipse) in modelos.items():
                mascara = {"scrfd": rosto, "posicao": so_posicao}.get(elipse, fixa_centro)
                x_d = quantize(inserir_fragil(x_r, mensagem_fragil(m_r, chave), emb_f, mascara, piso)[0])

                ataques = [(n, "benigno", lambda a=a, kw=kw: attack_layer(x_d, attack=a, **kw)[0])
                           for n, a, kw in BENIGNOS]
                # a manipulação mexe no rosto verdadeiro (geo), para os dois modelos
                ataques += [(n, "manipulação", lambda fn=fn: fn(x_d, rosto, outras, geo=geo))
                            for n, fn in MANIPULACOES.items()]
                for k, (nome, tipo, atacar) in enumerate(ataques):
                    semente(args.seed + 1000 * i + k)
                    a = quantize(atacar())
                    lida_r = (ext_r(a) >= 0).float()
                    lida_f = (ext_f(a) >= 0).float()
                    concord = (lida_f == mensagem_fragil(lida_r, chave)).float().mean(1)
                    robusta = (lida_r == m_r).float().mean(1)
                    for j in range(B):
                        linhas.append({"cenario": cenario, "modelo": modelo, "tipo": tipo, "ataque": nome,
                                       "concordancia": concord[j].item(), "robusta": robusta[j].item(),
                                       "alterada": float(concord[j].item() < args.limiar)})

    df = pd.DataFrame(linhas)
    res = df.groupby(["cenario", "modelo", "tipo", "ataque"], sort=False).mean().reset_index()
    for c in ["concordancia", "robusta", "alterada"]:
        res[c] *= 100
    res.round(2).to_csv(os.path.join(args.output, "resultados.csv"), index=False)

    tabela = res.pivot_table(index=["tipo", "ataque"], columns=["cenario", "modelo"], values="alterada", sort=False)
    pd.set_option("display.width", 250)
    print("\n% de imagens acusadas de ALTERADA")
    print("(benigno: quanto MENOR, melhor = alarme falso | manipulação: quanto MAIOR, melhor = detecção)\n")
    print(tabela.round(1).to_string())
    print(f"\nSalvo em {os.path.join(args.output, 'resultados.csv')}")


if __name__ == "__main__":
    main()
