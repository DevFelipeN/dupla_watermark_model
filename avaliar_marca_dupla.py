#!/usr/bin/env python3
"""
avaliar_marca_dupla.py

Avalia a MARCA DUPLA da versão 4: robusta v3 (origem) + semi-frágil treinada
com o train_fragil.py (integridade), lidas da mesma imagem.

Protocolo (como quem verifica faria, sem guardar a imagem original):
    1. lê a marca robusta -> mensagem de origem (30 bits);
    2. calcula a mensagem semi-frágil esperada = robusta lida XOR chave;
    3. lê a marca semi-frágil e mede a CONCORDÂNCIA com a esperada;
    4. concordância < --limiar  ->  imagem ALTERADA (deepfake).

Nos ataques benignos a imagem não deveria ser marcada como alterada (falso
positivo); nos deepfakes, deveria (detecção). Nenhum deepfake foi usado no
treino da semi-frágil.

Ataques: os 16 do evaluate.py + os 7 deepfakes, nas mesmas --num_images
imagens (metade de cada gênero) e com as mesmas origens (pares_origem.csv).

Saídas em --output:
    media_per_attack.csv        mesma tabela do evaluate.py (robusta: BER, Bit
                                Accuracy, Accuracy; PSNR, SSIM, LPIPS e
                                DISTS original x atacada) + colunas da
                                semi-frágil (Bit Accuracy, concordância e %
                                de imagens marcadas como alteradas)
    results_per_image.csv       uma linha por imagem e ataque
    qualidade.csv               original x dupla e original x só robusta
    exemplo.png                 original | só robusta | dupla | diferença da semi-frágil x20

Ambiente: watermark-v8.

Exemplo:
    python avaliar_marca_dupla.py --num_images 200 --output results/marca_dupla_200
"""

import os
import sys
import math
import random
import shutil
import argparse

import lpips
from tabulate import tabulate
from DISTS_pytorch import DISTS
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torchmetrics.functional.image import structural_similarity_index_measure

from attacks import AttackLayer
from deepfake_attack import DeepFakeAttack, DEEPFAKE_ROOT
from watermark_embedding import WatermarkEmbedding, watermark_embedding_tensor
from watermark_extraction import WatermarkExtraction
from marca_fragil import mascara_rosto, inserir_fragil, chave_fragil, mensagem_fragil
from pontos_rosto import carregar_pontos, pontos_do_lote, mascara_por_pontos
from train_fragil import carregar_robusta, ROBUSTA_PADRAO

# 20 identidades de origem usadas pelos deepfakes de troca de rosto (só com deepfakes)
ORIGENS_FACESWAP_DIR = os.path.join(DEEPFAKE_ROOT, "origens_faceswap")
SEMIFRAGIL_PADRAO = "output/checkpoints/semifragil.pth"

IMAGE_SIZE = 256

# Mesmos ataques e parâmetros do evaluate.py
ATTACK_CASES = [
    ("none",               "none",        {}),
    ("jpeg_test",          "jpeg_test",   {"quality": 75}),
    ("jpeg_test_forte50",  "jpeg_test",   {"quality": 50}),
    ("jpeg_test_forte30",  "jpeg_test",   {"quality": 30}),
    ("blur",               "blur",        {"sigma": 2.0}),
    ("median_blur",        "median_blur", {"kernel": 3}),
    ("median_blur_forte",  "median_blur", {"kernel": 5}),
    ("noise",              "noise",       {"std": 0.02}),
    ("noise_forte",        "noise",       {"std": 0.05}),
    ("salt_pepper",        "salt_pepper", {"amount": 0.01}),
    ("salt_pepper_forte",  "salt_pepper", {"amount": 0.05}),
    ("dropout",            "dropout",     {"p": 0.05}),
    ("dropout_forte10",    "dropout",     {"p": 0.1}),
    ("dropout_forte20",    "dropout",     {"p": 0.2}),
    ("resize",             "resize",      {"scale": 0.5}),
    ("resize_forte",       "resize",      {"scale": 0.3}),
]

DEEPFAKES = ["diffae", "blendface", "simswap", "inswapper", "stargan", "hisd", "liveportrait"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_images(root):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


def limit_by_gender(paths, limit):
    """N imagens, metade de cada gênero (igual ao --limit do evaluate.py)."""
    female = [p for p in paths if os.path.basename(os.path.dirname(p)) == "female"]
    male = [p for p in paths if os.path.basename(os.path.dirname(p)) == "male"]
    n_male = limit // 2
    return sorted(female[:limit - n_male] + male[:n_male])


def to_uint8(x_01):
    return (x_01[0].permute(1, 2, 0).detach().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)


def quantize(x_01):
    """Simula salvar em PNG de 8 bits."""
    return torch.round(x_01.clamp(0, 1) * 255) / 255


def auc(pos, neg):
    """Probabilidade de um deepfake ter concordância MENOR que uma imagem benigna
    (área sob a curva ROC; empates contam metade)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    menor = (pos[:, None] < neg[None, :]).mean()
    igual = (pos[:, None] == neg[None, :]).mean()
    return float(menor + 0.5 * igual)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragil", default=SEMIFRAGIL_PADRAO, help="Checkpoint da marca semi-frágil (padrão: o modelo atual)")
    parser.add_argument("--robusta", default=None,
                        help="Checkpoint da robusta (padrão: o salvo no checkpoint da semi-frágil, se existir; senão, o modelo atual)")
    parser.add_argument("--celeba_val_dir", default="Dataset/celeba_hq/val")
    parser.add_argument("--output", default="results/marca_dupla")
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--limiar", type=float, default=0.75,
                        help="Concordância da semi-frágil abaixo disso = imagem alterada (0,75 = pelo menos 23 de 30 bits)")
    parser.add_argument("--alpha", type=float, default=None, help="Alpha da semi-frágil (padrão: o do checkpoint)")
    parser.add_argument("--sem_deepfakes", action="store_true", help="Só os 16 ataques (rápido)")
    parser.add_argument("--deepfakes", default=",".join(DEEPFAKES), help="Deepfakes a rodar, separados por vírgula")
    parser.add_argument("--pontos", default="pontos_scrfd/val.npz",
                        help="Pontos do SCRFD das imagens (preparar_pontos_scrfd.py); usados quando o checkpoint "
                             "da semi-frágil foi treinado com a elipse pelo SCRFD")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shutil.rmtree(args.output, ignore_errors=True)
    os.makedirs(args.output, exist_ok=True)

    # -------------------------
    # Modelos
    # -------------------------
    ck_f = torch.load(args.fragil, map_location=device)
    salvo = ck_f.get("robust_checkpoint")
    robusta_path = args.robusta or (salvo if salvo and os.path.exists(salvo) else ROBUSTA_PADRAO)
    ck_r, emb_r, ext_r = carregar_robusta(robusta_path, IMAGE_SIZE, device)
    num_bits, grid = ck_r["num_bits"], ck_r["watermark_size"]

    emb_f = WatermarkEmbedding(grid, IMAGE_SIZE, num_bits=num_bits).to(device).eval()
    emb_f.load_state_dict(ck_f["embedding_model"])
    emb_f.alpha = args.alpha if args.alpha is not None else ck_f["alpha"]
    ext_f = WatermarkExtraction(grid, IMAGE_SIZE, use_hh=False, num_bits=num_bits).to(device).eval()
    ext_f.load_state_dict(ck_f["extraction_model"])
    mascara = mascara_rosto(IMAGE_SIZE).to(device)
    elipse = ck_f.get("elipse", "fixa")
    if elipse in ("scrfd", "posicao"):
        # a elipse de cada imagem segue o rosto achado pelo SCRFD (posicao: só desloca, tamanho fixo)
        pontos_val, _ = carregar_pontos(args.pontos)
        molde = np.array(ck_f["molde"], dtype=np.float32)
    piso = ck_f.get("mask_floor", 0.0)
    chave = chave_fragil(num_bits).to(device)
    print(f"Robusta: {robusta_path} (alpha {emb_r.alpha}) | semi-frágil: {args.fragil} (alpha {emb_f.alpha}, piso {piso}, elipse {elipse})")

    def ler_robusta(x):
        with torch.no_grad():
            return (ext_r(x.clamp(0, 1)) >= 0).float()

    def ler_fragil(x):
        with torch.no_grad():
            return (ext_f(x.clamp(0, 1)) >= 0).float()

    # -------------------------
    # Imagens e mensagens
    # -------------------------
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    images = limit_by_gender(find_images(args.celeba_val_dir), args.num_images)
    pairs_csv = os.path.join(args.output, "pares_origem.csv")
    if not args.sem_deepfakes:
        sys.path.append(ORIGENS_FACESWAP_DIR)
        from pares_origem import gerar_pares, salvar_pares, genero_pelo_caminho
        salvar_pares(
            gerar_pares([(os.path.splitext(os.path.basename(p))[0], genero_pelo_caminho(p)) for p in images]),
            pairs_csv
        )
    deepfake_params = {
        "diffae": {"attribute": "Young", "strength": 0.3, "image_resolution": IMAGE_SIZE},
        "blendface": {"pairs_csv": pairs_csv},
        "simswap": {"pairs_csv": pairs_csv},
        "inswapper": {"pairs_csv": pairs_csv},
        "stargan": {"attribute": "Male"},
        "hisd": {"tag": "Eyeglasses", "attribute": "with", "seed": args.seed},
        "liveportrait": {"pairs_csv": pairs_csv},
    }

    lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    dists_metric = DISTS().to(device).eval()
    originais, so_robusta, duplas, mensagens, qualidade = {}, {}, {}, {}, []

    for path in tqdm(images, desc="Marcando"):
        name = os.path.splitext(os.path.basename(path))[0]
        x = quantize(transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device))
        m_r = torch.randint(0, 2, (1, num_bits), device=device).float()
        m_f = mensagem_fragil(m_r, chave)
        with torch.no_grad():
            x_r = quantize(watermark_embedding_tensor(x, m_r, emb_r)[0])
            if elipse in ("scrfd", "posicao"):
                pts = pontos_do_lote([os.path.basename(path)], pontos_val, molde).to(device)
                mascara = mascara_por_pontos(pts, torch.from_numpy(molde), IMAGE_SIZE, so_posicao=elipse == "posicao")[0]
            x_d = quantize(inserir_fragil(x_r, m_f, emb_f, mascara, piso)[0])
        originais[name], so_robusta[name], duplas[name], mensagens[name] = x, x_r, x_d, (m_r, m_f)

        o, d = to_uint8(x).astype(np.float64), to_uint8(x_d).astype(np.float64)
        r = to_uint8(x_r).astype(np.float64)
        with torch.no_grad():
            qualidade.append({
                "imagem": name,
                "PSNR": 10 * math.log10(255 ** 2 / max(np.mean((o - d) ** 2), 1e-10)),
                "PSNR_so_robusta": 10 * math.log10(255 ** 2 / max(np.mean((o - r) ** 2), 1e-10)),
                "SSIM": structural_similarity_index_measure(x, x_d).item(),
                "LPIPS": lpips_metric(x * 2 - 1, x_d * 2 - 1).mean().item(),
            })

    resultados = []

    def registrar(ataque, tipo, name, atacada):
        m_r, m_f = mensagens[name]
        lida_r = ler_robusta(atacada)
        lida_f = ler_fragil(atacada)
        esperada = mensagem_fragil(lida_r, chave)
        concord = (lida_f == esperada).float().mean().item()
        ber = (lida_r != m_r).float().mean().item()

        # qualidade: imagem ORIGINAL (sem marca) x atacada, como no evaluate.py
        x = originais[name]
        atacada = atacada.clamp(0, 1)
        o, a = to_uint8(x).astype(np.float32), to_uint8(atacada).astype(np.float32)
        mse = np.mean((o - a) ** 2)
        with torch.no_grad():
            ssim = structural_similarity_index_measure(x, atacada).item()
            lp = lpips_metric(x * 2 - 1, atacada * 2 - 1).mean().item()
            di = dists_metric(x, atacada).mean().item()

        resultados.append({
            "Image": name, "Attack": ataque, "Tipo": tipo,
            "BER": ber, "Bit Accuracy": 1 - ber, "Accuracy": (1 - ber) * 100,
            "Exact Match": float(ber == 0),
            "PSNR": float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse),
            "SSIM": ssim, "LPIPS": lp, "DISTS": di,
            "Semi-fragil Bit Accuracy": (lida_f == m_f).float().mean().item(),
            "Concordancia": concord,
            "Alterada": float(concord < args.limiar),
        })

    # -------------------------
    # Imagem sem nenhuma marca e só com a robusta (a semi-frágil não deve "ler")
    # -------------------------
    for name in originais:
        registrar("sem_semifragil", "controle", name, so_robusta[name])

    # -------------------------
    # 16 ataques comuns
    # -------------------------
    attack_layer = AttackLayer().to(device).eval()
    for key, attack, params in tqdm(ATTACK_CASES, desc="Ataques"):
        for i, name in enumerate(originais):
            set_seed(args.seed + i)
            with torch.no_grad():
                atacada, _ = attack_layer(duplas[name], attack=attack, **params)
            registrar(key, "benigno", name, atacada.clamp(0, 1))

    # -------------------------
    # Deepfakes (mesmo fluxo do evaluate.py: ZIP -> modelo externo -> ZIP)
    # -------------------------
    if not args.sem_deepfakes:
        deepfake_attack = DeepFakeAttack()
        root = os.path.join(args.output, "deepfakes")
        for kind in [k.strip() for k in args.deepfakes.split(",") if k.strip()]:
            if kind not in deepfake_params:
                raise ValueError(f"Deepfake desconhecido: {kind}. Opções: {DEEPFAKES}")
            in_dir = os.path.join(root, f"{kind}_entrada")
            shutil.rmtree(in_dir, ignore_errors=True)
            os.makedirs(in_dir)
            for name in originais:
                Image.fromarray(to_uint8(duplas[name])).save(os.path.join(in_dir, name + ".png"))
            input_zip = shutil.make_archive(in_dir, "zip", in_dir)
            print(f"{kind}: {len(originais)} imagens...")
            deepfake_attack.apply_dataset(attack=kind, input_zip=input_zip, output_dir=root, **deepfake_params[kind])
            out_dir = os.path.join(root, kind, "output")
            for name in originais:
                out_path = os.path.join(out_dir, name + ".png")
                if not os.path.exists(out_path):            # rosto não detectado
                    print(f"[AVISO] {kind}: sem saída para {name}")
                    continue
                atacada = transform(Image.open(out_path).convert("RGB")).unsqueeze(0).to(device)
                registrar(kind, "deepfake", name, atacada)

    # -------------------------
    # Tabelas
    # -------------------------
    df = pd.DataFrame(resultados)
    df.to_csv(os.path.join(args.output, "results_per_image.csv"), index=False)
    q = pd.DataFrame(qualidade)
    q.to_csv(os.path.join(args.output, "qualidade_por_imagem.csv"), index=False)
    q_media = q.drop(columns="imagem").mean().to_frame("média").T
    q_media.to_csv(os.path.join(args.output, "qualidade.csv"), index=False)

    parametros = {k: p for k, _, p in ATTACK_CASES}
    parametros.update({"diffae": "attribute=Young", "stargan": "attribute=Male", "hisd": "Eyeglasses=with",
                       "liveportrait": "20 motoras (mesmo gênero)", "blendface": "20 origens (mesmo gênero)",
                       "simswap": "20 origens (mesmo gênero)", "inswapper": "20 origens (mesmo gênero)",
                       "sem_semifragil": "controle: só a robusta"})

    def parametro(ataque):
        p = parametros.get(ataque, "-")
        if isinstance(p, dict):
            return ", ".join(f"{'Q' if k == 'quality' else k}={v}" for k, v in p.items()) or "-"
        return p

    ordem = [k for k, _, _ in ATTACK_CASES] + [d for d in DEEPFAKES if d in set(df["Attack"])] + ["sem_semifragil"]
    linhas = []
    for ataque in ordem:
        g = df[df["Attack"] == ataque]
        linha = {
            "Attack": ataque, "Parameter": parametro(ataque),
            "BER": g["BER"].mean(), "BER Std": g["BER"].std(ddof=0),
            "Bit Accuracy": g["Bit Accuracy"].mean(), "Accuracy (%)": g["Accuracy"].mean(),
        }
        linha.update({
            "PSNR": g["PSNR"].mean(), "SSIM": g["SSIM"].mean(), "LPIPS": g["LPIPS"].mean(), "DISTS": g["DISTS"].mean(),
            "Semi-frágil Acc (%)": g["Semi-fragil Bit Accuracy"].mean() * 100,
            "Concordância (%)": g["Concordancia"].mean() * 100,
            "Alterada (%)": g["Alterada"].mean() * 100,
            "N": len(g),
        })
        linhas.append(linha)
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(os.path.join(args.output, "media_per_attack.csv"), index=False)

    mostrar = tabela.copy()
    for c, f in {"BER": "{:.2f}", "BER Std": "{:.2f}", "Bit Accuracy": "{:.4f}", "Accuracy (%)": "{:.2f}%",
                 "PSNR": "{:.2f}", "SSIM": "{:.4f}",
                 "LPIPS": "{:.4f}", "DISTS": "{:.4f}", "Semi-frágil Acc (%)": "{:.2f}%",
                 "Concordância (%)": "{:.2f}%", "Alterada (%)": "{:.2f}%"}.items():
        if c in mostrar:
            mostrar[c] = mostrar[c].map(f.format)

    benignos = df[df["Tipo"] == "benigno"]
    fakes = df[df["Tipo"] == "deepfake"]
    print("\n" + "=" * 100)
    print(f"QUALIDADE DA IMAGEM MARCADA ({len(originais)} imagens)")
    print("PSNR = original x dupla | PSNR_so_robusta = original x só a robusta")
    print("=" * 100)
    print(q_media.round(4).to_string(index=False))
    print("\n" + "=" * 100)
    print("RESULTADOS FINAIS — marca dupla")
    print("Colunas até DISTS: marca robusta e qualidade (original x atacada), como no evaluate.py")
    print("Semi-frágil Acc: bits da semi-frágil | Concordância: semi-frágil x esperada (robusta lida XOR chave)")
    print(f"Alterada: concordância < {args.limiar:.2f} (benignos = falso positivo; deepfakes = detecção)")
    print("=" * 100)
    print(tabulate(mostrar, headers="keys", tablefmt="grid", showindex=False))
    print("\nResumo:")
    print(f"    falsos positivos (16 ataques benignos): {benignos['Alterada'].mean() * 100:.2f}%")
    if len(fakes):
        print(f"    deepfakes detectados:                   {fakes['Alterada'].mean() * 100:.2f}%")
        print(f"    robusta nos deepfakes (bit acc):        {fakes['Bit Accuracy'].mean() * 100:.2f}%")
        print(f"    AUC (concordância, deepfake x benigno): {auc(fakes['Concordancia'], benignos['Concordancia']):.4f}")
    print(f"\nTabela: {os.path.join(args.output, 'media_per_attack.csv')}")

    # -------------------------
    # Exemplo visual (primeira imagem)
    # -------------------------
    name = list(originais)[0]
    o = to_uint8(originais[name])
    r = to_uint8(so_robusta[name])
    d = to_uint8(duplas[name])
    dif = np.clip(128 + (d.astype(np.float64) - r.astype(np.float64)) * 20, 0, 255).astype(np.uint8)
    Image.fromarray(np.concatenate([o, r, d, dif], 1)).save(os.path.join(args.output, "exemplo.png"))
    print(f"\nExemplo (original | só robusta | dupla | diferença da semi-frágil x20): {os.path.join(args.output, 'exemplo.png')}")
    print(f"CSVs em: {args.output}")


if __name__ == "__main__":
    main()
