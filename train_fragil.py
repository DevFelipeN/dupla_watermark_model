"""
train_fragil.py

Treina a marca SEMI-FRÁGIL da versão 4 (marca dupla), por cima da marca
robusta v3, que fica CONGELADA:

    imagem -> marca robusta v3 (congelada, imagem toda)  -> diz a ORIGEM
           -> marca semi-frágil (treinada, só no rosto)   -> diz se foi ALTERADA

A marca semi-frágil usa a mesma arquitetura da robusta (Y -> DWT -> DCT ->
CNN, extrator U-Net), mas:

    - a diferença que ela causa é multiplicada por uma elipse do rosto;
    - o extrator dela aprende a LER a mensagem depois de ataques benignos
      (JPEG, blur, ruído, resize, ...), BCE normal;
    - e a NÃO LER (saída ~0, ou seja, bits no acaso) quando o rosto foi
      manipulado (troca, regeneração, deformação ou mistura do rosto; ver
      marca_fragil.MANIPULACOES) e quando a imagem não tem a marca semi-frágil
      (só a robusta), para ele não aprender a "ler" a marca robusta;
    - uma perda extra mantém a marca robusta legível com a semi-frágil por cima.

Nenhum deepfake real é usado no treino: os 7 ficam para a avaliação
(avaliar_marca_dupla.py), o que mostra se a detecção generaliza.

Durante o treino as mensagens robusta e semi-frágil são sorteadas
independentemente. Na avaliação a semi-frágil é a robusta XOR uma chave
(marca_fragil.chave_fragil): quem verifica lê a robusta, calcula a semi-frágil
esperada e compara.

Exemplo (a partir dos pesos da robusta v3, alpha 5, como o modelo atual):
    python train_fragil.py --dataset_path Dataset/celeba_hq/train \
        --output output_fragil/fragil_alpha5_v2 --num_epochs 20
"""

import os
import random
import argparse
from datetime import datetime
from os.path import join

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from attacks import AttackLayer, ATTACK_SETS
from train import WatermarkDataset, find_images, set_seed, seed_worker
from watermark_embedding import WatermarkEmbedding, watermark_embedding_tensor
from watermark_extraction import WatermarkExtraction
from marca_fragil import mascara_rosto, inserir_fragil, MANIPULACOES

ROBUSTA_PADRAO = "output/checkpoints/v3_robusta_final.pth"


def carregar_robusta(caminho, resolucao, device):
    ck = torch.load(caminho, map_location=device)
    emb = WatermarkEmbedding(ck["watermark_size"], resolucao, num_bits=ck["num_bits"]).to(device).eval()
    emb.load_state_dict(ck["embedding_model"])
    emb.alpha = ck["alpha"]
    ext = WatermarkExtraction(ck["watermark_size"], resolucao, use_hh=ck.get("use_hh", False),
                              num_bits=ck["num_bits"]).to(device).eval()
    ext.load_state_dict(ck["extraction_model"])
    for p in list(emb.parameters()) + list(ext.parameters()):
        p.requires_grad = False
    return ck, emb, ext


def acuracia(logits, bits):
    return ((logits >= 0).float() == bits).float().mean().item()


def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


@torch.no_grad()
def validar(val_images, robusta, fragil, mascara, args, device):
    """Validação rápida em imagens fixas: a semi-frágil lê depois de ataques benignos
    e não lê depois das manipulações; a robusta continua legível."""
    emb_r, ext_r, emb_f, ext_f = robusta + fragil
    emb_f.eval(); ext_f.eval()
    attack_layer = AttackLayer().to(device)
    g = torch.Generator(device="cpu").manual_seed(0)
    rng_state = (random.getstate(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    random.seed(0); torch.manual_seed(0)

    soma = {}
    def add(k, v):
        soma.setdefault(k, []).append(v)

    nb = emb_r.num_bits
    for i in range(0, len(val_images), args.batch_size):
        x = val_images[i:i + args.batch_size].to(device)
        B = x.shape[0]
        m_r = torch.randint(0, 2, (B, nb), generator=g).float().to(device)
        m_f = torch.randint(0, 2, (B, nb), generator=g).float().to(device)
        x_r = watermark_embedding_tensor(x, m_r, emb_r)[0]
        x_d, _ = inserir_fragil(x_r, m_f, emb_f, mascara, args.mask_floor)
        x_d = torch.round(x_d * 255) / 255   # como salvar em PNG
        add("PSNR (original x dupla)", psnr(x_d, x))
        add("PSNR (só robusta x dupla)", psnr(x_d, x_r))

        for nome, (atk, kw) in {"sem ataque": ("none", {}), "JPEG 50": ("jpeg_test", {"quality": 50}),
                                "resize 0,5": ("resize", {"scale": 0.5}), "blur": ("blur", {"sigma": 2.0})}.items():
            a = attack_layer(x_d, attack=atk, **kw)[0].clamp(0, 1)
            add(f"semi-frágil | {nome}", acuracia(ext_f(a), m_f))
            add(f"robusta | {nome}", acuracia(ext_r(a), m_r))
        add("semi-frágil | só robusta (sem semi-frágil)", acuracia(ext_f(x_r), m_f))
        outras = x.roll(1, 0)
        for nome, fn in MANIPULACOES.items():
            a = fn(x_d, mascara, outras).clamp(0, 1)
            add(f"semi-frágil | {nome}", acuracia(ext_f(a), m_f))
            add(f"robusta | {nome}", acuracia(ext_r(a), m_r))

    random.setstate(rng_state[0]); torch.set_rng_state(rng_state[1]); torch.cuda.set_rng_state_all(rng_state[2])
    emb_f.train(); ext_f.train()
    return {k: float(np.mean(v)) for k, v in soma.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, help="Imagens de treino (ex.: Dataset/celeba_hq/train)")
    parser.add_argument("--val_path", default="Dataset/celeba_hq/val", help="Imagens da validação rápida do fim de cada época")
    parser.add_argument("--val_images", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--robust_checkpoint", default=ROBUSTA_PADRAO, help="Marca robusta (congelada)")
    parser.add_argument("--init_from", choices=["robusta", "zero"], default="robusta",
                        help="robusta = a semi-frágil começa com os pesos da robusta v3 (converge mais rápido); zero = pesos aleatórios")
    parser.add_argument("--resume", default=None, help="Checkpoint da semi-frágil para continuar o treino")
    parser.add_argument("--image_resolution", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=5.0, help="Força da marca semi-frágil (fixa; 5 no modelo atual)")
    parser.add_argument("--mask_floor", type=float, default=0.0,
                        help="Fração da marca semi-frágil fora do rosto (0 = só no rosto)")
    parser.add_argument("--benign_attack_set", choices=list(ATTACK_SETS), default="benigno_realista",
                        help="Ataques que a semi-frágil deve aguentar (ver attacks.ATTACK_SETS)")
    parser.add_argument("--manipulacoes", default=",".join(MANIPULACOES),
                        help="Manipulações de treino, separadas por vírgula: " + ", ".join(MANIPULACOES))
    parser.add_argument("--benign_after_manip", type=float, default=0.5,
                        help="Probabilidade de aplicar também um ataque benigno depois da manipulação "
                             "(os deepfakes reais também saem recomprimidos)")
    parser.add_argument("--fragility_weight", type=float, default=1.0, help="Peso da perda de 'não ler' nas manipulações")
    parser.add_argument("--no_mark_weight", type=float, default=0.5, help="Peso da perda de 'não ler' sem a marca semi-frágil")
    parser.add_argument("--robust_weight", type=float, default=1.0, help="Peso da perda que mantém a robusta legível")
    parser.add_argument("--image_loss_weight", type=float, default=10.0)
    parser.add_argument("--lpips_weight", type=float, default=0.005)
    parser.add_argument("--image_loss_ramp", type=int, default=3000, help="Passos até a loss da imagem chegar ao peso máximo")
    parser.add_argument("--image_loss_start_acc", type=float, default=0.85,
                        help="A loss da imagem só começa quando a semi-frágil lê, nos ataques benignos, essa acurácia "
                             "(média dos últimos 100 passos); com --resume ela já começa no peso máximo")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_min", type=float, default=1e-6, help="LR final (cosseno de --lr até --lr_min)")
    parser.add_argument("--limit", type=int, default=0, help="Usa só as N primeiras imagens de treino (teste rápido)")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manipulacoes = [m.strip() for m in args.manipulacoes.split(",") if m.strip()]
    for m in manipulacoes:
        if m not in MANIPULACOES:
            raise ValueError(f"Manipulação desconhecida: {m}. Opções: {list(MANIPULACOES)}")

    LOGS, CKPTS = join(args.output, "logs"), join(args.output, "checkpoints")
    for p in (LOGS, CKPTS):
        os.makedirs(p, exist_ok=True)
    writer = SummaryWriter(LOGS)
    EXP = "fragil_" + datetime.now().strftime("%d%m%Y_%H:%M:%S")

    # -------------------------
    # Marca robusta (congelada) e marca semi-frágil (treinada)
    # -------------------------
    ck_r, emb_r, ext_r = carregar_robusta(args.robust_checkpoint, args.image_resolution, device)
    num_bits, grid = ck_r["num_bits"], ck_r["watermark_size"]

    emb_f = WatermarkEmbedding(grid, args.image_resolution, num_bits=num_bits).to(device)
    ext_f = WatermarkExtraction(grid, args.image_resolution, use_hh=False, num_bits=num_bits).to(device)
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        emb_f.load_state_dict(ck["embedding_model"]); ext_f.load_state_dict(ck["extraction_model"])
        print(f"Continuando de {args.resume}")
    elif args.init_from == "robusta":
        emb_f.load_state_dict(ck_r["embedding_model"]); ext_f.load_state_dict(ck_r["extraction_model"])
        print("Marca semi-frágil começa com os pesos da robusta v3")
    emb_f.alpha = args.alpha

    mascara = mascara_rosto(args.image_resolution).to(device)
    print(f"Robusta (congelada): {args.robust_checkpoint} (alpha {emb_r.alpha})")
    print(f"Semi-frágil: alpha {args.alpha} | piso fora do rosto {args.mask_floor} | "
          f"ataques benignos ({args.benign_attack_set}): {ATTACK_SETS[args.benign_attack_set]}")
    print(f"Manipulações de treino: {manipulacoes}")

    # -------------------------
    # Dados
    # -------------------------
    transform = transforms.Compose([transforms.Resize((args.image_resolution,) * 2), transforms.ToTensor()])
    paths = find_images(args.dataset_path)
    if args.limit:
        paths = paths[:args.limit]
    dataset = WatermarkDataset(paths, transform=transform, watermark_size=grid, num_bits=num_bits)
    gen = torch.Generator(); gen.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                        worker_init_fn=seed_worker, generator=gen, drop_last=True)
    val_paths = find_images(args.val_path)[:args.val_images]
    val_images = torch.stack([transform(Image.open(p).convert("RGB")) for p in val_paths])
    print(f"Treino: {len(dataset)} imagens | validação rápida: {len(val_images)} imagens")

    # -------------------------
    # Otimização
    # -------------------------
    attack_layer = AttackLayer(attack_set=args.benign_attack_set).to(device)
    params = list(emb_f.parameters()) + list(ext_f.parameters())
    optimizer = Adam(params, lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(loader), eta_min=args.lr_min)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    lpips_metric = None
    if args.lpips_weight > 0:
        import lpips
        lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in lpips_metric.parameters():
            p.requires_grad = False

    def incerteza(logits):
        """'Não ler': BCE com alvo 0,5 empurra a saída para 0 (bit no acaso)."""
        return bce(logits, torch.full_like(logits, 0.5))

    step = 0
    # passos desde que a loss da imagem foi ativada (-1 = ainda não)
    passos_img = args.image_loss_ramp if args.resume else -1
    historico = []
    melhor_nota = None
    for epoch in range(args.num_epochs):
        print(f"\nÉpoca {epoch + 1}/{args.num_epochs}")
        emb_f.train(); ext_f.train()
        medias = {}
        bar = tqdm(loader)
        for images, _ in bar:
            step += 1
            x = images.to(device, non_blocking=True)
            B = x.shape[0]
            m_r = torch.randint(0, 2, (B, num_bits), device=device).float()
            m_f = torch.randint(0, 2, (B, num_bits), device=device).float()

            with torch.no_grad():
                x_r = watermark_embedding_tensor(x, m_r, emb_r)[0]
            x_d, modificacao = inserir_fragil(x_r, m_f, emb_f, mascara, args.mask_floor)

            # 1) ataque benigno: a semi-frágil deve ler
            atacada, info = attack_layer(x_d)
            atacada = atacada.clamp(0, 1)
            logits_b = ext_f(atacada)
            loss_bits = bce(logits_b, m_f)

            # 2) manipulação do rosto: a semi-frágil NÃO deve ler. O rosto colado
            # vem de outra imagem SEM marca (como a origem de um face swap real)
            nome_manip = random.choice(manipulacoes)
            manipulada = MANIPULACOES[nome_manip](x_d, mascara, x.roll(1, 0)).clamp(0, 1)
            if random.random() < args.benign_after_manip:
                manipulada = attack_layer(manipulada)[0].clamp(0, 1)
            logits_m = ext_f(manipulada)
            loss_frag = incerteza(logits_m)

            # 3) imagem sem a marca semi-frágil (só a robusta): a semi-frágil NÃO deve ler
            logits_n = ext_f(x_r)
            loss_sem = incerteza(logits_n)

            # 4) a robusta continua legível com a semi-frágil por cima (mesmo ataque benigno)
            logits_r = ext_r(atacada)
            loss_rob = bce(logits_r, m_r)

            # 5) qualidade: a semi-frágil deve mudar o mínimo a imagem que já tem a robusta
            loss_img = mse(x_d, x_r)
            if lpips_metric is not None:
                loss_img = loss_img + args.lpips_weight * lpips_metric(x_d * 2 - 1, x_r * 2 - 1).mean()
            peso_img = 0.0 if passos_img < 0 else args.image_loss_weight * min(1.0, passos_img / max(1, args.image_loss_ramp))

            loss = (loss_bits + args.fragility_weight * loss_frag + args.no_mark_weight * loss_sem
                    + args.robust_weight * loss_rob + peso_img * loss_img)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            valores = {
                "fragil_benigno": acuracia(logits_b, m_f),
                "fragil_manipulada": acuracia(logits_m, m_f),
                "fragil_sem_marca": acuracia(logits_n, m_f),
                "robusta": acuracia(logits_r, m_r),
                "psnr_total": psnr(x_d.detach(), x),
            }
            for k, v in valores.items():
                medias.setdefault(k, []).append(v)
            historico = (historico + [valores["fragil_benigno"]])[-100:]
            if passos_img >= 0:
                passos_img += 1
            elif len(historico) == 100 and np.mean(historico) >= args.image_loss_start_acc:
                passos_img = 0
                print(f"\nSemi-frágil lendo {np.mean(historico)*100:.1f}% nos ataques benignos: loss da imagem ativada")
            bar.set_postfix(frag=f"{valores['fragil_benigno']*100:.0f}",
                            manip=f"{valores['fragil_manipulada']*100:.0f}",
                            rob=f"{valores['robusta']*100:.0f}", psnr=f"{valores['psnr_total']:.1f}")

            if step % 100 == 0:
                for k, v in valores.items():
                    writer.add_scalar(f"treino/{k}", v, step)
                for k, v in {"total": loss, "bits": loss_bits, "fragilidade": loss_frag, "sem_marca": loss_sem,
                             "robusta": loss_rob, "imagem": loss_img}.items():
                    writer.add_scalar(f"loss/{k}", v.item(), step)
                writer.add_scalar("loss/peso_imagem", peso_img, step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)
                print(f"\nPasso {step} | semi-frágil benigno {valores['fragil_benigno']*100:.1f}% ({info['name']}) | "
                      f"semi-frágil manipulada {valores['fragil_manipulada']*100:.1f}% ({nome_manip}) | "
                      f"semi-frágil sem marca {valores['fragil_sem_marca']*100:.1f}% | robusta {valores['robusta']*100:.1f}% | "
                      f"PSNR {valores['psnr_total']:.2f} dB | peso imagem {peso_img:.2f}")
            if step % 1000 == 0:
                diff = ((x_d - x_r)[:8] * 20 + 0.5).clamp(0, 1)
                writer.add_image("imagens/dupla", make_grid(x_d[:8]), step)
                writer.add_image("imagens/diferenca_fragil_x20", make_grid(diff), step)
                writer.add_image("imagens/manipulada", make_grid(manipulada[:8]), step)

        # -------------------------
        # Fim da época
        # -------------------------
        print("Média da época: " + " | ".join(f"{k} {np.mean(v)*(1 if k.startswith('psnr') else 100):.2f}"
                                              for k, v in medias.items()))
        val = validar(val_images, (emb_r, ext_r), (emb_f, ext_f), mascara, args, device)
        print("Validação (bit acc %, PSNR dB):")
        for k, v in val.items():
            print(f"    {k:42s} {v if k.startswith('PSNR') else v * 100:7.2f}")
            writer.add_scalar("validacao/" + k, v, step)

        salvar(join(CKPTS, EXP + "_model.pth"), emb_f, ext_f, optimizer, args, ck_r, epoch, step, val)

        # melhor época: lê bem depois dos ataques comuns e mal depois das
        # manipulações. Só conta com a loss da imagem já no peso máximo, para
        # não escolher uma época de antes do ajuste da qualidade.
        comuns = [val[f"semi-frágil | {n}"] for n in ("sem ataque", "JPEG 50", "resize 0,5", "blur")]
        manip = [val[f"semi-frágil | {n}"] for n in MANIPULACOES]
        nota = float(np.mean(comuns) - np.mean(manip))
        writer.add_scalar("validacao/nota", nota, step)
        print(f"Nota da época (média comuns - média manipulações): {nota * 100:.2f}")
        if passos_img >= args.image_loss_ramp and (melhor_nota is None or nota > melhor_nota):
            melhor_nota = nota
            salvar(join(CKPTS, EXP + "_melhor.pth"), emb_f, ext_f, optimizer, args, ck_r, epoch, step, val)
            print(f"Melhor época até agora: {epoch + 1} -> {EXP}_melhor.pth")

    caminho = join(CKPTS, EXP + "_final.pth")
    salvar(caminho, emb_f, ext_f, optimizer, args, ck_r, epoch, step, val)
    n = sum(p.numel() for p in params)
    print(f"\nCheckpoint salvo em: {caminho}")
    if melhor_nota is not None:
        print(f"Melhor época (nota {melhor_nota * 100:.2f}): {join(CKPTS, EXP + '_melhor.pth')}")
    print(f"Parâmetros da marca semi-frágil: {n:,} ({n / 1e6:.3f} M)")
    writer.close()


def salvar(caminho, emb_f, ext_f, optimizer, args, ck_r, epoch, step, val):
    torch.save({
        "tipo": "fragil",
        "alpha": args.alpha,
        "use_hh": False,
        "num_bits": ck_r["num_bits"],
        "watermark_size": ck_r["watermark_size"],
        "mask_floor": args.mask_floor,
        "robust_checkpoint": args.robust_checkpoint,
        "benign_attack_set": args.benign_attack_set,
        "manipulacoes": args.manipulacoes,
        "args": vars(args),
        "epoch": epoch,
        "global_step": step,
        "validacao": val,
        "embedding_model": emb_f.state_dict(),
        "extraction_model": ext_f.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, caminho)


if __name__ == "__main__":
    main()
