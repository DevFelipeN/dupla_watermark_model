import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from PIL import Image
from tqdm import tqdm
from datetime import datetime
from os.path import join

from torch.optim import Adam
from torch.utils.data import (Dataset, DataLoader)
from torch.utils.tensorboard import (SummaryWriter)
from torchvision import transforms
from torchvision.utils import (make_grid)
from torch.optim.lr_scheduler import (ReduceLROnPlateau, CosineAnnealingLR)
from utils import (block_idct2d, inverse_arnold_transform)
from watermark_embedding import (WatermarkEmbedding, watermark_embedding_tensor)
from watermark_extraction import (WatermarkExtraction)
from attacks import AttackLayer, ATTACK_SETS
from collections import deque, defaultdict, Counter

attack_counter = Counter()

# ==========================================
# Reprodutibilidade
# ==========================================
def set_seed(seed):
    # random: sorteio dos ataques e de seus parâmetros (AttackLayer)
    # torch: pesos iniciais, marcas aleatórias e ruídos dos ataques
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    # Cada worker do DataLoader recebe uma seed derivada da seed principal
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ==========================================
# Criar lista de imagens
# ==========================================
def find_images(root):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith((
                    ".jpg",
                    ".png",
                    ".jpeg"
                )
            ):
                paths.append(
                    os.path.join(
                        dirpath,
                        f
                    )
                )
    return sorted(paths)

# ==========================================
# Dataset
# ==========================================
class WatermarkDataset(Dataset):
    def __init__(
        self,
        image_paths,
        transform=None,
        watermark_size=16,
        num_bits=0,
    ):
        self.image_paths = image_paths
        self.transform = transform
        self.watermark_size = watermark_size
        self.num_bits = num_bits

    def __len__(self):
        return len(
            self.image_paths
        )
    def __getitem__(self, idx):

        path = self.image_paths[idx]

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.num_bits:
            watermark = torch.randint(0, 2, (self.num_bits,), dtype=torch.float32)
        else:
            watermark = torch.randint(0, 2, (1,self.watermark_size,self.watermark_size), dtype=torch.float32)

        return img, watermark

# ==========================================
# Main
# ==========================================
def main():

    parser = argparse.ArgumentParser()

    # -------------------------
    # Arguments
    # -------------------------
    parser.add_argument("--dataset_path", type=str, required=True, help="Caminho para o dataset")
    parser.add_argument("--image_resolution", type=int, required=True)
    parser.add_argument("--watermark_size", type=int, required=True, help="Tamanho da marca em bits (qualquer valor, ex.: 30). O modelo converte os bits internamente num mapa de (image_resolution/16) x (image_resolution/16).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--arnold_iterations", type=int,default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="Seed para tornar o treino reproduzível.")
    parser.add_argument("--alpha_patience", type=int, default=2000, help="Passos sem redução do alpha antes de forçar a próxima redução.")
    parser.add_argument("--image_loss_weight", type=float, default=10.0, help=("Peso máximo da loss de qualidade da imagem."))
    parser.add_argument("--image_loss_await", type=int, default=1000, help=("Quantidade de passos de espera depois de atingir 90 porcento de Bit Accuracy."))
    parser.add_argument("--image_loss_ramp", type=int, default=3000, help=("Quantidade de passos para a loss da imagem atingir o peso máximo."))
    parser.add_argument("--attack_set", choices=list(ATTACK_SETS), default="v3", help="Ataques sorteados no treino: v1 = treino até 09/10/2026; v2 = JPEG real, median 3/5, dropout até 0.3, resize+JPEG e deformação geométrica; v3 = v2 sem a deformação geométrica.")
    parser.add_argument("--lpips_weight", type=float, default=0.0, help="Peso do LPIPS na loss da imagem (0 = só MSE, como na v1). A loss da imagem vira MSE + lpips_weight * LPIPS.")
    parser.add_argument("--resume", default=None, help="Checkpoint para continuar o treino (fine-tuning): carrega os pesos e já começa com a loss da imagem no peso máximo.")
    parser.add_argument("--start_alpha", type=float, default=None, help="Alpha inicial (padrão: 100, ou o alpha do checkpoint com --resume).")
    parser.add_argument("--min_alpha", type=float, default=3.0, help="Menor alpha permitido (o treino reduz o alpha até ele).")
    parser.add_argument("--lr_schedule", choices=["plateau", "cosine"], default="plateau", help="plateau = reduz o LR pela metade se a loss da marca parar de cair por 6 épocas (padrão, só com o alpha no mínimo); cosine = o LR desce em cosseno, a cada passo, de --lr até --lr_min ao longo de todo o treino (para a fase final/fine-tuning).")
    parser.add_argument("--lr_min", type=float, default=1e-6, help="LR final com --lr_schedule cosine.")
    parser.add_argument("--subband_weights", default=None, help="Pesos LL,LH,HL,HH (ex.: 0.3,0.35,0.35,0.001; normalizados pelo softmax dos logs). Padrão: os do watermark_embedding.py ou, com --resume, os do checkpoint.")
    parser.add_argument("--learnable_subbands", action="store_true", help="Deixa a rede aprender os pesos das sub-bandas (partindo de --subband_weights).")
    parser.add_argument("--subband_lr", type=float, default=1e-3, help="LR dos pesos das sub-bandas com --learnable_subbands (com o LR da rede eles quase não se movem).")
    
    args = (parser.parse_args())

    # Antes de criar modelos e o AttackLayer (que embaralha os ataques no __init__)
    set_seed(args.seed)

    # --watermark_size é o número de bits da marca. Internamente os bits viram
    # um mapa de grid_size x grid_size (1/8 da sub-banda: 8x8 para 128, 16x16
    # para 256), que a rede amplia até o tamanho da sub-banda.
    if args.image_resolution % 16 != 0:
        raise ValueError("image_resolution precisa ser múltiplo de 16.")

    num_bits = args.watermark_size
    grid_size = args.image_resolution // 16
    print(f"Marca: {num_bits} bits -> mapa interno {grid_size}x{grid_size}")
    print(f"Seed: {args.seed}")

    # -------------------------
    # CUDA
    # -------------------------
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    # -------------------------
    # Paths
    # -------------------------
    LOGS_PATH = join(args.output, "logs")
    CHECKPOINTS_PATH = join(args.output, "checkpoints")

    for path in [LOGS_PATH, CHECKPOINTS_PATH]:
        os.makedirs(path, exist_ok=True)

    writer = SummaryWriter(LOGS_PATH)

    # -------------------------
    # Logging steps
    # -------------------------
    plot_points = (list(range(0, 1000, 100)) + list(range(1000, 3000, 200)) + list(range(3000, 100000, 1000)))

    now = datetime.now()

    dt_string = now.strftime("%d%m%Y_%H:%M:%S")

    EXP_NAME = "watermark_Y_DCT_DWT_" + dt_string

    # -------------------------
    # Device
    # -------------------------
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Using device: "f" {device}")

    # -------------------------
    # Dataset
    # -------------------------
    transform = (transforms.Compose([transforms.Resize((args.image_resolution, args.image_resolution)), transforms.ToTensor()]))

    image_paths = find_images(args.dataset_path)
    #image_paths = image_paths[:1000]

    dataset = WatermarkDataset(image_paths=image_paths, transform=transform, watermark_size=grid_size, num_bits=num_bits)

    print(f"Loaded " f"{len(dataset)} "f"images.")

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    dataloader = DataLoader(dataset,batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=loader_generator)

    # -------------------------
    # Models
    # -------------------------
    embedding_model = WatermarkEmbedding(watermark_size=grid_size, image_resolution=args.image_resolution, learnable_weights=args.learnable_subbands, num_bits=num_bits).to(device)
    extraction_model = WatermarkExtraction(
        watermark_size=grid_size,
        image_resolution=args.image_resolution,
        use_hh=False,
        num_bits=num_bits,
    ).to(device)

    #-------------------------
    # Ataques
    #--------------------------
    attack_layer = AttackLayer(attack_set=args.attack_set).to(device)
    print(f"Ataques do treino ({args.attack_set}): {ATTACK_SETS[args.attack_set]}")

    # -------------------------
    # Optimizer
    # -------------------------
    subband_params = [embedding_model.subband_logits] if args.learnable_subbands else []
    network_params = [
        p for p in list(embedding_model.parameters()) + list(extraction_model.parameters())
        if all(p is not q for q in subband_params)
    ]
    param_groups = [{"params": network_params}]
    if subband_params:
        param_groups.append({"params": subband_params, "lr": args.subband_lr, "weight_decay": 0.0})
    optimizer = Adam(param_groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5)

    # -------------------------
    # Scheduler
    # -------------------------
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, min_lr=5e-6)
    cosine_scheduler = None
    if args.lr_schedule == "cosine":
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(dataloader), eta_min=args.lr_min)
        print(f"LR em cosseno: {args.lr} -> {args.lr_min} em {args.num_epochs * len(dataloader)} passos")

    # -------------------------
    # Loss
    # -------------------------
    criterion_imag = nn.MSELoss()

    # LPIPS na loss da imagem: penaliza o que o olho percebe (ex.: o padrão
    # em grade), não só o erro médio por pixel
    lpips_metric = None
    if args.lpips_weight > 0:
        import lpips
        lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for param in lpips_metric.parameters():
            param.requires_grad = False
    criterion_watermark = nn.BCEWithLogitsLoss()

    global_step = 0

    # --------------------------------------------------
    # Controle da ativação da loss da imagem
    #
    # -1:
    # a Bit Accuracy ainda não atingiu 90%
    #
    # 0 ou maior:
    # quantidade de passos desde que atingiu 90%
    # --------------------------------------------------
    steps_since_image_loss_activated = -1

    bit_acc_history = deque(maxlen=100)

    # Nomes exatos retornados em attack_info["name"] pelo AttackLayer
    ATTACK_NAMES = [
        "Identity", "JPEG", "Gaussian Blur", "Median Blur",
        "Gaussian Noise", "Salt & Pepper", "Dropout", "Resize"
    ]
    if args.attack_set in ("v2", "v3"):
        ATTACK_NAMES += ["Combined"]
    if args.attack_set == "v2":
        ATTACK_NAMES += ["Geometric"]
    bit_acc_by_attack = defaultdict(lambda: deque(maxlen=30))

    current_alpha = args.start_alpha if args.start_alpha is not None else 100
    min_alpha = args.min_alpha
    

    # Passos desde a última redução do alpha (para a redução forçada)
    steps_since_alpha_change = 0

    # -------------------------
    # Fine-tuning a partir de um checkpoint
    # -------------------------
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        if ck.get("num_bits") != num_bits or ck.get("watermark_size") != grid_size:
            raise ValueError("O checkpoint de --resume tem outro tamanho de marca/resolução.")
        embedding_model.load_state_dict(ck["embedding_model"])
        extraction_model.load_state_dict(ck["extraction_model"])
        if args.start_alpha is None:
            current_alpha = ck["alpha"]
        # o modelo já extrai a marca: loss da imagem no peso máximo desde o início
        steps_since_image_loss_activated = args.image_loss_await + args.image_loss_ramp
        print(f"Continuando de {args.resume} (alpha {current_alpha})")

    # pesos das sub-bandas pedidos na linha de comando (depois do --resume,
    # que carregaria os do checkpoint)
    if args.subband_weights:
        values = [float(v) for v in args.subband_weights.split(",")]
        if len(values) != 4 or min(values) <= 0:
            raise ValueError("--subband_weights precisa de 4 valores positivos: LL,LH,HL,HH")
        with torch.no_grad():
            embedding_model.subband_logits.copy_(torch.log(torch.tensor(values, device=device)))

    w = embedding_model.get_subband_weights().detach().view(4).tolist()
    print("Pesos das sub-bandas (LL, LH, HL, HH): " + ", ".join(f"{v:.4f}" for v in w)
          + (" (aprendíveis)" if args.learnable_subbands else " (fixos)"))

    embedding_model.alpha = current_alpha

    # ==================================
    # Training
    # ==================================
    for epoch in range(args.num_epochs):

        print(f"\nEpoch "f"{epoch+1}/"f"{args.num_epochs}")

        embedding_model.train()
        extraction_model.train()

        epoch_loss = 0.0
        epoch_watermark_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            total=len(dataloader)
        )

        for batch_idx, (images, watermarks) in enumerate(progress_bar):
            # --------------------------------------------------
            # Atualiza gradualmente o peso da loss da imagem
            # --------------------------------------------------

            image_loss_weight = min(
                max(
                    0.0,
                    args.image_loss_weight
                    *
                    (
                        steps_since_image_loss_activated
                        -
                        args.image_loss_await
                    )
                    /
                    args.image_loss_ramp
                ),
                args.image_loss_weight
            )

            global_step += 1

            # -----------------
            # Device
            # -----------------
            images = (
                images.to(
                    device,
                    non_blocking=True
                )
            )

            watermarks = (
                watermarks.to(
                    device,
                    non_blocking=True
                )
            )

            # -----------------
            # Embedding
            # -----------------
            watermarked_images, targets, modification = (
                watermark_embedding_tensor(
                    image_tensor=images,
                    watermark_image=watermarks,
                    embedding_model=embedding_model
                )
            )

            
            attacked_images, attack_info = attack_layer(
                watermarked_images
            )
            attack_counter[attack_info["name"]] += 1

            # Sem isso, um nome fora de ATTACK_NAMES impede o alpha de ser reduzido
            if attack_info["name"] not in ATTACK_NAMES:
                raise ValueError(
                    f"Ataque '{attack_info['name']}' não está em ATTACK_NAMES: {ATTACK_NAMES}"
                )
            

            # -----------------
            # Forward
            # -----------------
            decoder_output = extraction_model(attacked_images)

            # -----------------
            # Loss da marca
            # -----------------
            loss_watermark = (
                criterion_watermark(
                    decoder_output,
                    watermarks
                )
            )

            # -----------------
            # Loss da imagem
            # -----------------
            loss_mse = criterion_imag(
                watermarked_images,
                images
            )
            loss_image = loss_mse

            if lpips_metric is not None:
                loss_lpips = lpips_metric(
                    watermarked_images * 2 - 1,
                    images * 2 - 1
                ).mean()
                loss_image = loss_mse + args.lpips_weight * loss_lpips

            # -----------------
            # Loss total
            # -----------------
            loss = loss_watermark + image_loss_weight * loss_image
            #loss = loss_watermark
            # -----------------
            # Bits recuperados
            # -----------------
            predicted_bits = (
                torch.sigmoid(
                    decoder_output
                )
                >= 0.5
            ).float()

            # -----------------
            # Bit Accuracy
            # -----------------
            bit_acc = (
                predicted_bits
                ==
                watermarks
            ).float().mean()

            # -----------------
            # BER
            # -----------------
            ber = (
                predicted_bits
                !=
                watermarks
            ).float().mean()

            bit_acc_history.append(
                bit_acc.item()
            )
            bit_acc_by_attack[attack_info["name"]].append(
                bit_acc.item()
            )

            if steps_since_image_loss_activated == -1:
                if len(bit_acc_history) == 100:

                    mean_bit_acc = (
                        sum(bit_acc_history)
                        /
                        len(bit_acc_history)
                    )

                    if mean_bit_acc >= 0.90:

                        steps_since_image_loss_activated = 0

                        print(
                            "\n"
                            "Média da Bit Accuracy dos "
                            "últimos 100 batches: "
                            f"{mean_bit_acc * 100:.2f}%"
                            "\n"
                            "Contagem para ativar a loss "
                            "da imagem iniciada."
                        )
            else:

                steps_since_image_loss_activated += 1
                steps_since_alpha_change += 1

                ready = [
                    name for name in ATTACK_NAMES
                    if len(bit_acc_by_attack[name]) == 30
                ]

                reason = None

                if len(ready) == len(ATTACK_NAMES):

                    worst_attack = min(
                        ready,
                        key=lambda k: sum(bit_acc_by_attack[k]) / len(bit_acc_by_attack[k])
                    )
                    worst_acc = (
                        sum(bit_acc_by_attack[worst_attack])
                        / len(bit_acc_by_attack[worst_attack])
                    )

                    if worst_acc >= 0.90:
                        reason = f"pior ataque era: {worst_attack} = {worst_acc*100:.2f}%"

                # Sem isso, um ataque difícil pode travar o alpha por dezenas de
                # milhares de passos e o treino termina com alpha (e PSNR) diferente
                if reason is None and steps_since_alpha_change >= args.alpha_patience:
                    reason = f"forçado: {steps_since_alpha_change} passos sem redução"

                if reason is not None and current_alpha > min_alpha:
                    current_alpha *= 0.8

                    if current_alpha < min_alpha:
                        current_alpha = min_alpha

                    embedding_model.alpha = current_alpha
                    steps_since_alpha_change = 0

                    for name in ATTACK_NAMES:
                        bit_acc_by_attack[name].clear()

                    print(
                        f"\nNovo alpha = {current_alpha:.2f} ({reason})"
                    )


            # -----------------
            # PSNR
            # -----------------
            if loss_mse.item() == 0:
                psnr = torch.tensor(
                    float("inf"),
                    device=loss_mse.device
                )
            else:
                psnr = (10.0 * torch.log10(1.0 / loss_mse.detach()))

            epoch_loss += loss.item()
            epoch_watermark_loss += loss_watermark.item()

            pred_vis = (
                torch.sigmoid(
                    decoder_output.detach()
                )
            )

            # -----------------
            # Backprop
            # -----------------
            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(
                    embedding_model.parameters()
                )
                +
                list(
                    extraction_model.parameters()
                ),
                max_norm=1.0
            )

            optimizer.step()
            if cosine_scheduler is not None:
                cosine_scheduler.step()

            # -----------------
            # Logging
            # -----------------
            if global_step % 100 == 0:

                w = embedding_model.get_subband_weights().detach().view(4)
                for b, band in enumerate(["LL", "LH", "HL", "HH"]):
                    writer.add_scalar(f"subband_weights/Y_{band}", w[b].item(), global_step)

                mag = (embedding_model.alpha * modification).detach().abs().mean(dim=(0, 2, 3))   # (4,)
                for b, band in enumerate(["LL", "LH", "HL", "HH"]):
                    writer.add_scalar(f"strength/Y_{band}", mag[b].item(), global_step)

                print(
                    f"Y | Pesos [LL,LH,HL,HH]: {[round(x, 3) for x in w.tolist()]} | "
                    f"Intensidade: {[round(x, 3) for x in mag.tolist()]}"
                )

                writer.add_scalar(
                    "quality/PSNR",
                    psnr.item(),
                    global_step
                )

                writer.add_scalar(
                    "loss/total",
                    loss.item(),
                    global_step
                )

                writer.add_scalar(
                    "loss/watermark",
                    loss_watermark.item(),
                    global_step
                )

                writer.add_scalar(
                    "loss/image",
                    loss_image.item(),
                    global_step
                )

                writer.add_scalar(
                    "loss/image_weight",
                    image_loss_weight,
                    global_step
                )
                
                writer.add_scalar(
                    "embedding/alpha",
                    current_alpha,
                    global_step
                )
                writer.add_scalar(
                    "learning_rate",
                    optimizer
                    .param_groups[0]['lr'],
                    global_step
                )

                writer.add_image(
                    "marked_images",
                    make_grid(
                        watermarked_images,
                        normalize=True
                    ),
                    global_step
                )

                writer.add_image(
                    "attacked_images",
                    make_grid(
                        attacked_images,
                        normalize=True
                    ),
                    global_step
                )

                writer.add_image(
                    "watermark_gt",
                    make_grid(
                        watermarks.view(watermarks.shape[0], 1, 1, -1),
                        normalize=False 
                    ),
                    global_step
                )

                writer.add_image(
                    "watermark_pred",
                    make_grid(
                        pred_vis.view(pred_vis.shape[0], 1, 1, -1),
                        normalize=False
                    ),
                    global_step
                )

                writer.add_scalar(
                    "bit_accuracy",
                    bit_acc.item(),
                    global_step
                )

                writer.add_scalar(
                    "ber",
                    ber.item(),
                    global_step
                )

                print(
                    f"Step {global_step} | "
                    f"Alpha={current_alpha:.2f} | "
                    f"TotalLoss={loss.item():.6f} | "
                    f"WatermarkLoss={loss_watermark.item():.6f} | "
                    f"ImageLoss={loss_image.item():.6f} | "
                    f"ImageWeight={image_loss_weight:.6f} | "
                    f"WeightedImageLoss={(image_loss_weight * loss_image).item():.6f} | "
                    f"BitAcc={bit_acc.item()*100:.2f}% | "
                    f"BER={ber.item()*100:.2f}% | "
                    f"PSNR={psnr.item():.2f} dB | "
                    f"Attack={attack_info['name']} | "
                    f"Params={', '.join(f'{k}={v}' for k, v in attack_info.items() if k != 'name')}"
                )

            # -----------------
            # Checkpoint
            # -----------------
            if (global_step % 5000 == 0):
                torch.save({
                    "alpha": current_alpha,
                    "use_hh": False,
                    "num_bits": num_bits,
                    "watermark_size": grid_size,
                    "seed": args.seed,
                    "attack_set": args.attack_set,
                    "lpips_weight": args.lpips_weight,
                    "resumed_from": args.resume,
                    "subband_weights": embedding_model.get_subband_weights().detach().view(4).tolist(),
                    "learnable_subbands": args.learnable_subbands,
                    "epoch": epoch,
                    "global_step": global_step,
                    "embedding_model": embedding_model.state_dict(),
                    "extraction_model": extraction_model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }, join(CHECKPOINTS_PATH, EXP_NAME + "_model.pth"))

        # -------------------------
        # End of epoch
        # -------------------------
        epoch_loss /= len(dataloader)
        epoch_watermark_loss /= len(dataloader)

        # Enquanto o alpha cai, a loss sobe por design (marca mais fraca e rampa
        # da loss de imagem); o scheduler leria isso como estagnação e cortaria
        # o LR antes da hora. Por isso ele só passa a contar com o alpha no
        # mínimo, e olha só a loss da marca.
        if current_alpha <= min_alpha:
            if cosine_scheduler is None:
                scheduler.step(epoch_watermark_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch+1} | "
            f"AvgLoss={epoch_loss:.6f} | "
            f"AvgWatermarkLoss={epoch_watermark_loss:.6f} | "
            f"Alpha={current_alpha:.2f} | "
            f"LR={current_lr:.8f}"
        )

    print("\nAtaques utilizados nesta época:")
    for attack, count in sorted(attack_counter.items()):
        print(f"{attack:15}: {count}")

    print(
        f"\nAlpha final ao término do treino: {current_alpha:.4f}"
    )

    # -------------------------
    # Final checkpoint
    # -------------------------
    checkpoint_path = join(
        CHECKPOINTS_PATH,
        EXP_NAME + "_final.pth"
    )
    torch.save({
        "alpha": current_alpha,
        "use_hh": False,
        "num_bits": num_bits,
        "watermark_size": grid_size,
        "seed": args.seed,
        "attack_set": args.attack_set,
        "lpips_weight": args.lpips_weight,
        "resumed_from": args.resume,
        "subband_weights": embedding_model.get_subband_weights().detach().view(4).tolist(),
        "learnable_subbands": args.learnable_subbands,
        "epoch": epoch,
        "global_step": global_step,
        "embedding_model": embedding_model.state_dict(),
        "extraction_model": extraction_model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, checkpoint_path)

    # Tamanho do modelo como na literatura: parâmetros treináveis e pesos em
    # float32. O arquivo do checkpoint é ~3x maior porque também guarda o
    # estado do otimizador (Adam: 2 valores por parâmetro) e metadados.
    n_embedding = sum(p.numel() for p in embedding_model.parameters() if p.requires_grad)
    n_extraction = sum(p.numel() for p in extraction_model.parameters() if p.requires_grad)
    n_total = n_embedding + n_extraction
    size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)

    print(f"\nCheckpoint salvo em: {checkpoint_path}")
    print(f"Parâmetros: marcação {n_embedding:,} ({n_embedding / 1e6:.3f} M) | "
          f"extração {n_extraction:,} ({n_extraction / 1e6:.3f} M) | total {n_total / 1e6:.3f} M")
    print(f"Pesos (float32): {n_total * 4 / 2 ** 20:.2f} MB")
    print(f"Arquivo do checkpoint (com o estado do otimizador): {size_mb:.2f} MB")

    writer.close()

if __name__ == "__main__":
    main()