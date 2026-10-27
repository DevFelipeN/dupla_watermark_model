#!/usr/bin/env python3
"""
collect_reference.py

Localiza UMA imagem específica no dataset, aplica o watermark usando
o modelo de embedding e salva somente:

    <output_dir>/<reference_name>_original.png
    <output_dir>/<reference_name>_marcada.png

Não executa ataques, DeepFake, extração da marca ou cálculo de métricas.

Marca dupla (versão 4): com --fragil <checkpoint da semi-frágil>, a marca
semi-frágil é posta por cima da robusta e são salvas:

    <output_dir>/<reference_name>_original.png
    <output_dir>/<reference_name>_so_robusta.png   (só a marca robusta)
    <output_dir>/<reference_name>_marcada.png      (robusta + semi-frágil)

Exemplo:
    python collect_image_by_diff.py \
        --models_path output/checkpoints/watermark_Y_DCT_DWT_07102026_08:06:39_final.pth \
        --dataset_dir Dataset/celeba_hq/val \
        --output_dir "my_tests" \
        --reference_name "003251" \
        --reference_file "003251.jpg" \
        --image_resolution 256 \
        --watermark_size 30 \
        --arnold_iterations 0 \
        --cuda 0
"""

import os
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from watermark_embedding import (
    WatermarkEmbedding,
    watermark_embedding_tensor,
)


# ============================================================
# FIND IMAGES
# ============================================================
def find_image(root, filename):
    """
    Procura recursivamente pelo arquivo exato dentro do dataset.
    """
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f == filename:
                return os.path.join(dirpath, f)

    return None


# ============================================================
# MAIN
# ============================================================
def main():

    parser = argparse.ArgumentParser(
        description="Gera somente a imagem original e a imagem marcada "
                    "para uma referência específica."
    )

    parser.add_argument(
        "--models_path",
        required=True,
        help="Caminho do checkpoint do modelo de embedding."
    )

    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Pasta do dataset onde procurar a imagem."
    )

    parser.add_argument(
        "--output_dir",
        default="deepguard_tests",
        help="Pasta onde salvar as duas imagens."
    )

    parser.add_argument(
        "--reference_name",
        default="003251",
        help="Nome usado nos arquivos de saída. Ex.: 003251"
    )

    parser.add_argument(
        "--reference_file",
        default="003251.png",
        help="Nome exato da imagem a ser procurada."
    )

    parser.add_argument(
        "--image_resolution",
        type=int,
        default=128
    )

    parser.add_argument(
        "--watermark_size",
        type=int,
        default=None,
        help="Opcional: o tamanho da marca é lido do checkpoint."
    )

    parser.add_argument(
        "--arnold_iterations",
        type=int,
        default=0
    )

    parser.add_argument(
        "--cuda",
        type=int,
        default=0
    )

    parser.add_argument(
        "--fragil",
        default=None,
        help="Opcional: checkpoint da marca semi-frágil (train_fragil.py); "
             "ela é posta por cima da robusta."
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Opcional: força da marca (padrão: o alpha salvo no checkpoint)."
    )

    args = parser.parse_args()

    # ========================================================
    # DEVICE
    # ========================================================
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # ========================================================
    # OUTPUT
    # ========================================================
    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================================
    # LOCALIZAR IMAGEM
    # ========================================================
    image_path = find_image(
        args.dataset_dir,
        args.reference_file
    )

    if image_path is None:
        raise FileNotFoundError(
            f"\nImagem '{args.reference_file}' não encontrada em:\n"
            f"{args.dataset_dir}\n"
        )

    print(f"Imagem encontrada:")
    print(f"  {image_path}")

    # ========================================================
    # TRANSFORM
    # ========================================================
    image_transform = transforms.Compose([
        transforms.Resize(
            (args.image_resolution, args.image_resolution)
        ),
        transforms.ToTensor(),
    ])

    # ========================================================
    # CARREGAR IMAGEM ORIGINAL
    # ========================================================
    image = Image.open(image_path).convert("RGB")

    image_tensor = (
        image_transform(image)
        .unsqueeze(0)
        .to(device)
    )

    # ========================================================
    # CARREGAR CHECKPOINT (define o formato da marca)
    # ========================================================
    checkpoint = torch.load(args.models_path, map_location=device)

    # Mesma lógica do evaluate.py:
    # - novos: num_bits bits, mapa interno grid_size x grid_size
    # - antigos (sem num_bits): marca quadrada grid_size x grid_size
    num_bits = checkpoint.get("num_bits", 0)

    if num_bits:
        grid_size = checkpoint["watermark_size"]
        watermark_shape = (1, num_bits)
        print(f"Marca: {num_bits} bits (mapa interno {grid_size}x{grid_size})")
    else:
        upsample = checkpoint["embedding_model"]["watermark_encoder.0.weight"].shape[-1]
        grid_size = (args.image_resolution // 2) // upsample
        watermark_shape = (1, 1, grid_size, grid_size)
        print(f"Marca (modelo antigo, quadrada): {grid_size}x{grid_size}")

    # ========================================================
    # WATERMARK
    # ========================================================
    # Mantém a mesma lógica do código de avaliação:
    # uma watermark binária aleatória.
    watermark_tensor = torch.randint(
        0,
        2,
        watermark_shape,
        device=device,
        dtype=torch.float32,
    )

    # ========================================================
    # CARREGAR MODELO
    # ========================================================
    embedding_model = WatermarkEmbedding(
        watermark_size=grid_size,
        image_resolution=args.image_resolution,
        num_bits=num_bits or None,
    ).to(device)

    embedding_model.load_state_dict(checkpoint["embedding_model"])
    embedding_model.alpha = args.alpha if args.alpha is not None else checkpoint["alpha"]
    print(f"Alpha: {embedding_model.alpha}")

    embedding_model.eval()

    print("Modelo de embedding carregado.")

    # ========================================================
    # EMBEDDING
    # ========================================================
    with torch.no_grad():

        watermarked, _, _ = watermark_embedding_tensor(
            image_tensor=image_tensor,
            watermark_image=watermark_tensor,
            embedding_model=embedding_model,
            arnold_iterations=args.arnold_iterations,
        )

    so_robusta = None
    if args.fragil:
        from marca_fragil import mascara_rosto, inserir_fragil, chave_fragil, mensagem_fragil

        ck_f = torch.load(args.fragil, map_location=device)
        fragil_model = WatermarkEmbedding(
            watermark_size=grid_size,
            image_resolution=args.image_resolution,
            num_bits=num_bits or None,
        ).to(device).eval()
        fragil_model.load_state_dict(ck_f["embedding_model"])
        fragil_model.alpha = ck_f["alpha"]
        print(f"Semi-frágil: {args.fragil} (alpha {fragil_model.alpha})")

        # quantiza a robusta como uma imagem salva, e põe a semi-frágil por cima
        so_robusta = torch.round(watermarked.clamp(0, 1) * 255) / 255
        mensagem = mensagem_fragil(watermark_tensor, chave_fragil(num_bits))
        with torch.no_grad():
            watermarked, _ = inserir_fragil(
                so_robusta, mensagem, fragil_model,
                mascara_rosto(args.image_resolution).to(device), ck_f.get("mask_floor", 0.0)
            )

    # ========================================================
    # CONVERTER ORIGINAL
    # ========================================================
    original_np = (
        image_tensor[0]
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy() * 255
    )

    original_np = np.clip(
        original_np,
        0,
        255
    ).astype(np.uint8)

    # ========================================================
    # CONVERTER MARCADA
    # ========================================================
    watermarked_np = (
        watermarked[0]
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy() * 255
    )

    watermarked_np = np.clip(
        watermarked_np,
        0,
        255
    ).astype(np.uint8)

    # ========================================================
    # NOMES DE SAÍDA
    # ========================================================
    original_path = os.path.join(
        args.output_dir,
        f"{args.reference_name}_original.png"
    )

    marked_path = os.path.join(
        args.output_dir,
        f"{args.reference_name}_marcada.png"
    )

    # ========================================================
    # SALVAR
    # ========================================================
    cv2.imwrite(
        original_path,
        cv2.cvtColor(
            original_np,
            cv2.COLOR_RGB2BGR
        )
    )

    cv2.imwrite(
        marked_path,
        cv2.cvtColor(
            watermarked_np,
            cv2.COLOR_RGB2BGR
        )
    )

    if so_robusta is not None:
        so_robusta_path = os.path.join(args.output_dir, f"{args.reference_name}_so_robusta.png")
        so_robusta_np = (so_robusta[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
        cv2.imwrite(so_robusta_path, cv2.cvtColor(so_robusta_np, cv2.COLOR_RGB2BGR))
        print(f"Só a robusta:\n  {so_robusta_path}")

    # ========================================================
    # FINAL
    # ========================================================
    print("\n" + "=" * 60)
    print("IMAGENS DE REFERÊNCIA GERADAS")
    print("=" * 60)

    print(f"Original:")
    print(f"  {original_path}")

    print(f"Marcada:")
    print(f"  {marked_path}")

    print("\nNenhum ataque foi executado.")
    print("Nenhuma extração de watermark foi executada.")
    print("Nenhuma métrica foi calculada.")
    print("=" * 60)


if __name__ == "__main__":
    main()
