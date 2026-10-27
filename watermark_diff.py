#!/usr/bin/env python3
"""
watermark_diff.py

Recebe uma imagem original e sua versão marcada (com watermark embutido),
calcula a diferença pixel a pixel (o "ruído" da marca d'água) e salva:

  1. original.png            -> cópia da imagem original
  2. marcada.png              -> cópia da imagem marcada
  3. diferenca.png            -> mapa de diferença (amplificado, para ficar visível)
  4. grid_comparativo.png     -> as três empilhadas verticalmente, com legendas
                                  (original / marcada / diferença)

A diferença é calculada como |original - marcada| por pixel/canal. Como esse
valor costuma ser muito pequeno (a marca d'água é quase imperceptível), o
script AMPLIFICA o resultado por um fator (--amplify) só para fins de
visualização — o valor "cru" da diferença também fica disponível se você
passar --no_amplify_saved (salva a diferença real, sem amplificação).

Uso básico:
    python watermark_diff.py \
        --original ./my_tests/003251_original.png \
        --watermarked ./my_tests/003251_marcada.png \
        --output_dir ./my_tests/resultado_diff

Opções:
    --amplify 15          fator de amplificação visual da diferença (default: 15)
    --colormap jet        aplica um mapa de cor tipo "heatmap" na diferença
                           (precisa de matplotlib; sem essa flag, fica em cinza)
    --no_amplify_saved    salva diferenca.png com o valor real (sem amplificar)

Requisitos: pip install pillow numpy
Opcional (para --colormap): pip install matplotlib
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_image_as_array(path: Path, target_size=None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if target_size is not None and img.size != target_size:
        print(f"[AVISO] {path.name} tem tamanho {img.size}, redimensionando para {target_size} "
              f"para bater com a outra imagem.")
        img = img.resize(target_size, Image.LANCZOS)
    return np.array(img).astype(np.int16)  # int16 para evitar overflow na subtração


def compute_diff(orig: np.ndarray, wm: np.ndarray) -> np.ndarray:
    """Diferença absoluta por pixel/canal, valor real (0-255, sem amplificar)."""
    return np.abs(orig - wm).astype(np.uint8)


def amplify(diff: np.ndarray, factor: float) -> np.ndarray:
    amplified = diff.astype(np.float32) * factor
    return np.clip(amplified, 0, 255).astype(np.uint8)


def apply_colormap(diff_gray: np.ndarray, cmap_name: str) -> np.ndarray:
    """Converte um mapa de diferença (H,W,3) em escala de cinza equivalente
    para uma imagem colorida tipo heatmap, usando matplotlib."""
    try:
        import matplotlib.cm as cm
    except ImportError:
        raise SystemExit(
            "Para usar --colormap é preciso instalar matplotlib:\n"
            "  pip install matplotlib --break-system-packages"
        )
    # reduz para 1 canal (média dos 3 canais) antes de aplicar o colormap
    gray = diff_gray.mean(axis=2) / 255.0
    colormap = cm.get_cmap(cmap_name)
    colored = colormap(gray)[:, :, :3]  # remove canal alpha
    return (colored * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original", required=True, type=Path, help="Caminho da imagem original")
    parser.add_argument("--watermarked", required=True, type=Path, help="Caminho da imagem marcada")
    parser.add_argument("--output_dir", required=True, type=Path, help="Pasta onde salvar os resultados")
    parser.add_argument("--amplify", type=float, default=15.0,
                         help="Fator de amplificação visual da diferença (default: 15)")
    parser.add_argument("--colormap", default=None,
                         help="Nome de um colormap do matplotlib (ex: jet, inferno, viridis). "
                              "Sem essa opção, a diferença fica em tons de cinza.")
    parser.add_argument("--no_amplify_saved", action="store_true",
                         help="Salva diferenca.png com o valor real, sem amplificar")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    orig_img = Image.open(args.original).convert("RGB")
    orig_arr = load_image_as_array(args.original)
    wm_arr = load_image_as_array(args.watermarked, target_size=orig_img.size)
    wm_img = Image.fromarray(wm_arr.astype(np.uint8))

    diff_raw = compute_diff(orig_arr, wm_arr)  # valor real, 0-255
    diff_to_save = diff_raw if args.no_amplify_saved else amplify(diff_raw, args.amplify)

    if args.colormap:
        diff_visual = apply_colormap(diff_to_save, args.colormap)
    else:
        diff_visual = diff_to_save

    diff_img = Image.fromarray(diff_visual)

    # estatísticas úteis para o artigo (MSE, diferença máxima, média)
    mse = float(np.mean((orig_arr.astype(np.float32) - wm_arr.astype(np.float32)) ** 2))
    max_diff = int(diff_raw.max())
    mean_diff = float(diff_raw.mean())
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")

    # 1-3. salva as três imagens separadas
    orig_img.save(args.output_dir / "original.png")
    wm_img.save(args.output_dir / "marcada.png")
    diff_img.save(args.output_dir / "diferenca.png")

    # 4. monta o grid vertical SEM legendas
    #    com pequeno espaçamento entre as imagens

    spacing = 2  # pixels de distância entre as imagens

    width = max(
        orig_img.width,
        wm_img.width,
        diff_img.width
    )

    total_height = (
        orig_img.height
        + wm_img.height
        + diff_img.height
        + spacing * 2
    )

    grid = Image.new(
        "RGB",
        (width, total_height),
        "white"
    )

    y = 0

    for i, part in enumerate((orig_img, wm_img, diff_img)):
        grid.paste(part, (0, y))
        y += part.height

        if i < 2:
            y += spacing

    grid.save(
        args.output_dir / "grid_comparativo.png"
    )

    print(f"Salvo em: {args.output_dir}")
    print("  - original.png")
    print("  - marcada.png")
    print("  - diferenca.png")
    print("  - grid_comparativo.png")
    print()
    print("Estatísticas da diferença (valores reais, não amplificados):")
    print(f"  MSE:  {mse:.4f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Diferença média por pixel: {mean_diff:.4f}")
    print(f"  Diferença máxima:          {max_diff}")


if __name__ == "__main__":
    main()