#!/usr/bin/env python3
"""
preparar_pontos_scrfd.py

Detecta o rosto de cada imagem com o SCRFD-10G (InsightFace) e salva os 5
pontos (olhos, nariz e cantos da boca), em pixels da imagem 256x256. Roda uma
vez; o treino e a avaliação da semi-frágil só leem o arquivo, e o ambiente
deles não precisa do onnxruntime.

Saída, para cada pasta (--pastas): <saida>/<nome da pasta>.npz com
    nomes       nome do arquivo de cada imagem
    pontos      (N, 5, 2) em px de 256x256 (NaN onde não achou rosto)
    detectado   (N,) bool
    score       (N,) confiança do SCRFD
    molde       (5, 2) posição média dos pontos nas imagens da PRIMEIRA pasta
                (o treino): é onde a elipse da semi-frágil foi desenhada

A imagem é reduzida para 256x256 e ganha uma margem cinza de 64 px antes da
detecção: com o rosto ocupando a imagem inteira, o SCRFD deixa de achar metade
dos rostos do CelebA-HQ (ver detetores_rostos/README.md).

Ambiente: detectores (onnxruntime-gpu + insightface).

    ~/miniconda3/envs/detectores/bin/python preparar_pontos_scrfd.py
"""

import os
import glob
import ctypes
import argparse

import cv2
import numpy as np
from tqdm import tqdm

AQUI = os.path.dirname(os.path.abspath(__file__))
LADO = 256
MARGEM = 64


def carregar_cuda_do_torch():
    """O onnxruntime-gpu procura o cuDNN 9 e o cuBLAS no sistema; aqui eles só
    existem dentro dos pacotes nvidia-* do PyTorch. Carregá-los antes deixa o
    onnxruntime encontrá-los."""
    try:
        import nvidia
    except ImportError:
        return
    base = os.path.dirname(nvidia.__file__)
    for pasta in ["cuda_runtime", "cuda_nvrtc", "nvjitlink", "cublas", "cufft", "curand", "cudnn"]:
        for lib in sorted(glob.glob(os.path.join(base, pasta, "lib", "lib*.so*"))):
            if "nvblas" in lib:
                continue
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def listar(pasta):
    exts = (".jpg", ".jpeg", ".png")
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(pasta) for f in fs if f.lower().endswith(exts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pastas", default="Dataset/celeba_hq/train,Dataset/celeba_hq/val",
                   help="Pastas de imagens, separadas por vírgula (a primeira define o molde)")
    p.add_argument("--modelo", default=os.path.join(AQUI, "..", "..", "detetores_rostos", "modelos", "scrfd_10g.onnx"),
                   help="SCRFD-10G (det_10g.onnx do pacote buffalo_l do InsightFace)")
    p.add_argument("--saida", default="pontos_scrfd")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    os.makedirs(args.saida, exist_ok=True)

    if not args.cpu:
        carregar_cuda_do_torch()
    import onnxruntime
    onnxruntime.set_default_logger_severity(3)
    from insightface.model_zoo import get_model
    prov = ["CPUExecutionProvider"] if args.cpu else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    det = get_model(args.modelo, providers=prov)
    det.prepare(ctx_id=-1 if args.cpu else 0, input_size=(640, 640), det_thresh=0.5)

    molde = None
    for pasta in args.pastas.split(","):
        arqs = listar(pasta)
        nomes, pontos, score = [], np.full((len(arqs), 5, 2), np.nan, np.float32), np.zeros(len(arqs), np.float32)
        for i, arq in enumerate(tqdm(arqs, desc=pasta)):
            nomes.append(os.path.basename(arq))
            img = cv2.resize(cv2.imread(arq), (LADO, LADO), interpolation=cv2.INTER_AREA)
            img = cv2.copyMakeBorder(img, MARGEM, MARGEM, MARGEM, MARGEM, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            caixas, kps = det.detect(img, input_size=(640, 640))
            if len(caixas):
                j = int(np.argmax(caixas[:, 4]))
                pontos[i] = kps[j] - MARGEM
                score[i] = caixas[j, 4]
        detectado = ~np.isnan(pontos[:, 0, 0])
        if molde is None:
            molde = np.nanmean(pontos[detectado], 0)
        nome = os.path.basename(os.path.normpath(pasta))
        np.savez(os.path.join(args.saida, f"{nome}.npz"), nomes=np.array(nomes), pontos=pontos,
                 detectado=detectado, score=score, molde=molde)
        print(f"{pasta}: {detectado.sum()}/{len(arqs)} rostos -> {os.path.join(args.saida, nome + '.npz')}")
    print("Molde (olho esq., olho dir., nariz, boca esq., boca dir.):\n" + np.array2string(molde, precision=1))


if __name__ == "__main__":
    main()
