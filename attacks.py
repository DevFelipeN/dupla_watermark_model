import random
import io
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import kornia
import cv2
import subprocess
from deepfake_attack import DeepFakeAttack


# =======================================================
# Differentiable JPEG (adaptado do código de referência)
# =======================================================

_LUMA_TABLE_50 = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=torch.float32)

_CHROMA_TABLE_50 = torch.tensor([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=torch.float32)


def _get_quantization_table(quality: int, is_luma: bool) -> torch.Tensor:
    if not 1 <= quality <= 95:
        raise ValueError("JPEG quality deve estar entre 1 e 95.")

    base = _LUMA_TABLE_50 if is_luma else _CHROMA_TABLE_50
    if quality < 50:
        scale = 5000.0 / quality
    else:
        scale = 200.0 - 2.0 * quality

    return torch.clamp(
        torch.floor((scale * base + 50.0) / 100.0),
        1.0,
        255.0,
    )


def _build_dct_matrix(n: int) -> torch.Tensor:
    mat = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i == 0:
                mat[i, j] = 1.0 / np.sqrt(n)
            else:
                mat[i, j] = np.sqrt(2.0 / n) * np.cos(
                    np.pi * (2 * j + 1) * i / (2 * n)
                )
    return mat


class _StraightThroughRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


_round_ste = _StraightThroughRound.apply


class DiffJPEG(nn.Module):
    """JPEG diferenciável baseado na implementação do código de referência.

    Entrada e saída: [B, 3, H, W] em [0, 1].
    Internamente a implementação trabalha em [-1, 1].
    """

    def __init__(self, quality: int = 75):
        super().__init__()
        self.quality = int(quality)

        dct = _build_dct_matrix(8)
        self.register_buffer("dct", dct)
        self.register_buffer("dct_t", dct.T)

        self.register_buffer(
            "q_luma", _get_quantization_table(self.quality, True)
        )
        self.register_buffer(
            "q_chroma", _get_quantization_table(self.quality, False)
        )

        rgb2ycbcr = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.16874, -0.33126, 0.5],
            [0.5, -0.41869, -0.08131],
        ], dtype=torch.float32)
        ycbcr2rgb = torch.inverse(rgb2ycbcr)
        self.register_buffer("rgb2ycbcr", rgb2ycbcr)
        self.register_buffer("ycbcr2rgb", ycbcr2rgb)

    def _rgb_to_ycbcr(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, 3)
        y = x_flat @ self.rgb2ycbcr.T
        return y.reshape(B, H, W, 3).permute(0, 3, 1, 2)

    def _ycbcr_to_rgb(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, 3)
        rgb = x_flat @ self.ycbcr2rgb.T
        return rgb.reshape(B, H, W, 3).permute(0, 3, 1, 2)

    def _to_blocks(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // 8, 8, W // 8, 8)
        return x.permute(0, 1, 2, 4, 3, 5)

    def _from_blocks(self, x, H, W):
        x = x.permute(0, 1, 2, 4, 3, 5)
        return x.contiguous().view(x.shape[0], x.shape[1], H, W)

    def _block_dct(self, blocks):
        return self.dct @ blocks @ self.dct_t

    def _block_idct(self, blocks):
        return self.dct_t @ blocks @ self.dct

    def forward(self, x):
        # x aqui está em [0, 1]
        B, C, H, W = x.shape
        if C != 3:
            raise ValueError("DiffJPEG espera imagens RGB com 3 canais.")

        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # [0,1] -> [-1,1] -> [0,1]
        x_01 = x.clamp(0.0, 1.0)
        ycbcr = self._rgb_to_ycbcr(x_01)

        y_ch = (ycbcr[:, 0:1] - 0.5) * 255.0
        cb_ch = ycbcr[:, 1:2] * 255.0
        cr_ch = ycbcr[:, 2:3] * 255.0

        Hp, Wp = x.shape[2], x.shape[3]

        def _compress_channel(ch, is_luma):
            blocks = self._to_blocks(ch)
            dct_blocks = self._block_dct(blocks)
            q_table = self.q_luma if is_luma else self.q_chroma
            q_table = q_table.view(1, 1, 1, 1, 8, 8)
            quantized = _round_ste(dct_blocks / q_table)
            dequantized = quantized * q_table
            idct_blocks = self._block_idct(dequantized)
            return self._from_blocks(idct_blocks, ch.shape[2], ch.shape[3])

        y_out = _compress_channel(y_ch, is_luma=True)

        # JPEG 4:2:0 para crominância
        cb_down = F.avg_pool2d(cb_ch, 2)
        cr_down = F.avg_pool2d(cr_ch, 2)

        ch2, cw2 = cb_down.shape[2], cb_down.shape[3]
        pad_h2 = (8 - ch2 % 8) % 8
        pad_w2 = (8 - cw2 % 8) % 8

        if pad_h2 or pad_w2:
            cb_down = F.pad(cb_down, (0, pad_w2, 0, pad_h2), mode="reflect")
            cr_down = F.pad(cr_down, (0, pad_w2, 0, pad_h2), mode="reflect")

        cb_comp = _compress_channel(cb_down, is_luma=False)
        cr_comp = _compress_channel(cr_down, is_luma=False)

        if pad_h2 or pad_w2:
            cb_comp = cb_comp[:, :, :ch2, :cw2]
            cr_comp = cr_comp[:, :, :ch2, :cw2]

        cb_out = F.interpolate(
            cb_comp, size=(Hp, Wp), mode="bilinear", align_corners=False
        )
        cr_out = F.interpolate(
            cr_comp, size=(Hp, Wp), mode="bilinear", align_corners=False
        )

        y_out = y_out / 255.0 + 0.5
        cb_out = cb_out / 255.0
        cr_out = cr_out / 255.0

        ycbcr_out = torch.cat([y_out, cb_out, cr_out], dim=1)
        rgb_out = self._ycbcr_to_rgb(ycbcr_out)
        rgb_out = torch.clamp(rgb_out, 0.0, 1.0)

        if pad_h or pad_w:
            rgb_out = rgb_out[:, :, :H, :W]

        return rgb_out



# Ataques sorteados no treino (attack=None). A avaliação sempre passa o
# ataque e os parâmetros, então não é afetada por esta escolha.
ATTACK_SETS = {
    # treino até 09/10/2026 (versoes_salvas/v1_alpha3_09102026)
    "v1": [
        "none", "jpeg", "blur", "median_blur", "noise", "salt_pepper",
        "dropout", "dropout",   # 2x: dropout é o ataque mais fraco
        "resize",
    ],
    # v2: mais perto da avaliação. JPEG real (além do simulado), median 3 ou
    # 5, dropout até 0.3, resize seguido de JPEG e pequenas deformações
    # geométricas (reencenação/LivePortrait).
    "v2": [
        "none", "jpeg", "jpeg_real", "blur", "median_blur", "noise",
        "salt_pepper", "dropout", "dropout", "resize", "combined", "geometric",
    ],
    # v3: v2 sem a deformação geométrica. No treino do zero ela era o ataque
    # que segurava a queda do alpha e coincidiu com a perda de robustez ao
    # dropout, sem ajudar no LivePortrait (5% -> 9% no fine-tuning).
    "v3": [
        "none", "jpeg", "jpeg_real", "blur", "median_blur", "noise",
        "salt_pepper", "dropout", "dropout", "resize", "combined",
    ],
    # benigno: edições do dia a dia que a marca SEMI-FRÁGIL (versão 4) deve
    # aguentar; é o v3 com dropout uma vez só e sem salt & pepper (estragos
    # locais e fortes, mais perto de uma manipulação do que de uma edição comum).
    "benigno": [
        "none", "jpeg", "jpeg_real", "blur", "median_blur", "noise",
        "dropout", "resize", "combined",
    ],
    # benigno_realista: só as edições do dia a dia, nas faixas da avaliação
    # realista da marca dupla (JPEG 50-95, resize 0,5-0,9, blur, median 3,
    # ruído leve). Sem dropout e sem salt & pepper, que destroem a imagem:
    # treinar a semi-frágil para aguentá-los a deixaria robusta demais.
    "benigno_realista": [
        "none", "jpeg", "jpeg_real", "blur", "median_blur", "noise",
        "resize", "combined",
    ],
}


class AttackLayer(nn.Module):

    def __init__(self, attack_set="v1"):
        super().__init__()

        # =======================================================
        # ATAQUES DISPONÍVEIS
        # =======================================================

        if attack_set not in ATTACK_SETS:
            raise ValueError(f"attack_set deve ser um de {list(ATTACK_SETS)}")
        self.attack_set = attack_set
        self.attacks = list(ATTACK_SETS[attack_set])
        '''
        self.attacks = [
            "none",
            "jpeg"
        ]
        '''
        random.shuffle(self.attacks)

        self.deepfake_attack = DeepFakeAttack()

    # -------------------------------------------------------
    # Gaussian Noise
    #
    # Treinamento:
    #   std aleatório entre 0.01 e 0.05
    #
    # Teste:
    #   std pode ser informado manualmente
    # -------------------------------------------------------

    @staticmethod
    def add_gaussian_noise(images, std=None):

        if std is None:
            std = random.uniform(0.01, 0.05)

        noise = torch.randn_like(images) * std

        return torch.clamp(
            images + noise,
            0.0,
            1.0
        ), std

    # -------------------------------------------------------
    # Gaussian Blur
    #
    # Treinamento:
    #   sigma aleatório entre 1.0 e 2.0
    #
    # Teste:
    #   sigma pode ser informado manualmente
    # -------------------------------------------------------

    @staticmethod
    def gaussian_blur(images, sigma=None):

        if sigma is None:
            sigma = random.uniform(1.0, 2.0)

        attacked = TF.gaussian_blur(
            images,
            kernel_size=3,
            sigma=sigma
        )

        return attacked, sigma

    # -------------------------------------------------------
    # Median Blur
    #
    # Treinamento:
    #   kernel = 3
    #
    # Teste:
    #   kernel pode ser informado manualmente
    # -------------------------------------------------------

    @staticmethod
    def median_blur(images, kernel=3):

        # Garante kernel ímpar e >= 3
        kernel = int(kernel)

        if kernel < 3:
            kernel = 3

        if kernel % 2 == 0:
            kernel += 1

        attacked = kornia.filters.median_blur(
            images,
            (kernel, kernel)
        )

        return attacked, kernel

    # -------------------------------------------------------
    # Salt & Pepper
    #
    # Treinamento:
    #   amount aleatório entre 0.01 e 0.05
    #
    # Teste:
    #   amount pode ser informado manualmente
    # -------------------------------------------------------

    @staticmethod
    def salt_pepper(images, amount=None):

        if amount is None:
            amount = random.uniform(0.01, 0.05)

        images = images.clone()

        rnd = torch.rand_like(images)

        images[rnd < amount / 2] = 0.0
        images[rnd > 1 - amount / 2] = 1.0

        return images, amount

    # -------------------------------------------------------
    # Dropout
    #
    # Treinamento:
    #   p aleatório entre 0.03 e 0.15
    #
    # Teste:
    #   p pode ser informado manualmente
    # -------------------------------------------------------

    @staticmethod
    def dropout(images, p=None):

        if p is None:
            p = random.uniform(0.03, 0.15)

        mask = (
            torch.rand_like(images[:, :1]) > p
        ).float()

        return images * mask, p

    # -------------------------------------------------------
    # Resize
    #
    # Treinamento:
    #   scale aleatório entre 0.5 e 0.9
    #
    # Teste:
    #   scale pode ser informado manualmente
    # -------------------------------------------------------

    @staticmethod
    def resize(images, scale=None):

        if scale is None:
            # 0.5 exato apaga LH/HL/HH (equivale à LL da Haar); sorteado
            # na faixa contínua ele quase nunca aparece, então forçamos 25%
            if random.random() < 0.25:
                scale = 0.5
            else:
                scale = random.uniform(0.3, 0.9)

        _, _, H, W = images.shape

        new_h = max(
            1,
            int(H * scale)
        )

        new_w = max(
            1,
            int(W * scale)
        )

        resized = F.interpolate(
            images,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False
        )

        attacked = F.interpolate(
            resized,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        return attacked, scale

    # -------------------------------------------------------
    # JPEG
    #
    # Mantém a implementação atual.
    #
    # Treinamento:
    #   se quality=None, escolhe aleatoriamente
    #
    # Teste:
    #   quality pode ser definido manualmente.
    #
    # Exemplo:
    #   quality=50
    #   quality=70
    #   quality=80
    #   quality=90
    # -------------------------------------------------------

    @staticmethod
    def jpeg_compression(images, quality=None):
        """
        Entrada/saída em [0, 1].
        RGB -> YCbCr -> DCT 8x8 -> quantização -> IDCT -> RGB.
        """
        if quality is None:
            quality = random.randint(30, 95)
        quality = int(quality)

        if not 1 <= quality <= 95:
            raise ValueError(
                "JPEG quality deve estar entre 1 e 95."
            )

        jpeg = DiffJPEG(quality=quality).to(images.device)

        attacked = jpeg(
            images.float().clamp(0.0, 1.0)
        )

        attacked = attacked.to(images.dtype)

        return attacked, quality

    @staticmethod
    def jpeg_real(images, quality=None):
        """
        JPEG REAL no forward (OpenCV/libjpeg-turbo, o mesmo resultado do PIL
        usado no jpeg_test) e gradiente do DiffJPEG no backward. O DiffJPEG
        sozinho é só uma aproximação; com isso a rede treina contra os valores
        exatos do JPEG de verdade. ~26 ms por batch de 32 (256x256).
        """
        if quality is None:
            quality = random.randint(30, 95)
        quality = int(quality)

        x = images.float().clamp(0.0, 1.0)
        simulated = DiffJPEG(quality=quality).to(images.device)(x)

        u8 = (x.detach() * 255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        real = np.stack([
            cv2.imdecode(cv2.imencode(".jpg", im[:, :, ::-1], params)[1], cv2.IMREAD_COLOR)[:, :, ::-1]
            for im in u8
        ])
        real = torch.from_numpy(real.copy()).to(images.device).permute(0, 3, 1, 2).float() / 255.0

        attacked = simulated + (real - simulated).detach()
        return attacked.to(images.dtype), quality

    @staticmethod
    def geometric(images, max_angle=2.0, max_shift=3.0, max_scale=0.03):
        """
        Pequena deformação geométrica (rotação, deslocamento e escala) com
        bordas refletidas. Simula, de forma barata e diferenciável, o que a
        reencenação (LivePortrait) faz: mover os pixels de lugar sem destruir
        o conteúdo. Cada imagem do batch recebe uma transformação diferente.
        """
        B, _, H, W = images.shape
        dev = images.device

        angle = (torch.rand(B, device=dev) * 2 - 1) * max_angle * np.pi / 180
        scale = 1 + (torch.rand(B, device=dev) * 2 - 1) * max_scale
        shift = (torch.rand(B, 2, device=dev) * 2 - 1) * max_shift
        tx, ty = shift[:, 0] * 2 / W, shift[:, 1] * 2 / H   # pixels -> coordenadas [-1, 1]

        cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
        theta = torch.stack([
            torch.stack([cos, -sin, tx], dim=1),
            torch.stack([sin, cos, ty], dim=1),
        ], dim=1)

        grid = F.affine_grid(theta, images.shape, align_corners=False)
        attacked = F.grid_sample(images, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        return attacked, (max_angle, max_shift, max_scale)

    @staticmethod
    def jpeg_test(images, quality=None, subsampling=None):
        """
        subsampling: None (padrão do PIL/libjpeg), 0 (4:4:4, sem subsampling),
                    1 (4:2:2) ou 2 (4:2:0). Use 0 para isolar o efeito da
                    quantização em si, sem a perda adicional de crominância.
        """
        if quality is None:
            quality = random.randint(30, 95)

        quality = int(quality)
        if not 1 <= quality <= 100:
            raise ValueError("JPEG quality deve estar entre 1 e 100.")

        x = images.clamp(0.0, 1.0) * 2.0 - 1.0

        results = []
        x_np = ((x.detach().cpu() + 1.0) / 2.0 * 255.0).round().clamp(0, 255).byte()

        save_kwargs = {"format": "JPEG", "quality": quality}
        if subsampling is not None:
            save_kwargs["subsampling"] = subsampling

        for i in range(x.shape[0]):
            img = Image.fromarray(x_np[i].permute(1, 2, 0).numpy())
            buf = io.BytesIO()
            img.save(buf, **save_kwargs)
            buf.seek(0)
            img_jpeg = Image.open(buf).convert('RGB')
            t = torch.from_numpy(np.array(img_jpeg)).float() / 255.0 * 2.0 - 1.0
            results.append(t.permute(2, 0, 1))

        compressed = torch.stack(results, dim=0).to(images.device)
        compressed = x + (compressed - x).detach()

        attacked = (compressed + 1.0) / 2.0
        return attacked.to(images.dtype), quality
    # -------------------------------------------------------
    # Forward
    # -------------------------------------------------------

    def forward(
        self,
        images,
        attack=None,
        quality=None,
        sigma=None,
        kernel=None,
        std=None,
        amount=None,
        p=None,
        scale=None,
        subsampling=None,
        attribute=None,
        strength=None
    ):

        # ===================================================
        # ESCOLHA DO ATAQUE
        # ===================================================

        if attack is None:

            if len(self.attacks) == 0:
                self.attacks = list(ATTACK_SETS[self.attack_set])
                '''
                self.attacks = [
                    "none",
                    "jpeg"
                ]
                '''

                random.shuffle(self.attacks)

            attack = self.attacks.pop()

            # faixas mais leves para o conjunto benigno_realista
            if self.attack_set == "benigno_realista":
                if attack in ("jpeg", "jpeg_real") and quality is None:
                    quality = random.randint(50, 95)
                elif attack == "resize" and scale is None:
                    scale = random.uniform(0.5, 0.9)
                elif attack == "median_blur" and kernel is None:
                    kernel = 3
                elif attack == "noise" and std is None:
                    std = random.uniform(0.01, 0.03)

            # parâmetros sorteados do conjunto v2 (o v1 usa os padrões antigos)
            elif self.attack_set != "v1":
                if attack == "median_blur" and kernel is None:
                    kernel = random.choice([3, 5])
                elif attack == "dropout" and p is None:
                    p = random.uniform(0.03, 0.3)

        else:

            attack = attack.lower()

        # ===================================================
        # NONE
        # ===================================================

        if attack == "none":

            return images, {
                "name": "Identity"
            }

        # ===================================================
        # JPEG
        # ===================================================

        elif attack == "jpeg":

            attacked, used_quality = self.jpeg_compression(
                images,
                quality=quality
            )

            return attacked, {
                "name": "JPEG",
                "quality": used_quality
            }

        elif attack == "jpeg_real":

            attacked, used_quality = self.jpeg_real(
                images,
                quality=quality
            )

            return attacked, {
                "name": "JPEG",
                "quality": used_quality,
                "real": True
            }

        elif attack == "combined":

            # resize e depois JPEG real (ex.: foto reduzida e recomprimida)
            resized, used_scale = self.resize(
                images,
                scale=scale if scale is not None else random.uniform(0.5, 0.9)
            )
            attacked, used_quality = self.jpeg_real(
                resized,
                quality=quality if quality is not None else random.randint(50, 90)
            )

            return attacked, {
                "name": "Combined",
                "scale": used_scale,
                "quality": used_quality
            }

        elif attack == "geometric":

            attacked, used = self.geometric(images)

            return attacked, {
                "name": "Geometric",
                "max_angle": used[0],
                "max_shift": used[1],
                "max_scale": used[2]
            }

        elif attack == "jpeg_test":

            attacked, quality = self.jpeg_test(
                images,
                quality=quality
            )

            return attacked, {
                "name": "JPEG",
                "quality": quality
            }

        # ===================================================
        # GAUSSIAN BLUR
        # ===================================================

        elif attack == "blur":

            attacked, used_sigma = self.gaussian_blur(
                images,
                sigma=sigma
            )

            return attacked, {
                "name": "Gaussian Blur",
                "kernel": 3,
                "sigma": used_sigma
            }

        # ===================================================
        # MEDIAN BLUR
        # ===================================================

        elif attack == "median_blur":

            if kernel is None:
                kernel = 3

            attacked, used_kernel = self.median_blur(
                images,
                kernel=kernel
            )

            return attacked, {
                "name": "Median Blur",
                "kernel": used_kernel
            }

        # ===================================================
        # GAUSSIAN NOISE
        # ===================================================

        elif attack == "noise":

            attacked, used_std = self.add_gaussian_noise(
                images,
                std=std
            )

            return attacked, {
                "name": "Gaussian Noise",
                "std": used_std
            }

        # ===================================================
        # SALT & PEPPER
        # ===================================================

        elif attack == "salt_pepper":

            attacked, used_amount = self.salt_pepper(
                images,
                amount=amount
            )

            return attacked, {
                "name": "Salt & Pepper",
                "amount": used_amount
            }

        # ===================================================
        # DROPOUT
        # ===================================================

        elif attack == "dropout":

            attacked, used_p = self.dropout(
                images,
                p=p
            )

            return attacked, {
                "name": "Dropout",
                "p": used_p
            }

        # ===================================================
        # RESIZE
        # ===================================================

        elif attack == "resize":

            attacked, used_scale = self.resize(
                images,
                scale=scale
            )

            return attacked, {
                "name": "Resize",
                "scale": used_scale
            }

        # ===================================================
        # ATAQUE DEEPFAKE
        # ===================================================

        elif attack == "blendface":

            attacked = self.deepfake_attack.blendface(images)

            return attacked, {
                "name": "BlendFace"
            }
        
        elif attack == "diffae":

            attacked = self.deepfake_attack.diffae_manipulate(
                images,
                attribute=attribute if attribute is not None else "Smiling",
                strength=strength if strength is not None else 0.3
            )

            return attacked, {
                "name": "DiffAE",
                "attribute": attribute,
                "strength": strength
            }

        raise ValueError(
            f"Ataque desconhecido: {attack}"
        )