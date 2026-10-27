import torch
import torch.nn as nn
import utils

from pytorch_wavelets import ( DWTForward )


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation: aprende a pesar cada canal (sub-banda)
    de acordo com sua confiabilidade sob o ataque atual, em vez de
    tratar todos os canais de entrada com peso fixo.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        hidden = max(channels // reduction, 1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape

        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w


class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        num_groups=8
    ):
        super().__init__()

        # num_groups precisa dividir out_channels; se não dividir,
        # cai para 1 grupo (equivalente a InstanceNorm nesse caso)
        groups = num_groups if out_channels % num_groups == 0 else 1

        self.block = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        x
    ):
        return self.block(x)


class WatermarkExtraction(
    nn.Module
):

    def __init__(self, watermark_size=16, image_resolution=128, use_hh=True, num_bits=None):
        super().__init__()

        # num_bits=N: saída é um vetor de N bits (ver WatermarkEmbedding)
        self.num_bits = num_bits

        self.watermark_size = watermark_size
        self.use_hh = use_hh
        in_ch = 4 if use_hh else 3

        # -------------------------
        # As sub-bandas que chegam no extrator têm metade da
        # resolução da imagem (por causa da DWT J=1). O fator
        # de downsample final é exato e conhecido de antemão,
        # então criamos a camada aqui no __init__ (registrada
        # nos parâmetros do módulo desde o início) em vez de
        # criá-la dinamicamente no forward — assim o otimizador,
        # que é montado logo após instanciar o modelo, já
        # enxerga e treina esses pesos desde o primeiro passo.
        # -------------------------
        feature_size = image_resolution // 2

        if feature_size % watermark_size != 0:
            raise ValueError(
                f"image_resolution/2 ({feature_size}) precisa ser "
                f"múltiplo de watermark_size ({watermark_size})."
            )

        self.downsample_factor = feature_size // watermark_size

        self.dwt = DWTForward(J=1, wave="haar")

        self.se_in = SEBlock(channels=in_ch)

        # -------------------------
        # Encoder
        # -------------------------
        self.enc1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        # -------------------------
        # Bottleneck
        # -------------------------
        self.bottleneck = DoubleConv(64, 128)

        # -------------------------
        # Decoder
        # -------------------------
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        # -------------------------
        # Output
        # -------------------------
        self.out_refine = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True)
        )

        self.out_down = nn.Conv2d(
            32, 32,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor
        )
        if num_bits:
            self.out_fc = nn.Linear(32 * watermark_size * watermark_size, num_bits)
        else:
            self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def extract_frequency_tensor(self, images, block_size=8):

        if images.max() <= 1.0:
            images = images * 255.0

        R = images[:, 0]
        G = images[:, 1]
        B = images[:, 2]

        Y = 0.299 * R + 0.587 * G + 0.114 * B

        cA, (cH, cV, cD) = utils.dwt2_torch(Y, self.dwt)    

        cA_DCT = utils.block_dct2d(cA, block_size)
        cH_DCT = utils.block_dct2d(cH, block_size)
        cV_DCT = utils.block_dct2d(cV, block_size)
        cD_DCT = utils.block_dct2d(cD, block_size)

        chans = [cA_DCT, cH_DCT, cV_DCT]

        if self.use_hh:
            chans.append(cD_DCT)

        return torch.stack(chans, dim=1)

    def forward(self, x):
        x = self.extract_frequency_tensor(x)
        x = self.se_in(x)

        # -------------------------
        # Encoder
        # -------------------------
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # -------------------------
        # Bottleneck
        # -------------------------
        b = self.bottleneck(p2)

        # -------------------------
        # Decoder
        # -------------------------
        u2 = self.up2(b)
        u2 = torch.cat((u2, e2), dim=1)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat((u1, e1), dim=1)
        u1 = self.dec1(u1)

        feat = self.out_refine(u1)
        feat = self.out_down(feat)

        if self.num_bits:
            out = self.out_fc(feat.flatten(1))   # (B, num_bits)
        else:
            out = self.out_conv(feat)            # (B, 1, watermark_size, watermark_size)

        return out
