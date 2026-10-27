import torch
import torch.nn as nn
import utils

from pytorch_wavelets import ( DWTForward, DWTInverse )

class WatermarkEmbedding(nn.Module):

    def __init__(self, watermark_size=16, image_resolution=128,
             learnable_weights=False, per_channel_weights=False, num_bits=None):
        super().__init__()

        # num_bits=None: a marca é uma imagem watermark_size x watermark_size.
        # num_bits=N: a marca é um vetor de N bits, convertido por uma camada
        # linear num mapa watermark_size x watermark_size (qualquer N funciona).
        self.num_bits = num_bits
        self.watermark_size = watermark_size

        self.alpha = None
        self.learnable_weights = learnable_weights
        self.per_channel_weights = per_channel_weights   # sem efeito nesta versão (só o canal Y)

        # pesos iniciais (LL, LH, HL, HH); o softmax normaliza para somar 1
        init_logits = torch.log(torch.tensor([0.3, 0.35, 0.35, 1e-3]))

        if learnable_weights:
            self.subband_logits = nn.Parameter(init_logits)
        else:
            self.register_buffer("subband_logits", init_logits)   # fixo

        # OPÇÃO 3: a marca só usa os coeficientes DCT de frequência baixa e
        # média de cada bloco 8x8 (1 <= u+v <= 7: sem o DC, que deixaria os
        # blocos visíveis, e sem as frequências altas, que concentram a
        # energia em pontos). Coeficiente (u, v) do bloco fica em (8*by+u, 8*bx+v).
        u = torch.arange(8).view(8, 1)
        v = torch.arange(8).view(1, 8)
        band = ((u + v >= 1) & (u + v <= 7)).float()
        sub = image_resolution // 2
        self.register_buffer("dct_band_mask", band.repeat(sub // 8, sub // 8).view(1, 1, sub, sub))

        self.dwt = DWTForward(J=1, wave="haar")
        self.idwt = DWTInverse(wave='haar')

        self.subband_encoder = nn.Sequential(

            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(
                inplace=True
            )
        )

        # ==================================================
        # Encoder da marca
        # ==================================================

        feature_size = image_resolution // 2   # sub-bandas têm metade da resolução (DWT J=1)

        if feature_size % watermark_size != 0:
            raise ValueError(
                f"image_resolution/2 ({feature_size}) precisa ser "
                f"múltiplo de watermark_size ({watermark_size})."
            )

        upsample_factor = feature_size // watermark_size

        if num_bits:
            self.message_encoder = nn.Linear(num_bits, watermark_size * watermark_size)

        self.watermark_encoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1,
                out_channels=32,
                kernel_size=upsample_factor,
                stride=upsample_factor
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # ==================================================
        # Rede de fusão
        # ==================================================

        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(
                inplace=True
            )
        )
        self.out_conv = nn.Conv2d(
            32,
            4,
            kernel_size=3,
            padding=1
        )

    def get_subband_weights(self):
        w = torch.softmax(self.subband_logits, dim=0)   # (4,)
        return w.view(1, 4, 1, 1)    

    def forward(
        self,
        image_subbands,
        watermark
    ):

        # ==================================
        # Processa as seis sub-bandas
        # ==================================

        subband_features = (
            self.subband_encoder(
                image_subbands
            )
        )

        # Formato:
        #
        # [B, 64, H, W]


        # ==================================
        # Processa a marca
        # ==================================

        if self.num_bits:
            # (B, num_bits) -> (B, 1, watermark_size, watermark_size)
            watermark = self.message_encoder(watermark).view(
                -1, 1, self.watermark_size, self.watermark_size
            )

        watermark_features = self.watermark_encoder(watermark)
        # Formato:
        #
        # [B, 64, H, W]


        # ==================================
        # Junta as características
        # ==================================

        combined_features = (
            subband_features + watermark_features
        )

        features = self.fusion(
            combined_features
        )

        modification = torch.tanh(
            self.out_conv(features)
        ) * self.dct_band_mask

        subband_weights = self.get_subband_weights().to(modification.dtype)   # (1,4,1,1)

        # Aplica os pesos à modificação
        weighted_modification = (
            modification * subband_weights
        )

        # Aplica alpha global
        watermarked_subbands = (
            image_subbands
            +
            self.alpha
            *
            weighted_modification
        )

        return watermarked_subbands, weighted_modification

def watermark_embedding_tensor(image_tensor, watermark_image, embedding_model, arnold_iterations=0, block_size=8):

    if image_tensor.max() <= 1.0:
        image_tensor = image_tensor * 255.
    image_tensor = image_tensor.float()          # (B,3,H,W)

    R = image_tensor[:, 0]
    G = image_tensor[:, 1]
    B = image_tensor[:, 2]

    # Luminância (mesma fórmula do JPEG, ITU-R BT.601)
    Y = 0.299 * R + 0.587 * G + 0.114 * B        # (B,H,W)

    dwt = embedding_model.dwt
    idwt = embedding_model.idwt

    # ==================================================
    # DWT PRIMEIRO
    # ==================================================

    cA, (cH, cV, cD) = utils.dwt2_torch(
        Y,
        dwt
    )

    # ==================================================
    # DCT EM CADA SUB-BANDA
    # ==================================================

    cA_DCT = utils.block_dct2d(
        cA,
        block_size
    )

    cH_DCT = utils.block_dct2d(
        cH,
        block_size
    )

    cV_DCT = utils.block_dct2d(
        cV,
        block_size
    )

    cD_DCT = utils.block_dct2d(
        cD,
        block_size
    )

    # ==================================================
    # JUNTA AS QUATRO SUB-BANDAS
    # ==================================================

    image_subbands = torch.stack(
        [
            cA_DCT,
            cH_DCT,
            cV_DCT,
            cD_DCT
        ],
        dim=1
    )

    if watermark_image.dim() == 4:
        watermark_input = utils.arnold_transform(watermark_image, arnold_iterations)
    else:
        # marca em vetor (num_bits): Arnold só se aplica a imagens quadradas
        watermark_input = watermark_image
    watermark_input = watermark_input.float() * 2.0 - 1.0

    watermarked_subbands, modification = embedding_model(image_subbands, watermark_input)

    # ==================================================
    # SEPARA AS SUB-BANDAS MARCADAS
    # ==================================================

    cA_DCT_W = watermarked_subbands[:, 0]
    cH_DCT_W = watermarked_subbands[:, 1]
    cV_DCT_W = watermarked_subbands[:, 2]
    cD_DCT_W = watermarked_subbands[:, 3]

    # ==================================================
    # IDCT EM CADA SUB-BANDA
    # ==================================================

    cA_W = utils.block_idct2d(
        cA_DCT_W,
        block_size
    )

    cH_W = utils.block_idct2d(
        cH_DCT_W,
        block_size
    )

    cV_W = utils.block_idct2d(
        cV_DCT_W,
        block_size
    )

    cD_W = utils.block_idct2d(
        cD_DCT_W,
        block_size
    )

    # ==================================================
    # IDWT
    # ==================================================

    Y_W = utils.idwt2_torch(
        (
            cA_W,
            (
                cH_W,
                cV_W,
                cD_W
            )
        ),
        idwt
    )

    # soma a mudança de Y igualmente em R, G e B (Cb e Cr ficam intactos)
    delta_Y = Y_W - Y
    I_W = image_tensor + delta_Y.unsqueeze(1)
    I_W = torch.clamp(I_W, 0, 255) / 255.

    return I_W, watermark_image, modification

