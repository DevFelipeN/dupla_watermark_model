import torch
from pytorch_wavelets import ( DWTForward, DWTInverse )
import torch.nn.functional as F
import torch_dct as dct

from pytorch_wavelets import DWTForward


def arnold_transform(image, iterations=1):

    n = image.shape[-1]

    if image.shape[-2] != n:
        raise ValueError(
            "Arnold Transform requires a square image"
        )

    device = image.device

    x = torch.arange(
        n,
        device=device
    )

    y = torch.arange(
        n,
        device=device
    )

    xx, yy = torch.meshgrid(
        x,
        y,
        indexing="ij"
    )

    new_x = (
        2 * xx + yy
    ) % n

    new_y = (
        xx + yy
    ) % n

    transformed_image = image.clone()

    for _ in range(iterations):

        temp_image = torch.zeros_like(
            transformed_image
        )

        temp_image[
            ...,
            new_x,
            new_y
        ] = transformed_image[
            ...,
            xx,
            yy
        ]

        transformed_image = temp_image

    return transformed_image

def inverse_arnold_transform(image, iterations=1):

    n = image.shape[-1]

    if image.shape[-2] != n:
        raise ValueError(
            "Arnold Transform requires a square image"
        )

    device = image.device

    x = torch.arange(
        n,
        device=device
    )

    y = torch.arange(
        n,
        device=device
    )

    xx, yy = torch.meshgrid(
        x,
        y,
        indexing="ij"
    )

    old_x = (
        xx - yy
    ) % n

    old_y = (
        -xx + 2 * yy
    ) % n

    transformed_image = image.clone()

    for _ in range(iterations):

        temp_image = torch.zeros_like(
            transformed_image
        )

        temp_image[
            ...,
            old_x,
            old_y
        ] = transformed_image[
            ...,
            xx,
            yy
        ]

        transformed_image = temp_image

    return transformed_image


    return (
        image > threshold
    ).float()

def block_dct2d(image, block_size=8):
    B, H, W = image.shape
    blocks = (
        image
        .unfold(1, block_size, block_size)
        .unfold(2, block_size, block_size)
    )
    # (B, H_blocks, W_blocks, 8, 8)
    blocks_dct = dct.dct_2d(
        blocks,
        norm='ortho'
    )
    return (
        blocks_dct
        .permute(
            0, 1, 3,
            2, 4
        )
        .reshape(
            B,
            H,
            W
        )
    )

def block_idct2d(dct_coeffs,block_size=8):

    B, H, W = dct_coeffs.shape

    blocks = (
        dct_coeffs
        .unfold(
            1,
            block_size,
            block_size
        )
        .unfold(
            2,
            block_size,
            block_size
        )
    )

    # (B,H_blocks,W_blocks,8,8)

    blocks_idct = dct.idct_2d(
        blocks,
        norm='ortho'
    )

    return (
        blocks_idct
        .permute(
            0,
            1,
            3,
            2,
            4
        )
        .reshape(
            B,
            H,
            W
        )
    )

def dwt2_torch(x, dwt):

    yl, yh = dwt(
        x.unsqueeze(1)
    )

    cA = yl.squeeze(1)

    cH = yh[0][:,0,0]
    cV = yh[0][:,0,1]
    cD = yh[0][:,0,2]

    return (
        cA,
        (
            cH,
            cV,
            cD
        )
    )

def idwt2_torch(coeffs,idwt):

    cA, (
        cH,
        cV,
        cD
    ) = coeffs

    yl = cA.unsqueeze(1)

    yh = [
        torch.stack(
            (
                cH,
                cV,
                cD
            ),
            dim=1
        ).unsqueeze(1)
    ]

    reconstructed = idwt(
        (
            yl,
            yh
        )
    )

    return reconstructed.squeeze(1)