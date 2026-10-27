import subprocess
import torch
from PIL import Image
from torchvision import transforms
import os
import shutil
import subprocess
import tempfile

# Caminhos dos modelos de deepfake e dos ambientes conda de cada um (usados só na
# avaliação com deepfakes). Em outra máquina, defina as variáveis de ambiente
# DEEPFAKE_ROOT (pasta com os modelos) e CONDA_ENVS (pasta dos ambientes conda).
DEEPFAKE_ROOT = os.environ.get(
    "DEEPFAKE_ROOT", "/home/cnt-computadores/Área de trabalho/FelipeNascimento/models_deepfake")
CONDA_ENVS = os.environ.get("CONDA_ENVS", "/home/cnt-computadores/miniconda3/envs")


def conda_python(env):
    return os.path.join(CONDA_ENVS, env, "bin", "python")


BLENDFACE_PYTHON = conda_python("blendeface")
BLENDFACE_DIR = os.path.join(DEEPFAKE_ROOT, "BlendFace", "swapping")
BLENDFACE_WEIGHTS = os.path.join(BLENDFACE_DIR, "checkpoints", "blendswap.pth")
BLENDFACE_SOURCE = os.path.join(BLENDFACE_DIR, "examples", "source.png")

SIMSWAP_PYTHON = conda_python("simswap")
SIMSWAP_DIR = os.path.join(DEEPFAKE_ROOT, "SimSwap")
SIMSWAP_SOURCE = os.path.join(SIMSWAP_DIR, "source.jpg")

INSWAPPER_PYTHON = conda_python("inswapper")
INSWAPPER_DIR = os.path.join(DEEPFAKE_ROOT, "inswapper")
INSWAPPER_SOURCE = SIMSWAP_SOURCE    # mesma identidade de origem do SimSwap

STARGAN_PYTHON = conda_python("stargan")
STARGAN_DIR = os.path.join(DEEPFAKE_ROOT, "stargan")

HISD_PYTHON = STARGAN_PYTHON         # mesmo ambiente do StarGAN
HISD_DIR = os.path.join(DEEPFAKE_ROOT, "HiSD")

LIVEPORTRAIT_PYTHON = conda_python("liveportrait")
LIVEPORTRAIT_DIR = os.path.join(DEEPFAKE_ROOT, "LivePortrait")

class DeepFakeAttack:

    def apply_dataset(
        self,
        attack,
        input_zip,
        output_dir,
        **kwargs
    ):
        """
        Recebe o tipo do DeepFake e um dataset ZIP.
        Executa o modelo externo uma única vez sobre o dataset.
        Retorna o ZIP resultante.
        """

        if attack == "diffae":
            return self.diffae_dataset(
                input_zip=input_zip,
                output_dir=output_dir,
                **kwargs
            )

        elif attack == "simswap":
            return self.simswap_dataset(
                input_zip=input_zip,
                output_dir=output_dir,
                **kwargs
            )

        elif attack == "inswapper":
            return self.inswapper_dataset(
                input_zip=input_zip,
                output_dir=output_dir,
                **kwargs
            )

        elif attack == "stargan":
            return self.stargan_dataset(
                input_zip=input_zip,
                output_dir=output_dir,
                **kwargs
            )

        elif attack == "hisd":
            return self.hisd_dataset(
                input_zip=input_zip,
                output_dir=output_dir,
                **kwargs
            )

        elif attack == "liveportrait":
            return self.liveportrait_dataset(
                input_zip=input_zip,
                output_dir=output_dir,
                **kwargs
            )

        elif attack == "blendface":
            return self.blendface_dataset(
                input_zip=input_zip,
                output_dir=output_dir,
                **kwargs
            )

        else:
            raise ValueError(
                f"DeepFake não suportado: {attack}"
            )

    def blendface(self, images):
        # images = batch [B, C, H, W]
        # Uma chamada do BlendFace por imagem (carrega o modelo a cada vez).
        # Para muitas imagens, use apply_dataset(attack="blendface").
        outputs = []

        with tempfile.TemporaryDirectory() as tmp:

            target_image = os.path.join(tmp, "target.png")
            output_image = os.path.join(tmp, "output.png")

            for image in images:

                transforms.ToPILImage()(image.cpu()).save(target_image)

                subprocess.run(
                    [
                        BLENDFACE_PYTHON,
                        "inference.py",
                        "-w", BLENDFACE_WEIGHTS,
                        "-t", target_image,
                        "-s", BLENDFACE_SOURCE,
                        "-o", output_image
                    ],
                    cwd=BLENDFACE_DIR,
                    check=True
                )

                result = Image.open(output_image).convert("RGB")
                outputs.append(transforms.ToTensor()(result))

        return torch.stack(outputs).to(images.device)

    def blendface_dataset(
        self,
        input_zip,
        output_dir,
        source_image=BLENDFACE_SOURCE,
        pairs_csv=None
    ):
        """
        Troca de rosto em todas as imagens do ZIP, carregando o BlendFace
        uma única vez. A identidade vem de pairs_csv (uma origem por imagem,
        coluna origem_alinhada_112) ou, sem ele, de source_image
        (112x112, alinhada no padrão InsightFace).
        """

        input_zip = os.path.abspath(input_zip)
        output_dir = os.path.abspath(output_dir)
        source_image = os.path.abspath(source_image)

        output_name = "blendface_dataset"

        attack_root = os.path.join(output_dir, "blendface")
        input_dir = os.path.join(attack_root, "input")
        output_dir_images = os.path.join(attack_root, "output")
        output_zip = os.path.join(attack_root, output_name + ".zip")

        # limpa arquivos de execuções anteriores
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(output_dir_images, ignore_errors=True)

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir_images, exist_ok=True)

        # 1. Extrair ZIP de entrada
        shutil.unpack_archive(input_zip, input_dir)

        # 2. Executar BlendFace UMA VEZ
        subprocess.run(
            [
                BLENDFACE_PYTHON,
                "blendface_dataset_infer.py",
                "--input_dir", input_dir,
                "--output_dir", output_dir_images,
                "--source_image", source_image,
                "--weight_path", BLENDFACE_WEIGHTS,
                "--device", "cuda:0",
            ] + (["--pairs_csv", os.path.abspath(pairs_csv)] if pairs_csv else []),
            cwd=BLENDFACE_DIR,
            check=True
        )

        # 3. Compactar resultados
        shutil.make_archive(
            os.path.join(attack_root, output_name),
            "zip",
            output_dir_images
        )

        return output_zip, {
            "attack": "blendface",
            "source_image": source_image,
            "output_zip": output_zip
        }

    def diffae_dataset(
        self,
        input_zip,
        output_dir,
        attribute="Smiling",
        strength=0.3,
        image_resolution=256
    ):


        DIFFAE_PYTHON = conda_python("diffae")

        # DiffAE na mesma resolução do modelo de marca d'água
        # (diffae128 usa FFHQ128; diffae256 usa FFHQ256)
        if image_resolution not in (128, 256):
            raise ValueError(f"Não há DiffAE para {image_resolution}x{image_resolution} (só 128 ou 256).")

        DIFFAE_DIR = os.path.join(DEEPFAKE_ROOT, f"diffae{image_resolution}")

        # converter caminhos relativos para absolutos
        input_zip = os.path.abspath(input_zip)
        output_dir = os.path.abspath(output_dir)

        output_name = "diffae_dataset"

        attack_root = os.path.join(
            output_dir,
            "diffae"
        )

        input_dir = os.path.join(
            attack_root,
            "input"
        )

        output_dir_images = os.path.join(
            attack_root,
            "output"
        )

        output_zip = os.path.join(
            attack_root,
            output_name + ".zip"
        )

        # limpa arquivos de execuções anteriores
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(output_dir_images, ignore_errors=True)

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir_images, exist_ok=True)

        # 1. Extrair ZIP de entrada
        shutil.unpack_archive(
            input_zip,
            input_dir
        )

        # 2. Executar DiffAE UMA VEZ
        command = [
            DIFFAE_PYTHON,
            "diffae_dataset_infer.py",

            "--input_dir",
            input_dir,

            "--output_dir",
            output_dir_images,

            "--attribute",
            attribute,

            "--strength",
            str(strength),

            "--device",
            "cuda:0",
        ]

        subprocess.run(
            command,
            cwd=DIFFAE_DIR,
            check=True
        )

        # 3. Compactar resultados
        shutil.make_archive(
            os.path.join(
                attack_root,
                output_name
            ),
            "zip",
            output_dir_images
        )

        return output_zip, {
            "attack": "diffae",
            "attribute": attribute,
            "strength": strength,
            "output_zip": output_zip
        }

    def _run_dataset_tool(self, name, input_zip, output_dir, python, cwd, script, extra_args):
        """
        Mesmo fluxo do DiffAE/BlendFace: descompacta o ZIP, roda o modelo
        externo UMA VEZ sobre a pasta e compacta o resultado.
        """
        input_zip = os.path.abspath(input_zip)
        attack_root = os.path.join(os.path.abspath(output_dir), name)
        input_dir = os.path.join(attack_root, "input")
        output_dir_images = os.path.join(attack_root, "output")

        # limpa arquivos de execuções anteriores
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(output_dir_images, ignore_errors=True)
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir_images, exist_ok=True)

        shutil.unpack_archive(input_zip, input_dir)

        subprocess.run(
            [python, script, "--input_dir", input_dir, "--output_dir", output_dir_images] + extra_args,
            cwd=cwd,
            check=True
        )

        # sem_rosto.txt (imagens em que o rosto não foi detectado) não vai para o ZIP
        report = os.path.join(output_dir_images, "sem_rosto.txt")
        if os.path.exists(report):
            shutil.move(report, os.path.join(attack_root, "sem_rosto.txt"))

        output_zip = shutil.make_archive(os.path.join(attack_root, f"{name}_dataset"), "zip", output_dir_images)
        return output_zip

    def simswap_dataset(self, input_zip, output_dir, source_image=SIMSWAP_SOURCE, pairs_csv=None):
        """
        Troca de rosto com SimSwap (recorte 224, máscara por segmentação do rosto),
        carregando o modelo uma vez. Saída em PNG, mesma resolução da entrada.
        """
        source_image = os.path.abspath(source_image)
        output_zip = self._run_dataset_tool(
            "simswap", input_zip, output_dir, SIMSWAP_PYTHON, SIMSWAP_DIR,
            "simswap_dataset_infer.py",
            ["--source_image", source_image] + (["--pairs_csv", os.path.abspath(pairs_csv)] if pairs_csv else [])
        )
        return output_zip, {"attack": "simswap", "source_image": source_image, "output_zip": output_zip}

    def inswapper_dataset(self, input_zip, output_dir, source_image=INSWAPPER_SOURCE, pairs_csv=None):
        """
        Troca de rosto com InsightFace InSwapper (o do Roop; recorte 128, máscara
        retangular suavizada), carregando o modelo uma vez. Saída em PNG.
        """
        source_image = os.path.abspath(source_image)
        output_zip = self._run_dataset_tool(
            "inswapper", input_zip, output_dir, INSWAPPER_PYTHON, INSWAPPER_DIR,
            "inswapper_dataset_infer.py",
            ["--source_image", source_image] + (["--pairs_csv", os.path.abspath(pairs_csv)] if pairs_csv else [])
        )
        return output_zip, {"attack": "inswapper", "source_image": source_image, "output_zip": output_zip}

    def stargan_dataset(self, input_zip, output_dir, attribute="Male"):
        """
        Edição de atributos com StarGAN v1 (CelebA 256x256), carregando o gerador
        uma vez. Mesmo protocolo do SepMark: atributos reais do list_attr_celeba.txt
        com um deles alterado (Male = troca de gênero). Saída em PNG.
        """
        output_zip = self._run_dataset_tool(
            "stargan", input_zip, output_dir, STARGAN_PYTHON, STARGAN_DIR,
            "stargan_dataset_infer.py",
            ["--attribute", attribute]
        )
        return output_zip, {"attack": "stargan", "attribute": attribute, "output_zip": output_zip}

    def hisd_dataset(self, input_zip, output_dir, tag="Eyeglasses", attribute="with", seed=42):
        """
        Edição de atributos com HiSD (CVPR 2021, CelebA-HQ 256), carregando o
        modelo uma vez. Estilo gerado de ruído com seed fixa por imagem; altera
        só a tag pedida (óculos, franja ou cor do cabelo). Saída em PNG.
        """
        output_zip = self._run_dataset_tool(
            "hisd", input_zip, output_dir, HISD_PYTHON, HISD_DIR,
            "hisd_dataset_infer.py",
            ["--tag", tag, "--attribute", attribute, "--seed", str(seed)]
        )
        return output_zip, {"attack": "hisd", "tag": tag, "attribute": attribute, "output_zip": output_zip}

    def liveportrait_dataset(self, input_zip, output_dir, pairs_csv, relative_motion=True):
        """
        Reencenação facial com LivePortrait (guiado por imagem), carregando os
        modelos uma vez. A imagem motora de cada alvo é a mesma origem do
        pairs_csv usada nos face swaps: o alvo recebe a pose e a expressão dela,
        e o rosto é colado de volta na foto original. Saída em PNG.
        """
        output_zip = self._run_dataset_tool(
            "liveportrait", input_zip, output_dir, LIVEPORTRAIT_PYTHON, LIVEPORTRAIT_DIR,
            "liveportrait_dataset_infer.py",
            ["--pairs_csv", os.path.abspath(pairs_csv), "--relative_motion", str(relative_motion).lower()]
        )
        return output_zip, {"attack": "liveportrait", "pairs_csv": pairs_csv, "output_zip": output_zip}

    def faceswap(self, image):
        # lógica do FaceSwap
        pass

    def outro_modelo(self, image):
        # lógica de outro modelo
        pass