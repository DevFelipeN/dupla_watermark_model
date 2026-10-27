# Marca dupla para rostos 256×256: robusta + semi-frágil (versão 4)

Duas marcas d'água invisíveis de **30 bits** na mesma imagem de rosto (CelebA-HQ, 256×256), lidas sem precisar da imagem original:

| | Marca robusta (versão 3, congelada) | Marca semi-frágil (nova) |
|---|---|---|
| Onde fica | imagem toda | **só no rosto** (elipse suave) |
| Ataques comuns (JPEG, blur, ruído, resize…) | sobrevive | **sobrevive** |
| Deepfake | sobrevive → diz a **origem** | **quebra** → diz que a imagem foi **alterada** |

A robusta é a da versão 3 ([robust-watermark-model](https://github.com/DevFelipeN/robust-watermark-model)), sem nenhuma mudança. Este projeto acrescenta a semi-frágil, treinada por cima dela.

## Como funciona

**Marcar:** a imagem recebe a robusta v3 e depois a semi-frágil. A semi-frágil usa a mesma rede de marcação da v3 (Y → DWT Haar → DCT 8×8 → CNN → máscara mid-band → IDCT → IDWT), mas a diferença que ela causa é multiplicada por uma **elipse suave do rosto**, então o fundo e o cabelo só têm a robusta. A mensagem da semi-frágil não é livre: é a mensagem robusta **XOR uma chave** fixa (`marca_fragil.chave_fragil`).

**Verificar** (`avaliar_marca_dupla.py`):

1. lê a robusta → mensagem de origem (30 bits);
2. calcula a semi-frágil esperada = robusta lida XOR chave;
3. lê a semi-frágil e mede a **concordância** (fração de bits iguais à esperada);
4. concordância abaixo do limiar (padrão 0,8) → imagem **alterada**.

Cada marca tem o seu extrator (mesma arquitetura U-Net da v3, pesos diferentes).

![Arquitetura da marca dupla](my_tests/arquitetura/arquitetura_marca_dupla.png)

Figura gerada por `gerar_figura_marca_dupla.py` (PNG e PDF em `my_tests/arquitetura/`).

### Treino da semi-frágil (`train_fragil.py`)

A robusta fica congelada. A semi-frágil começa dos pesos da robusta v3 e, a cada batch:

1. a imagem recebe a robusta e depois a semi-frágil (alpha fixo, 5);
2. **ataque comum** (`--benign_attack_set benigno_realista`: nenhum, JPEG simulado e real com qualidade 50–95, blur, median 3, ruído leve, resize 0,5–0,9 e resize + JPEG) → o extrator da semi-frágil deve **ler** a mensagem (BCE);
3. **manipulação do rosto** (`marca_fragil.MANIPULACOES`, sorteada) → o extrator deve **não ler**: o alvo da BCE é 0,5 em todos os bits, ou seja, bits no acaso. Em metade das vezes um ataque comum vem depois, porque deepfakes reais também saem recomprimidos:
   - `troca_rosto`: cola o rosto de outra imagem (sem marca), com a cor ajustada e a borda suave;
   - `regenera_rosto`: o rosto perde os detalhes finos e ganha ruído novo, como a saída de um gerador;
   - `deforma_rosto`: campo de deformação suave no rosto, como uma reencenação;
   - `mistura_rosto`: mistura parcial de dois rostos;
   - `troca_miolo` (sutil): troca só olhos, nariz e boca, com borda bem suave;
   - `troca_detalhes` (sutil): mantém cor e forma do rosto e troca só a textura fina (onde a marca fica);
   - `edita_regiao` (sutil): troca uma região pequena (olhos, boca ou nariz);
4. **imagem só com a robusta** → o extrator da semi-frágil também deve não ler, para não aprender a "ler" a robusta, que sobrevive aos deepfakes;
5. **a robusta continua legível** com a semi-frágil por cima (BCE do extrator robusto congelado);
6. **qualidade**: MSE + 0,005 × LPIPS entre a imagem com as duas marcas e a só com a robusta. Só entra depois que a semi-frágil lê 85% nos ataques comuns e cresce até o peso máximo em 3.000 passos.

**Nenhum dos 7 deepfakes reais entra no treino.** Todos ficam para a avaliação, o que mostra se a detecção generaliza.

No fim de cada época, uma validação rápida (200 imagens do `val`) mostra a acurácia da semi-frágil e da robusta sem ataque, com JPEG 50, resize 0,5, blur e após cada manipulação, além do PSNR, e salva a melhor época em `<nome>_melhor.pth`.

## Modelo atual

| Item | Valor |
|---|---|
| Robusta | `output/checkpoints/v3_robusta_final.pth` (12 MB, o mesmo `v3_final.pth` da versão 3) |
| Semi-frágil | `output/checkpoints/semifragil_final.pth` (12 MB, a melhor época do treino) |
| Resolução / marcas | 256×256 / 30 bits cada |
| Alpha | 5 nas duas (salvo em cada checkpoint) |
| Treino da semi-frágil | 20 épocas a partir dos pesos da robusta, batch 16, LR em cosseno 5e-5 → 1e-6 (≈1h25 numa RTX 3090) |
| Dataset | CelebA-HQ 256×256: 28.000 imagens de treino e 2.000 de validação |

> Os dois checkpoints estão incluídos no repositório e são os padrões de todos os scripts. Eles só carregam com o `watermark_embedding.py` daqui (a máscara DCT é o buffer `dct_band_mask`).

## Instalação

### 1. Ambiente

```bash
conda env create -f environment.yml
conda activate watermark-v3
```

O mesmo ambiente da versão 3 (Python 3.12, PyTorch 2.11 com CUDA 12.8, `pytorch_wavelets`, `torch-dct`, `kornia`, `lpips`, `DISTS-pytorch`, `torchmetrics`, …). Para outra versão de CUDA, troque o índice do PyTorch no arquivo. Na primeira execução, o LPIPS e o DISTS baixam os próprios pesos.

### 2. Dataset

CelebA-HQ em 256×256, separado por gênero na validação (a avaliação usa metade de cada gênero):

```text
Dataset/
└── celeba_hq/
    ├── train/
    │   ├── female/   (imagens .jpg)
    │   └── male/
    └── val/
        ├── female/
        └── male/
```

Crie a pasta `Dataset/` na raiz do projeto (ela não está no repositório: a licença do CelebA-HQ não permite redistribuir as imagens). Uma fonte é o Kaggle (`kaggle datasets download -d lamsimon/celebahq`), que já vem em `train/` e `val/` com `female/` e `male/`. As imagens são redimensionadas para 256×256 pelos scripts. `Dataset/` também pode ser um link simbólico para um dataset que já esteja em outro lugar (`ln -s /caminho/do/dataset Dataset`).

### 3. Modelos treinados

Já vêm no repositório, em `output/checkpoints/` (veja [Modelo atual](#modelo-atual)). Para treinar a semi-frágil de novo, veja [Treino](#treino).

## Uso

Todos os comandos rodam de dentro desta pasta, com o ambiente ativado.

### Avaliar a marca dupla (métricas principais)

```bash
python avaliar_marca_dupla.py --num_images 2000 --sem_deepfakes --output results/marca_dupla_2000
```

- Marca as imagens de validação com as duas marcas, aplica os **16 ataques** do `evaluate.py` da versão 3 e lê as duas marcas de cada imagem atacada.
- `--num_images N`: usa N imagens (metade de cada gênero; padrão 200).
- `--limiar`: concordância abaixo da qual a imagem é considerada alterada (padrão 0,8).
- `--fragil`, `--robusta`: outros checkpoints (padrão: os do modelo atual).
- Use um `--output` diferente a cada execução: a pasta é sobrescrita.

Saídas em `--output`:

- `media_per_attack.csv`: a mesma tabela do `evaluate.py` (robusta: BER, Bit Accuracy, Accuracy; PSNR, SSIM, LPIPS e DISTS entre a original e a atacada) e mais três colunas da semi-frágil:
  - `Semi-frágil Acc (%)`: fração de bits certos da semi-frágil;
  - `Concordância (%)`: semi-frágil lida × esperada (robusta lida XOR chave);
  - `Alterada (%)`: imagens com concordância abaixo do limiar (falso positivo nos ataques comuns, detecção nos deepfakes).

  A última linha (`sem_semifragil`) é um controle: a imagem só com a robusta, em que a semi-frágil deve dar ≈50%.
- `results_per_image.csv`: uma linha por imagem e ataque.
- `qualidade.csv` e `qualidade_por_imagem.csv`: original × dupla e original × só robusta.
- `exemplo.png`: original | só robusta | dupla | diferença da semi-frágil ×20.

O terminal mostra também a taxa de falsos positivos nos ataques comuns, a taxa de detecção nos deepfakes, a origem recuperada nos deepfakes e a AUC.

### Avaliar com deepfakes (opcional)

```bash
python avaliar_marca_dupla.py --num_images 200 --output results/marca_dupla_deepfakes_200
```

Sem `--sem_deepfakes`, roda também os 7 deepfakes: DiffAE (atributo `Young`), BlendFace, SimSwap, InSwapper, StarGAN (`Male`), HiSD (óculos) e LivePortrait (`--deepfakes inswapper,simswap` escolhe quais). Cada imagem recebe uma identidade de origem fixa do mesmo gênero (`pares_origem.csv`). Exige os modelos de deepfake e um ambiente conda para cada um, que **não estão neste repositório**. Os caminhos ficam no início de `deepfake_attack.py` e podem ser trocados pelas variáveis de ambiente `DEEPFAKE_ROOT` (pasta com os modelos: `BlendFace/`, `SimSwap/`, `inswapper/`, `stargan/`, `HiSD/`, `LivePortrait/`, `diffae256/` e `origens_faceswap/`) e `CONDA_ENVS` (pasta dos ambientes conda, um por modelo: `blendeface`, `simswap`, `inswapper`, `stargan`, `liveportrait`, `diffae`). O DiffAE leva ≈11 s por imagem.

### Marcar uma imagem e ver a diferença

```bash
python collect_image_by_diff.py --models_path output/checkpoints/v3_robusta_final.pth \
    --fragil output/checkpoints/semifragil_final.pth \
    --dataset_dir Dataset/celeba_hq/val --output_dir my_tests/exemplo \
    --reference_name 003251 --reference_file 003251.jpg --image_resolution 256

# diferença da semi-frágil (só robusta × dupla)
python watermark_diff.py --original my_tests/exemplo/003251_so_robusta.png \
    --watermarked my_tests/exemplo/003251_marcada.png --output_dir my_tests/exemplo/diff_semifragil

# diferença das duas marcas (original × dupla)
python watermark_diff.py --original my_tests/exemplo/003251_original.png \
    --watermarked my_tests/exemplo/003251_marcada.png --output_dir my_tests/exemplo/diff_dupla
```

O primeiro salva a original, a imagem só com a robusta e a imagem com as duas marcas; o `watermark_diff.py` gera a diferença amplificada (`diferenca.png`), um grid comparativo e imprime o PSNR. Troque `--reference_file` para usar outra imagem do dataset.

### Treino

```bash
python train_fragil.py --dataset_path Dataset/celeba_hq/train \
    --output output_fragil/fragil_alpha5 --num_epochs 20
```

Opções principais:

| Opção | Padrão | O que faz |
|---|---|---|
| `--robust_checkpoint` | `output/checkpoints/v3_robusta_final.pth` | marca robusta (congelada) |
| `--alpha` | 5 | força da semi-frágil (fixa) |
| `--mask_floor` | 0 | fração da semi-frágil fora do rosto (0 = só no rosto) |
| `--fragility_weight` | 1,0 | peso de "não ler" nas manipulações |
| `--no_mark_weight` | 0,5 | peso de "não ler" sem a marca semi-frágil |
| `--robust_weight` | 1,0 | peso que mantém a robusta legível |
| `--benign_attack_set` | `benigno_realista` | ataques que a semi-frágil deve aguentar (`benigno` = com dropout e JPEG até 30; `v3` = os mesmos da robusta) |
| `--manipulacoes` | as 7 | manipulações de treino, separadas por vírgula |
| `--init_from` | `robusta` | `zero` = começa com pesos aleatórios |
| `--resume` | — | continua o treino de um checkpoint da semi-frágil |
| `--limit` | 0 | usa só N imagens (teste rápido) |

- O treino salva em `<output>/checkpoints/` o `*_melhor.pth` (melhor época, o usado no modelo atual) e o `*_final.pth`, e os logs do TensorBoard em `<output>/logs/` (`tensorboard --logdir <output>/logs`). Checkpoints e logs não vão para o repositório (`.gitignore`), exceto os dois de `output/checkpoints/`.
- Use um `--output` diferente para cada treino.

## Resultados do modelo atual

`avaliar_marca_dupla.py`, alpha 5 nas duas marcas, limiar 0,8, seed 42.

### Qualidade (2.000 imagens de validação)

| | PSNR | SSIM | LPIPS | DISTS |
|---|---|---|---|---|
| Original × só robusta | 47,69 dB | 0,991 | 0,0034 | 0,031 |
| Original × robusta + semi-frágil | 46,02 dB | 0,988 | 0,0053 | 0,036 |

A semi-frágil custa ≈1,7 dB de PSNR, concentrados no rosto.

### Ataques comuns (2.000 imagens)

"Robusta" é a Bit Accuracy da marca de origem; "Concordância" é a semi-frágil lida × esperada; "Alterada" é a fração de imagens que a verificação marcaria como alteradas (aqui, **falso positivo**).

| Ataque | Parâmetro | Robusta | Concordância | Alterada |
|---|---|---|---|---|
| Nenhum | – | 100% | 99,96% | 0% |
| JPEG | Q=75 | 99,97% | 98,45% | 0% |
| JPEG | Q=50 | 99,55% | 93,29% | 1,25% |
| JPEG | Q=30 | 96,17% | 80,03% | 39,3% |
| Blur | σ=2 | 99,90% | 97,02% | 0,65% |
| Median | 3×3 | 99,92% | 97,62% | 0,3% |
| Median | 5×5 | 99,71% | 87,52% | 12,2% |
| Ruído | σ=0,02 | 99,997% | 99,80% | 0% |
| Ruído | σ=0,05 | 99,90% | 95,51% | 0% |
| Salt & pepper | 1% | 99,995% | 96,65% | 0% |
| Salt & pepper | 5% | 99,22% | 77,35% | 55,3% |
| Dropout | 5% | 99,89% | 77,95% | 49,4% |
| Dropout | 10% | 99,48% | 71,23% | 74,8% |
| Dropout | 20% | 97,28% | 64,76% | 89,5% |
| Resize | 0,5 | 99,69% | 90,48% | 8,8% |
| Resize | 0,3 | 98,64% | 82,18% | 31,5% |
| Controle: só a robusta | – | 100% | 50,53% | 99,9% |

- A robusta continua acima de 96% de Bit Accuracy em todos os ataques, com a semi-frágil por cima.
- Nas edições comuns e leves (JPEG 75 e 50, blur, median 3, ruído, salt & pepper 1%), a semi-frágil quase nunca acusa alteração (≤1,25%).
- Ataques que destroem a imagem (JPEG 30, salt & pepper 5%, dropout, resize 0,3) também quebram a semi-frágil: eles ficam fora do conjunto benigno do treino (veja [Limitações](#limitações)).
- No controle, sem a semi-frágil, a concordância fica em ≈50% (acaso) e a imagem é acusada como alterada: o extrator da semi-frágil não lê a robusta.

### Deepfakes (200 imagens, 100 femininas e 100 masculinas)

"Alterada" aqui é a **taxa de detecção**.

| Deepfake | Robusta (origem) | Concordância | Alterada |
|---|---|---|---|
| SimSwap | 99,90% | 51,08% | 99,5% |
| InSwapper | 99,60% | 50,60% | 99,5% |
| LivePortrait | 83,05% | 53,95% | 96,5% |
| BlendFace | 100% | 82,47% | 37,5% |
| DiffAE (Young) | 99,97% | 91,67% | 15,5% |
| HiSD (óculos) | 99,37% | 89,95% | 3,0% |
| StarGAN (Male) | 99,82% | 94,25% | 2,0% |

- **Troca de rosto completa** (SimSwap, InSwapper) e **reencenação** (LivePortrait) são detectadas em quase todas as imagens, e a robusta continua dizendo a origem.
- **BlendFace** (troca só do miolo do rosto) é detectado em parte.
- **Edições de atributo** (DiffAE, HiSD, StarGAN) preservam a maior parte da textura do rosto e quase não quebram a semi-frágil.

### Aparência das marcas

![Original, duas marcas e diferenças](my_tests/marca_dupla_alpha5_003251/comparacao_marca_dupla.png)

*Imagem 003251: original, com as duas marcas, e as diferenças da robusta (×15), da semi-frágil (×30, concentrada no rosto) e total (×15); embaixo, recorte ampliado.*

### Limitações

- **Fronteira benigno × malicioso:** JPEG 30, dropout, salt & pepper 5% e resize 0,3 derrubam a semi-frágil junto com os deepfakes. O limiar e o conjunto de ataques considerados benignos precisam ser definidos e justificados.
- **Edição de atributos** (DiffAE, HiSD, StarGAN) é pouco detectada.
- **Máscara fixa:** a elipse serve para rostos alinhados e centralizados (CelebA-HQ). Para fotos não alinhadas, seria preciso um detector de rosto.
- **LivePortrait** continua sendo o caso mais difícil para a robusta (83% de Bit Accuracy).

## Arquivos principais

| Arquivo | Função |
|---|---|
| `marca_fragil.py` | elipse do rosto, inserção da semi-frágil, manipulações de treino e chave |
| `train_fragil.py` | treino da semi-frágil com a robusta congelada |
| `avaliar_marca_dupla.py` | avaliação da marca dupla (origem + detecção) com 16 ataques e, opcionalmente, 7 deepfakes |
| `attacks.py` | ataques da versão 3 + os conjuntos `benigno` e `benigno_realista` |
| `watermark_embedding.py`, `watermark_extraction.py` | redes de marcação e extração (as mesmas da versão 3) |
| `deepfake_attack.py` | chamada dos modelos de deepfake externos |
| `collect_image_by_diff.py`, `watermark_diff.py` | marcar uma imagem (com `--fragil`, as duas marcas) e visualizar a diferença |
| `gerar_figura_marca_dupla.py`, `gerar_figura_arquitetura.py` | figuras da arquitetura (`my_tests/arquitetura/`) |
| `train.py` | treino da robusta (o mesmo da versão 3) |
| `output/checkpoints/v3_robusta_final.pth`, `output/checkpoints/semifragil_final.pth` | checkpoints do modelo atual |
| `utils.py` | DWT/DCT em blocos e funções auxiliares |
| `environment.yml` | ambiente conda |
