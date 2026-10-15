# Kinveil 模型清单

本文档记录 Kinveil 当前代码会使用或已经预留支持的模型。它是一份模型清单，**不代表这些模型已经全部安装**。

## 下载方式总览

| 模型 | 用途 | 获取方式 | 模型文件约占用 |
| --- | --- | --- | ---: |
| InsightFace `buffalo_l` | 人脸检测与人脸向量 | 首次运行人脸功能时自动下载 | 约 326 MB |
| 3D-Speaker CAMPPlus | 声纹向量与说话人识别 | `kinveil voices setup-models` | 28.3 MB |
| Silero VAD | 判断何时有人说话 | `kinveil voices setup-models` | 约 0.6 MB |
| Pyannote Segmentation 3.0 | 重叠语音检测 | `kinveil voices setup-models` | 约 5.7 MB |
| SpeechBrain SepFormer WHAM | 双人重叠语音分离 | 单独命令下载，可选 | 约 113 MB |
| Paraformer Chinese/English INT8 | 默认中文/英文语音转文字 | `kinveil voices setup-models` | 约 217 MB |
| Whisper Small Multilingual INT8 | 多语言后备语音转文字 | 通过已有路径指定 | 约 375 MB |
| Kokoro Multi-Lang v1.0 | 本地文字转语音 | 当前需要手动下载，可选 | 约 350 MB |

全部可选模型都安装时，模型文件合计约 **1.2–1.4 GB**。这不包括 Python、ONNX Runtime、PyTorch、SpeechBrain 等运行库；启用 SepFormer 后，相关 Python 运行库还可能占用约 **0.5–1.5 GB**。

## 默认存放位置

Kinveil 自己管理的模型默认保存在：

```text
~/Library/Application Support/Kinveil/models/
├── voice/
│   ├── 3dspeaker-campplus-zh-cn-16k-common.onnx
│   ├── silero-vad.onnx
│   ├── pyannote-segmentation-3.0.onnx
│   ├── sepformer-wham/
│   └── sherpa-onnx-whisper-small/
└── tts/
    └── kokoro-multi-lang-v1_0/
```

设置 `KINVEIL_DATA_DIR` 或相关命令的 `--data-dir` 后，可以改变 Kinveil 数据目录。InsightFace 自行下载的人脸模型是例外，默认保存在：

```text
~/.insightface/models/buffalo_l/
```

模型和运行数据均不应提交到 Git 仓库。

## 1. InsightFace `buffalo_l`

- **状态：** 首次启动 `kinveil faces watch` 或 `kinveil start` 时，由 InsightFace 自动下载。
- **用途：** 检测摄像头画面中的人脸，并生成用于身份比对的人脸特征向量。
- **下载地址：** [InsightFace v0.7 `buffalo_l.zip`](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip)
- **默认目录：** `~/.insightface/models/buffalo_l/`
- **大小：** 解压后约 326 MB；压缩包大小会随发布版本有所不同。
- **Kinveil 实际使用：** 人脸检测和人脸识别是身份比对的核心；关键点用于嘴唇运动，人脸姿态可用于把正脸和侧脸向量分组保存。年龄和性别模型当前不参与 Kinveil 的身份流程。

常见文件包括：

| 文件 | 作用 |
| --- | --- |
| `det_10g.onnx` | SCRFD 人脸检测 |
| `w600k_r50.onnx` | ResNet-50 人脸识别与特征向量生成 |
| `1k3d68.onnx` | 3D 人脸关键点 |
| `2d106det.onnx` | 2D 人脸关键点 |
| `genderage.onnx` | 年龄与性别估计；Kinveil 当前不使用 |

> **许可证注意：** InsightFace 源代码采用 MIT License，但官方预训练模型权重通常仅允许非商业研究使用。公开仓库不应重新分发这些权重；商业使用前必须向 InsightFace 确认并取得相应授权。参见 [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)。

## 2. 3D-Speaker CAMPPlus

- **状态：** 可由 Kinveil 自动下载并校验 SHA-256。
- **用途：** 从一段有效语音中生成声纹向量，用来注册和识别说话人。
- **输入：** 16 kHz 单声道音频。
- **文件名：** `3dspeaker-campplus-zh-cn-16k-common.onnx`
- **大小：** 约 28.3 MB。
- **SHA-256：** `f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11`
- **来源：** [3D-Speaker](https://github.com/modelscope/3D-Speaker)
- **下载文件：** [Sherpa-ONNX CAMPPlus](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx)
- **许可证：** Apache-2.0。

## 3. Silero VAD

- **状态：** 可由 Kinveil 自动下载并校验 SHA-256。
- **用途：** Voice Activity Detection，只负责判断音频中何时存在语音、何时开始和结束，不负责识别说话内容或身份。
- **文件名：** `silero-vad.onnx`
- **大小：** 约 629 KB。
- **SHA-256：** `9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6`
- **来源：** [Silero VAD](https://github.com/snakers4/silero-vad)
- **下载文件：** [Sherpa-ONNX Silero VAD](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx)
- **许可证：** MIT。

## 4. Pyannote Segmentation 3.0

- **状态：** 可由 Kinveil 自动下载、解压并校验 SHA-256。
- **用途：** 说话人分段和重叠语音检测。Kinveil 用它判断当前是否可能有两个人同时说话，以避免把混合语音强行绑定给某个人。
- **安装后的文件名：** `pyannote-segmentation-3.0.onnx`
- **模型大小：** 约 5.7 MB；下载压缩包约 7 MB。
- **模型 SHA-256：** `220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079`
- **压缩包 SHA-256：** `24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488`
- **来源：** [Pyannote Segmentation 3.0](https://huggingface.co/pyannote/segmentation-3.0)
- **下载文件：** [Sherpa-ONNX Pyannote Segmentation 3.0](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2)
- **许可证：** MIT。

一次安装上述三个默认音频模型和 Paraformer 中文/英文转写模型：

```bash
kinveil voices setup-models --accept-model-licenses
```

命令会下载 CAMPPlus、Silero VAD、Pyannote Segmentation 3.0 和 Paraformer INT8，并通过固定的 SHA-256 检查下载及模型文件是否完整。

## 5. SpeechBrain SepFormer WHAM（可选）

- **状态：** 可用单独命令下载；需要先安装 `separation` 依赖。
- **用途：** 尝试把两个人同时说话的混合音频分离成两个音轨，分离后再分别提取声纹和转写。
- **模型仓库：** [`speechbrain/sepformer-wham`](https://huggingface.co/speechbrain/sepformer-wham)
- **Kinveil 固定版本：** `3f70635a1a640713e5af867b35de7929834bdeda`
- **默认目录：** `~/Library/Application Support/Kinveil/models/voice/sepformer-wham/`
- **运行所需模型：** 约 113 MB，主要是 `masknet.ckpt`。
- **模型仓库总量：** 约 319 MB，其中包括推理不需要的训练优化器状态。
- **许可证：** Apache-2.0。
- **限制：** 模型以 8 kHz 的 WHAM 英语混合语音训练；对中文和真实房间环境可能仍有作用，但分离质量需要实测，不能保证每次都能正确分开。

安装和下载：

```bash
python -m pip install -e '.[separation]'
kinveil voices setup-separation-model --accept-model-license
```

Hugging Face 缓存中可能会保留另一份下载文件，因此磁盘实际增加量可能高于模型目录本身。

## 6. Paraformer Chinese/English INT8（默认）

- **状态：** 由 `kinveil voices setup-models --accept-model-licenses` 自动下载。
- **用途：** 针对普通话和英语的本地语音转文字，作为 `kinveil start --debug` 的默认识别器。
- **模型来源：** [Sherpa-ONNX Paraformer zh 2024-03-09](https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2024-03-09)
- **默认目录：** `~/Library/Application Support/Kinveil/models/voice/sherpa-onnx-paraformer-zh-2024-03-09/`
- **运行文件：** `model.int8.onnx`（约 217 MB）和 `tokens.txt`（约 74 KB）。
- **许可证：** Apache-2.0（FunASR Paraformer）。

```bash
kinveil start --debug --paraformer-model-dir /path/to/sherpa-onnx-paraformer-zh-2024-03-09
```

## 7. Whisper Small Multilingual INT8（多语言后备）

- **状态：** 不再默认下载；已有模型仍可通过 `--whisper-model-dir` 显式使用。
- **用途：** 完全在本地把语音转换成文字。`kinveil start --debug` 会把识别到的文字输出到控制台。
- **模型来源：** [Sherpa-ONNX Whisper Small](https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small)
- **默认目录：** `~/Library/Application Support/Kinveil/models/voice/sherpa-onnx-whisper-small/`
- **许可证：** Whisper 模型为 MIT；使用和分发时也应保留下载包内的第三方声明。

Kinveil 运行时至少需要以下文件：

| 文件 | 约占用 |
| --- | ---: |
| `small-encoder.int8.onnx` | 112.4 MB |
| `small-decoder.int8.onnx` | 262.2 MB |
| `small-tokens.txt` | 0.8 MB |
| **合计** | **约 375 MB** |

Kinveil 直接下载并校验上面三个运行文件，不下载额外的 FP32 模型。

指定模型目录启动：

```bash
kinveil start --debug --whisper-model-dir /path/to/sherpa-onnx-whisper-small
```

普通的 `kinveil start` 不会为了显示界面而自动运行语音转文字；`--debug` 才会把转写结果打印到控制台。

### 其他多语言版本

Kinveil 默认使用 Small。加载器也兼容 Sherpa-ONNX 导出的 Base 多语言
INT8 模型，可通过 `--whisper-model-dir` 指定，但准确率通常低于 Small。

| 版本 | 参数量 | INT8 运行文件约占用 | 相对速度 | 建议用途 |
| --- | ---: | ---: | ---: | --- |
| Base | 74M | 161 MB | 较快 | 资源受限设备 |
| Small | 244M | 375 MB | 较慢 | Kinveil 默认，更重视中文准确率 |

这里的速度比例来自 OpenAI 公布的跨型号相对速度，只能用于估算；实际
ONNX CPU 延迟需要在目标 Mac 上用相同录音测量。模型文件来源：
[Whisper Base](https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base) 和
[Whisper Small](https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small)。

## 8. Kokoro Multi-Lang v1.0（可选）

- **状态：** Kinveil 已预留本地 TTS 接口，但当前需要手动下载并解压。
- **用途：** 把 Kinveil 生成的文字回复转换成语音 WAV 文件，支持中文、英文和混合文本。
- **下载压缩包：** [kokoro-multi-lang-v1_0.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2)
- **上游模型：** [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- **默认目录：** `~/Library/Application Support/Kinveil/models/tts/kokoro-multi-lang-v1_0/`
- **解压后大小：** 约 350 MB。
- **许可证：** Apache-2.0；同时应保留模型包中的数据来源和第三方许可证声明。

主要文件：

| 文件或目录 | 作用 | 约占用 |
| --- | --- | ---: |
| `model.onnx` | Kokoro 声学与合成模型 | 310 MB |
| `voices.bin` | 说话人风格数据 | 26 MB |
| `lexicon-*.txt` | 中英文发音词典 | 约 14 MB |
| `tokens.txt`、`espeak-ng-data/` 等 | 音素和文本规范化资源 | 少量 |

该模型包含 53 种声音风格。中文女声 ID 为 45–48，中文男声 ID 为 49–52；Kinveil 当前默认使用 `voice-id 48`。

示例：

```bash
kinveil tts synthesize "你好，我是 Kinveil。" \
  --model-dir "$HOME/Library/Application Support/Kinveil/models/tts/kokoro-multi-lang-v1_0" \
  --voice-id 48 \
  --output reply.wav
```

## 隐私与版本管理

- 模型文件本身不包含 Kinveil 保存的人脸或声纹身份数据。
- 用户的人脸向量、声纹向量和身份数据库位于 Kinveil 数据目录，而不是模型目录。
- 模型、数据库、录音、截图和生成音频都不应提交到公开仓库。
- 下载预训练权重不等于自动取得商业使用权；发布产品前应逐项复核模型许可证及训练数据限制。
- 自动下载的模型使用固定下载地址和 SHA-256，或固定模型版本；Kokoro 等手动模型目前需要使用者自行记录下载版本。
