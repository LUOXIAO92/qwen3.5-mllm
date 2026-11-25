# 在 Google Colab 上训练 projector（任务 C1）

本目录让 `pipeline/train/train.py` 能在 Google Colab 上训练，并且能接着本机已经开始的训练续训。难点在训练音频：v2b 的训练清单引用约 480 万个 FLAC 文件，共 614 GB。Colab 的本地磁盘只有根目录 `/` 可写（236 GB），`/kaggle/input` 是只读的；Google Drive 容量够，但几百万个小文件的上传、列目录、打开和随机读取都很慢。

做法是把训练音频打包成约 1 GB 一个的 tar 文件（分片）放在 Drive 上。FLAC 文件的字节原样写进 tar，不重新编码、不截断，读出的字节与原文件逐字节相同。分片按训练计划排列：训练计划只由训练配置决定，每一步读哪些文件可以提前算出。这一轮中读取不止一次的文件（被加权多抽的无人声片段、虚拟切分后几段共用的整段录音，约 48 GB）放进常驻分片，每次开始训练前整体复制到本地；其余只读一次的文件按读取它的步数排进顺序分片。训练期间后台的窗口管理程序按训练进度，把接下来要用的顺序分片从 Drive 整个顺序复制到本地，用过的删掉，本地只保留一段滚动窗口。训练程序按索引中记录的位置一次读出一个文件的字节，再交给 torchcodec 解码，结果与直接解码原文件相同。分片格式的细节见 `pipeline/train/shards.py`。

分片也可以按路径排列（`--order path`），供随机读取的存储使用，例如用 gcsfuse 挂载的 Google Cloud Storage 存储桶，见第 5 节。

| 文件 | 用途 |
| --- | --- |
| `pack_shards.py` | 本机：按训练计划生成索引、分批打包分片、核对、列出分片文件名 |
| `bundle.py` | 本机：把背景声列表、dev 清单连同音频打成 tar，在 Colab 上解开后路径仍然有效 |
| `setup.sh`、`requirements-train.txt` | Colab：建立训练环境，放回训练清单、背景声、dev 清单和模型，修正模型目录的分词器文件（#77） |
| `shard_window.py` | Colab：开始训练前复制常驻分片和起点附近的顺序分片，训练期间在后台维护窗口 |
| `bench_storage.py` | Colab：比较 Drive、存储桶和本地磁盘的顺序复制与随机读取速度 |

下文的 `<工作目录>` 是本机的工作目录（环境变量 `WORK_DIR`），`$DATASET` 是本机的数据集目录。命令都在仓库根目录运行，本机用 `<工作目录>/envs/mllm-train/bin/python`。

## 1. 本机：生成索引和分片

### 1.1 索引

索引记录每个音频文件在哪个分片、从第几个字节开始、多长。生成索引只读清单、算训练计划、取文件大小，不写分片。训练配置直接取要续训的那次训练的 `train_config.json`，例如 4B：

```bash
python -m pipeline.colab.pack_shards index \
    --out-dir <分片目录>/v2b-e0 \
    --run-config <工作目录>/data/T62/v2b-4b/train_config.json
```

训练计划用训练程序自己的函数算出（读清单、选训练目标、抽样权重、按时长分批），所以与训练时一致；索引中记下这些配置和按读取顺序算出的摘要，`train.py` 开始时重新算一遍核对，不一致就不开始训练。v2b 的结果见 1.5 节。还没有开始的训练可以在 `--` 之后写出与 `run.sh` 相同的参数代替 `--run-config`。

索引写出后，各分片的内容就固定了。之后改动训练清单或训练配置，需要重新生成索引和全部分片。

### 1.2 速度测试用的索引

速度测试从第 30,000 步的检查点续训到第 40,000 步（第 3 节）。这段步数单独生成一个索引，与完整索引放在同一个目录：

```bash
python -m pipeline.colab.pack_shards index \
    --out-dir <分片目录>/v2b-e0 \
    --name index-30000-40000 \
    --steps 30000-40000 \
    --run-config <工作目录>/data/T62/v2b-4b/train_config.json
```

它的顺序分片与完整索引的同名分片逐字节相同，测试时上传的顺序分片以后正式训练可以直接用；常驻分片只含这段步数用到的文件，文件名带步数范围（`pool-30000-40000-*.tar`），与完整的常驻分片不冲突。

### 1.3 分批打包、核对、上传

本机磁盘放不下全部分片时分批生成：每批按编号范围生成约 100 个（约 100 GB），核对后上传，再删除本地副本。已经完整的分片（大小与索引一致）会跳过，中断后重新运行同一条命令即可。

```bash
# 速度测试用的分片：这段步数的常驻分片和顺序分片
python -m pipeline.colab.pack_shards pack --index <分片目录>/v2b-e0/index-30000-40000.json --pool --steps 30000-40000
# 完整的常驻分片，再按编号分批生成顺序分片
python -m pipeline.colab.pack_shards pack --index <分片目录>/v2b-e0/index.json --pool
python -m pipeline.colab.pack_shards pack --index <分片目录>/v2b-e0/index.json --shards 49-148
# 核对：tar 成员表与索引一致，随机抽查 500 个成员与原文件逐字节相同
python -m pipeline.colab.pack_shards verify --index <分片目录>/v2b-e0/index.json --samples 500
# 查看本机和 Drive 上各有哪些完整的分片
python -m pipeline.colab.pack_shards status --index <分片目录>/v2b-e0/index.json --dirs <分片目录>/v2b-e0
```

本机实测（v2b 中连续 4,000 个 Common Voice 文件，8 个读取线程，训练同时在读同一块磁盘）每秒约 180 MB、2,700 个文件，按这个速度打包全部分片约需 1 小时，主要时间花在上传。每个文件在 tar 中多占约 1 KB（头和补齐到 512 字节），v2b 全部分片比原文件多 0.6%。

索引文件（`index.json`、`index.tsv.gz`，速度测试的是 `index-30000-40000.*`）和全部分片上传到 Drive 上的同一个目录，下文写作 `$SHARDS_DRIVE`。分片是标准 tar，索引丢失时也能用 `tar -tvf` 列出、用 `tar -xf` 解开。

### 1.4 其他要上传的文件

```bash
# 训练清单：压缩后上传，Colab 上解压回与本机相同的路径
gzip -c $DATASET/audio/qwen3.5-asr/v2/train/manifest-v2b.jsonl > manifest-v2b.jsonl.gz
# 背景声：列表和它引用的音频（约 9 GB），路径相对 $DATASET
python -m pipeline.colab.bundle \
    --manifest $DATASET/audio/qwen3.5-asr/nonspeech/v2/background.jsonl \
    --root $DATASET \
    --output noise-v2.tar
# dev 清单和 FLEURS dev 音频（约 1.2 GB），路径相对工作目录；清单中的路径改写为相对清单所在目录
python -m pipeline.colab.bundle \
    --manifest <工作目录>/data/T10/fleurs_dev.jsonl \
    --root <工作目录> \
    --output fleurs-dev.tar
```

Drive 上的目录（`$DRIVE_DIR` 是 Drive 上本项目的目录）：

| 位置 | 内容 |
| --- | --- |
| `$DRIVE_DIR/upload/manifest-v2b.jsonl.gz` | 训练清单 |
| `$DRIVE_DIR/upload/noise-v2.tar` | 背景声 |
| `$DRIVE_DIR/upload/fleurs-dev.tar` | dev 清单与音频 |
| `$DRIVE_DIR/upload/qwen3.5-4b-mllm/` | 融合模型目录（仓库中没有，约 9.8 GB） |
| `$DRIVE_DIR/upload/qwen3.5-2b-mllm/` | 全量微调用的 2B 融合模型目录（约 5.7 GB），见单元格 7 的全量微调变体 |
| `$DRIVE_DIR/upload/v2b-2b-checkpoint-0005000/` | 全量微调的起点：本机 `<工作目录>/data/T61/v2b-2b/checkpoint-0005000/` 整个目录（`projector.safetensors` 与 `projector.json`，约 34 MB） |
| `$DRIVE_DIR/upload/replay-noloop-text.tar`、`replay-noloop-teacher.tar` | #71 C 组的自生成回放数据（文本约 1.4 GB、top-32 约 17 GB），见单元格 7 的 #71 C 组 |
| `$DRIVE_DIR/shards/v2b-e0/` | 索引和全部分片（`$SHARDS_DRIVE`） |
| `$DRIVE_DIR/runs/v2b-4b/` | 训练输出目录：本机训练的输出目录复制过来，不含 `dataset/` |
| `$DRIVE_DIR/wheels/` | 安装包缓存，第一次运行 `setup.sh` 时生成 |

要续训本机的训练，先在本机停下训练（训练程序收到 SIGTERM 时保存检查点后退出），再把它的输出目录上传到 `$DRIVE_DIR/runs/v2b-4b/`。`dataset/` 子目录是清单行的缓存（约 1.4 GB），每次开始训练时重新生成，不需要上传。

### 1.5 v2b 的数字

按 4B 训练的 `train_config.json`（每批 16 条、每批 360 秒、30 秒以内、种子 1234、空目标 3%）生成的索引：

| 项目 | 数值 |
| --- | --- |
| 清单行数 / 这一轮的读取次数 | 4,886,954 / 5,001,574 |
| 读到的文件 | 4,795,603 个，原文件共 613.64 GB，打包后 617.39 GB |
| 步数 | 第 0–317,482 步 |
| 常驻分片（编号 0–48） | 49 个，118,339 个文件，48.31 GB |
| 顺序分片（编号 49–618） | 570 个，4,677,264 个文件，569.07 GB；每个分片中位数服务 558 步，每步约 1.79 MB |
| 接下来 31,600 步的顺序分片 | 约 56.6 GB |
| 速度测试的第 30,000–40,000 步 | 顺序分片 `stream-00053.tar`–`stream-00071.tar`（编号 102–120，19 个，19.00 GB），常驻分片 `pool-30000-40000-*.tar`（4 个，9,748 个文件，3.69 GB），共 22.69 GB |
| 生成索引的耗时与内存 | 约 6 分钟，峰值 14 GB；索引文件 `index.tsv.gz` 约 100 MB |

读取不止一次的文件中，无人声片段 35,386 个、9.59 GB（按 3% 的占比被多抽，平均每个读约 4.2 次），虚拟切分后几段共用的整段录音 82,953 个、38.63 GB（同一段录音的几段在不同的步数读取）。

## 2. Colab 单元格

运行时选 GPU（RTX PRO 6000 Blackwell 96 GB 或其他），依次运行下面的单元格。每个新会话都从第 1 个单元格开始：本地磁盘在会话结束时清空，环境和本地分片都要重建。

**单元格 1：挂载 Google Drive。**

```python
from google.colab import drive
drive.mount("/content/drive")
```

**单元格 2：设置路径。** `DATASET` 必须与本机的 `$DATASET` 完全相同：训练清单和背景声列表要放回与本机相同的路径，`train.py` 续训时按路径字符串核对它们。Colab 以 root 运行，可以建立任意目录。

```python
import os

os.environ["DRIVE_DIR"] = "/content/drive/MyDrive/qwen3.5-asr"
os.environ["DATASET"] = "<与本机相同的 $DATASET 路径>"
os.environ["WORK_DIR"] = "/content/work"
os.environ["REPO"] = "/content/qwen3.5-asr"
os.environ["SHARDS_DRIVE"] = os.environ["DRIVE_DIR"] + "/shards/v2b-e0"
os.environ["SHARDS_LOCAL"] = "/content/shards"
os.environ["INDEX"] = os.environ["SHARDS_DRIVE"] + "/index.json"
os.environ["OUT"] = os.environ["DRIVE_DIR"] + "/runs/v2b-4b"
os.environ["PY"] = os.environ["WORK_DIR"] + "/envs/mllm-train/bin/python"
```

**单元格 3：取得代码。**

```bash
%%bash
git clone -b exp/v2 https://github.com/LUOXIAO92/qwen3.5-mllm "$REPO"
```

**单元格 4：建立环境，放回训练要读的小文件。** `setup.sh` 用 Colab 的 Python 3.13 在 `$WORK_DIR/envs/mllm-train` 建立虚拟环境（`run.sh` 固定使用这个位置），按 `requirements-train.txt` 安装与本机相同版本的包；第一次运行时把安装包缓存到 `$DRIVE_DIR/wheels/`，以后从缓存安装。缓存的标记 `wheels/.complete` 记着生成它的 `requirements-train.txt` 的 SHA-256，需求改了（例如 #66 加入的 torchao）就再运行一次 `pip wheel`，缓存中已有的安装包不再下载，只补缺少的；同一个会话中已经建好的虚拟环境也按新的需求补装。接着解压训练清单、解开背景声和 dev 清单、复制模型。每一步完成后留下标记，同一个会话中重复运行会跳过已完成的步骤。

最后一步检查仓库根目录下每个融合模型目录的分词器（#77），每次运行都做。修正之前上传到 Drive 的模型目录里，`tokenizer_config.json` 写的分词器类是 `Qwen2Tokenizer`，`tokenizer.json` 的预分词规则把泰语的元音与声调符号、阿拉伯语和波斯语的元音符号等组合用符号从字母上切下来，与原始 Qwen3.5 的切法不同。`setup.sh` 用 `python -m pipeline.tokenizer_fix <模型目录> --verify` 改正复制到本地的这两个文件（类名改为 `Qwen3_5Tokenizer`，预分词规则恢复原始写法；已经正确时不改动），再用 transformers 加载核对。Drive 上的文件保持原样，不需要重新上传。训练和评测加载模型时也做同样的检查，分词规则不对就停下并给出修正命令。修正之前开始的训练在修正之后续训，从续训的那一步起，泰语和带元音符号的阿拉伯语等的训练目标按原始规则分词。

```bash
%%bash
bash "$REPO/pipeline/colab/setup.sh"
```

**单元格 5：复制开始训练所需的分片。** 复制全部常驻分片，以及从续训起点（输出目录中最近的检查点）开始 2,000 步的顺序分片，复制完才结束。完整的常驻分片约 48 GB，Drive 的复制速度见窗口程序打出的 MB/s。

```bash
%%bash
cd "$REPO"
"$PY" -m pipeline.colab.shard_window prepare \
    --index "$INDEX" \
    --source "$SHARDS_DRIVE" \
    --pool-dir "$SHARDS_LOCAL/pool" \
    --window-dir "$SHARDS_LOCAL/window" \
    --output-dir "$OUT"
```

**单元格 6：在后台维护窗口。** 第一优先是常驻分片和接下来 `--ahead-steps` 步（默认 31,600 步，每步 16 条时约 50 万条样本、约 56 GB）的顺序分片；这些都在本地后，在 `--disk-budget-gb`（常驻与窗口合计）之内继续按步数顺序预取。第一优先还缺分片时不做第二优先的预取。顺序分片只在它的最后一步早于最近的检查点、也早于当前步数减 `--margin-steps`（默认 500）时删除，训练中途重启时从检查点续训要用的数据都还在。日志每次复制、删除各一行，每 5 分钟一行状态，其中「领先」是本地连续可用的数据比训练进度多出的步数。训练完成（`progress.json` 为 `finished`）后程序自己退出。

```bash
%%bash
cd "$REPO"
nohup "$PY" -m pipeline.colab.shard_window run \
    --index "$INDEX" \
    --source "$SHARDS_DRIVE" \
    --pool-dir "$SHARDS_LOCAL/pool" \
    --window-dir "$SHARDS_LOCAL/window" \
    --output-dir "$OUT" \
    --ahead-steps 31600 \
    --disk-budget-gb 150 \
    > "$WORK_DIR/shard_window.log" 2>&1 &
```

本地磁盘的分配大约是：模型 10 GB、虚拟环境约 8 GB、背景声 9 GB、训练清单 4 GB、清单行缓存 1.4 GB、dev 音频 1.2 GB，分片 150 GB（常驻 48 GB、第一优先约 56 GB、其余为预取），剩余约 50 GB。

**单元格 7：训练。** 设置与本机的 v2b 训练相同（`run.sh` 的默认值加 `<工作目录>/data/T53/train_v2b.sh` 中的 `--epochs 1 --warmup-steps 12000 --eval-every 5000 --save-every 2500` 和背景声），另加从分片读取音频的几项。这几项和 `--num-workers`、`--model-dir`、`--output-dir` 都不影响批次，不在续训核对的配置中，本机开始的训练可以加上它们续训。

```bash
%%bash
cd "$REPO"
pipeline/train/run.sh "$DATASET/audio/qwen3.5-asr/v2/train/manifest-v2b.jsonl" original "$OUT" \
    --model-dir "$REPO/qwen3.5-4b-mllm" \
    --epochs 1 \
    --warmup-steps 12000 \
    --eval-every 5000 \
    --save-every 2500 \
    --noise-manifest "$DATASET/audio/qwen3.5-asr/nonspeech/v2/background.jsonl" \
    --audio-shards "$INDEX" \
    --audio-shard-dirs "$SHARDS_LOCAL/window:$SHARDS_LOCAL/pool" \
    --audio-shard-fallback "$SHARDS_DRIVE" \
    --rows-cache-dir "$WORK_DIR/rows/v2b-4b" \
    --num-workers 16 \
    --stop-after-minutes 1260 \
    2>&1 | tee -a "$WORK_DIR/train.log"
```

- 峰值学习率用 `run.sh` 的默认值 3e-3。续训时峰值学习率取自检查点中学习率调度器保存的数值（`base_lrs`），命令行的 `--lr` 不起作用，与它不同时训练开始时打印一行提示；从 #63 的低学习率状态接着训（#64，或用它做速度测试）时仍按那一组的 1e-3 或 7e-4 继续。加上对应的 `--lr 1e-3` 或 `--lr 7e-4`，检查点中记录的训练配置才与实际一致。
- `--audio-shard-dirs` 是本地查找分片的目录，依次是窗口目录和常驻目录。本地没有某个分片时到 `--audio-shard-fallback`（Drive）读取，并在日志中记一行 `[shard-fallback]`；窗口程序跟得上时不应出现这样的行。
- `--rows-cache-dir` 把清单行的缓存放在本地磁盘。输出目录在 Drive 上，缓存放在那里会让读取进程通过 Drive 访问它。
- `--num-workers` 是数据读取进程数。机器有 48 个 vCPU，16 是起点，具体数值按速度测试中的 `audio_h_per_h` 调整。
- `--stop-after-minutes 1260`（21 小时）让训练在 Colab 会话的 24 小时上限之前保存检查点并停下，下一个会话从单元格 1 开始，`run.sh` 默认 `--resume`，从最近的检查点续训。会话意外断开时最多损失 `--save-every` 的 2,500 步。
- 96 GB 显存下 4B 每批 16 条、360 秒应当不需要梯度检查点，所以没有加本机用的 `--checkpoint-layers 24`。显存不够时训练程序会自己逐步打开梯度检查点并重试同一批，结果不变。

**单元格 7（全量微调变体，#66）。** 从 2B 只训练 projector 的 v2b 训练（#61）第 5,000 步的检查点出发，语言模型连同 projector 一起训练 30,000 步，音频塔和视觉塔冻结。数据与上面相同：同一份 v2b 训练清单和训练计划，从计划的第 0 步开始，直接用 `$SHARDS_DRIVE` 上的完整索引和分片（`--max-steps` 只决定学习率日程的长度，不改变训练计划，开始时的分片核对照常通过）。开始前做三件事：

1. 在 Drive 上放好 2B 的融合模型 `$DRIVE_DIR/upload/qwen3.5-2b-mllm/`，以及起点检查点 `$DRIVE_DIR/upload/v2b-2b-checkpoint-0005000/`：把本机 `<工作目录>/data/T61/v2b-2b/checkpoint-0005000/` 整个目录上传，里面只有 `projector.safetensors` 和 `projector.json`。
2. 单元格 2 中把输出目录改为新的目录，并让 `setup.sh` 复制 2B 模型：

   ```python
   os.environ["OUT"] = os.environ["DRIVE_DIR"] + "/runs/v2b-2b-full"
   os.environ["MODEL_NAME"] = "qwen3.5-2b-mllm"
   ```

3. 依次运行单元格 3–6（单元格 5、6 用同一个 `$OUT`；新的输出目录中没有检查点，从第 0 步开始准备分片），再运行下面的单元格代替单元格 7。

```bash
%%bash
cd "$REPO"
pipeline/train/run.sh "$DATASET/audio/qwen3.5-asr/v2/train/manifest-v2b.jsonl" original "$OUT" \
    --model-dir "$REPO/qwen3.5-2b-mllm" \
    --train-language-model \
    --language-model-lr 1e-5 \
    --lr 1e-4 \
    --warmup-steps 1000 \
    --max-steps 30000 \
    --eval-every 2500 \
    --save-every 2500 \
    --init-projector "$DRIVE_DIR/upload/v2b-2b-checkpoint-0005000" \
    --noise-manifest "$DATASET/audio/qwen3.5-asr/nonspeech/v2/background.jsonl" \
    --audio-shards "$INDEX" \
    --audio-shard-dirs "$SHARDS_LOCAL/window:$SHARDS_LOCAL/pool" \
    --audio-shard-fallback "$SHARDS_DRIVE" \
    --rows-cache-dir "$WORK_DIR/rows/v2b-2b-full" \
    --num-workers 16 \
    --stop-after-minutes 1260 \
    2>&1 | tee -a "$WORK_DIR/train.log"
```

- 语言模型参数组的峰值学习率 1e-5，projector 参数组 1e-4，两组都在前 1,000 步线性预热，再按余弦降到峰值的 0.1 倍。续训时两组的峰值学习率沿用检查点中保存的数值；`--train-language-model` 和 `--language-model-lr` 在续训时也要照样给出，所以重新运行同一个单元格即可。
- `--init-projector` 只在第一次开始时载入起点的 projector；续训时 projector 取自输出目录中最近的检查点，这一项不起作用。
- 精度：语言模型以 bfloat16 计算，默认由优化器更新 float32 主权重（其他选择见下文「全量微调的精度」）。2B 的语言模型约 18.8 亿个参数，主权重、它的梯度和 AdamW 状态比只训练 projector 多约 30 GB 显存；按 0.8B 的实测估计，每批 16 条、360 秒的峰值约 50–55 GB（`pipeline/train/__init__.py`「全量微调」），96 GB 显存应当不需要梯度检查点，显存不够时训练程序会逐步打开梯度检查点并重试同一批。
- 检查点：每 2,500 步多一个 `language_model.safetensors`（bfloat16，约 3.8 GB，评测用），12 个检查点共约 46 GB；最近检查点的 `trainer_state.pt` 含主权重和 AdamW 状态，约 22.6 GB，每次保存都整个重写，旧的随即删除。删掉的文件进入 Drive 的回收站，30,000 步下来回收站中约有 270 GB 旧的 `trainer_state.pt`，需要时手动清空。
- 这一段训练读第 0–29,999 步的数据：常驻分片 48 GB 加约 54 GB 顺序分片，单元格 6 的 `--ahead-steps 31600` 已经覆盖全程。
- 评测在本机进行（Colab 上只建立了训练环境）：把输出目录下载到本机（`trainer_state.pt` 不需要），再运行 `MODEL_DIR=<仓库>/qwen3.5-2b-mllm pipeline/eval/run.sh <下载的输出目录> <评测清单目录> <结果目录>`。它选 dev 损失最低的检查点，并自动载入其中的 `language_model.safetensors`。
- 纯续写格式（#71）：在训练单元格中加 `--prompt-format cpt`，样本变为音频标记后直接接转写和 `<|endoftext|>`（`pipeline/CONVENTIONS.md`「训练与评测的对话格式」），dev 损失也按这个格式计算。它不改变训练计划，现有的分片索引照常可用；它在续训核对的配置中，续训时要照样给出。`run.sh` 从训练配置读出这一项，评测时自动用同样的提示格式。

**全量微调的精度（#66）。** 语言模型参数组怎样更新由四个训练参数选择（`pipeline/train/precision.py`），默认值就是上面 2B 那次训练的做法：float32 主权重、float32 梯度、float32 AdamW 状态，每个语言模型参数 18 字节。4B 的语言模型约 42.1 亿个参数，这一项约 75.7 GB，所以 4B 全量微调的小规模测试（每批 16 条、360 秒）要对 16 层做梯度检查点，峰值 88 GB。省显存的组合（全部选项、显存表和 0.8B 上的比较见 `pipeline/train/__init__.py`「全量微调」）：

| 追加的参数 | 每参数字节 | 4B |
| --- | ---: | ---: |
| （默认） | 18 | 75.7 GB |
| `--language-model-master-weights none --language-model-grad-dtype bf16 --language-model-rounding stochastic` | 12 | 50.5 GB |
| 同上，再加 `--language-model-optimizer-states 8bit` | 6.03 | 25.4 GB |

- `none` 不另存 float32 主权重，优化器直接更新模型中的 bfloat16 参数；`stochastic` 是 torchao 的随机舍入，学习率 1e-5 时每步的更新多半小于 bfloat16 的舍入间距，就近舍入会把它们舍掉。0.8B 上同样的数据训练 2,000 步，这两种组合与默认设置的训练损失、dev 损失相差很小，与几种数值上近似等价的做法之间的差别相当（第 2,000 步的 dev 损失都在 1.266–1.275 之间）；就近舍入（不加 `--language-model-rounding stochastic`）时只有约四分之一的权重动过，`--language-model-optimizer-states bf16` 的轨迹也与默认设置不同，都不建议使用。
- 这几项在续训核对的配置中，一段训练开始后不能更换；以前开始的训练（配置中没有这几项）按默认值续训。
- 除默认设置外用 torchao 的优化器（`requirements-train.txt` 中的 torchao 0.18.0），它逐个张量调用编译过的更新函数：第一步多约 1 分钟编译，之后每步多出逐个张量调用的开销（单独测量优化器一步：torch 的 fused AdamW 在 0.8B 上 15.5 ms，torchao 55–80 ms，4B 的 8 bit 状态 104 ms；0.8B 全量微调从每秒 6.0 步降到 4.3–4.7 步）。8 bit 状态更新词嵌入兼输出层时另有临时显存，4B 约 5.1 GB。省下的显存用来少做或不做梯度检查点。
- 4B 的建议：在上面的全量微调单元格中把 `--model-dir` 换成 `$REPO/qwen3.5-4b-mllm`、`--init-projector` 换成 4B 的起点检查点，加上 `--language-model-master-weights none --language-model-grad-dtype bf16 --language-model-rounding stochastic --language-model-optimizer-states 8bit` 和 `--pack-sequences`，不给 `--checkpoint-layers`，让训练程序在显存不足时自己增加做梯度检查点的层。语言模型一侧常驻约 25 GB（默认 76 GB），加上不做检查点的中间结果（每批 360 秒约 48 GB，拼接后略少）和 9 GB 冻结权重，估计只需少量层或不需要梯度检查点；4B 的一步本来就长，torchao 多出的约 0.1 秒与省下的重算大致抵消。想少改一项时，去掉 `--language-model-optimizer-states 8bit`（fp32 状态，约 50 GB），0.8B 上两者与默认设置一样接近。
- 学习率：同一组 0.8B 测试中，默认精度下把语言模型峰值学习率从 1e-5 降到 2e-6，前 2,000 步的 dev 损失反而更低（1.140 对 1.269）。2B 用 1e-5 训练 3 万步效果很好，较长训练中哪个学习率更好还不知道；4B 是否降低学习率，可以先各跑一小段比较 dev 损失再决定。

**把语言模型参数放到主机内存（#66）。** 显存仍然不够时，三个训练参数把语言模型的一部分放到主机内存中（做法、支持的精度组合和本机测量见 `pipeline/train/__init__.py`「把语言模型参数放到主机内存」）：`--offload-optimizer`（梯度、主权重和 AdamW 状态放主机内存，显存中只留 bfloat16 权重）、`--lm-blocks-to-swap N`（末尾 N 层连权重也放主机内存，需要梯度检查点）、`--offload-activations`（梯度检查点留下的层输入放主机内存，反传时提前一层取回）。前两项要求 fp32 的 AdamW 状态和 `--grad-accum 1`。它们不在续训核对的配置中，同一段训练可以中途打开或关闭（例如 G4 上开始的训练在小显存的机器上加 `--offload-optimizer` 续训）。代价是 CPU 上的 AdamW：本机 16 核约每秒 14–15 亿个参数，主机内存每个放到主机的参数 16 字节（无主权重时 12 字节），另加训练进程本身约 20 GB。本机 RTX 5090 32 GB 上：2B 默认精度流式加载 12 层每秒 1.53 步（显存 22.6 GB）；4B 用 `--offload-optimizer --gradient-checkpointing --offload-activations` 加无主权重、随机舍入、fp32 状态，每秒 0.29 步，显存 11.8 GB，训练进程占主机内存 75 GB。G4 的 96 GB 显存放得下 4B 的默认精度（16 层梯度检查点，每秒 2.39 步），用这些选项只会更慢；它们的用处是在显存更小的机器上训练。开始前用 `free -g` 确认主机内存够用：4B 全部放到主机时，无主权重约需 75 GB，默认的 fp32 主权重约需 92 GB。

**单元格 7（#71 C 组：纯 CPT 全量微调加自生成回放）。** 第一轮的全部步骤由无人值守的驱动程序 `pipeline/colab/c_runs.py` 完成：冒烟测试（回放单元测试在 GPU 上跑一遍；C-mix 用 ntp 训练 100 步，第 1 步另加最长的回放样本看峰值显存，第 50 步停下再续训，核对续训后取到的回放样本与日志一致；C-mix 用 kl32 加尾部项训练 20 步）→ 用修正后的分词器（#77）从随机初始化重新训练 CPT 格式的 projector，取训练损失突降那一步的检查点（每 100 步保存）→ C0、C-free_raw、C-free_chat、C-cont_raw、C-cont_chat、C-mix 六个运行依次全量微调 3 万步并评测（FLEURS test、指令遵循探针、每字困惑度、回放差距、思考检查）→ 参考模型的思考检查、回放差距与困惑度 → 汇总。各阶段的设置、续跑和停下的规则见 `c_runs.py` 开头，回放本身见 `pipeline/train/replay.py`。

开始前在本机把回放数据打包成两个 tar（路径相对 `$DATASET`，不含 `prompt_chat`），上传到 `$DRIVE_DIR/upload/`：

```bash
cd $DATASET
tar -cf replay-noloop-text.tar --exclude='*/teacher' --exclude='text/qwen3.5-asr/replay-noloop/prompt_chat' text/qwen3.5-asr/replay-noloop
tar -cf replay-noloop-teacher.tar text/qwen3.5-asr/replay-noloop/*/*/teacher
```

`setup.sh` 看到这两个文件就解到 `$DATASET` 下与本机相同的位置（文本约 1.4 GB、top-32 约 17 GB，Drive 上约 35–50 MB/s 时复制约 6–8 分钟，本地盘多占约 19 GB）；没有时跳过，不影响其他训练。单元格 4 运行 `setup.sh` 时与全量微调变体一样让它复制 2B 模型（`MODEL_NAME=qwen3.5-2b-mllm`），之后准备 FLEURS test（`python -m pipeline.eval.fleurs_prepare`，放在 `$WORK_DIR/data/T02`，同 #71 第一轮）。单元格 5、6 不用运行（驱动程序自己为每个运行准备分片窗口，领先 1 万步、预算 80 GB），在后台启动驱动程序：

```python
import os
import subprocess

driver = subprocess.Popen(
    args = [os.environ["PY"], "-m", "pipeline.colab.c_runs"],
    cwd = os.environ["REPO"],
    start_new_session = True,
    stdout = open(file = os.environ["WORK_DIR"] + "/c71-driver.out", mode = "a"),
    stderr = subprocess.STDOUT,
)
```

- 开始时打印并在 `status.json` 中写下估计：冒烟测试与第一次复制常驻分片约 33 分钟，projector 约 12 分钟，C0 训练约 110 分钟、加回放的运行约 127 分钟（冒烟测试后按实测速度重估），每个运行的评测约 57 分钟（其中思考检查约 18 分钟），参考模型约 27 分钟；合计约 19 小时、约 190 个计算单元（G4 约 10 单元/小时，`C71_CU_PER_HOUR` 可改）。只看估计时加 `C71_ESTIMATE_ONLY=1`。
- 状态与日志：`$WORK_DIR/c71/status.json`、`driver.log`，随时复制到 `$DRIVE_DIR/runs/c71-status/`；每个训练的日志在 `$WORK_DIR/<运行名>.log`，回放的取样记录在输出目录的 `replay_log.jsonl`。运行名为 `v2b-2b-cpt77-<c0|c-free_raw|...>`，输出与评测结果在 Drive 的 `runs/<运行名>/`（评测在其中的 `eval/`），参考模型在 `runs/c71-ref/`，汇总在 `runs/c71-summary/`。
- Drive 上每个运行的检查点只有 projector、日志和 `projector.json`；最近检查点的 `trainer_state.pt`（约 26 GB）在训练停下或结束时复制过去（只留这一份），训练完成时再复制最后检查点的 `language_model.safetensors`。训练中途也要定期保存续训状态时设 `C71_STATE_SYNC_STEPS=10000`（Drive 上删除的旧文件会进回收站、占用空间）。
- 会话时限：默认在驱动程序启动后 23 小时内收尾（`C71_DEADLINE_HOURS`）：训练在时限前保存并停下，评测放不下时不开始，状态为 `paused`、退出码 75。新会话从单元格 1 开始，跑完单元格 4 和 FLEURS 准备后重新启动驱动程序，已完成的阶段跳过，停下的训练从 Drive 上的 `trainer_state.pt` 续训。冒烟测试失败、训练非正常退出、找不到突降点时状态为 `blocked`，停下等协调者处理。
- 只训练、不评测：`C71_TRAIN_ONLY=1`（训练全部完成后状态为 `trained`；评测以后去掉它重新运行驱动程序补做，或在本机用 `pipeline/eval/c71_local.py` 进行，评测项目相同，见 `pipeline/README.md`「7. 评测」）。
- 只跑其中几个运行：`C71_RUNS=c0,c-mix`；第二轮的 kl32 运行（回放损失为参考模型 top-32 的截断前向 KL，不加尾部项）要点名：`c-cont_raw-kl32`（与 c-cont_raw 只差损失）、`c-cont_think-kl32`（续写组加思考模式组）、`c-cont_raw-kl32tail`（与 c-cont_raw-kl32 只差尾部项：top-32 之外合成一个桶，目标为参考模型的剩余概率）；损失图第二轮用 `c71_plot --runs` 指定运行；回放比例 `C71_RATIO`（默认 0.5）；一步损失的分母 `C71_REPLAY_NORM`（默认 asr，按 token 归一化：分母为 (1 + r) × ASR 损失 token 数，回放的梯度份额约为 r / (1 + r)；step 为 ASR 与回放的全部损失 token 一起取平均，回放整条加入、集中在少数步里，按数据长度模拟的实际梯度份额只有 4%–20%，见 `pipeline/train/replay.py`「损失归一化与实际权重」）；思考检查的生成上限 `C71_THINK_MAX_NEW`（默认 16384，降低可省时间，比较的各模型要用同一个上限）。

**单元格 8：查看进度。**

```bash
%%bash
tail -n 3 "$WORK_DIR/train.log"; tail -n 3 "$WORK_DIR/shard_window.log"
grep -c shard-fallback "$WORK_DIR/train.log"; df -h / | tail -1
```

**单元格 9：结束会话前。** Drive 的写入在后台同步，结束会话前先把输出目录同步完。

```python
from google.colab import drive
drive.flush_and_unmount()
```

## 3. 速度测试（第 30,000–40,000 步）

用 1.2 节的索引从第 30,000 步的检查点续训到第 40,000 步，比较 Colab 与本机的速度，确认分片读取和窗口程序没有拖慢训练。

1. 本机按 1.2、1.3 节生成并上传 `index-30000-40000.*`、`pool-30000-40000-*.tar` 和第 30,000–40,000 步的顺序分片（`pack_shards list --index ... --pool --steps 30000-40000` 列出文件名），共 23 个分片、22.69 GB。
2. 准备一个测试用的输出目录，例如 `$DRIVE_DIR/runs/speed-30000/`：放入本机训练的 `train_config.json`、含 `trainer_state.pt` 的 `checkpoint-0030000/`，以及内容为 `checkpoint-0030000` 的 `latest` 文件。
3. 在 Colab 上把单元格 2 的 `INDEX` 改为 `$SHARDS_DRIVE/index-30000-40000.json`、`OUT` 改为测试目录，依次运行单元格 3–6，单元格 7 再加 `--stop-at-step 40000`；起点取自 #63 的 lr 1e-3 时再加 `--lr 1e-3`。
4. 比较测试目录的 `train_log.jsonl` 与本机同一段步数的 `audio_h_per_h`（每墙钟小时处理的音频小时数）。训练日志中不应有 `[shard-fallback]` 和 `[bad-audio]`；窗口日志中的「领先」应保持为正。速度不如预期时再试不同的 `--num-workers`。

`--steps` 之外的样本不在这个索引中，读到时按 `[bad-audio]` 跳过，所以测试必须停在第 40,000 步；训练开始时会打印索引覆盖的步数范围。

## 4. 训练开始时的核对

`train.py` 读到按训练计划排列的索引时，先核对索引记录的配置与本次训练的 `train_manifest`、`target`、`batch_size`、`max_batch_seconds`、`seed` 等决定训练计划的各项（`train.PLAN_KEYS`）相同，再按本次训练的计划重新算出读取顺序的摘要，与索引中的比较；任何一项不同都不开始训练。开始时还打印一行 `audio_shards` 概况：索引中有多少行、本地目录中常驻分片和顺序分片各有几个。

## 5. 存储桶测试

比较 Google Cloud Storage 存储桶（东京 `asia-northeast1` 和美国的一个区域）与 Drive 的读取速度和流量费用。按路径排列的分片（`pack_shards index --order path`）与训练计划无关，适合从挂载的存储桶随机读取；测试时直接用第 3 节的测试分片，读取方式相同。训练程序按位置读取，用 gcsfuse 挂载的存储桶就是 `--audio-shard-dirs` 中的又一个目录。项目编号和桶名都放在 Colab 的变量里，不写进仓库。

1. **在 Colab 中登录，查看虚拟机所在的区域。**

   ```python
   from google.colab import auth
   auth.authenticate_user()
   import os
   os.environ["GCP_PROJECT"] = "<项目 ID>"
   os.environ["BUCKET_TOKYO"] = "<东京的桶名>"
   os.environ["BUCKET_US"] = "<美国的桶名>"
   ```

   ```bash
   %%bash
   gcloud config set project "$GCP_PROJECT"
   curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | sed 's|.*/||'
   ```

2. **先设预算提醒。** 在 Cloud 控制台的「结算 → 预算和提醒」中为这个项目建一个小额预算（例如 10 美元），阈值 50%、90%、100% 发邮件。

3. **建两个存储桶**：东京一个，美国的选与虚拟机同一区域（或离它最近的区域）。加上标签，次日在结算报告中按标签区分两个桶。

   ```bash
   %%bash
   gcloud storage buckets create "gs://$BUCKET_TOKYO" --location=asia-northeast1 --default-storage-class=STANDARD --uniform-bucket-level-access
   gcloud storage buckets create "gs://$BUCKET_US" --location=<美国区域，例如 us-central1> --default-storage-class=STANDARD --uniform-bucket-level-access
   gcloud storage buckets update "gs://$BUCKET_TOKYO" --update-labels=c1-test=tokyo
   gcloud storage buckets update "gs://$BUCKET_US" --update-labels=c1-test=us
   ```

4. **在 Colab 中把同一批测试分片从 Drive 复制到两个桶。** 写入存储桶的流量不收费，在 Colab 中复制也省去从本机上传两次。只测 `bench_storage.py` 时用第 30,000–34,000 步就够（12 个分片、11.69 GB）；还要从存储桶跑第 3 节的训练速度测试时，把下面的 `30000-34000` 换成 `30000-40000`（23 个分片、22.69 GB）。

   ```bash
   %%bash
   cd "$REPO"
   TEST_INDEX="$SHARDS_DRIVE/index-30000-40000.json"
   for bucket in "$BUCKET_TOKYO" "$BUCKET_US"; do
     { ls "$SHARDS_DRIVE"/index-30000-40000.*
       "$PY" -m pipeline.colab.pack_shards list --index "$TEST_INDEX" --pool --steps 30000-34000 --prefix "$SHARDS_DRIVE"
     } | gcloud storage cp -I "gs://$bucket/shards/"
   done
   ```

5. **安装 gcsfuse 并以只读方式挂载两个桶。** 不给 `--cache-dir` 时 gcsfuse 不在本地缓存整个文件，随机读取按字节范围向存储桶请求；元数据缓存不过期（`--metadata-cache-ttl-secs -1`），分片写入后不再改变。参数名以所装版本的 `gcsfuse --help` 为准。

   ```bash
   %%bash
   echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -c -s) main" > /etc/apt/sources.list.d/gcsfuse.list
   curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg > /usr/share/keyrings/cloud.google.asc
   apt-get update -q && apt-get install -y -q gcsfuse
   for bucket in "$BUCKET_TOKYO" "$BUCKET_US"; do
     mkdir -p "/content/gcs/$bucket"
     gcsfuse --implicit-dirs -o ro --metadata-cache-ttl-secs -1 --max-conns-per-host 100 "$bucket" "/content/gcs/$bucket"
   done
   ```

6. **测速。** `bench_storage.py` 先打印虚拟机的可用区，再对每个位置测两项：顺序复制几个整分片（MB/s），以及多线程随机读取分片中的音频文件（每秒请求数、MB/s、延迟的 p50、p95、p99，加 `--decode` 时另给出解码耗时）。`--gs` 另用 `gcloud storage cp` 复制同样的分片做对照。每项结果打印一行 JSON，最后是一张表。本地的目录用单元格 5 复制下来的窗口目录。

   ```bash
   %%bash
   cd "$REPO"
   "$PY" -m pipeline.colab.bench_storage \
       --index "$SHARDS_DRIVE/index-30000-40000.json" \
       --dir drive="$SHARDS_DRIVE" \
       --dir tokyo="/content/gcs/$BUCKET_TOKYO/shards" \
       --dir us="/content/gcs/$BUCKET_US/shards" \
       --dir local="$SHARDS_LOCAL/window" \
       --gs tokyo="gs://$BUCKET_TOKYO/shards" \
       --gs us="gs://$BUCKET_US/shards" \
       --threads 16 \
       --reads 2000 \
       --decode | tee -a "$WORK_DIR/bench_storage.log"
   ```

   还可以按第 3 节的速度测试直接从存储桶训练：不运行窗口程序，`--audio-shard-dirs "/content/gcs/$BUCKET_TOKYO/shards"`，不加 `--audio-shard-fallback`，比较 `audio_h_per_h`。

7. **次日查看费用。** 在「结算 → 报告」中按 SKU 分组、按标签 `c1-test` 筛选，看每个桶的网络出站（名称含 Data Transfer 或 Egress 的 SKU）和操作次数的费用。存储桶与虚拟机在同一区域时出站不收费，跨区域、跨洲按距离计费。

8. **测完删除两个桶。**

   ```bash
   %%bash
   gcloud storage rm --recursive "gs://$BUCKET_TOKYO" "gs://$BUCKET_US"
   ```

## 6. 注意事项

- 一个分片目录只放一套索引生成的分片。同名的顺序分片来自不同的训练计划时内容不同，训练程序只按大小判断分片是否完整。
- Colab 的 Python 是 3.13，本机是 3.14，包的版本相同。数据读取进程固定用 forkserver 启动（`train.WORKER_START_METHOD`，Python 3.14 的默认值），读取进程不继承主进程中几百万行清单字典。torchcodec 使用 Colab 系统的 FFmpeg，FLAC 是无损格式，解码结果与本机相同。
- 输出目录在 Drive 上：每 2,500 步保存的检查点、`progress.json` 和日志都写到 Drive。训练程序只保留最近一个检查点的 `trainer_state.pt`，删掉的旧文件进入 Drive 的回收站并占用空间，需要时手动清空。
- 窗口程序读 `$OUT/latest` 和 `$OUT/progress.json` 得知训练进度。换输出目录时，单元格 5、6、7 的 `$OUT` 要一致。
- 拼接批次 `--pack-sequences`（任务 #66，见 `pipeline/train/__init__.py`「交叉熵实现与拼接批次」）需要 flash-attn 和 causal-conv1d。PyPI 上没有与 torch 2.13.0+cu130、Python 3.13 对应的安装包，所以不放在 `requirements-train.txt` 中，而是事先把两个 `.whl` 放到 Drive 的 `$DRIVE_DIR/wheels/packing/`，`setup.sh` 看到这个目录就安装并导入核对；目录不存在时跳过。缺少这两个包而用了 `--pack-sequences` 时，训练程序开始前就停止，不会悄悄退回会让样本互相串扰的实现。`--loss-impl torch` 不需要额外的包。

  拼接批次用的安装包在本机这样准备（`<工作目录>/tmp/C1/wheel-build/build.sh`）：flash-attn 用社区预编译包 [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels) v0.9.47 的 `flash_attn-2.8.3+cu130torch2.13-cp313-cp313-linux_x86_64.whl`；causal-conv1d 没有对应的预编译包（它的 `setup.py` 默认下载的现成包与 torch 2.13.0 稳定版不兼容，导入时缺符号 `c10::cuda::CUDAErrorLogCapture`），在一个装了 torch 2.13.0+cu130 的 Python 3.13 临时环境中从源码编译：

  ```bash
  CUDA_HOME=/usr/local/cuda-13.0 CAUSAL_CONV1D_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0;12.0" MAX_JOBS=8 \
      <Python 3.13 临时环境>/bin/python -m pip wheel --no-build-isolation --no-deps -w <输出目录> \
      "git+https://github.com/Dao-AILab/causal-conv1d.git@cd81f0413cad2fc1e6f17e785ac39f59aae690cd"
  ```

  `CAUSAL_CONV1D_FORCE_BUILD=TRUE` 让它在本机编译而不去下载现成包；编译需要 `ninja` 在 PATH 上。

## 6. 监控图

训练由命令行会话在后台运行时，浏览器连不上那台虚拟机。`monitor.py` 与训练同时在后台运行，每 2 分钟把训练损失与 dev 损失、训练速度、窗口领先步数、GPU 利用率和显存画成一张图，连同状态摘要写到 `$OUT/monitor/dashboard.png` 和 `status.json`（输出目录在 Drive 上，所以在 Drive 里就能打开）。它只需要 Colab 自带的 python3 和 matplotlib，训练结束后自己退出。

```bash
%%bash
cd "$REPO"
nohup python3 -m pipeline.colab.monitor --output-dir "$OUT" \
    --train-log "$WORK_DIR/train.log" --window-log "$WORK_DIR/shard_window.log" \
    > "$WORK_DIR/monitor.log" 2>&1 &
```

加上 `--upload gs://<存储桶>/runs/<运行名>/monitor` 时，每次写完把这两个文件上传到存储桶，本机用 `gcloud storage cat gs://.../status.json` 随时可读，浏览器标签页关闭后也能看到训练状况。在浏览器中的 notebook 里先运行一次下面的授权单元格（弹窗中点同意），Colab 的 gcloud 才能写存储桶：

```python
from google.colab import auth
auth.authenticate_user()
```

训练结束后释放运行时，避免空转计费：训练单元格最后先 `drive.flush_and_unmount()` 把输出目录写回 Drive，再 `google.colab.runtime.unassign()`。

只在 Drive 上看时，在另一个 notebook（CPU 运行时即可）里挂载 Drive 后，这个单元格每分钟刷新一次显示：

```python
import time
from IPython.display import Image, clear_output

path = "/content/drive/MyDrive/qwen3.5-asr/runs/<输出目录名>/monitor/dashboard.png"
while True:
    clear_output(wait = True)
    display(Image(filename = path))
    time.sleep(60)
```
