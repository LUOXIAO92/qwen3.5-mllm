"""音频 projector 训练（实验 v1，任务 T10；空目标、加权抽样与混音，实验 v2，任务 T1）。

只训练阶段 1 模型中的 model.audio_projector（Linear → SiLU → Linear），音频塔、视觉塔和语言模型冻结。
输入使用 CONVENTIONS 中「训练与评测的对话格式」：用户消息是一段音频加指令 Transcribe the audio.，
用模型自带的 chat template 展开（add_generation_prompt = True、enable_thinking = False），助手回合
以空思考块开头；只对空思考块之后的目标文本和结尾的 <|im_end|> 计算损失。

模块
====

- data.py：读取训练清单、选择训练目标（script 朗读稿，或 original 原文，中文转简体）、按时长分组成批、
  用阶段 1 processor 做特征提取与音频标记展开，再用 mm_assistant_mask 在最终的 input_ids 上按 token 匹配助手回合生成标签。
- check_labels.py：核对计损失的内容正好是目标文本加 <|im_end|>（空目标样本只有 <|im_end|>），提示部分与推理时一致；
  更换模板或 processor 后运行。
- sampling.py：按清单字段（lang、source 等）设定抽样权重，空目标样本按设定占比抽取；每轮的抽样和分批只由
  (seed, 轮次) 决定，续训时与不中断的训练一致。
- mixing.py：在数据读取进程中给语音样本实时混入背景声，信噪比在设定范围内均匀随机，背景声可按类别加权。
- model.py：加载模型（冻结部分 bfloat16，projector float32），损失计算。词表约 25 万，词表投影和
  交叉熵合在一起分块计算，只在计损失的位置进行。
- train.py：训练入口，梯度累积、线性预热加余弦退火、定期 dev 损失、检查点与断点续训。
- checkpoint.py：检查点格式与导出完整模型。
- progress.py：进度日志行与持续更新的 progress.json。
- show_batch.py：打印样本的标记序列与标签掩码，核对音频标记数和损失位置。
- decode.py：贪心解码，统计与训练目标逐字一致的比例。
- export.py：把 projector 合入阶段 1 模型，导出完整模型目录。
- bench.py：不同批量下的显存与吞吐。
- fleurs_dev.py：把 FLEURS dev 清单转成训练清单格式，并抽取过拟合测试子集。
- tests/：空目标标签、混音信噪比与比例、加权抽样比例与续训一致性的单元测试，
  在仓库根目录运行 python -m unittest discover -s pipeline/train/tests -t .。

训练清单
========

每行至少包含 sample_id、audio_path、duration_s、lang、script、original（CONVENTIONS 中「核对通过的
训练清单」）。相对的 audio_path 以 --audio-root 为基准，未给出时以清单所在目录为基准。

检查点
======

<output-dir>/checkpoint-<7 位步数>/ 下：

- projector.safetensors：只含完整模型 state_dict 中以 model.audio_projector. 开头的四个键
  （linear_1.weight、linear_1.bias、linear_2.weight、linear_2.bias），键名与完整模型一致，float32。
  可以用 load_state_dict(strict = False) 直接载入完整模型。
- projector.json：训练目标、步数、轮数、最近训练损失、dev 损失、冻结参数核对结果和完整训练配置。
- trainer_state.pt：优化器、学习率调度器、随机数状态和数据位置，只在最近的检查点中保留。

<output-dir>/ 下另有 train_config.json、train_log.jsonl（每 --log-every 步一行，见下文）、
eval_log.jsonl（dev 损失，按语言）、frozen_check.json（训练前后冻结参数逐张量的 SHA-256 比较）、
progress.json（见下文）和 latest（最近检查点的目录名）。

进度
====

每 --log-every 步在标准输出打印一行，例如：

    [步 300/30000 1.0%] 3.57 it/s | 57.1 样本/s | 727 音频h/h | loss 3.6621 | lr 1.81e-03 | 显存 20.9 GB | 已用 0:02:04 | 剩余 3:25:12

依次是步数/总步数、每秒优化步数、每秒样本数、每墙钟小时处理的音频小时数、这一段的平均损失、学习率、
这一段的峰值显存、已用时间（含续训前各段）、预计剩余时间。速度按训练本身计算，不含 dev 评估和保存；
剩余时间按最近 20 段的墙钟速度估计，包含这些开销。同样的内容以 JSON 追加到 train_log.jsonl
（字段 step、total_steps、epoch、loss、token_acc、lr、grad_norm、it_per_s、samples_per_s、
audio_h_per_h、mem_gb、elapsed_s、eta_s）。

<output-dir>/progress.json 在开始、每写一行日志、每次 dev 评估和保存、停止、完成或出错时重写：
status（running、stopped、finished、failed）、step、total_steps、percent、elapsed、eta、
eta_finish_at（预计完成时刻）、window（最近一段的速度、损失、学习率、显存）、max_mem_gb、
last_eval（最近一次 dev 损失）、last_checkpoint、updated_at，停止时还有 stop_reason，出错时有 error。

显存与吞吐
==========

RTX 5090 32 GB，FLEURS dev（≤ 30 秒，平均约 12 秒）实测，包含数据读取、前向、反传和优化器更新：

| 每批条数 | 每秒样本 | 每秒音频秒数 | 每步秒数 | 峰值显存 | 最长 30 秒样本组成一批 |
| --- | --- | --- | --- | --- | --- |
| 8 | 48 | 538 | 0.17 | 10.1 GB | 12.1 GB |
| 16 | 65 | 770 | 0.25 | 20.1 GB | 21.1 GB |
| 24 | 69 | 754 | 0.35 | 24.5 GB | 超出 |
| 32 | 超出 | | | | |

只取 10 秒以内的样本（平均约 8 秒）时，每批 16 条为 97 条/秒、715 音频秒/秒、10.8 GB；每批 32 条为
105 条/秒、824 音频秒/秒、19.2 GB；48 条超出。显存约为 3.5 GB 加每个 token（批大小 × 填充后长度）
2.2 MB，一秒音频连同提示和文字约 17 个 token。冻结的语言模型反传要保存中间结果，这是显存的主要来源。
吞吐在每批 16 条左右已接近饱和，更大的批量主要用梯度累积实现。打开 --deterministic 后每秒音频秒数
降为 597（约慢 23%）。

WSL 下显存用满后会借用内存继续运行，速度骤降而不报错。用 --max-duration-s 和 --max-batch-seconds
（每批「条数 × 批内最长时长」的上限）控制显存，并留意进度中的显存（mem_gb、max_mem_gb）。

v1 超参数建议
=============

依据：在 FLEURS dev 80% 的句子（约 17 小时，18 种语言）上训练、在其余 20% 句子上看损失，每批 16 条、
1500 步。峰值学习率 3e-3 时留出损失从 4.1 降到 3.21 且仍在下降；1e-3 只降到 3.45；1e-2 停在 3.40
左右，训练损失继续下降。用 3e-3 训练的 projector，把音频换成同语言其他样本后留出损失从 3.18 升到 3.84，
说明模型已经在使用音频内容。

- --batch-size 16 --grad-accum 1 --max-duration-s 30；若训练音频大多在 10 秒左右，可用
  --batch-size 32 --max-batch-seconds 400，10 秒以内的样本上吞吐比每批 16 条高约 15%。
- --lr 3e-3 --warmup-steps 500 --min-lr-ratio 0.1（余弦退火），--weight-decay 0 --max-grad-norm 1.0。
- 数据按每种语言最多 30 小时、18 种语言计约 540 小时，每秒处理约 780 音频秒，一轮约 42 分钟；平均
  每条 8 秒时一轮约 1.5 万步。建议 --epochs 2，约 1.5 小时，朗读稿与原文两版共约 3 小时。
- --eval-manifest 用 FLEURS dev，--eval-max-per-lang 50，--eval-every 2000 --save-every 2000，
  按 dev 损失（eval_log.jsonl 中的 lang_mean_loss）选检查点。
- 分段占用 GPU：每段加 --resume --stop-after-minutes 14，退出码 75 时再次运行，退出码 0 表示完成。

冒烟规模（约 250 条）建议
=========================

目的是在冒烟清单上过拟合，确认整条流程可以训练。--batch-size 16 --lr 3e-3 --warmup-steps 50
--max-steps 1500（约 95 轮），--eval-manifest 用同一份清单、--eval-every 250、--save-every 250。
在 FLEURS dev 中每种语言 14 条（共 252 条、约 50 分钟音频，≤ 30 秒）上实测：第 1000 步损失 0.034，
第 1500 步 0.021，贪心解码 252 条全部与目标一致；每秒 4.9 步、77 条、约 930 音频h/h，峰值显存
15.7 GB，训练用时约 7 分钟（另加每段启动约 1 分钟）。

环境
====

基于项目 conda 环境（torch 2.13、transformers 5.16.1、torchcodec）建立的 venv，额外安装
flash-linear-attention 与 fla-core 0.6.0（安装时加 --no-deps，避免改动 torch 与 transformers）。
没有 fla 时语言模型的线性注意力层退回 PyTorch 实现，结果相同但更慢。

注意事项
========

Python 3.14 的多进程默认使用 forkserver，数据读取进程启动时要重新导入 torch 和 transformers，
每次新建数据读取进程需要十几秒；训练只建一次，dev 评估的读取进程常驻。fla 的 Triton 内核对每个
新的批大小做一次自动调优，第一次遇到时多花几秒到十几秒，之后不再重复。

默认的注意力反传有微小的随机差异，同一配置训练两次结果不逐位相同；需要逐位复现时加 --deterministic。
分词器会对文本做 NFC 规范化，训练目标实际是规范化后的文本。
"""
