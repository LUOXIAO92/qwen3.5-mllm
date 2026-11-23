"""IFEval 的官方打分逻辑（任务 #71 的文本基准，打分入口见 score.py）。

instructions.py、instructions_registry.py、instructions_util.py、evaluation_lib.py 原样取自 google-research/google-research 的
instruction_following_eval（提交 e6890f85757dd84e27ca6df2dd30651dafad28e0，Apache-2.0，许可全文见同目录 LICENSE），
保留原来的版权与许可头，代码风格保持上游原样，便于与上游逐行对照。改动只有三处，都写在各文件头的 Modified 注释中：
导入路径改成本包；absl.logging 换成标准库 logging；immutabledict 换成标准库 types.MappingProxyType（只读映射）。

依赖 langdetect 与 nltk（≥ 3.9.1，带 punkt_tab 数据；nltk 3.9.1 把原代码中 punkt 的 english.pickle 路径映射到 punkt_tab）。
项目的 envs/mllm-train、envs/mllm-eval 都没有这两个包，打分在单独的 python 中进行（只用 CPU），见 score.py。
"""
