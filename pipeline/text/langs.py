"""各语言的静态配置：来源目录名、语言检测标签、文字、估计语速。

其他模块只从这里读取语言相关的取值，新增语言或来源时只改这一个文件。
"""

from __future__ import annotations

LANGS = ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "id", "ms", "th", "vi", "fil", "ar", "tr", "fa"]

# 不用空格分词的语言：规范化键时空白全部去掉，句子切分按字符处理。
NO_SPACE_LANGS = {"zh", "ja", "th"}

# 有大小写的文字：句首必须大写。
CASED_LANGS = {"en", "de", "fr", "ru", "pt", "es", "it", "id", "ms", "vi", "fil", "tr"}

# GlotLID v3 的语言标签，取值是「语言_文字」，接受的标签集合。
GLOTLID_LABELS = {
    "zh": {"cmn_Hani"},
    "en": {"eng_Latn"},
    "ja": {"jpn_Jpan"},
    "ko": {"kor_Hang"},
    "de": {"deu_Latn"},
    "fr": {"fra_Latn"},
    "ru": {"rus_Cyrl"},
    "pt": {"por_Latn"},
    "es": {"spa_Latn"},
    "it": {"ita_Latn"},
    "id": {"ind_Latn"},
    "ms": {"zsm_Latn"},
    "th": {"tha_Thai"},
    "vi": {"vie_Latn"},
    "fil": {"fil_Latn"},
    "ar": {"arb_Arab", "ars_Arab"},
    "tr": {"tur_Latn"},
    "fa": {"fas_Arab"},
}

# 期望的文字（Unicode 字符名的第一个词）。
EXPECTED_SCRIPTS = {
    "zh": {"CJK"},
    "ja": {"CJK", "HIRAGANA", "KATAKANA"},
    "ko": {"HANGUL", "CJK"},
    "ru": {"CYRILLIC"},
    "th": {"THAI"},
    "ar": {"ARABIC"},
    "fa": {"ARABIC"},
}
# 其余语言（拉丁文字）：
for _lang in LANGS:
    EXPECTED_SCRIPTS.setdefault(_lang, {"LATIN"})

# 期望文字占全部字母的最低比例。拉丁字母语言要求更严，防止混入西里尔、汉字等。
MIN_SCRIPT_RATIO = {
    "zh": 0.85, "ja": 0.85, "ko": 0.85,
    "ru": 0.9, "th": 0.9, "ar": 0.9, "fa": 0.9,
}
DEFAULT_MIN_SCRIPT_RATIO = 0.97

# 估计语速（字符/秒，含标点和空格）：FLEURS 开发集和测试集的真人朗读，
# 取每句「字符数 / 音频时长」的中位数，样本数 700–1500 句。录音首尾带少量静音，
# 所以估计值略偏长；TTS 合成的语速另行实测后可以替换这张表。
CHARS_PER_SECOND = {
    "zh": 3.69, "en": 13.53, "ja": 4.03, "ko": 4.87, "de": 12.33, "fr": 15.24,
    "ru": 12.22, "pt": 11.26, "es": 12.54, "it": 10.53, "id": 11.30, "ms": 12.91,
    "th": 10.15, "vi": 10.95, "fil": 8.89, "ar": 10.11, "tr": 10.34, "fa": 7.88,
}

# 长度档的时长边界（秒）：短 <=5，中 5–15，长 15–30。
MIN_DURATION_S = 1.0
SHORT_MAX_S = 5.0
MEDIUM_MAX_S = 15.0
LONG_MAX_S = 30.0

# Tatoeba 的语言代码。
TATOEBA_CODES = {
    "zh": ["cmn"], "en": ["eng"], "ja": ["jpn"], "ko": ["kor"], "de": ["deu"],
    "fr": ["fra"], "ru": ["rus"], "pt": ["por"], "es": ["spa"], "it": ["ita"],
    "id": ["ind"], "ms": ["zsm", "zlm"], "th": ["tha"], "vi": ["vie"],
    "fil": ["tgl"], "ar": ["ara"], "tr": ["tur"], "fa": ["pes"],
}

# Common Voice 句库在仓库 server/data 下的目录名。
COMMON_VOICE_DIRS = {
    "zh": "zh-CN", "en": "en", "ja": "ja", "ko": "ko", "de": "de", "fr": "fr",
    "ru": "ru", "pt": "pt", "es": "es", "it": "it", "id": "id", "ms": "ms",
    "th": "th", "vi": "vi", "fil": "tl", "ar": "ar", "tr": "tr", "fa": "fa",
}

# singletongue/wikipedia-paragraphs 覆盖的语言及其 config 前缀。
WIKI_PARAGRAPH_LANGS = {
    "zh": "zhwiki", "en": "enwiki", "ja": "jawiki", "ko": "kowiki", "de": "dewiki",
    "fr": "frwiki", "ru": "ruwiki", "pt": "ptwiki", "es": "eswiki", "it": "itwiki",
    "ar": "arwiki",
}
