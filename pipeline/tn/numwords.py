"""数词读法。

英语、德语、法语、西班牙语、葡萄牙语、意大利语、俄语调用 num2words（LGPL-2.1，以依赖方式调用），
输出经过逐语言核对；中文、日语、韩语、泰语、越南语、印尼语、马来语、土耳其语、波斯语和阿拉伯语年份
规则简单、num2words 的输出有错或不支持，在这里自己实现。所有函数只处理非负整数。
"""

from __future__ import annotations

from functools import lru_cache

# ---------------------------------------------------------------------------
# num2words 封装
# ---------------------------------------------------------------------------

N2W_LANG = {"en": "en", "de": "de", "fr": "fr", "es": "es", "pt": "pt_BR", "it": "it", "ru": "ru"}


@lru_cache(maxsize = 4096)
def n2w(n: int, lang: str, to: str = "cardinal", **kwargs) -> str:
    from num2words import num2words

    text = num2words(number = n, lang = N2W_LANG[lang], to = to, **kwargs)
    # one thousand, nine hundred → one thousand nine hundred；葡萄牙语 mil, novecentos 同理（数内逗号会让 TTS 停顿）
    text = text.replace(",", "")
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# 中文
# ---------------------------------------------------------------------------

ZH_DIGITS = "零一二三四五六七八九"
_ZH_GROUP_UNITS = ["", "万", "亿", "万亿"]


def _zh_group(g: int, first: bool) -> str:
    """0 < g < 10000 的一组；first 表示整个数的最高一组（此时 10–19 读「十几」）。"""
    out = ""
    zero = False
    for value, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        d = g // value % 10
        if d == 0:
            zero = zero or bool(out)
            continue
        if zero:
            out += "零"
            zero = False
        if unit == "十" and d == 1 and first and not out:
            out += "十"
        elif unit == "千" and d == 2:
            out += "两千"
        else:
            out += ZH_DIGITS[d] + unit
    return out


def zh_cardinal(n: int) -> str:
    """中文基数：1050 → 一千零五十，20000 → 两万，2 → 二（「两」由调用方按量词决定）。"""
    if n == 0:
        return "零"
    groups = []
    while n:
        groups.append(n % 10000)
        n //= 10000
    if len(groups) > len(_ZH_GROUP_UNITS):
        raise ValueError("数值过大")
    out = ""
    gap = False
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            gap = gap or bool(out)
            continue
        if out and (gap or g < 1000):
            out += "零"
        if g == 2 and i > 0:
            out += "两"
        else:
            out += _zh_group(g = g, first = not out)
        out += _ZH_GROUP_UNITS[i]
        gap = False
    return out


def zh_digits(digits: str) -> str:
    """逐位读：1937 → 一九三七。"""
    return "".join(ZH_DIGITS[int(d)] for d in digits)


# ---------------------------------------------------------------------------
# 日语（汉字数字，读音交给 TTS 前端）
# ---------------------------------------------------------------------------

JA_DIGITS = "〇一二三四五六七八九"
_JA_GROUP_UNITS = ["", "万", "億", "兆"]


def _ja_group(g: int, big: bool) -> str:
    out = ""
    for value, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        d = g // value % 10
        if d == 0:
            continue
        if d == 1 and unit and not (unit == "千" and big):
            out += unit
        else:
            out += JA_DIGITS[d] + unit
    return out


def ja_cardinal(n: int) -> str:
    """日语基数（汉字）：1950 → 千九百五十，41000 → 四万一千，10000000 → 一千万。"""
    if n == 0:
        return "〇"
    groups = []
    while n:
        groups.append(n % 10000)
        n //= 10000
    if len(groups) > len(_JA_GROUP_UNITS):
        raise ValueError("数值过大")
    out = ""
    for i in range(len(groups) - 1, -1, -1):
        if groups[i]:
            out += _ja_group(g = groups[i], big = i > 0) + _JA_GROUP_UNITS[i]
    return out


def ja_digits(digits: str) -> str:
    return "".join(JA_DIGITS[int(d)] for d in digits)


# ---------------------------------------------------------------------------
# 韩语
# ---------------------------------------------------------------------------

KO_SINO = "영일이삼사오육칠팔구"
_KO_GROUP_UNITS = ["", "만", "억", "조"]
_KO_NATIVE_ATTR = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]
_KO_NATIVE_TENS = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]


def _ko_group(g: int) -> str:
    out = ""
    for value, unit in ((1000, "천"), (100, "백"), (10, "십"), (1, "")):
        d = g // value % 10
        if d == 0:
            continue
        out += unit if (d == 1 and unit) else KO_SINO[d] + unit
    return out


def ko_sino(n: int) -> str:
    """韩语汉字词数词：1944 → 천구백사십사，10000 → 만。"""
    if n == 0:
        return "영"
    groups = []
    while n:
        groups.append(n % 10000)
        n //= 10000
    if len(groups) > len(_KO_GROUP_UNITS):
        raise ValueError("数值过大")
    out = ""
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if not g:
            continue
        out += ("" if (g == 1 and i == 1) else _ko_group(g = g)) + _KO_GROUP_UNITS[i]
    return out


def ko_native_attr(n: int) -> str:
    """韩语固有词数词的冠形形式（用在量词前），只支持 1–99：3 → 세，20 → 스무，21 → 스물한。"""
    if not 1 <= n <= 99:
        raise ValueError("固有词数词只支持 1–99")
    t, u = divmod(n, 10)
    if u == 0:
        return "스무" if t == 2 else _KO_NATIVE_TENS[t]
    return _KO_NATIVE_TENS[t] + _KO_NATIVE_ATTR[u]


def ko_digits(digits: str) -> str:
    return " ".join(KO_SINO[int(d)] for d in digits)


# ---------------------------------------------------------------------------
# 泰语
# ---------------------------------------------------------------------------

TH_DIGITS = ["ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า"]


def _th_block(n: int, tail: bool) -> str:
    """0 < n < 1000000；tail 表示前面还有「ล้าน」，此时个位 1 也读 เอ็ด。"""
    out = ""
    for value, unit in ((100000, "แสน"), (10000, "หมื่น"), (1000, "พัน"), (100, "ร้อย"), (10, "สิบ"), (1, "")):
        d = n // value % 10
        if d == 0:
            continue
        if unit == "สิบ":
            out += ("" if d == 1 else "ยี่" if d == 2 else TH_DIGITS[d]) + "สิบ"
        elif unit == "":
            out += "เอ็ด" if (d == 1 and (n > 10 or tail)) else TH_DIGITS[d]
        else:
            out += TH_DIGITS[d] + unit
    return out


def th_cardinal(n: int) -> str:
    """泰语基数：11 → สิบเอ็ด，21 → ยี่สิบเอ็ด，100 → หนึ่งร้อย。"""
    if n == 0:
        return TH_DIGITS[0]
    if n >= 1000000:
        high, low = divmod(n, 1000000)
        return th_cardinal(n = high) + "ล้าน" + (_th_block(n = low, tail = True) if low else "")
    return _th_block(n = n, tail = False)


def th_digits(digits: str) -> str:
    return "".join(TH_DIGITS[int(d)] for d in digits)


# ---------------------------------------------------------------------------
# 越南语
# ---------------------------------------------------------------------------

VI_DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_VI_SCALES = ["", "nghìn", "triệu", "tỷ"]


def _vi_group(g: int, full: bool) -> str:
    """0 < g < 1000；full 表示前面还有更高的组，此时百位为 0 也读「không trăm」，十位为 0 读「linh」。"""
    h, rest = divmod(g, 100)
    t, u = divmod(rest, 10)
    parts = []
    if h or full:
        parts.append(VI_DIGITS[h] + " trăm")
    if t == 0:
        if u:
            parts.append(("linh " if parts else "") + VI_DIGITS[u])
    elif t == 1:
        parts.append("mười")
        if u:
            parts.append("lăm" if u == 5 else VI_DIGITS[u])
    else:
        parts.append(VI_DIGITS[t] + " mươi")
        if u:
            parts.append({1: "mốt", 4: "tư", 5: "lăm"}.get(u, VI_DIGITS[u]))
    return " ".join(parts)


def vi_cardinal(n: int) -> str:
    """越南语基数：2026 → hai nghìn không trăm hai mươi sáu，21 → hai mươi mốt，105 → một trăm linh năm。"""
    if n == 0:
        return VI_DIGITS[0]
    groups = []
    while n:
        groups.append(n % 1000)
        n //= 1000
    if len(groups) > len(_VI_SCALES):
        raise ValueError("数值过大")
    parts = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if not g:
            continue
        word = _vi_group(g = g, full = bool(parts))
        parts.append(word + (" " + _VI_SCALES[i] if _VI_SCALES[i] else ""))
    return " ".join(parts)


def vi_digits(digits: str) -> str:
    return " ".join(VI_DIGITS[int(d)] for d in digits)


# ---------------------------------------------------------------------------
# 印尼语、马来语
# ---------------------------------------------------------------------------

_ID_DIGITS = ["nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan"]
_MS_DIGITS = ["kosong", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "lapan", "sembilan"]
_ID_SCALES = [(10 ** 12, "triliun"), (10 ** 9, "miliar"), (10 ** 6, "juta"), (1000, "ribu")]
_MS_SCALES = [(10 ** 12, "trilion"), (10 ** 9, "bilion"), (10 ** 6, "juta"), (1000, "ribu")]


def _idms_below_1000(n: int, digits: list) -> list:
    h, r = divmod(n, 100)
    parts = []
    if h:
        parts.append("seratus" if h == 1 else digits[h] + " ratus")
    if r == 0:
        pass
    elif r < 10:
        parts.append(digits[r])
    elif r == 10:
        parts.append("sepuluh")
    elif r == 11:
        parts.append("sebelas")
    elif r < 20:
        parts.append(digits[r - 10] + " belas")
    else:
        t, u = divmod(r, 10)
        parts.append(digits[t] + " puluh" + (" " + digits[u] if u else ""))
    return parts


def idms_cardinal(n: int, lang: str) -> str:
    """印尼语、马来语基数：1000 → seribu，1730 → seribu tujuh ratus tiga puluh；马来语 8 为 lapan。"""
    digits = _MS_DIGITS if lang == "ms" else _ID_DIGITS
    scales = _MS_SCALES if lang == "ms" else _ID_SCALES
    if n == 0:
        return digits[0]
    parts = []
    for value, word in scales:
        q, n = divmod(n, value)
        if not q:
            continue
        if value == 1000 and q == 1:
            parts.append("seribu")
        else:
            parts.append(idms_cardinal(n = q, lang = lang) + " " + word)
    parts.extend(_idms_below_1000(n = n, digits = digits))
    return " ".join(parts)


def idms_digits(digits: str, lang: str) -> str:
    table = _MS_DIGITS if lang == "ms" else _ID_DIGITS
    return " ".join(table[int(d)] for d in digits)


# ---------------------------------------------------------------------------
# 土耳其语
# ---------------------------------------------------------------------------

_TR_UNITS = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TR_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
_TR_ORDINAL = {
    "bir": "birinci", "iki": "ikinci", "üç": "üçüncü", "dört": "dördüncü", "beş": "beşinci",
    "altı": "altıncı", "yedi": "yedinci", "sekiz": "sekizinci", "dokuz": "dokuzuncu", "on": "onuncu",
    "yirmi": "yirminci", "otuz": "otuzuncu", "kırk": "kırkıncı", "elli": "ellinci", "altmış": "altmışıncı",
    "yetmiş": "yetmişinci", "seksen": "sekseninci", "doksan": "doksanıncı", "yüz": "yüzüncü", "bin": "bininci",
    "milyon": "milyonuncu", "milyar": "milyarıncı",
}


def _tr_below_1000(n: int) -> list:
    h, r = divmod(n, 100)
    t, u = divmod(r, 10)
    parts = []
    if h:
        parts.append("yüz" if h == 1 else _TR_UNITS[h] + " yüz")
    if t:
        parts.append(_TR_TENS[t])
    if u:
        parts.append(_TR_UNITS[u])
    return parts


def tr_cardinal(n: int) -> str:
    """土耳其语基数，分开写：2010 → iki bin on，1000 → bin，100 → yüz。"""
    if n == 0:
        return "sıfır"
    parts = []
    for value, word in ((10 ** 9, "milyar"), (10 ** 6, "milyon"), (1000, "bin")):
        q, n = divmod(n, value)
        if not q:
            continue
        if value == 1000 and q == 1:
            parts.append("bin")
        else:
            parts.append(tr_cardinal(n = q) + " " + word)
    parts.extend(_tr_below_1000(n = n))
    return " ".join(parts)


def tr_ordinal(n: int) -> str:
    """土耳其语序数：最后一个词换成序数形式，4 → dördüncü，21 → yirmi birinci。"""
    words = tr_cardinal(n = n).split()
    words[-1] = _TR_ORDINAL[words[-1]]
    return " ".join(words)


def tr_digits(digits: str) -> str:
    return " ".join(_TR_UNITS[int(d)] if d != "0" else "sıfır" for d in digits)


# ---------------------------------------------------------------------------
# 波斯语
# ---------------------------------------------------------------------------

_FA_UNITS = ["", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"]
_FA_TEENS = ["ده", "یازده", "دوازده", "سیزده", "چهارده", "پانزده", "شانزده", "هفده", "هجده", "نوزده"]
_FA_TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
_FA_HUNDREDS = ["", "صد", "دویست", "سیصد", "چهارصد", "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد"]


def _fa_below_1000(n: int) -> list:
    h, r = divmod(n, 100)
    parts = []
    if h:
        parts.append(_FA_HUNDREDS[h])
    if 10 <= r < 20:
        parts.append(_FA_TEENS[r - 10])
    else:
        t, u = divmod(r, 10)
        if t:
            parts.append(_FA_TENS[t])
        if u:
            parts.append(_FA_UNITS[u])
    return parts


def fa_cardinal(n: int) -> str:
    """波斯语基数，各部分用「و」连接：1894 → هزار و هشتصد و نود و چهار。"""
    if n == 0:
        return "صفر"
    parts = []
    for value, word in ((10 ** 9, "میلیارد"), (10 ** 6, "میلیون"), (1000, "هزار")):
        q, n = divmod(n, value)
        if not q:
            continue
        if value == 1000 and q == 1:
            parts.append("هزار")
        else:
            parts.append(fa_cardinal(n = q) + " " + word)
    parts.extend(_fa_below_1000(n = n))
    return " و ".join(parts)


def fa_digits(digits: str) -> str:
    return " ".join(_FA_UNITS[int(d)] if d != "0" else "صفر" for d in digits)


# ---------------------------------------------------------------------------
# 阿拉伯语（只用于年份，属格）
# ---------------------------------------------------------------------------

_AR_UNITS = ["", "واحد", "اثنين", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"]
_AR_TEENS = {
    11: "أحد عشر", 12: "اثني عشر", 13: "ثلاثة عشر", 14: "أربعة عشر", 15: "خمسة عشر",
    16: "ستة عشر", 17: "سبعة عشر", 18: "ثمانية عشر", 19: "تسعة عشر",
}
_AR_TENS = ["", "عشرة", "عشرين", "ثلاثين", "أربعين", "خمسين", "ستين", "سبعين", "ثمانين", "تسعين"]
_AR_HUNDREDS = ["", "مئة", "مئتين", "ثلاثمئة", "أربعمئة", "خمسمئة", "ستمئة", "سبعمئة", "ثمانمئة", "تسعمئة"]


def ar_year_genitive(n: int) -> str:
    """阿拉伯语年份（属格，用在 عام、سنة、في、منذ 之后）：2025 → ألفين وخمسة وعشرين。只支持 100–2999。"""
    if not 100 <= n <= 2999:
        raise ValueError("阿拉伯语年份只支持 100–2999")
    thousands, rest = divmod(n, 1000)
    hundreds, r = divmod(rest, 100)
    parts = []
    if thousands:
        parts.append("ألف" if thousands == 1 else "ألفين")
    if hundreds:
        parts.append(_AR_HUNDREDS[hundreds])
    if r:
        if r < 10:
            parts.append(_AR_UNITS[r])
        elif r == 10:
            parts.append(_AR_TENS[1])
        elif r < 20:
            parts.append(_AR_TEENS[r])
        else:
            t, u = divmod(r, 10)
            parts.append(_AR_UNITS[u] + " و" + _AR_TENS[t] if u else _AR_TENS[t])
    return " و".join(parts)
