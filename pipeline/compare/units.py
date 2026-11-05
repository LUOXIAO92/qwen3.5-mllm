"""比对用的单位、货币、时刻词表，以及各语言的书写变体合并规则。

朗读稿把「12 km」「$5」「3:15」写成读法（doze quilômetros、five dollars、三点十五分），ASR 常写回符号。
比对时两边紧跟在数后面的单位读法和单位符号都换成同一种单位记号（extend.py），记号带一组单位编号，
读法对符号时编号有交集即一致；两边都是读法时仍要求词形一致。货币符号写在数前（$5、Rp 5.000）时
移到数后，与读法的语序一致。

单位编号是语言无关的短名：km、m、cm、mm、kg、g、deg（度）、degc（摄氏度）、degf（华氏度）、
dollar、euro、yen、yuan 等。一个词可以对应几个编号，例如英语 pound 是英镑也是磅，
「度」「graus」既可以对 °，也可以对 °C；「摄氏度」「graus Celsius」只对 °C。

时刻的写法：两边都把「时」类词（点、時、시、Uhr、heures、giờ……）接在不超过 24 的数后时换成时刻记号，
分钟后面的「分」类词去掉；符号一边「14:30」「14h30」「14.30」也换成时刻记号。分钟为 00 时读法通常不读出，
所以符号一边去掉「00」。时刻符号记号在比较时可以不对应任何东西（「five thirty」对「5:30」）。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 符号写法（各语言通用）。键为规范化后的写法：小写，「km/h」中的斜线切分后成为「km h」。
# ---------------------------------------------------------------------------

SYMBOLS = {
    "km": "km", "m": "m", "cm": "cm", "mm": "mm",
    "kg": "kg", "g": "g", "mg": "mg",
    "km h": "km/h", "kmh": "km/h",
    "°": "deg", "°c": "degc", "°f": "degf",
    "h": "hour", "min": "minute",
    "$": "dollar", "us$": "dollar", "usd": "dollar",
    "€": "euro", "eur": "euro",
    "£": "pound", "gbp": "pound",
    "¥": ("yen", "yuan"), "jpy": "yen", "cny": "yuan", "rmb": "yuan",
    "₩": "won", "krw": "won",
    "₫": "dong", "vnd": "dong",
    "฿": "baht", "thb": "baht",
    "₺": "lira",
    "₽": "ruble",
    "₱": "peso",
    "₹": "rupee",
    "r$": "real", "brl": "real",
}

# 单字符的符号（切分时单独成为记号）；「°」后紧跟 c、f 时合成「°c」「°f」
SYMBOL_CHARS = "$€£¥₩₫฿₺₽₱₹°"
# 「R$」「US$」：字母紧接「$」
DOLLAR_PREFIXES = {"r", "us"}

# 可以写在数前面的符号（读法在数后）
PREFIX = {
    "$", "us$", "r$", "€", "£", "¥", "₩", "₫", "฿", "₺", "₽", "₱", "₹",
    "usd", "eur", "gbp", "jpy", "cny", "rmb", "krw", "vnd", "thb", "brl",
}

# 各语言自己的符号写法（「mi」在西语、意大利语中是普通词，只用于英语；俄语「г」常表示「год」，不收）
LANG_SYMBOLS = {
    "en": {"mi": "mi", "ft": "ft", "lb": "lb", "lbs": "lb", "mph": "mph"},
    "ru": {"км": "km", "м": "m", "см": "cm", "мм": "mm", "кг": "kg", "руб": "ruble", "км ч": "km/h"},
    "id": {"rp": "rupiah", "rm": "ringgit"},
    "ms": {"rm": "ringgit", "rp": "rupiah"},
    "tr": {"tl": "lira"},
    "vi": {"đ": "dong"},
    "ar": {"كم": "km", "كغ": "kg", "كجم": "kg"},
    "th": {"กม": "km", "กก": "kg", "ซม": "cm", "มม": "mm"},
}
LANG_PREFIX = {"id": {"rp", "rm"}, "ms": {"rm", "rp"}}


# ---------------------------------------------------------------------------
# 读法。每种语言一个字典：单位编号 → 读法列表；编号可以是元组（一个词对应几个编号）。
# ---------------------------------------------------------------------------

DEGREE = ("deg", "degc")


def _ru(stem: str, soft: bool = False) -> list:
    """俄语名词的各格形式（单复数），用于 километр、доллар、рубль 等。"""
    if soft:
        base = stem[:-1]
        return [stem] + [base + e for e in ("я", "ей", "ю", "ям", "ем", "ём", "ями", "е", "ях")]
    return [stem] + [stem + e for e in ("а", "ов", "у", "ам", "ом", "ами", "е", "ах")]


READINGS = {
    "en": {
        "km": ["kilometer", "kilometers", "kilometre", "kilometres"],
        "m": ["meter", "meters", "metre", "metres"],
        "cm": ["centimeter", "centimeters", "centimetre", "centimetres"],
        "mm": ["millimeter", "millimeters", "millimetre", "millimetres"],
        "kg": ["kilogram", "kilograms", "kilogramme", "kilogrammes", "kilo", "kilos"],
        "g": ["gram", "grams", "gramme", "grammes"],
        "mi": ["mile", "miles"],
        "ft": ["foot", "feet"],
        ("pound", "lb"): ["pound", "pounds"],
        DEGREE: ["degree", "degrees"],
        "degc": ["degree celsius", "degrees celsius", "degree centigrade", "degrees centigrade"],
        "degf": ["degree fahrenheit", "degrees fahrenheit"],
        "km/h": ["kilometers per hour", "kilometres per hour", "kilometer per hour", "kilometre per hour"],
        "mph": ["miles per hour", "mile per hour"],
        "dollar": ["dollar", "dollars"],
        "euro": ["euro", "euros"],
        "yen": ["yen"],
        "yuan": ["yuan"],
        "won": ["won"],
        "baht": ["baht"],
        "rupiah": ["rupiah", "rupiahs"],
        "ringgit": ["ringgit", "ringgits"],
        "peso": ["peso", "pesos"],
        "rupee": ["rupee", "rupees"],
        "ruble": ["ruble", "rubles", "rouble", "roubles"],
        "hour": ["hour", "hours"],
        "minute": ["minute", "minutes"],
    },
    "de": {
        "km": ["kilometer", "kilometern", "kilometers"],
        "m": ["meter", "metern", "meters"],
        "cm": ["zentimeter", "zentimetern"],
        "mm": ["millimeter", "millimetern"],
        "kg": ["kilogramm", "kilogramms", "kilo", "kilos"],
        "g": ["gramm", "gramms"],
        DEGREE: ["grad"],
        "degc": ["grad celsius"],
        "degf": ["grad fahrenheit"],
        "km/h": ["kilometer pro stunde", "stundenkilometer", "stundenkilometern"],
        "dollar": ["dollar", "dollars"],
        "euro": ["euro", "euros"],
        "pound": ["pfund"],
        "yen": ["yen"],
        "hour": ["stunde", "stunden"],
        "minute": ["minute", "minuten"],
    },
    "fr": {
        "km": ["kilomètre", "kilomètres"],
        "m": ["mètre", "mètres"],
        "cm": ["centimètre", "centimètres"],
        "mm": ["millimètre", "millimètres"],
        "kg": ["kilogramme", "kilogrammes", "kilo", "kilos"],
        "g": ["gramme", "grammes"],
        DEGREE: ["degré", "degrés"],
        "degc": ["degré celsius", "degrés celsius"],
        "degf": ["degré fahrenheit", "degrés fahrenheit"],
        "km/h": ["kilomètres par heure", "kilomètre par heure", "kilomètres heure", "kilomètre heure"],
        "dollar": ["dollar", "dollars"],
        "euro": ["euro", "euros"],
        "pound": ["livre", "livres", "livre sterling", "livres sterling"],
        "yen": ["yen", "yens"],
        "hour": ["heure", "heures"],
        "minute": ["minute", "minutes"],
    },
    "es": {
        "km": ["kilómetro", "kilómetros"],
        "m": ["metro", "metros"],
        "cm": ["centímetro", "centímetros"],
        "mm": ["milímetro", "milímetros"],
        "kg": ["kilogramo", "kilogramos", "kilo", "kilos"],
        "g": ["gramo", "gramos"],
        DEGREE: ["grado", "grados"],
        "degc": ["grado celsius", "grados celsius", "grado centígrado", "grados centígrados"],
        "degf": ["grado fahrenheit", "grados fahrenheit"],
        "km/h": ["kilómetros por hora", "kilómetro por hora"],
        "dollar": ["dólar", "dólares"],
        "peso": ["peso", "pesos"],
        "euro": ["euro", "euros"],
        "pound": ["libra", "libras"],
        "yen": ["yen", "yenes"],
        "hour": ["hora", "horas"],
        "minute": ["minuto", "minutos"],
    },
    "pt": {
        "km": ["quilômetro", "quilômetros", "quilómetro", "quilómetros"],
        "m": ["metro", "metros"],
        "cm": ["centímetro", "centímetros"],
        "mm": ["milímetro", "milímetros"],
        "kg": ["quilograma", "quilogramas", "quilo", "quilos"],
        "g": ["grama", "gramas"],
        DEGREE: ["grau", "graus"],
        "degc": ["grau celsius", "graus celsius"],
        "degf": ["grau fahrenheit", "graus fahrenheit"],
        "km/h": ["quilômetros por hora", "quilómetros por hora", "quilômetro por hora", "quilómetro por hora"],
        "dollar": ["dólar", "dólares"],
        "real": ["real", "reais"],
        "euro": ["euro", "euros"],
        "pound": ["libra", "libras"],
        "yen": ["iene", "ienes"],
        "hour": ["hora", "horas"],
        "minute": ["minuto", "minutos"],
    },
    "it": {
        "km": ["chilometro", "chilometri"],
        "m": ["metro", "metri"],
        "cm": ["centimetro", "centimetri"],
        "mm": ["millimetro", "millimetri"],
        "kg": ["chilogrammo", "chilogrammi", "chilo", "chili"],
        "g": ["grammo", "grammi"],
        "ft": ["piede", "piedi"],
        DEGREE: ["grado", "gradi"],
        "degc": ["grado celsius", "gradi celsius", "grado centigrado", "gradi centigradi"],
        "degf": ["grado fahrenheit", "gradi fahrenheit"],
        "km/h": ["chilometri orari", "chilometri all ora", "chilometri allora"],
        "dollar": ["dollaro", "dollari"],
        "euro": ["euro"],
        "pound": ["sterlina", "sterline"],
        "yen": ["yen"],
        "hour": ["ora", "ore"],
        "minute": ["minuto", "minuti"],
    },
    "ru": {
        "km": _ru(stem = "километр"),
        "m": _ru(stem = "метр"),
        "cm": _ru(stem = "сантиметр"),
        "mm": _ru(stem = "миллиметр"),
        "kg": _ru(stem = "килограмм"),
        "g": _ru(stem = "грамм"),
        DEGREE: _ru(stem = "градус"),
        "degc": [w + " цельсия" for w in _ru(stem = "градус")],
        "degf": [w + " фаренгейта" for w in _ru(stem = "градус")],
        "km/h": [w + " в час" for w in _ru(stem = "километр")],
        "dollar": _ru(stem = "доллар"),
        "euro": ["евро"],
        "ruble": _ru(stem = "рубль", soft = True),
        "yen": ["иена", "иены", "иен", "иене", "иену", "иеной", "иенам", "иенами", "иенах"],
        "hour": _ru(stem = "час"),
        "minute": ["минута", "минуты", "минут", "минуту", "минуте", "минутой", "минутам", "минутами", "минутах"],
    },
    "tr": {
        "km": ["kilometre"],
        "m": ["metre"],
        "cm": ["santimetre"],
        "mm": ["milimetre"],
        "kg": ["kilogram", "kilo"],
        "g": ["gram"],
        DEGREE: ["derece"],
        "degc": ["santigrat derece", "derece santigrat"],
        "degf": ["fahrenhayt derece", "derece fahrenhayt"],
        "dollar": ["dolar"],
        "euro": ["euro", "avro"],
        "lira": ["lira", "türk lirası"],
        "pound": ["sterlin"],
        "yen": ["yen"],
        "hour": ["saat"],
        "minute": ["dakika"],
    },
    "id": {
        "km": ["kilometer", "kilo meter"],
        "m": ["meter"],
        "cm": ["sentimeter", "senti meter"],
        "mm": ["milimeter"],
        "kg": ["kilogram", "kilo gram", "kilo"],
        "g": ["gram"],
        DEGREE: ["derajat"],
        "degc": ["derajat celsius", "derajat celcius"],
        "degf": ["derajat fahrenheit"],
        "dollar": ["dolar"],
        "euro": ["euro"],
        "rupiah": ["rupiah"],
        "yen": ["yen"],
        "ringgit": ["ringgit"],
        "hour": ["jam"],
        "minute": ["menit"],
    },
    "ms": {
        "km": ["kilometer", "kilometre"],
        "m": ["meter", "metre"],
        "cm": ["sentimeter", "sentimetre"],
        "mm": ["milimeter", "milimetre"],
        "kg": ["kilogram", "kilo"],
        "g": ["gram"],
        DEGREE: ["darjah"],
        "degc": ["darjah celsius"],
        "degf": ["darjah fahrenheit"],
        "dollar": ["dolar"],
        "euro": ["euro"],
        "ringgit": ["ringgit"],
        "rupiah": ["rupiah"],
        "yen": ["yen"],
        "hour": ["jam"],
        "minute": ["minit"],
    },
    "fil": {
        "km": ["kilometro", "kilometer"],
        "m": ["metro"],
        "kg": ["kilo", "kilogramo"],
        "g": ["gramo"],
        "dollar": ["dolyar", "dolar"],
        "peso": ["piso", "peso"],
        "hour": ["oras"],
        "minute": ["minuto"],
    },
    "vi": {
        "km": ["ki lô mét", "kilômét", "kilomet", "cây số"],
        "m": ["mét"],
        "cm": ["xen ti mét", "xăng ti mét", "centimet"],
        "mm": ["mi li mét", "milimet"],
        "kg": ["ki lô gam", "kilôgam", "ki lô", "kí lô", "cân"],
        "g": ["gam"],
        DEGREE: ["độ"],
        "degc": ["độ c", "độ xê", "độ celsius"],
        "degf": ["độ f", "độ ép", "độ fahrenheit"],
        "dollar": ["đô la", "đô"],
        "dong": ["đồng"],
        "euro": ["euro", "ơ rô"],
        "yen": ["yên"],
        "hour": ["giờ", "tiếng"],
        "minute": ["phút"],
    },
    "th": {
        "km": ["กิโลเมตร"],
        ("km", "kg"): ["กิโล"],
        "m": ["เมตร"],
        "cm": ["เซนติเมตร"],
        "mm": ["มิลลิเมตร"],
        "kg": ["กิโลกรัม"],
        "g": ["กรัม"],
        DEGREE: ["องศา"],
        "degc": ["องศาเซลเซียส"],
        "degf": ["องศาฟาเรนไฮต์"],
        "baht": ["บาท"],
        "dollar": ["ดอลลาร์"],
        "euro": ["ยูโร"],
        "yen": ["เยน"],
        "hour": ["ชั่วโมง"],
        "minute": ["นาที"],
    },
    "ar": {
        "km": ["كيلومتر", "كيلومترات", "كيلومترا", "كيلو متر"],
        "m": ["متر", "مترا", "أمتار"],
        "cm": ["سنتيمتر", "سنتيمترات", "سنتيمترا", "سانتيمتر"],
        "mm": ["مليمتر", "مليمترات", "ملليمتر"],
        "kg": ["كيلوغرام", "كيلوغرامات", "كيلوجرام", "كيلوجرامات", "كيلو"],
        "g": ["غرام", "غرامات", "جرام", "جرامات"],
        DEGREE: ["درجة", "درجات"],
        "degc": ["درجة مئوية", "درجات مئوية"],
        "degf": ["درجة فهرنهايت", "درجات فهرنهايت"],
        "dollar": ["دولار", "دولارات", "دولارا"],
        "euro": ["يورو"],
        "hour": ["ساعة", "ساعات"],
        "minute": ["دقيقة", "دقائق"],
    },
    "fa": {
        "km": ["کیلومتر"],
        "m": ["متر"],
        "cm": ["سانتیمتر", "سانتی متر", "سانتی"],
        "mm": ["میلیمتر", "میلی متر"],
        "kg": ["کیلوگرم", "کیلو"],
        "g": ["گرم"],
        DEGREE: ["درجه"],
        "degc": ["درجه سانتیگراد", "درجه سانتی گراد", "درجه سلسیوس"],
        "degf": ["درجه فارنهایت"],
        "dollar": ["دلار"],
        "euro": ["یورو"],
        "rial": ["ریال"],
        "toman": ["تومان"],
        "hour": ["ساعت"],
        "minute": ["دقیقه"],
    },
    "zh": {
        "km": ["公里"],
        "m": ["米"],
        "cm": ["厘米"],
        "mm": ["毫米"],
        "kg": ["公斤"],
        "g": ["克"],
        DEGREE: ["度"],
        "degc": ["摄氏度"],
        "degf": ["华氏度"],
        "km/h": ["公里每小时"],
        "dollar": ["美元", "美金"],
        "yuan": ["元", "块", "块钱"],
        "euro": ["欧元"],
        "pound": ["英镑"],
        "yen": ["日元"],
        "won": ["韩元"],
        "hour": ["小时", "个小时"],
        "minute": ["分钟"],
    },
    "ja": {
        ("km", "kg"): ["キロ"],
        "km": ["キロメートル"],
        "m": ["メートル"],
        "cm": ["センチ", "センチメートル"],
        "mm": ["ミリ", "ミリメートル"],
        "kg": ["キログラム"],
        "g": ["グラム"],
        DEGREE: ["度"],
        "dollar": ["ドル"],
        "euro": ["ユーロ"],
        "yen": ["円"],
        "yuan": ["元"],
        "won": ["ウォン"],
        "pound": ["ポンド"],
        "hour": ["時間"],
        "minute": ["分", "分間"],
    },
    "ko": {
        "km": ["킬로미터"],
        ("km", "kg"): ["킬로"],
        "m": ["미터"],
        "cm": ["센티미터", "센티"],
        "mm": ["밀리미터", "밀리"],
        "kg": ["킬로그램"],
        "g": ["그램"],
        DEGREE: ["도"],
        "dollar": ["달러", "불"],
        "won": ["원"],
        "euro": ["유로"],
        "yen": ["엔"],
        "yuan": ["위안"],
        "pound": ["파운드"],
        "hour": ["시간"],
        "minute": ["분"],
    },
}

# ---------------------------------------------------------------------------
# 时刻
# ---------------------------------------------------------------------------

TIME = {
    "zh": {"hour": ["点", "点钟"], "minute": ["分"], "half": ["半"]},
    "ja": {"hour": ["時"], "minute": ["分"], "half": ["半"]},
    "ko": {"hour": ["시"], "minute": ["분"], "half": ["반"]},
    "en": {"hour": ["oclock", "o clock"]},
    "de": {"hour": ["uhr"]},
    "fr": {"hour": ["heure", "heures"], "hour_sym": ["h"], "minute": ["minute", "minutes"]},
    "es": {"hour": ["hora", "horas"], "hour_sym": ["h"], "minute": ["minuto", "minutos"], "conn": ["y"]},
    "pt": {"hour": ["hora", "horas"], "hour_sym": ["h"], "minute": ["minuto", "minutos"], "conn": ["e"]},
    "it": {"minute": ["minuto", "minuti"], "conn": ["e"]},
    "ru": {"hour": ["час", "часа", "часов"], "minute": ["минута", "минуты", "минут"]},
    "vi": {"hour": ["giờ"], "hour_sym": ["h"], "minute": ["phút"], "half": ["rưỡi"]},
    "th": {"hour": ["นาฬิกา", "โมง"], "minute": ["นาที"]},
    "fa": {"minute": ["دقیقه"], "conn": ["و"]},
}

# ---------------------------------------------------------------------------
# 书写变体：拉丁字母上的变音符号
# ---------------------------------------------------------------------------

# 各语言字母表中带变音符号的字母，这些字母保留；其他拉丁字母上的变音符号在比较时去掉
# （德语句子中的「São」「Stéfano」与「Sao」「Stefano」读音相同）。越南语的变音符号表示声调，全部保留。
NATIVE_LETTERS = {
    "en": "",
    "de": "äöü",
    "fr": "àâçéèêëîïôùûüÿ",
    "es": "áéíóúüñ",
    "pt": "áâãàçéêíóôõúü",
    "it": "àèéìíîòóùú",
    "tr": "çğöşüâîû",
    "id": "",
    "ms": "",
    "fil": "ñ",
}
KEEP_ALL_LETTERS = {"vi"}


@dataclass
class Extras:
    """一种语言比对时用的扩展词表，词都已按该语言规范化。"""

    units: dict = field(default_factory = dict)        # 词 → (来源 sym/word, 单位编号集合)
    prefix: set = field(default_factory = set)         # 可以写在数前的符号
    max_chars: int = 1                                 # 单位词最长几个字（不计空格）
    hour: set = field(default_factory = set)
    hour_sym: set = field(default_factory = set)
    minute: set = field(default_factory = set)
    half: set = field(default_factory = set)
    conn: set = field(default_factory = set)
    native_letters: str | None = None                  # None 表示不合并变音符号


def _ids(key) -> frozenset:
    return frozenset(key) if isinstance(key, tuple) else frozenset({key})


def build_extras(spec) -> Extras:
    """按 spec.code 建立该语言的扩展词表。"""
    code = spec.code
    ext = Extras()

    def register(word: str, origin: str, ids: frozenset) -> None:
        key = spec.norm(word = word)
        if not key:
            return
        old = ext.units.get(key)
        if old is not None:
            # 同一个词既是符号又是读法时按符号处理（与读法比较时只看编号）；编号取并集
            origin = "sym" if "sym" in (origin, old[0]) else origin
            ids = ids | old[1]
        ext.units[key] = (origin, ids)
        ext.max_chars = max(ext.max_chars, len(key.replace(" ", "")))

    symbols = dict(SYMBOLS)
    symbols.update(LANG_SYMBOLS.get(code, {}))
    for word, key in symbols.items():
        register(word = word, origin = "sym", ids = _ids(key = key))
    for key, words in READINGS.get(code, {}).items():
        for word in words:
            register(word = word, origin = "word", ids = _ids(key = key))
    if code == "es":
        # 拉美西语的「$」也读作 pesos
        register(word = "$", origin = "sym", ids = frozenset({"dollar", "peso"}))
    for word in PREFIX | LANG_PREFIX.get(code, set()):
        ext.prefix.add(spec.norm(word = word))
    time = TIME.get(code, {})
    for name in ("hour", "hour_sym", "minute", "half", "conn"):
        getattr(ext, name).update(spec.norm(word = w) for w in time.get(name, []))
    if code not in KEEP_ALL_LETTERS:
        ext.native_letters = NATIVE_LETTERS.get(code, "")
    return ext


def fold_latin(text: str, native: str) -> str:
    """去掉拉丁字母上不属于该语言字母表的变音符号。text 已经过 casefold 和 NFC。"""
    if text.isascii():
        return text
    out = []
    for ch in text:
        if ch in native or ch.isascii():
            out.append(ch)
            continue
        decomposed = unicodedata.normalize("NFD", ch)
        base = decomposed[0]
        if len(decomposed) > 1 and "a" <= base <= "z":
            out.append(base)
        else:
            out.append(ch)
    return "".join(out)
