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

# 平方、立方：切分前把「²」「³」「㎢」「㎡」换成私用区字符 U+E002、U+E003（NFKC 会把它们变成普通数字）
SQUARE_MARKS = str.maketrans({
    "²": "\ue002", "³": "\ue003",
    "㎢": "km\ue002", "㎡": "m\ue002", "㎠": "cm\ue002", "㎥": "m\ue003",
})
SQUARE_CHARS = "\ue002\ue003"

SYMBOLS = {
    "km": "km", "m": "m", "cm": "cm", "mm": "mm", "nm": "nm", "μm": "um",
    "kg": "kg", "g": "g", "mg": "mg", "t": "ton",
    "l": "liter", "ml": "ml",
    "km\ue002": "km2", "m\ue002": "m2", "cm\ue002": "cm2", "m\ue003": "m3",
    "ha": "hectare",
    "mi": "mi", "ft": "ft", "lb": "lb", "lbs": "lb", "mph": "mph",
    "km h": "km/h", "kmh": "km/h",
    "v": "volt", "kv": "kilovolt",
    "w": "watt", "kw": "kilowatt", "mw": ("megawatt", "milliwatt"), "gw": "gigawatt", "kwh": "kwh",
    "hz": "hertz", "khz": "kilohertz", "mhz": "megahertz", "ghz": "gigahertz",
    "kb": "kb", "mb": "mb", "gb": "gb", "tb": "tb",
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

# 罗马数字后面可以紧跟的序数后缀（XIXe、XIº）
ROMAN_SUFFIXES = {
    "fr": ("e", "er", "re", "es", "ers", "res", "ème", "èmes", "eme", "è"),
    "es": ("º", "ª", "o", "a"), "pt": ("º", "ª", "o", "a"), "it": ("º", "ª", "o", "a"),
}

# 单字符的符号（切分时单独成为记号）；「°」后紧跟 c、f 时合成「°c」「°f」
SYMBOL_CHARS = "$€£¥₩₫฿₺₽₱₹°&§"
# 「R$」「US$」：字母紧接「$」
DOLLAR_PREFIXES = {"r", "us"}

# 可以写在数前面的符号（读法在数后）
PREFIX = {
    "$", "us$", "r$", "€", "£", "¥", "₩", "₫", "฿", "₺", "₽", "₱", "₹",
    "usd", "eur", "gbp", "jpy", "cny", "rmb", "krw", "vnd", "thb", "brl",
}

# 不作单位符号的写法：「mi」在西语、意大利语、法语中是普通词（mi casa、mi-temps），「t」「l」是法语、
# 意大利语的省略形式，「ha」是西语、意大利语、葡语的动词
NOT_SYMBOLS = {
    "es": {"mi", "ha"}, "it": {"mi", "ha", "t", "l"}, "fr": {"mi", "t", "l"}, "pt": {"ha"}, "fil": {"mi"},
}

# 各语言自己的符号写法（俄语「г」常表示「год」，不收）
LANG_SYMBOLS = {
    "ru": {
        "км": "km", "м": "m", "см": "cm", "мм": "mm", "кг": "kg", "руб": "ruble", "км ч": "km/h",
        "км\ue002": "km2", "м\ue002": "m2", "т": "ton", "л": "liter", "га": "hectare",
        "г": "year_ru", "гг": "year_ru", "в": "century_ru", "вв": "century_ru",
        "квт": "kilowatt", "мвт": "megawatt", "гц": "hertz", "кгц": "kilohertz", "мгц": "megahertz",
    },
    "id": {"rp": "rupiah", "rm": "ringgit", "km persegi": "km2", "m persegi": "m2"},
    "ms": {"rm": "ringgit", "rp": "rupiah", "km persegi": "km2", "m persegi": "m2"},
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
        "nm": ["nanometer", "nanometers", "nanometre", "nanometres"],
        "um": ["micrometer", "micrometers", "micrometre", "micrometres", "micron", "microns"],
        "ton": ["ton", "tons", "tonne", "tonnes"],
        "liter": ["liter", "liters", "litre", "litres"],
        "ml": ["milliliter", "milliliters", "millilitre", "millilitres"],
        "hectare": ["hectare", "hectares"],
        "km2": ["square kilometer", "square kilometers", "square kilometre", "square kilometres"],
        "m2": ["square meter", "square meters", "square metre", "square metres"],
        "volt": ["volt", "volts"],
        "kilovolt": ["kilovolt", "kilovolts"],
        "watt": ["watt", "watts"],
        "kilowatt": ["kilowatt", "kilowatts"],
        "megawatt": ["megawatt", "megawatts"],
        "gigawatt": ["gigawatt", "gigawatts"],
        "hertz": ["hertz"],
        "kilohertz": ["kilohertz"],
        "megahertz": ["megahertz"],
        "gigahertz": ["gigahertz"],
        "kb": ["kilobyte", "kilobytes"],
        "mb": ["megabyte", "megabytes"],
        "gb": ["gigabyte", "gigabytes"],
        "tb": ["terabyte", "terabytes"],
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
        "ton": ["tonne", "tonnen"],
        "liter": ["liter", "litern"],
        "hectare": ["hektar"],
        "km2": ["quadratkilometer", "quadratkilometern"],
        "m2": ["quadratmeter", "quadratmetern"],
        "volt": ["volt"],
        "kilovolt": ["kilovolt"],
        "watt": ["watt"],
        "kilowatt": ["kilowatt"],
        "megawatt": ["megawatt"],
        "hertz": ["hertz"],
        "megahertz": ["megahertz"],
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
        "dollar": ["dollar", "dollars", "de dollars"],
        "euro": ["euro", "euros", "d'euros"],
        "pound": ["livre", "livres", "livre sterling", "livres sterling", "de livres"],
        "yen": ["yen", "yens"],
        "hour": ["heure", "heures"],
        "minute": ["minute", "minutes"],
        "ton": ["tonne", "tonnes"],
        "liter": ["litre", "litres"],
        "hectare": ["hectare", "hectares"],
        "km2": ["kilomètre carré", "kilomètres carrés"],
        "m2": ["mètre carré", "mètres carrés"],
        "volt": ["volt", "volts"],
        "watt": ["watt", "watts"],
        "kilowatt": ["kilowatt", "kilowatts"],
        "megawatt": ["mégawatt", "mégawatts"],
        "hertz": ["hertz"],
        "megahertz": ["mégahertz"],
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
        "dollar": ["dólar", "dólares", "de dólares"],
        "peso": ["peso", "pesos", "de pesos"],
        "euro": ["euro", "euros", "de euros"],
        "pound": ["libra", "libras", "de libras"],
        "yen": ["yen", "yenes"],
        "hour": ["hora", "horas"],
        "minute": ["minuto", "minutos"],
        "ton": ["tonelada", "toneladas"],
        "liter": ["litro", "litros"],
        "hectare": ["hectárea", "hectáreas"],
        "km2": ["kilómetro cuadrado", "kilómetros cuadrados"],
        "m2": ["metro cuadrado", "metros cuadrados"],
        "volt": ["voltio", "voltios"],
        "watt": ["vatio", "vatios"],
        "kilowatt": ["kilovatio", "kilovatios"],
        "megawatt": ["megavatio", "megavatios"],
        "hertz": ["hercio", "hercios", "hertz"],
        "megahertz": ["megahercio", "megahercios"],
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
        "dollar": ["dólar", "dólares", "de dólares"],
        "real": ["real", "reais", "de reais"],
        "euro": ["euro", "euros", "de euros"],
        "pound": ["libra", "libras", "de libras"],
        "yen": ["iene", "ienes"],
        "hour": ["hora", "horas"],
        "minute": ["minuto", "minutos"],
        "ton": ["tonelada", "toneladas"],
        "liter": ["litro", "litros"],
        "hectare": ["hectare", "hectares"],
        "km2": ["quilômetro quadrado", "quilômetros quadrados", "quilómetro quadrado", "quilómetros quadrados"],
        "m2": ["metro quadrado", "metros quadrados"],
        "volt": ["volt", "volts"],
        "watt": ["watt", "watts"],
        "kilowatt": ["quilowatt", "quilowatts"],
        "megawatt": ["megawatt", "megawatts"],
        "hertz": ["hertz"],
        "megahertz": ["megahertz"],
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
        "dollar": ["dollaro", "dollari", "di dollari"],
        "euro": ["euro", "di euro"],
        "pound": ["sterlina", "sterline"],
        "yen": ["yen"],
        "hour": ["ora", "ore"],
        "minute": ["minuto", "minuti"],
        "ton": ["tonnellata", "tonnellate"],
        "liter": ["litro", "litri"],
        "hectare": ["ettaro", "ettari"],
        "km2": ["chilometro quadrato", "chilometri quadrati"],
        "m2": ["metro quadrato", "metri quadrati"],
        "volt": ["volt"],
        "watt": ["watt"],
        "kilowatt": ["chilowatt"],
        "megawatt": ["megawatt"],
        "hertz": ["hertz"],
        "megahertz": ["megahertz"],
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
        "ton": ["тонна", "тонны", "тонн", "тонну", "тонне", "тонной", "тоннам", "тоннами", "тоннах"],
        "liter": _ru(stem = "литр"),
        "hectare": _ru(stem = "гектар"),
        "km2": [a + " " + b for a in ("квадратный", "квадратного", "квадратных", "квадратным", "квадратными")
                for b in _ru(stem = "километр")],
        "volt": _ru(stem = "вольт"),
        "kilovolt": _ru(stem = "киловольт"),
        "watt": _ru(stem = "ватт"),
        "kilowatt": _ru(stem = "киловатт"),
        "megawatt": _ru(stem = "мегаватт"),
        "hertz": _ru(stem = "герц"),
        "megahertz": _ru(stem = "мегагерц"),
        "year_ru": ["год", "года", "году", "годом", "годе", "годы", "годов", "годам", "годами", "годах"],
        "century_ru": _ru(stem = "век"),
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
        "km2": ["kilometrekare", "kilometre kare"],
        "m2": ["metrekare", "metre kare"],
        "cm2": ["santimetrekare", "santimetre kare"],
        "m3": ["metreküp", "metre küp"],
        "mg": ["miligram"],
        "ton": ["ton"],
        "liter": ["litre"],
        "hectare": ["hektar"],
        "km/h": ["kilometre saat"],
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
        "km2": ["kilometer persegi"],
        "m2": ["meter persegi"],
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
        "km2": ["kilometer persegi", "kilometre persegi"],
        "m2": ["meter persegi", "metre persegi"],
        "km/h": ["kilometer sejam", "kilometre sejam", "kilometer per jam", "kilometre per jam"],
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
        "hectare": ["héc ta", "hecta"],
        "liter": ["lít"],
        "ml": ["mi li lít", "mililít"],
        "km/h": ["ki lô mét trên giờ", "ki lô mét một giờ"],
        "mg": ["mi li gam", "miligam"],
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
        ("km",): ["千米"],
        ("kg",): ["千克"],
        "nm": ["纳米"],
        "um": ["微米"],
        "ton": ["吨"],
        "liter": ["升", "公升"],
        "ml": ["毫升"],
        "mg": ["毫克"],
        "hectare": ["公顷"],
        "km2": ["平方公里", "平方千米"],
        "m2": ["平方米"],
        "cm2": ["平方厘米"],
        "mi": ["英里"],
        "ft": ["英尺"],
        ("pound", "lb"): ["磅"],
        "volt": ["伏", "伏特"],
        "kilovolt": ["千伏"],
        "watt": ["瓦", "瓦特"],
        "kilowatt": ["千瓦"],
        "megawatt": ["兆瓦"],
        "hertz": ["赫兹", "赫"],
        "kilohertz": ["千赫", "千赫兹"],
        "megahertz": ["兆赫", "兆赫兹"],
        "gigahertz": ["吉赫", "吉赫兹"],
        "kb": ["千字节"],
        "mb": ["兆字节"],
        "gb": ["吉字节"],
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
        "nm": ["ナノメートル"],
        "um": ["マイクロメートル", "ミクロン"],
        "ton": ["トン"],
        "liter": ["リットル"],
        "ml": ["ミリリットル"],
        "mg": ["ミリグラム"],
        "hectare": ["ヘクタール"],
        "km2": ["平方キロメートル", "平方キロ"],
        "m2": ["平方メートル"],
        "mi": ["マイル"],
        "ft": ["フィート"],
        "lb": ["ポンド"],
        "volt": ["ボルト"],
        "kilovolt": ["キロボルト"],
        "watt": ["ワット"],
        "kilowatt": ["キロワット"],
        "megawatt": ["メガワット"],
        "hertz": ["ヘルツ"],
        "kilohertz": ["キロヘルツ"],
        "megahertz": ["メガヘルツ"],
        "gigahertz": ["ギガヘルツ"],
        "kb": ["キロバイト"],
        "mb": ["メガバイト"],
        "gb": ["ギガバイト"],
        "tb": ["テラバイト"],
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
        "nm": ["나노미터"],
        "um": ["마이크로미터", "미크론"],
        "ton": ["톤"],
        "liter": ["리터"],
        "ml": ["밀리리터"],
        "mg": ["밀리그램"],
        "hectare": ["헥타르"],
        "km2": ["제곱킬로미터", "평방킬로미터"],
        "m2": ["제곱미터", "평방미터"],
        "mi": ["마일"],
        "ft": ["피트"],
        "lb": ["파운드"],
        "volt": ["볼트"],
        "kilovolt": ["킬로볼트"],
        "watt": ["와트"],
        "kilowatt": ["킬로와트"],
        "megawatt": ["메가와트"],
        "hertz": ["헤르츠"],
        "kilohertz": ["킬로헤르츠"],
        "megahertz": ["메가헤르츠"],
        "gigahertz": ["기가헤르츠"],
        "kb": ["킬로바이트"],
        "mb": ["메가바이트"],
        "gb": ["기가바이트"],
        "tb": ["테라바이트"],
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
    "fa": {"minute": ["دقیقه"], "conn": ["و"], "conn_needs_minute": True},
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


# ---------------------------------------------------------------------------
# 缩写与符号：写法 → 读法。键是切分后的写法（点去掉，「z. B.」成为「z b」），读法按自然写法给出。
# general 在任何位置都换；number 只在后面紧跟数时换（No. 5、§ 4、n° 3）。
# ---------------------------------------------------------------------------

ABBREVIATIONS = {
    "en": {
        "general": {
            "mr": ["mister"], "mrs": ["missus"], "dr": ["doctor"], "prof": ["professor"],
            "st": ["saint", "street"], "mt": ["mount"], "ft": ["fort"], "gen": ["general"],
            "col": ["colonel"], "lt": ["lieutenant"], "sgt": ["sergeant"], "capt": ["captain"],
            "gov": ["governor"], "sen": ["senator"], "rep": ["representative"], "rev": ["reverend"],
            "hon": ["honourable", "honorable"], "jr": ["junior"], "sr": ["senior"], "vs": ["versus"],
            "etc": ["et cetera", "etcetera"], "e g": ["for example"], "i e": ["that is"],
            "approx": ["approximately"], "ltd": ["limited"], "corp": ["corporation"], "bros": ["brothers"],
            "&": ["and"],
        },
        "number": {
            "no": ["number"], "nos": ["numbers"], "nr": ["number"], "vol": ["volume"], "fig": ["figure"],
            "p": ["page"], "pp": ["pages"], "§": ["section", "paragraph"], "c": ["circa"], "ca": ["circa"],
        },
    },
    "de": {
        "general": {
            "z b": ["zum beispiel"], "d h": ["das heißt"], "usw": ["und so weiter"], "etc": ["et cetera"],
            "bzw": ["beziehungsweise"], "ca": ["circa", "zirka"], "vgl": ["vergleiche"],
            "ggf": ["gegebenenfalls"], "evtl": ["eventuell"], "inkl": ["inklusive"], "insb": ["insbesondere"],
            "bspw": ["beispielsweise"], "z t": ["zum teil"], "v a": ["vor allem"], "u u": ["unter umständen"],
            "i d r": ["in der regel"], "s o": ["siehe oben"], "s u": ["siehe unten"],
            "u ä": ["und ähnliches"], "o ä": ["oder ähnliches"], "v chr": ["vor christus"],
            "n chr": ["nach christus"], "dr": ["doktor"], "prof": ["professor"], "st": ["sankt"],
            "mr": ["mister"], "z zt": ["zurzeit"], "u v a": ["und viele andere"], "u v m": ["und vieles mehr"],
            "u a": ["unter anderem", "und andere"], "&": ["und"],
            "jh": ["jahrhundert", "jahrhunderts"], "jhd": ["jahrhundert", "jahrhunderts"],
            "jhdt": ["jahrhundert", "jahrhunderts"], "jhdts": ["jahrhunderts"],
        },
        "number": {
            "nr": ["nummer"], "abs": ["absatz"], "art": ["artikel"], "abb": ["abbildung"], "bd": ["band"],
            "kap": ["kapitel"], "s": ["seite"], "§": ["paragraf", "paragraph", "paragrafen", "paragraphen"],
            "§§": ["paragrafen", "paragraphen"],
        },
    },
    "fr": {
        "general": {
            "etc": ["et cetera"], "mme": ["madame"], "mmes": ["mesdames"], "mlle": ["mademoiselle"],
            "mgr": ["monseigneur"], "dr": ["docteur"], "st": ["saint"], "ste": ["sainte"],
            "av j c": ["avant jésus christ"], "apr j c": ["après jésus christ"], "av": ["avenue"],
            "bd": ["boulevard"], "env": ["environ"], "p ex": ["par exemple"], "c à d": ["c'est-à-dire"],
            "jr": ["junior"], "mr": ["mister"], "m": ["monsieur"], "mm": ["messieurs"], "&": ["et"],
        },
        "number": {
            "no": ["numéro"], "n°": ["numéro"], "p": ["page"], "vol": ["volume"], "art": ["article"],
            "c": ["circa"], "§": ["paragraphe"],
        },
    },
    "es": {
        "general": {
            "etc": ["etcétera"], "dr": ["doctor"], "dra": ["doctora"], "sr": ["señor"], "sra": ["señora"],
            "srta": ["señorita"], "sres": ["señores"], "lic": ["licenciado"], "prof": ["profesor"],
            "st": ["saint"], "mr": ["míster"], "ud": ["usted"], "uds": ["ustedes"],
            "a c": ["antes de cristo"], "d c": ["después de cristo"], "ee uu": ["estados unidos"],
            "aprox": ["aproximadamente"], "av": ["avenida"], "avda": ["avenida"], "jr": ["júnior"],
            "s n m": ["sobre el nivel del mar"], "msnm": ["metros sobre el nivel del mar"], "&": ["y"],
        },
        "number": {
            "n°": ["número"], "no": ["número"], "núm": ["número"], "pág": ["página"], "vol": ["volumen"],
            "art": ["artículo"], "cap": ["capítulo"], "c": ["circa"], "§": ["párrafo"], "s": ["siglo"],
            "sig": ["siglo"],
        },
    },
    "pt": {
        "general": {
            "etc": ["etcétera"], "dr": ["doutor"], "dra": ["doutora"], "sr": ["senhor"], "sra": ["senhora"],
            "srs": ["senhores"], "prof": ["professor"], "st": ["saint"], "mr": ["míster"],
            "a c": ["antes de cristo"], "d c": ["depois de cristo"], "ltda": ["limitada"], "jr": ["júnior"],
            "d": ["dom", "dona"], "&": ["e"],
        },
        "number": {
            "n°": ["número"], "no": ["número"], "pág": ["página"], "vol": ["volume"], "art": ["artigo"],
            "cap": ["capítulo"], "c": ["cerca de"], "§": ["parágrafo"], "séc": ["século"], "sec": ["século"],
        },
    },
    "it": {
        "general": {
            "ecc": ["eccetera"], "etc": ["eccetera", "etcetera"], "mons": ["monsignor", "monsignore"],
            "dott": ["dottor", "dottore"], "sig": ["signor", "signore"], "sig ra": ["signora"],
            "avv": ["avvocato"], "ing": ["ingegner", "ingegnere"], "prof": ["professor", "professore"],
            "dr": ["dottor", "dottore"], "st": ["saint"], "mr": ["mister"], "jr": ["junior"],
            "a c": ["avanti cristo"], "d c": ["dopo cristo"], "s l m": ["sul livello del mare"], "&": ["e"],
        },
        "number": {
            "n": ["numero"], "n°": ["numero"], "no": ["numero"], "nr": ["numero"], "pag": ["pagina"],
            "vol": ["volume"], "art": ["articolo"], "cap": ["capitolo"], "c": ["circa"], "§": ["paragrafo"],
            "sec": ["secolo"],
        },
    },
    "ar": {
        "general": {"م": ["للميلاد", "ميلادي", "ميلادية", "ميلاديا"], "ه": ["للهجرة", "هجري", "هجرية"]},
        "number": {},
    },
    "th": {
        "general": {"พศ": ["พุทธศักราช"], "คศ": ["คริสต์ศักราช"]},
        "number": {},
    },
    "ms": {
        # 22hb November：haribulan（日）
        "general": {"hb": ["haribulan", "hari bulan"]},
        "number": {},
    },
    "fil": {
        "general": {"atbp": ["at iba pa"]},
        "number": {},
    },
    "ru": {
        "general": {
            "т е": ["то есть"], "т д": ["так далее"], "т п": ["тому подобное"], "н э": ["нашей эры"],
            "им": ["имени"], "см": ["смотри"], "&": ["и"],
        },
        "number": {"ок": ["около"], "no": ["номер"], "§": ["параграф"]},
    },
}

# 读法数后的连接成分：菲律宾语以辅音结尾的数词修饰后面的词时要加 na（apat na taon、anim na porsiyento），
# 写成阿拉伯数字时通常不写（4 taon、6%）
LINKERS = {"fil": ["na"]}

# 斜线日期的读法：日/月读作「日 tháng 月」，月/年读作「月 năm 年」，日/月/年读作「日 tháng 月 năm 年」
SLASH_DATES = {"vi": {"month": ["tháng"], "year": ["năm"]}}

# 前面是这些词时，单个字母 I、V、X 也认作罗马数字（siglo V a. C.、в V веке）
CENTURY_WORDS = {
    "es": {"siglo", "siglos", "s"}, "pt": {"século", "séculos", "séc"}, "it": {"secolo", "secoli", "sec"},
    "fr": {"siècle", "siècles"}, "en": {"century"},
}

# 罗马数字作君主序号时读法前面的冠词（Henry the Eighth、Wilhelm der Zweite）
ROMAN_ARTICLES = {"en": ["the"], "de": ["der", "die", "das", "dem", "den", "des"]}

# 英语日期的「the … of」：11 December ↔ the eleventh of December，May 11 ↔ May the eleventh
EN_MONTHS = [
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
]

# 数词与后缀连写成一个词（семнадцатилетний ↔ 17-летний，hundertjährige ↔ 100-jährige，
# neunzehnhundertachtziger ↔ 1980er）。decade 表示只在整十的数后拆
COMPOUND_SUFFIXES = {
    "ru": {
        "any": [
            "летний", "летнего", "летнему", "летним", "летнем", "летняя", "летней", "летнюю", "летнее",
            "летние", "летних", "летними", "летие", "летия", "летию", "летием", "летии",
            "процентный", "процентная", "процентное", "процентного", "процентной",
        ],
    },
    "de": {
        "any": ["jährige", "jährigen", "jähriger", "jähriges", "jährigem", "jährig", "fach", "fache", "fachen"],
        "decade": ["er", "ern"],
    },
}


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
    conn_needs_minute: bool = False                    # 连接词式的时刻必须带「分」类词（波斯语）
    native_letters: str | None = None                  # None 表示不合并变音符号
    abbr: dict = field(default_factory = dict)         # 缩写写法 → 读法集合
    abbr_words: dict = field(default_factory = dict)   # 读法 → 读法集合（与 abbr 的集合比较）
    abbr_number: set = field(default_factory = set)    # 只在后面接数时才换的写法和读法
    abbr_chars: int = 1
    unit_suffixes: frozenset = frozenset()            # 单位词后可以连写的词尾（土耳其语 kilometreye、km'ye）
    date_month: frozenset = frozenset()               # 斜线日期 7/1 读出的「月」词（越南语 bảy tháng một）
    date_year: frozenset = frozenset()                # 斜线日期 5/2023 读出的「年」词（越南语 tháng năm năm 2023）
    linkers: frozenset = frozenset()                  # 读法数后的连接成分，阿拉伯数字写法不写出（菲律宾语 apat na taon ↔ 4 taon）
    roman_articles: list = field(default_factory = list)
    months: set = field(default_factory = set)
    compound_any: list = field(default_factory = list)
    compound_decade: list = field(default_factory = list)


def _ids(key) -> frozenset:
    return frozenset(key) if isinstance(key, tuple) else frozenset({key})


def build_extras(spec) -> Extras:
    """按 spec.code 建立该语言的扩展词表。"""
    code = spec.code
    ext = Extras()

    def register(word: str, origin: str, ids: frozenset) -> None:
        key = phrase(text = word, spec = spec) if origin == "word" and not spec.scan else spec.norm(word = word)
        if not key:
            return
        old = ext.units.get(key)
        if old is not None:
            # 同一个词既是符号又是读法时按符号处理（与读法比较时只看编号）；编号取并集
            origin = "sym" if "sym" in (origin, old[0]) else origin
            ids = ids | old[1]
        ext.units[key] = (origin, ids)
        ext.max_chars = max(ext.max_chars, len(key.replace(" ", "")))

    symbols = {k: v for k, v in SYMBOLS.items() if k not in NOT_SYMBOLS.get(code, set())}
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
    ext.conn_needs_minute = bool(time.get("conn_needs_minute"))
    if code not in KEEP_ALL_LETTERS:
        ext.native_letters = NATIVE_LETTERS.get(code, "")
    for cls, table in ABBREVIATIONS.get(code, {}).items():
        for key, words in table.items():
            norm_key = spec.norm(word = key)
            readings = frozenset(phrase(text = w, spec = spec) for w in words)
            ext.abbr[norm_key] = ext.abbr.get(norm_key, frozenset()) | readings
            for reading in readings:
                ext.abbr_words[reading] = frozenset({reading})
            if cls == "number":
                ext.abbr_number.add(norm_key)
                ext.abbr_number.update(readings)
            ext.abbr_chars = max([ext.abbr_chars, len(norm_key)] + [len(r) for r in readings])
    ext.unit_suffixes = spec.suffixes
    dates = SLASH_DATES.get(code, {})
    ext.date_month = frozenset(phrase(text = w, spec = spec) for w in dates.get("month", []))
    ext.date_year = frozenset(phrase(text = w, spec = spec) for w in dates.get("year", []))
    ext.linkers = frozenset(spec.norm(word = w) for w in LINKERS.get(code, []))
    ext.roman_articles = [phrase(text = w, spec = spec) for w in ROMAN_ARTICLES.get(code, [])]
    if code == "en":
        ext.months = {spec.norm(word = w) for w in EN_MONTHS}
    suffixes = COMPOUND_SUFFIXES.get(code, {})
    ext.compound_any = sorted((spec.norm(word = w) for w in suffixes.get("any", [])), key = len, reverse = True)
    ext.compound_decade = sorted((spec.norm(word = w) for w in suffixes.get("decade", [])), key = len, reverse = True)
    return ext


def phrase(text: str, spec) -> str:
    """一段读法切分后的形式（与比较单元中的普通文字一致）：撇号去掉，连字符成为空格。"""
    from .parse import tokenize

    joiner = "" if spec.scan else " "
    return joiner.join(t.text for t in tokenize(text = text, spec = spec) if t.kind in ("word", "char"))


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
