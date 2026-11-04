"""意大利语：基数、年份、「21 maggio 1974」日期、小数（virgola，逐位读）、百分数、单位、货币（€、$）。

以 1 结尾的数（uno/una、ventuno/ventuna）随名词变性，只在确定为阳性时改写，阳性名词前 1 读 un。
复合数词末尾的 tre 带重音（ventitré、duemilaventitré）。
"""

from __future__ import annotations

import re

from ..numwords import n2w
from .romance import Romance


class It(Romance):
    code = "it"
    point = "virgola"
    percent = "per cento"
    units = {
        "km/h": ("chilometro orario", "chilometri orari"), "km²": ("chilometro quadrato", "chilometri quadrati"),
        "m²": ("metro quadrato", "metri quadrati"), "km": ("chilometro", "chilometri"), "cm": ("centimetro", "centimetri"),
        "mm": ("millimetro", "millimetri"), "kg": ("chilogrammo", "chilogrammi"), "m": ("metro", "metri"),
        "g": ("grammo", "grammi"), "°C": ("grado Celsius", "gradi Celsius"), "℃": ("grado Celsius", "gradi Celsius"),
        "°F": ("grado Fahrenheit", "gradi Fahrenheit"), "℉": ("grado Fahrenheit", "gradi Fahrenheit"),
    }
    currencies = {"€": ("euro", "euro"), "$": ("dollaro", "dollari")}
    scales = ("mila", "milioni", "milione", "miliardi", "miliardo")
    scale_links = {"milioni": "di", "milione": "di", "miliardi": "di", "miliardo": "di"}
    months = (
        "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre",
        "novembre", "dicembre",
    )
    year_before = {
        "nel", "dal", "al", "del", "fino", "anno", "anni", "entro", "tra", "fra", "e", "il", "dell", "sul", "verso",
        "prima", "dopo", "nell",
    }
    masc_after = {
        "anni", "giorni", "mesi", "secoli", "gradi", "chilometri", "metri", "minuti", "secondi", "punti", "uomini",
        "abitanti", "euro", "dollari", "milioni", "gol", "voti", "paesi", "chili",
    }
    first_day = "primo"

    def card(self, n: int) -> str:
        # num2words 对 223、2023 输出 duecentoventitre、duemilaventitre，词尾的 tre 应写作 tré
        return re.sub(pattern = r"(?<=[a-z])tre$", repl = "tré", string = n2w(n = n, lang = "it"))

    def ordinal(self, n: int) -> str:
        return n2w(n = n, lang = "it", to = "ordinal")

    def digits(self, digits: str) -> str:
        return " ".join(n2w(n = int(d), lang = "it") for d in digits)

    def masc(self, n: int) -> str:
        return "un" if n == 1 else self.card(n = n)

    def gender_sensitive(self, n: int) -> bool:
        return n % 10 == 1 and n % 100 != 11


LANG = It
