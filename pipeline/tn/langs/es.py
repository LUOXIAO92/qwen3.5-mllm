"""西班牙语：基数、年份、「11 de mayo de 2025」日期、小数（coma）、百分数、单位、货币（€、$）。

uno/una、veintiuno/veintiuna 和 doscientos–novecientos（千位组同样）随名词变性，只在确定为阳性时改写；
阳性名词前 uno 缩为 un（un kilómetro、veintiún años）。序数（1.º、1.ª）有 primer/tercer 的缩略，暂不处理。
"""

from __future__ import annotations

import re

from ..numwords import n2w
from .romance import Romance

_APOCOPE_BEFORE_SCALE = re.compile(pattern = r"\b(veinti)?uno(?= (?:mil|millón|millones|billón|billones)\b)")


def apocope(text: str) -> str:
    """uno → un、veintiuno → veintiún（用在名词或 mil、millones 之前）。"""
    def repl(m):
        return "veintiún" if m.group(1) else "un"

    return _APOCOPE_BEFORE_SCALE.sub(repl = repl, string = text)


class Es(Romance):
    code = "es"
    point = "coma"
    frac_as_number = True
    percent = "por ciento"
    units = {
        "km/h": ("kilómetro por hora", "kilómetros por hora"), "km²": ("kilómetro cuadrado", "kilómetros cuadrados"),
        "m²": ("metro cuadrado", "metros cuadrados"), "km": ("kilómetro", "kilómetros"), "cm": ("centímetro", "centímetros"),
        "mm": ("milímetro", "milímetros"), "kg": ("kilogramo", "kilogramos"), "m": ("metro", "metros"),
        "g": ("gramo", "gramos"), "°C": ("grado Celsius", "grados Celsius"), "℃": ("grado Celsius", "grados Celsius"),
        "°F": ("grado Fahrenheit", "grados Fahrenheit"), "℉": ("grado Fahrenheit", "grados Fahrenheit"),
    }
    currencies = {"€": ("euro", "euros"), "$": ("dólar", "dólares")}
    scales = ("mil", "millones", "millón")
    scale_links = {"millones": "de", "millón": "de"}
    months = (
        "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "setiembre",
        "octubre", "noviembre", "diciembre",
    )
    year_before = {
        "en", "de", "desde", "hasta", "año", "del", "entre", "y", "hacia", "para", "durante", "antes", "después",
        "a", "o", "el", "finales", "principios", "mediados",
    }
    masc_after = {
        "años", "días", "meses", "siglos", "grados", "kilómetros", "metros", "minutos", "segundos", "puntos",
        "hombres", "habitantes", "euros", "dólares", "millones", "goles", "partidos", "votos", "países", "kilos",
        "pesos", "centímetros",
    }
    first_day = "primero"
    date_link = r"\s(?:de\s)?"

    def card(self, n: int) -> str:
        return apocope(text = n2w(n = n, lang = "es"))

    def ordinal(self, n: int) -> str:
        return n2w(n = n, lang = "es", to = "ordinal")

    def digits(self, digits: str) -> str:
        return " ".join(n2w(n = int(d), lang = "es") for d in digits)

    def masc(self, n: int) -> str:
        text = self.card(n = n)
        if text.endswith("veintiuno"):
            return text[:-len("veintiuno")] + "veintiún"
        if text == "uno" or text.endswith(" uno"):
            return text[:-len("uno")] + "un"
        return text

    def gender_sensitive(self, n: int) -> bool:
        units, thousands = n % 1000, n // 1000 % 1000
        if units % 10 == 1 and units % 100 != 11:
            return True
        if units // 100 >= 2 or thousands // 100 >= 2:
            return True
        return thousands != 1 and thousands % 10 == 1 and thousands % 100 != 11


LANG = Es
