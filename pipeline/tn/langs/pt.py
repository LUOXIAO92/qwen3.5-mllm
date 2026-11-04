"""葡萄牙语：基数、年份、「21 de maio de 1974」日期、小数（vírgula）、百分数、单位、货币（€、$）。

数词用巴西拼写（dezesseis、quilômetro）。um/uma、dois/duas 和 duzentos–novecentos（千位组同样）
随名词变性，只在确定为阳性时改写。十亿以上巴西（bilhão）与葡萄牙（bilião）数值不同，丢弃。
"""

from __future__ import annotations

from ..numwords import n2w
from ..rules import Skip
from .romance import Romance


class Pt(Romance):
    code = "pt"
    point = "vírgula"
    frac_as_number = True
    percent = "por cento"
    units = {
        "km/h": ("quilômetro por hora", "quilômetros por hora"),
        "km²": ("quilômetro quadrado", "quilômetros quadrados"), "m²": ("metro quadrado", "metros quadrados"),
        "km": ("quilômetro", "quilômetros"), "cm": ("centímetro", "centímetros"), "mm": ("milímetro", "milímetros"),
        "kg": ("quilograma", "quilogramas"), "m": ("metro", "metros"), "g": ("grama", "gramas"),
        "°C": ("grau Celsius", "graus Celsius"), "℃": ("grau Celsius", "graus Celsius"),
        "°F": ("grau Fahrenheit", "graus Fahrenheit"), "℉": ("grau Fahrenheit", "graus Fahrenheit"),
    }
    currencies = {"€": ("euro", "euros"), "$": ("dólar", "dólares")}
    scales = ("mil", "milhões", "milhão")
    scale_links = {"milhões": "de", "milhão": "de"}
    months = (
        "janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro",
        "novembro", "dezembro",
    )
    year_before = {
        "em", "de", "desde", "até", "ano", "entre", "e", "a", "para", "durante", "antes", "após", "depois", "o",
        "final", "início", "meados",
    }
    masc_after = {
        "anos", "dias", "meses", "séculos", "graus", "quilômetros", "quilómetros", "metros", "minutos", "segundos",
        "pontos", "homens", "habitantes", "euros", "dólares", "reais", "milhões", "gols", "golos", "jogos", "votos",
        "países", "quilos",
    }
    first_day = "primeiro"
    date_link = r"\s(?:de\s)?"

    def card(self, n: int) -> str:
        return n2w(n = n, lang = "pt")

    def ordinal(self, n: int) -> str:
        return n2w(n = n, lang = "pt", to = "ordinal")

    def digits(self, digits: str) -> str:
        return " ".join(n2w(n = int(d), lang = "pt") for d in digits)

    def say(self, n, ctx, card = None):
        if n.value >= 10 ** 9:
            return Skip(reason = f"十亿以上的数在巴西和葡萄牙读法不同「{n.written}」")
        return super().say(n = n, ctx = ctx, card = card)

    def gender_sensitive(self, n: int) -> bool:
        units, thousands = n % 1000, n // 1000 % 1000
        if units % 10 in (1, 2) and units % 100 not in (11, 12):
            return True
        if units // 100 >= 2 or thousands // 100 >= 2:
            return True
        return thousands != 1 and thousands % 10 in (1, 2) and thousands % 100 not in (11, 12)


LANG = Pt
