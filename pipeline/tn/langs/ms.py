"""马来语（马来西亚）：规则同印尼语，数词 8 为 lapan、10^9 为 bilion，小数点读 perpuluhan；
千分位与小数点和英语相同（1,000.5）。年份按完整数值读（seribu sembilan ratus enam puluh tujuh）。
时刻「pukul 7.30」读 pukul tujuh tiga puluh；负数的读法（negatif、minus、tolak）不统一，不改写。
"""

from __future__ import annotations

from .id import Id


class Ms(Id):
    code = "ms"
    point = "perpuluhan"
    percent = "peratus"
    minus = None
    units = {
        "km/h": ("kilometer sejam", "kilometer sejam"), "km²": ("kilometer persegi", "kilometer persegi"),
        "m²": ("meter persegi", "meter persegi"), "km": ("kilometer", "kilometer"), "cm": ("sentimeter", "sentimeter"),
        "mm": ("milimeter", "milimeter"), "kg": ("kilogram", "kilogram"), "m": ("meter", "meter"),
        "g": ("gram", "gram"), "°C": ("darjah Celsius", "darjah Celsius"), "℃": ("darjah Celsius", "darjah Celsius"),
        "°F": ("darjah Fahrenheit", "darjah Fahrenheit"), "°": ("darjah", "darjah"),
    }
    currencies = {"$": ("dolar", "dolar"), "€": ("euro", "euro"), "RM": ("ringgit", "ringgit"), "Rp": ("rupiah", "rupiah")}
    scales = ("ribu", "juta", "bilion", "trilion")


LANG = Ms
