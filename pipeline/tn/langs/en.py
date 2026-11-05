"""英语：基数、序数（21st）、年份（nineteen sixty）、年代（1990s、'90s、the 60s）、日期（October 13, 1960；
13 October 1960）、时刻、单位、货币，以及常用缩写、字母缩写和罗马数字。

1000–2099 之间不带千分位的四位数按上下文判断：前面是 in、since、the、月份名等、后面不是复数名词时读作年份；
前面是 about、over 这类表示数量的词、或后面是复数名词且前面不是介词和限定词时读作基数；其余丢弃。
「13 October 1960」读作「the thirteenth of October nineteen sixty」，原文前面已有 the 时不再加。
缩写只在读法固定时展开（etc.、e.g.、i.e.、vs.，以及后面紧跟人名的 Dr.、Mr.、St.），字母缩写读作逐个字母
（U.S. → U S）。罗马数字只处理两类：前面是君主、教皇名时读作「the Eighth」（Henry VIII），前面是 War、Part、
Chapter 这类词时读作基数（World War Two）；其余丢弃。
"""

from __future__ import annotations

import re

from ..numwords import n2w
from ..rules import UNKNOWN_UNITS, Rule, Skip, alternation
from .base import Lang, word_after, word_before
from .euro import abbreviation_rules, initialism_rule, roman_value, sentence_end_after

MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December",
)
# 前面是这些词时，四位数可以读作年份
YEAR_BEFORE = {
    "in", "since", "from", "through", "until", "till", "during", "circa", "early", "late", "mid", "year", "by",
    "of", "to", "and", "or", "between", "before", "after", "around", "as", "the", "a", "an", "his", "her",
    "its", "their", "our", "my", "your", "summer", "spring", "autumn", "fall", "winter", "act", "acts", "born",
    "died", "until", "season", "census", "for", "at", "on", "fiscal", "calendar", "model", "class", "vintage",
} | {m.lower() for m in MONTHS}
# 前面是这些词时，四位数多为数量
QUANTITY_BEFORE = {
    "over", "under", "nearly", "almost", "some", "than", "only", "just", "total", "least", "most", "exactly",
    "additional", "another", "with", "had", "has", "have", "employs", "employed", "killed", "were", "are",
    "numbered", "numbering",
}
# 表示约数的词：后面是复数名词时是数量，否则是年份（about 1715, while …）
APPROX_BEFORE = {"about", "around", "approximately", "roughly", "estimated"}
# 后面常跟年份的动词和介词
YEAR_VERBS = {
    "dated", "published", "effective", "beyond", "into", "built", "founded", "established", "released",
    "completed", "opened", "closed", "elected", "retired", "married", "onwards", "unveiled", "launched",
}
# 年份后面可以跟的复数名词（the 1998 elections）
YEAR_PLURAL_NOUNS = {
    "elections", "olympics", "games", "playoffs", "finals", "championships", "riots", "protests", "floods",
    "attacks", "awards", "rankings", "series", "midterms", "primaries", "uprisings", "talks", "negotiations",
    "earthquakes", "hurricanes", "winter", "summer", "season", "census", "miniseries", "olympics", "notes",
    "onwards", "afterwards",
}
# 不以 s 结尾的复数名词和量词
QUANTITY_AFTER = {
    "people", "men", "women", "children", "feet", "aircraft", "personnel", "staff", "cattle", "sheep", "fish",
    "deer", "square", "cubic", "more", "persons", "times", "percent", "per", "thousand", "million", "billion",
}
ORDINAL_SUFFIX = {1: "st", 2: "nd", 3: "rd"}
# 句号后常见的句首词，用来判断缩写的点是否同时是句号
STARTERS = {
    "The", "He", "She", "It", "They", "We", "I", "In", "On", "At", "This", "These", "That", "Those", "A", "An",
    "After", "Before", "During", "His", "Her", "Its", "Their", "There", "But", "However", "As", "When", "By",
    "From", "Since", "For", "With", "Although", "While", "Many", "Some", "Most", "Today", "Later", "Then",
}
ABBREVIATIONS = {
    "Mr.": ("Mister", "name"), "Mrs.": ("Missus", "name"), "Dr.": ("Doctor", "name"),
    "Prof.": ("Professor", "name"), "St.": ("Saint", "name"), "Mt.": ("Mount", "name"), "Ft.": ("Fort", "name"),
    "Gen.": ("General", "name"), "Col.": ("Colonel", "name"), "Lt.": ("Lieutenant", "name"),
    "Sgt.": ("Sergeant", "name"), "Capt.": ("Captain", "name"), "Gov.": ("Governor", "name"),
    "Sen.": ("Senator", "name"), "Rep.": ("Representative", "name"), "Rev.": ("Reverend", "name"),
    "Hon.": ("Honourable", "name"),
    "vs.": ("versus", "any"), "etc.": ("et cetera", "any"), "e.g.": ("for example", "any"),
    "i.e.": ("that is", "any"), "approx.": ("approximately", "any"), "Approx.": ("Approximately", "any"), "Ltd.": ("Limited", "any"),
    "Corp.": ("Corporation", "any"), "Bros.": ("Brothers", "any"), "Ph.D.": ("P H D", "any"),
    "a.m.": ("A M", "any"), "p.m.": ("P M", "any"),
}
# 编号缩写后接数字：No. 5 → number five
NUMBER_ABBREVIATIONS = {"No": "number", "Nos": "numbers", "Vol": "Volume", "Fig": "Figure", "Nr": "number"}
# 前面是这些名字时，罗马数字是君主、教皇的序号（Henry the Eighth）
REGNAL_NAMES = {
    "Henry", "Edward", "George", "William", "Richard", "James", "Charles", "Elizabeth", "Mary", "Louis",
    "Philip", "Frederick", "John", "Paul", "Pius", "Benedict", "Gregory", "Innocent", "Leo", "Clement",
    "Alexander", "Nicholas", "Peter", "Ivan", "Catherine", "Napoleon", "Ramesses", "Ramses", "Ptolemy",
    "Amenhotep", "Thutmose", "Ferdinand", "Alfonso", "Otto", "Wilhelm", "Gustav", "Gustavus", "Christian",
    "Haakon", "Olav", "Harald", "Frederik", "Rama", "Constantine", "Michael", "Manuel", "Basil", "Justinian",
    "Theodosius", "Valentinian", "Selim", "Mehmed", "Murad", "Suleiman", "Mustafa", "Sancho", "Pedro",
    "Stephen", "Ladislaus", "Casimir", "Sigismund", "Vladimir", "Victor", "Umberto", "Leopold", "Joseph",
    "Francis", "Rudolf", "Albert", "Baldwin", "Boniface", "Urban", "Sixtus", "Julius", "Adrian", "Celestine",
    "Martin", "Eugene", "Felix", "Honorius", "Anastasius", "Hadrian", "Darius", "Xerxes", "Artaxerxes",
    "Antiochus", "Seleucus", "Herod", "Tiglath", "Shalmaneser", "Sargon", "Nebuchadnezzar", "Hussein",
    "Abdullah", "Faisal", "Farouk", "Fuad", "Hassan", "Mohammed", "Muhammad", "Bayezid", "Osman", "Abdulhamid",
    "Malcolm", "Duncan", "Kenneth", "Robert", "David", "Alexander", "Llywelyn", "Canute", "Cnut", "Eric",
    "Magnus", "Olaf", "Valdemar", "Waldemar", "Margaret", "Anne", "Victoria", "Isabella", "Isabel", "Juan",
    "Carlos", "Felipe", "Fernando", "Jaime", "Afonso", "João", "Sebastian", "Manuel", "Rainier", "Albert",
    "Wenceslaus", "Ottokar", "Vratislaus", "Bolesław", "Władysław", "Mieszko", "Andrew", "Béla", "Géza",
    "Stefan", "Simeon", "Tsar", "Kamehameha", "Akhenaten", "Seti", "Sobekhotep", "Mentuhotep", "Senusret",
    "Amenemhat", "Psamtik", "Nectanebo", "Shoshenq", "Osorkon", "Takelot", "Ardashir", "Kavad", "Conrad",
    "Khosrow", "Shapur", "Hormizd", "Bahram", "Yazdegerd", "Peroz", "Donald", "Andronicus", "Andronikos", "Vasily",
    "Humphrey", "Bohemond", "Raymond", "Guy", "Amalric", "Tancred", "Roger", "Bela", "Andrew", "Coloman",
    "Lothair", "Pepin", "Carloman", "Arnulf", "Berengar", "Ludwig", "Friedrich", "Heinrich", "Karl", "Maximilian",
}
# 前面是这些词时，罗马数字读作基数（World War Two、Part Two）
ROMAN_CARDINAL_BEFORE = {
    "War", "Part", "Chapter", "Volume", "Book", "Act", "Phase", "Stage", "Type", "Class", "Title", "Article",
    "Section", "Bowl", "Vatican", "Grade", "Level", "Tier", "Category", "Appendix", "Annex", "Canto", "Psalm",
    "Division", "Camp", "Mark", "Convention", "WrestleMania", "Chronicles", "Panzer", "Schedule",
}
# 后面是这些词时，罗马数字读作序数（the III Corps → the Third Corps）
ROMAN_ORDINAL_AFTER = {"Corps"}


def ordinal_suffix(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return "th"
    return ORDINAL_SUFFIX.get(n % 10, "th")


def pluralize(words: str) -> str:
    """年份读法的最后一个词变复数：nineteen ninety → nineteen nineties。"""
    head, _, last = words.rpartition(" ")
    if last.endswith("y"):
        last = last[:-1] + "ies"
    else:
        last += "s"
    return (head + " " if head else "") + last


def is_plural_noun(word: str) -> bool:
    """小写、以 s 结尾的词大多是复数名词（people 这类不规则复数另列）。"""
    if not word or not word[0].islower():
        return False
    if word in QUANTITY_AFTER:
        return True
    return len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")) and word not in YEAR_PLURAL_NOUNS


class En(Lang):
    code = "en"
    point = "point"
    percent = "percent"
    units = {
        "km/h": ("kilometer per hour", "kilometers per hour"), "mph": ("mile per hour", "miles per hour"),
        "km²": ("square kilometer", "square kilometers"), "m²": ("square meter", "square meters"),
        "km": ("kilometer", "kilometers"), "cm": ("centimeter", "centimeters"), "mm": ("millimeter", "millimeters"),
        "kg": ("kilogram", "kilograms"), "m": ("meter", "meters"), "ft": ("foot", "feet"), "mi": ("mile", "miles"),
        "lb": ("pound", "pounds"), "lbs": ("pound", "pounds"),
        "°C": ("degree Celsius", "degrees Celsius"), "℃": ("degree Celsius", "degrees Celsius"),
        "°F": ("degree Fahrenheit", "degrees Fahrenheit"), "℉": ("degree Fahrenheit", "degrees Fahrenheit"),
    }
    currencies = {"$": ("dollar", "dollars"), "€": ("euro", "euros"), "£": ("pound", "pounds")}
    scales = ("thousand", "million", "billion", "trillion")

    def card(self, n: int) -> str:
        return n2w(n = n, lang = "en")

    def attributive(self, m, ctx) -> bool:
        """前面是 a/an 时金额、单位作定语：a $70 million impact → a seventy million dollar impact。"""
        return word_before(text = ctx.text, start = m.start()) in ("a", "an")

    def unit_rules(self) -> list:
        rules = super().unit_rules()
        for rule in rules:
            rule.handle = self.singular_after_article(handle = rule.handle, table = self.units)
        return rules

    def currency_rules(self) -> list:
        rules = super().currency_rules()
        for rule in rules:
            rule.handle = self.singular_after_article(handle = rule.handle, table = self.currencies)
        return rules

    def singular_after_article(self, handle, table: dict):
        def wrapped(m, ctx):
            out = handle(m = m, ctx = ctx)
            if isinstance(out, str) and self.attributive(m = m, ctx = ctx):
                for singular, plural in table.values():
                    if out.endswith(" " + plural):
                        return out[:-len(plural)] + singular
            return out

        return wrapped

    def digits(self, digits: str) -> str:
        return " ".join(n2w(n = int(d), lang = "en") for d in digits)

    def year(self, n: int) -> str:
        return n2w(n = n, lang = "en", to = "year")

    # ------------------------------------------------------------------
    # 规则
    # ------------------------------------------------------------------

    def rules(self) -> list:
        """缩写、年代简写和 mid-1960s 要在公共保护规则之前处理（保护规则会把 '90、mid-1960 当作撇号年份和型号）。"""
        early = [
            Rule(pattern = re.compile(pattern = r"['’](?P<d>[1-9]0)s\b"), handle = self.short_decade),
            Rule(pattern = re.compile(pattern = r"\bmid-(?P<num>1\d{3}|20\d{2})(?P<s>s)?\b"), handle = self.mid_year),
            Rule(pattern = re.compile(pattern = r"(?<![\w.])c\.\s?(?P<num>\d{3,4})(?![\d]|[.,]\d)"), handle = self.circa),
        ]
        early += abbreviation_rules(table = ABBREVIATIONS, starters = STARTERS)
        early.append(Rule(pattern = re.compile(pattern = r"(?<![\w.])(?:Jr|Sr)\.(?![^\W\d_])"), handle = self.junior))
        early.append(initialism_rule(starters = STARTERS))
        early.append(Rule(
            pattern = re.compile(
                pattern = rf"(?<![\w.])(?P<abbr>{alternation(words = NUMBER_ABBREVIATIONS)})\.\s?(?P<num>\d+)(?![\d]|[.,]\d)",
            ),
            handle = self.numbered,
        ))
        early.append(Rule(pattern = re.compile(pattern = r"(?<=\s)&(?=\s)"), handle = self.ampersand))
        early.append(Rule(pattern = re.compile(pattern = r"(?<![\w])(?P<roman>[IVXLCDM]{2,})(?![\w])"), handle = self.roman))
        return early + super().rules()

    def extra_rules(self) -> list:
        months = alternation(words = MONTHS)
        return [
            Rule(
                pattern = re.compile(pattern = r"\b(?:at|by|from|until|till|before|after)\s\d{1,2}\.\d{2}(?!\d)"),
                handle = lambda m, ctx: Skip(reason = f"时刻写法（7.30）「{m.group()}」"),
            ),
            Rule(
                pattern = re.compile(
                    pattern = (
                        rf"(?P<month>{months})\s(?P<day>\d{{1,2}})(?P<suf>st|nd|rd|th)?"
                        rf"(?:(?P<sep>,?\s)(?P<year>\d{{4}}))?(?!\d|[.,]\d)"
                    ),
                ),
                handle = self.month_day,
            ),
            Rule(
                pattern = re.compile(
                    pattern = (
                        rf"(?<![\d.,])(?P<day>\d{{1,2}})(?P<suf>st|nd|rd|th)?\s(?P<of>of\s)?(?P<month>{months})\b"
                        rf"(?:(?P<sep>,?\s)(?P<year>\d{{4}})(?!\d|[.,]\d))?"
                    ),
                ),
                handle = self.day_month,
            ),
            Rule(
                pattern = re.compile(
                    pattern = r"(?P<cur>[$€£])(?P<num>\d+\.\d+)\s(?P<scale>million|billion|trillion)\b",
                ),
                handle = self.decimal_money,
            ),
            Rule(pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>\d+)(?P<suf>st|nd|rd|th)\b"), handle = self.ordinal),
            Rule(pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>1\d{2}0|20\d0)s\b"), handle = self.decade),
            Rule(pattern = re.compile(pattern = r"(?<![\d.,'’])(?P<d>[2-9]0)s\b"), handle = self.bare_decade),
            Rule(pattern = re.compile(pattern = r"(?<![\d.,])(?P<h>\d{1,2}):(?P<mm>\d{2})(?![\d:])"), handle = self.time),
            Rule(pattern = re.compile(pattern = rf"(?P<num>{self.num})-(?P<word>[a-z]+)"), handle = self.compound),
            Rule(pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>\d{4})(?![\d]|[.,]\d)"), handle = self.four_digits),
        ]

    # ------------------------------------------------------------------
    # 缩写、符号、罗马数字
    # ------------------------------------------------------------------

    def junior(self, m, ctx):
        """Jr.、Sr. 只在人名之后展开（Martin Luther King Jr. → Junior）。"""
        before = ctx.text[:m.start()].rstrip().rstrip(",")
        if not re.search(pattern = r"\b[A-Z][a-z]+$", string = before):
            return Skip(reason = f"缩写读法取决于上下文「{m.group()}」")
        spoken = "Junior" if m.group().startswith("J") else "Senior"
        if sentence_end_after(text = ctx.text, end = m.end(), starters = STARTERS):
            spoken += "."
        ctx.kind(name = "缩写", surface = m.group())
        return spoken

    def numbered(self, m, ctx):
        n = int(m.group("num"))
        if m.group("num").startswith("0"):
            return Skip(reason = f"编号写法无法确定「{m.group()}」")
        ctx.kind(name = "缩写", surface = m.group("abbr") + ".")
        return NUMBER_ABBREVIATIONS[m.group("abbr")] + " " + ctx.num(spoken = self.card(n = n), written = m.group("num"))

    def ampersand(self, m, ctx):
        ctx.kind(name = "符号", surface = "&")
        return "and"

    def roman(self, m, ctx):
        token = m.group("roman")
        value = roman_value(text = token)
        # 与 detect 一致：两个字母的 DC、MD、CD 等不算罗马数字
        if value is None or (len(token) < 3 and not set(token) <= set("IVXL")):
            return None
        before = re.search(pattern = r"([A-Za-z][a-z]+)\s$", string = ctx.text[:m.start()])
        prev = before.group(1) if before else ""
        prev = prev[:1].upper() + prev[1:] if prev in ("grade", "division", "part", "type", "class") else prev
        if word_after(text = ctx.text, end = m.end()) in ROMAN_ORDINAL_AFTER:
            ctx.kind(name = "罗马数字", surface = m.group())
            ordinal = ctx.num(spoken = n2w(n = value, lang = "en", to = "ordinal"), written = str(value))
            return ordinal[:1].upper() + ordinal[1:]
        if prev in REGNAL_NAMES:
            ctx.kind(name = "罗马数字", surface = f"{prev} {m.group()}")
            ordinal = ctx.num(spoken = n2w(n = value, lang = "en", to = "ordinal"), written = str(value))
            return "the " + ordinal[:1].upper() + ordinal[1:]
        if prev in ROMAN_CARDINAL_BEFORE:
            ctx.kind(name = "罗马数字", surface = f"{prev} {m.group()}")
            words = ctx.num(spoken = self.card(n = value), written = str(value))
            return words[:1].upper() + words[1:]
        return Skip(reason = f"罗马数字的读法取决于上下文「{m.group()}」")

    # ------------------------------------------------------------------
    # 年份、年代、日期
    # ------------------------------------------------------------------

    def short_decade(self, m, ctx):
        """'90s → nineties。"""
        ctx.kind(name = "年代", surface = m.group())
        return ctx.num(spoken = pluralize(words = self.card(n = int(m.group("d")))), written = m.group("d") + "s")

    def bare_decade(self, m, ctx):
        """the 60s → the sixties（只在 the、early、late 等词之后）。"""
        if word_before(text = ctx.text, start = m.start()) not in ("the", "early", "late", "mid", "his", "her", "their", "its"):
            return None
        ctx.kind(name = "年代", surface = m.group())
        return ctx.num(spoken = pluralize(words = self.card(n = int(m.group("d")))), written = m.group("d") + "s")

    def mid_year(self, m, ctx):
        words = self.year(n = int(m.group("num")))
        if m.group("s"):
            if not m.group("num").endswith("0"):
                return Skip(reason = f"年代写法无法确定「{m.group()}」")
            words = pluralize(words = words)
        ctx.kind(name = "年代" if m.group("s") else "年份", surface = m.group())
        return "mid-" + ctx.num(spoken = words, written = m.group("num") + (m.group("s") or ""))

    def circa(self, m, ctx):
        n = int(m.group("num"))
        if not 100 <= n <= 2099:
            return Skip(reason = f"缩写读法取决于上下文「{m.group()}」")
        ctx.kind(name = "缩写", surface = "c.")
        ctx.kind(name = "年份", surface = m.group("num"))
        return "circa " + ctx.num(spoken = self.year(n = n) if n >= 1000 else self.card(n = n), written = m.group("num"))

    def month_day(self, m, ctx):
        day = int(m.group("day"))
        if not 1 <= day <= 31 or (m.group("suf") and m.group("suf") != ordinal_suffix(n = day)):
            return Skip(reason = f"日期写法无法确定「{m.group()}」")
        out = m.group("month") + " " + ctx.num(spoken = n2w(n = day, lang = "en", to = "ordinal"), written = f"{day}th")
        if m.group("year"):
            year = int(m.group("year"))
            out += m.group("sep") + ctx.num(spoken = self.year(n = year), written = m.group("year"))
        ctx.kind(name = "日期", surface = m.group())
        return out

    def day_month(self, m, ctx):
        """13 October 1960 → the thirteenth of October nineteen sixty。"""
        day = int(m.group("day"))
        if not 1 <= day <= 31 or (m.group("suf") and m.group("suf") != ordinal_suffix(n = day)):
            return Skip(reason = f"日期写法无法确定「{m.group()}」")
        before = ctx.text[:m.start()]
        # 「3 and 5 May」「3–5 May」：前一个日子要读成序数，暂不处理
        if re.search(pattern = r"\d(?:st|nd|rd|th)?\s*(?:and|or|to|&|,|[-–—])\s*$", string = before):
            return Skip(reason = f"并列或范围的日期「{m.group()}」")
        prev = word_before(text = ctx.text, start = m.start())
        lead = "" if prev == "the" else "the "
        out = lead + ctx.num(spoken = n2w(n = day, lang = "en", to = "ordinal"), written = f"{day}th")
        out += " of " + m.group("month")
        if m.group("year"):
            out += m.group("sep") + ctx.num(spoken = self.year(n = int(m.group("year"))), written = m.group("year"))
        ctx.kind(name = "日期", surface = m.group())
        return out

    def decimal_money(self, m, ctx):
        """$1.1 billion → one point one billion dollars。"""
        n = self.parse(written = m.group("num"))
        if n is None:
            return Skip(reason = f"数字写法无法确定「{m.group()}」")
        forms = self.currencies[m.group("cur")]
        following = ctx.text[m.end():m.end() + 30].lower()
        if any(form in following for form in forms):
            return Skip(reason = f"货币符号与货币名重复「{m.group()}」")
        ctx.kind(name = "货币", surface = m.group())
        word = forms[0] if self.attributive(m = m, ctx = ctx) else forms[1]
        return self.join(self.say(n = n, ctx = ctx), m.group("scale"), word)

    def ordinal(self, m, ctx):
        n = int(m.group("num"))
        if m.group("suf") != ordinal_suffix(n = n) or (len(m.group("num")) > 1 and m.group("num").startswith("0")):
            return Skip(reason = f"序数写法无法确定「{m.group()}」")
        ctx.kind(name = "序数", surface = m.group())
        return ctx.num(spoken = n2w(n = n, lang = "en", to = "ordinal"), written = m.group())

    def decade(self, m, ctx):
        n = int(m.group("num"))
        ctx.kind(name = "年代", surface = m.group())
        return ctx.num(spoken = pluralize(words = self.year(n = n)), written = m.group())

    def time(self, m, ctx):
        h, mm = int(m.group("h")), int(m.group("mm"))
        if h > 24 or mm > 59:
            return Skip(reason = f"时刻写法无法确定「{m.group()}」")
        if mm == 0 and not 1 <= h <= 12:
            return Skip(reason = f"整点的 24 小时制读法不固定「{m.group()}」")
        ctx.kind(name = "时刻", surface = m.group())
        hour = ctx.num(spoken = self.card(n = h), written = m.group("h"))
        if mm == 0:
            return hour + " o'clock"
        minute = self.card(n = mm) if mm >= 10 else "oh " + self.card(n = mm)
        return hour + " " + ctx.num(spoken = minute, written = m.group("mm"))

    def compound(self, m, ctx):
        """24-year-old → twenty-four-year-old。"""
        n = self.parse(written = m.group("num"))
        if n is None or not n.is_int or 1000 <= n.value <= 2099:
            return None
        ctx.kind(name = "基数", surface = m.group())
        return self.say(n = n, ctx = ctx) + "-" + m.group("word")

    def four_digits(self, m, ctx):
        """1000–2099 的四位数：年份、数量，或因上下文不明而丢弃。"""
        n = int(m.group("num"))
        if not 1000 <= n <= 2099:
            return None
        text = ctx.text
        rest = text[m.end():]
        # 后面紧跟单位、百分号，或前面是货币符号：交给单位、百分数、货币规则
        if re.match(pattern = rf"\s?(?:{alternation(words = UNKNOWN_UNITS)}|mph|lbs?)(?![A-Za-z])|\s?[%‰]", string = rest):
            return None
        if text[:m.start()].rstrip()[-1:] and text[:m.start()].rstrip()[-1] in "$€£":
            return None
        before = word_before(text = text, start = m.start())
        after = word_after(text = text, end = m.end())
        lead = text[:m.start()].rstrip()
        starts = not lead or lead[-1] in ".!?(\"“,;:—–"
        plural_after = is_plural_noun(word = after) and not rest.lstrip().startswith(("-", "–"))
        prev_proper = re.search(pattern = r"(?:^|\s)[A-Z][a-z]+$", string = lead) is not None and not starts
        possessive = re.search(pattern = r"\w['’]s$", string = lead) is not None
        if plural_after:
            year_like = starts or before in YEAR_BEFORE or possessive or prev_proper
            if before in QUANTITY_BEFORE or before in APPROX_BEFORE or not year_like:
                ctx.kind(name = "基数", surface = m.group())
                return ctx.num(spoken = self.card(n = n), written = m.group())
            return Skip(reason = f"四位数是年份还是数量需要上下文「{m.group()} {after}」")
        if before in QUANTITY_BEFORE:
            return Skip(reason = f"四位数是年份还是数量需要上下文「{before} {m.group()}」")
        year_like = (
            starts or before in YEAR_BEFORE or before in APPROX_BEFORE or before in YEAR_VERBS or possessive
            or prev_proper or (before.endswith(".") and before[:-1] in YEAR_BEFORE)
        )
        if year_like:
            ctx.kind(name = "年份", surface = m.group())
            return ctx.num(spoken = self.year(n = n), written = m.group())
        return Skip(reason = f"四位数是年份还是数量需要上下文「{m.group()}」")

LANG = En
