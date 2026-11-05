"""俄语：数词的格、性、数由后面的名词和前面的介词决定，名词的词形用 pymorphy3 分析。

规则（任何一处无法唯一确定词形时整条丢弃）：

1. 基数加名词：按数的末位分三类（1；2–4；其余），与名词的格和数对照。名词是单数属格、数以 2–4 结尾时
   数用主格（два дома、две книги）；名词是复数与格、工具格、前置格时数用同一个格（в пяти домах、
   с двадцатью одним человеком）；名词是复数属格时，2–4 结尾的数用属格（двух студентов），其余的数既可能
   是主格也可能是属格，由前面的词决定：支配属格的介词或比较词（до、более、около）用属格，宾格介词
   （на、за、через、в）、动词、句首用主格（погибли 25 человек → двадцать пять человек）。
2. 年份：后接 год 的各种形式，或 г.、гг. 缩写（в 1990 г. → в тысяча девятьсот девяностом году），
   格由 год 的词形或前面的介词决定；年代写法（1990-х）同样。
3. 日期（13 сентября 1990 года）：日读作中性序数，没有介词时用属格（тринадцатого сентября），
   к 用与格，на、по 用宾格。
4. 带词尾的序数（5-й、1-го、23-м、1990-х）：与后面名词的性、数、格一致，词尾必须与读法相符。
5. 罗马数字：后接名词时读作与名词一致的序数（XX век、в XIX веке、III степени），后接 в.、вв. 时按介词
   选择 век 的格；前面是人名时读作与人名同格的序数（Пётр I → Пётр Первый，Петра I → Петра Первого）。
6. 单位、百分号和数量缩写（км、кг、%、тыс.、млн、руб.、чел.）：格由前面的词决定，读成相应的名词形式。
7. 常用缩写：т. е.、т. д.、т. п.、н. э.、им.、ок.。

小数的格变化暂不处理。依赖 pymorphy3（MIT，词典数据来自 OpenCorpora，CC BY-SA）。
"""

from __future__ import annotations

import re
from functools import lru_cache

from ..numwords import n2w
from ..rules import Rule, Skip, alternation
from .base import Lang, word_before
from .euro import (
    abbreviation_rules, dotted_decimal_guard, number_word_guard, number_words, roman_value, sentence_end_after,
)

CASE_NAMES = {
    "nomn": "nominative", "gent": "genitive", "datv": "dative", "accs": "accusative", "ablt": "instrumental",
    "loct": "prepositional",
}
CASE_ALIAS = {"gen2": "gent", "loc2": "loct", "acc2": "accs", "gen1": "gent", "loc1": "loct", "voct": "nomn"}
GENDERS = {"masc": "m", "femn": "f", "neut": "n", "ms-f": "m"}

# год 的词形 → (num2words 的格, 是否复数)
GOD_FORMS = {
    "году": (None, False), "года": ("genitive", False), "год": ("nominative", False),
    "годом": ("instrumental", False), "годах": ("prepositional", True), "годов": ("genitive", True),
    "годы": ("nominative", True), "годам": ("dative", True), "годами": ("instrumental", True),
}
PREPOSITIONAL_BEFORE = {"в", "во", "о", "об", "при"}
DATIVE_BEFORE = {"к", "ко", "по"}
# 支配属格的介词、比较词和名词
GENITIVE_BEFORE = {
    "более", "менее", "около", "ок.", "свыше", "от", "до", "из", "без", "для", "после", "кроме", "среди", "у",
    "порядка", "больше", "меньше", "вокруг", "против", "вместо", "возле", "мимо", "накануне", "ради", "вдоль",
    "внутри", "вне", "сверх", "посреди", "из-за", "из-под", "начала", "конца", "середины", "половины",
    "течение", "возрасте", "протяжении", "числа", "количестве", "размере", "свыше", "старше", "младше",
    "выше", "ниже", "дольше", "раньше", "позже", "позднее", "ранее", "начале", "конце", "середине",
    "половине", "рубеже", "начало", "конец", "середина", "составе",
}
# 后面是复数属格名词时，前面这些词表示数用主格（宾格同主格）
ACCUSATIVE_BEFORE = {"на", "за", "через", "про", "сквозь", "в", "во", "под"}
# 数前可以跳过的副词、语气词
TRANSPARENT = {
    "всего", "лишь", "только", "почти", "примерно", "приблизительно", "ещё", "еще", "уже", "также", "ровно",
    "целых", "минимум", "максимум", "как", "даже", "именно", "лишь", "уж", "всё", "все",
}
# 支配属格的动词（достигает 25 метров → двадцати пяти метров）
GENITIVE_VERBS = {
    "достигать", "достичь", "достигнуть", "лишиться", "лишаться", "избежать", "избегать", "бояться", "касаться",
    "коснуться",
}
MONTHS_GEN = (
    "января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября",
    "декабря",
)
ABBREVIATIONS = {
    "т. е.": ("то есть", "any"), "т.е.": ("то есть", "any"), "т. д.": ("так далее", "any"),
    "т.д.": ("так далее", "any"), "т. п.": ("тому подобное", "any"), "т.п.": ("тому подобное", "any"),
    "н. э.": ("нашей эры", "any"), "н.э.": ("нашей эры", "any"), "им.": ("имени", "name"),
    "ок.": ("около", "number"),
}
# 君主、教皇名（原形）→ 性。人名的格由 pymorphy3 分析，只取原形在此表中的分析
REGNAL_LEMMAS = {
    name: "m" for name in """
    пётр павел николай александр иван василий фёдор алексей михаил людовик генрих карл фридрих вильгельм
    георг эдуард ричард яков филипп фердинанд франциск франц иоанн лев пий бенедикт григорий иннокентий
    климент урбан бонифаций сикст юлий адриан хуан педро альфонсо рамзес птолемей тутмос аменхотеп сети
    дарий ксеркс артаксеркс антиох селевк митридат ирод константин юстиниан феодосий валентиниан мехмед
    мурад селим сулейман ахмед мустафа густав кристиан фредерик хокон олаф харальд эрик магнус вальдемар
    сигизмунд казимир владислав болеслав стефан андрей бела иосиф леопольд рудольф альбрехт оттон конрад
    лотарь людвиг максимилиан вацлав пржемысл оттокар хетум левон тигран ашот баграт давид георгий ираклий
    хосров шапур ардашир кавад бахрам йездегерд пероз ормизд мануил андроник роман лжедмитрий ярослав
    изяслав святослав всеволод мстислав владимир юрий даниил дмитрий борис симеон кирилл
    """.split()
} | {name: "f" for name in "екатерина елизавета анна мария виктория изабелла тамара клеопатра маргарита".split()}
STARTERS = {
    "В", "На", "С", "По", "Он", "Она", "Они", "Это", "Его", "Её", "Их", "После", "До", "Для", "При", "Также",
    "Кроме", "Однако", "Но", "И", "А", "К", "Из", "От", "За", "Через", "Сейчас", "Позже", "Затем", "Тогда",
}
# 单位和数量缩写 → 名词原形
UNIT_LEMMAS = {
    "км": "километр", "м": "метр", "см": "сантиметр", "мм": "миллиметр", "кг": "килограмм", "т": "тонна",
    "га": "гектар", "тыс.": "тысяча", "млн": "миллион", "млн.": "миллион", "млрд": "миллиард",
    "млрд.": "миллиард", "руб.": "рубль", "чел.": "человек", "%": "процент",
}


@lru_cache(maxsize = 1)
def analyzer():
    try:
        import pymorphy3
    except ImportError as exc:
        raise RuntimeError("俄语读法需要 pymorphy3（MIT）：uv pip install pymorphy3") from exc
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize = 50000)
def parses(word: str) -> tuple:
    """词的分析结果：(词性, 格, 数, 性, 有生, 得分, 是否专名, 原形)。"""
    out = []
    for p in analyzer().parse(word = word):
        tag = p.tag
        case = CASE_ALIAS.get(tag.case, tag.case)
        proper = bool({"Name", "Surn", "Patr", "Geox", "Orgn"} & tag.grammemes)
        out.append((tag.POS, case, tag.number, GENDERS.get(tag.gender), tag.animacy == "anim", p.score, proper, p.normal_form))
    return tuple(out)


def noun_parses(word: str) -> list:
    """名词分析（得分不低于 0.1；小写词不取人名、地名分析）。"""
    out = []
    if len(word) < 3:
        # г、м、т 这类单字母词多为缩写（1934 г），不作名词
        return out
    for pos, case, number, gender, anim, score, proper, _ in parses(word = word):
        if pos != "NOUN" or score < 0.1 or case is None:
            continue
        if proper and word[:1].islower():
            continue
        out.append((case, number, gender, anim))
    return out


def category(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return "one"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return "few"
    return "many"


def card_form(n: int, case: str, gender: str = "m", animate: bool = False) -> str:
    text = n2w(
        n = n,
        lang = "ru",
        case = CASE_NAMES[case],
        gender = gender or "m",
        animate = animate,
    )
    # одна тысяча пятьсот → тысяча пятьсот，одного миллиона → миллиона
    return re.sub(pattern = r"^(?:один|одна|одно|одного|одной|одному|одним|одном|одну|одною)\s(?=тысяч|миллион|миллиард)", repl = "", string = text)


def ord_form(n: int, case: str, gender: str = "m", plural: bool = False, animate: bool = False) -> str:
    return n2w(
        n = n,
        lang = "ru",
        to = "ordinal",
        case = CASE_NAMES[case],
        gender = gender or "m",
        plural = plural,
        animate = animate,
    )


def is_verb(word: str) -> str | None:
    """词是动词（含不定式、短尾形动词、副动词）时返回其原形。"""
    for pos, _, _, _, _, score, _, lemma in parses(word = word.lower()):
        if pos in ("VERB", "INFN", "PRTS", "GRND") and score >= 0.3:
            return lemma
    return None


class Ru(Lang):
    code = "ru"

    def card(self, n: int) -> str:
        return card_form(n = n, case = "nomn")

    # ------------------------------------------------------------------
    # 上下文
    # ------------------------------------------------------------------

    def tokens_before(self, text: str, start: int) -> list:
        """数前面的词和标点（最近的在前），到句首为止。"""
        head = text[:start]
        cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
        if cut >= 0:
            head = head[cut + 2:]
        return [t.lower() for t in reversed(re.findall(pattern = r"[^\W\d_]+(?:-[^\W\d_]+)?\.?|[,—–:(;]", string = head))]

    def context_case(self, text: str, start: int, strict: bool = False) -> str | None:
        """复数属格名词前的数用主格还是属格：介词、比较词、动词、句首决定，无法判断时返回 None。"""
        tokens = self.tokens_before(text = text, start = start)
        while tokens and tokens[0] in TRANSPARENT:
            tokens = tokens[1:]
        if not tokens or tokens[0] in ("—", "–", ":", "("):
            return "nomn"
        w = tokens[0]
        if w in GENITIVE_BEFORE or w in ("с", "со"):
            return "gent"
        if w in ACCUSATIVE_BEFORE:
            return None if strict and w in ("в", "во") else "nomn"
        if w in (",", ";", "и", "а", "или", "но", "не"):
            return None
        lemma = is_verb(word = w)
        if lemma:
            if len(tokens) > 1 and tokens[1] == "не":
                return None
            return "gent" if lemma in GENITIVE_VERBS else "nomn"
        return None

    def noun_after(self, text: str, end: int) -> str | None:
        """数后面的名词，最多跳过两个形容词或形动词（25 новых домов）。"""
        rest = text[end:]
        words = re.findall(pattern = r"^\s+([а-яёА-ЯЁ]+)(?:\s+([а-яёА-ЯЁ]+))?(?:\s+([а-яёА-ЯЁ]+))?", string = rest)
        if not words:
            return None
        for w in words[0]:
            if not w:
                return None
            kinds = {pos for pos, _, _, _, _, score, _, _ in parses(word = w.lower()) if score >= 0.1}
            if "NOUN" in kinds:
                return w
            if not kinds <= {"ADJF", "PRTF"}:
                return None
        return None

    # ------------------------------------------------------------------
    # 规则
    # ------------------------------------------------------------------

    def rules(self) -> list:
        words = set()
        for case in CASE_NAMES:
            words |= number_words(card = lambda n, case = case: card_form(n = n, case = case))
        early = [dotted_decimal_guard(), number_word_guard(words = words, conjunctions = ("и",))]
        early += abbreviation_rules(table = ABBREVIATIONS, starters = STARTERS) + [
            Rule(
                pattern = re.compile(pattern = r"(?<![\w.])см\.(?=\s+[а-яА-Я])"),
                handle = self.see,
            ),
            Rule(
                pattern = re.compile(
                    pattern = (
                        r"(?<![\w.])(?P<roman>[IVXLCDM]+)(?P<chain>(?:\s?[,и]\s?(?:и\s)?[IVXLCDM]+)*)"
                        r"(?:\s(?P<abbr>вв?\.)|(?=\s+[а-яё]|\s*$|[\s,.;:)]))"
                    ),
                ),
                handle = self.roman,
            ),
        ]
        return early + super().rules()

    def extra_rules(self) -> list:
        forms = alternation(words = GOD_FORMS)
        months = alternation(words = MONTHS_GEN)
        units = alternation(words = UNIT_LEMMAS)
        return [
            Rule(
                pattern = re.compile(
                    pattern = r"(?<![\w.])(?P<pre>[Сс]о?)\s(?P<num>\d{3,4})\sпо\s(?P<num2>\d{3,4})\s(?P<god>год|годы)(?![а-яё])",
                ),
                handle = self.year_span,
            ),
            Rule(
                pattern = re.compile(
                    pattern = (
                        rf"(?<![\d.,])(?:(?P<num>\d{{3,4}})(?P<suf>-(?:х|е|м|ми|го|му|й))?\s(?P<god>{forms})"
                        rf"|(?<=[Вв]\s)(?P<short>\d{{1,2}})\s(?P<god2>году))(?![а-яё])"
                    ),
                ),
                handle = self.year,
            ),
            Rule(
                pattern = re.compile(
                    pattern = r"(?<![\d.,])(?P<num>\d{3,4})(?P<suf>-(?:х|е|м|ми))?\s?(?P<abbr>гг?\.)(?![а-яё])",
                ),
                handle = self.year_abbreviation,
            ),
            Rule(pattern = re.compile(pattern = r"№\s?(?P<num>\d+)(?![\d]|[.,]\d)"), handle = self.numero),
            Rule(
                pattern = re.compile(
                    pattern = (
                        rf"(?<![\d.,])(?P<day>\d{{1,2}})(?:\s(?P<link>и|или|по)\s(?P<day2>\d{{1,2}}))?\s(?P<month>{months})"
                        rf"(?:\s(?P<year>\d{{4}})(?![\d.,]|\s?(?:г\b|гг\b|года|году|год)))?(?![а-яё])"
                    ),
                ),
                handle = self.date,
            ),
            Rule(
                pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>\d+)-(?P<suf>летн[а-яё]+|летие|летия|летию|летием)\b"),
                handle = self.years_compound,
            ),
            Rule(
                pattern = re.compile(
                    pattern = r"(?<![\d.,])(?P<num>\d+)-(?P<suf>[а-яё]{1,3})(?![а-яё-])",
                ),
                handle = self.ordinal_suffix,
            ),
            Rule(
                pattern = re.compile(pattern = rf"(?<![\d.,])(?P<num>\d+(?:[ ]\d{{3}})*)\s?(?P<unit>{units})(?![а-яё\w])"),
                handle = self.unit,
            ),
        ]

    def percent_rules(self) -> list:
        """百分号由 unit 规则处理（25 % → двадцать пять процентов）。"""
        return []

    # ------------------------------------------------------------------
    # 缩写
    # ------------------------------------------------------------------

    def see(self, m, ctx):
        """см. 后接文字时是 смотри（数字后的 см 是厘米，由单位规则处理）。"""
        ctx.kind(name = "缩写", surface = "см.")
        return "смотри"

    # ------------------------------------------------------------------
    # 年份
    # ------------------------------------------------------------------

    def year(self, m, ctx):
        """后接 год 的年份；一两位数只在「в 59 году」中读作年份（54 года 多为年数）。"""
        if m.group("short"):
            spoken = ord_form(n = int(m.group("short")), case = "loct")
            ctx.num(spoken = spoken, written = m.group("short"))
            ctx.kind(name = "年份", surface = m.group())
            return spoken + " году"
        n = int(m.group("num"))
        case, plural = GOD_FORMS[m.group("god")]
        cat = category(n = n)
        if n < 1000 and not m.group("suf") and (
            (m.group("god") == "год" and cat == "one") or (m.group("god") == "года" and cat == "few")
        ):
            return Skip(reason = f"数后接 год 可能是年数而不是年份「{m.group()}」")
        if case is None:
            before = word_before(text = ctx.text, start = m.start())
            if before in PREPOSITIONAL_BEFORE:
                case = "prepositional"
            elif before in DATIVE_BEFORE:
                case = "dative"
            else:
                return Skip(reason = f"「году」的格需要前面的介词判断「{m.group()}」")
        suf = m.group("suf")
        if plural and not suf:
            # 「в 1920 годах」：复数 год 前的单个年份写法不规范（年代还是并列的年份），丢弃
            return Skip(reason = f"复数 год 前的年份写法不规范「{m.group()}」")
        if suf and plural != (suf in ("-х", "-е", "-м", "-ми")):
            return Skip(reason = f"年代写法与 год 的词形不一致「{m.group()}」")
        spoken = n2w(
            n = n,
            lang = "ru",
            to = "ordinal",
            case = case,
            plural = plural,
            animate = False,
        )
        if suf and not spoken.endswith(suf[1:]):
            return Skip(reason = f"年代写法与 год 的词形不一致「{m.group()}」")
        ctx.num(spoken = spoken, written = m.group("num"))
        ctx.kind(name = "年代" if plural else "年份", surface = m.group())
        return spoken + " " + m.group("god")

    def year_span(self, m, ctx):
        """С 1836 по 1839 год → С тысяча восемьсот тридцать шестого по тысяча восемьсот тридцать девятый год。"""
        first = ctx.num(spoken = ord_form(n = int(m.group("num")), case = "gent"), written = m.group("num"))
        second = ctx.num(spoken = ord_form(n = int(m.group("num2")), case = "accs"), written = m.group("num2"))
        ctx.kind(name = "年份", surface = m.group())
        return f"{m.group('pre')} {first} по {second} {m.group('god')}"

    def numero(self, m, ctx):
        """№ 5 → номер пять。"""
        if m.group("num").startswith("0"):
            return Skip(reason = f"编号写法无法确定「{m.group()}」")
        ctx.kind(name = "符号", surface = "№")
        return "номер " + ctx.num(spoken = card_form(n = int(m.group("num")), case = "nomn"), written = m.group("num"))

    def year_abbreviation(self, m, ctx):
        """в 1990 г. → в тысяча девятьсот девяностом году；в 1990-х гг. → в тысяча девятьсот девяностых годах。"""
        n = int(m.group("num"))
        plural = m.group("abbr") == "гг."
        suf = m.group("suf")
        if plural and not suf:
            return Skip(reason = f"年份范围或并列「{m.group()}」")
        if suf and not plural and suf != "-м":
            return Skip(reason = f"年代写法无法确定「{m.group()}」")
        tokens = self.tokens_before(text = ctx.text, start = m.start())
        w = tokens[0] if tokens else ""
        if w in PREPOSITIONAL_BEFORE:
            case = "loct"
        elif w in GENITIVE_BEFORE or w in ("с", "со") or w in MONTHS_GEN:
            case = "gent"
        elif w in ("к", "ко"):
            case = "datv"
        else:
            return Skip(reason = f"「г.」的格需要前面的介词判断「{m.group()}」")
        god = {
            ("loct", False): "году", ("gent", False): "года", ("datv", False): "году",
            ("loct", True): "годах", ("gent", True): "годов", ("datv", True): "годам",
        }[(case, plural)]
        spoken = ord_form(n = n, case = case, plural = plural)
        if suf and not spoken.endswith(suf[1:]):
            return Skip(reason = f"年代写法与 год 的词形不一致「{m.group()}」")
        ctx.num(spoken = spoken, written = m.group("num"))
        ctx.kind(name = "缩写", surface = m.group("abbr"))
        ctx.kind(name = "年代" if plural else "年份", surface = m.group())
        out = spoken + " " + god
        if sentence_end_after(text = ctx.text, end = m.end(), starters = STARTERS):
            out += "."
        return out

    # ------------------------------------------------------------------
    # 日期
    # ------------------------------------------------------------------

    def date(self, m, ctx):
        """13 сентября 1990 → тринадцатого сентября тысяча девятьсот девяностого（日为中性序数）。"""
        rest = ctx.text[m.end():].lstrip()
        if rest[:1] in ("—", "–", ":"):
            return Skip(reason = f"日期可能是主格（标题、定义）「{m.group()}」")
        tokens = self.tokens_before(text = ctx.text, start = m.start())
        w = tokens[0] if tokens else ""
        if w in ("к", "ко"):
            case = "datv"
        elif w in ("на", "по"):
            case = "accs"
        elif w in ("в", "во", "о", "об", "при", "за", "под", "перед", "между", "сегодня", "был", "была", "было", "это"):
            return Skip(reason = f"日期的格需要上下文「{m.group()}」")
        else:
            case = "gent"
        day = int(m.group("day"))
        if not 1 <= day <= 31:
            return Skip(reason = f"日期写法无法确定「{m.group()}」")
        out = ctx.num(spoken = ord_form(n = day, case = case, gender = "n"), written = m.group("day"))
        if m.group("day2"):
            day2 = int(m.group("day2"))
            link = m.group("link")
            # с 13 по 15 сентября：по 之后用宾格
            case2 = "accs" if link == "по" else case
            if link == "по" and w not in ("с", "со"):
                return Skip(reason = f"日期范围「{m.group()}」")
            out += f" {link} " + ctx.num(spoken = ord_form(n = day2, case = case2, gender = "n"), written = m.group("day2"))
        out += " " + m.group("month")
        if m.group("year"):
            out += " " + ctx.num(spoken = ord_form(n = int(m.group("year")), case = "gent"), written = m.group("year"))
        ctx.kind(name = "日期", surface = m.group())
        return out

    # ------------------------------------------------------------------
    # 序数、复合词
    # ------------------------------------------------------------------

    def years_compound(self, m, ctx):
        """25-летний → двадцатипятилетний（数的属格与 летний 连写；90–99、100 以上和 1 不处理）。"""
        n = int(m.group("num"))
        if n < 2 or n >= 90 or m.group("num").startswith("0"):
            return Skip(reason = f"复合词的读法不固定「{m.group()}」")
        stem = card_form(n = n, case = "gent")
        ctx.num(spoken = stem, written = m.group("num"))
        ctx.kind(name = "复合词", surface = m.group())
        return stem.replace(" ", "") + m.group("suf")

    def ordinal_suffix(self, m, ctx):
        """5-й、1-го、23-м、1990-х：读法与后面名词一致，并且必须以写出的词尾结尾。"""
        n = int(m.group("num"))
        suf = m.group("suf")
        if m.group("num").startswith("0") or n == 0:
            return Skip(reason = f"序数写法无法确定「{m.group()}」")
        noun = self.noun_after(text = ctx.text, end = m.end())
        options = set()
        if noun:
            for case, number, gender, anim in noun_parses(word = noun.lower() if noun[0].islower() else noun):
                plural = number == "plur"
                options.add(ord_form(
                    n = n,
                    case = case,
                    gender = gender,
                    plural = plural,
                    animate = anim,
                ))
        else:
            # 没有名词：只有词尾能唯一确定读法的写法（1990-х、5-го、5-му、5-ми、5-я、5-ю、1990-е）
            fixed = {
                "х": [("gent", "m", True)], "го": [("gent", "m", False)], "му": [("datv", "m", False)],
                "ми": [("ablt", "m", True)], "я": [("nomn", "f", False)], "ю": [("accs", "f", False)],
            }
            if suf == "е" and n >= 1000 and n % 10 == 0:
                fixed["е"] = [("nomn", "m", True)]
            for case, gender, plural in fixed.get(suf, []):
                options.add(ord_form(
                    n = n,
                    case = case,
                    gender = gender,
                    plural = plural,
                ))
        options = {o for o in options if o.endswith(suf)}
        if len(options) != 1:
            return Skip(reason = f"序数的词形取决于上下文「{m.group()}」")
        spoken = options.pop()
        ctx.kind(name = "序数", surface = m.group())
        return ctx.num(spoken = spoken, written = m.group())

    # ------------------------------------------------------------------
    # 罗马数字
    # ------------------------------------------------------------------

    def roman(self, m, ctx):
        tokens = [m.group("roman")] + re.findall(pattern = r"[IVXLCDM]+", string = m.group("chain"))
        values = [roman_value(text = t) for t in tokens]
        if any(v is None for v in values):
            return None
        single = len(tokens) == 1 and len(tokens[0]) == 1
        text = ctx.text
        abbr = m.group("abbr")
        if text[m.end():m.end() + 1] in "—–-":
            return Skip(reason = f"罗马数字的范围「{m.group()}」")
        options = set()
        if abbr:
            # в XIX в. → в девятнадцатом веке；начала XX в. → двадцатого века
            w = (self.tokens_before(text = text, start = m.start()) or [""])[0]
            case = "loct" if w in ("в", "во") else "gent" if w in GENITIVE_BEFORE or w in ("с", "со") else "datv" if w in ("к", "ко") else None
            if case is None or (abbr == "вв." and len(tokens) == 1):
                return Skip(reason = f"「в.」的格需要前面的介词判断「{m.group()}」")
            plural = abbr == "вв."
            noun = {
                ("loct", False): "веке", ("gent", False): "века", ("datv", False): "веку",
                ("loct", True): "веках", ("gent", True): "веков", ("datv", True): "векам",
            }[(case, plural)]
            spoken = [ord_form(n = v, case = case) for v in values]
            out = self.join_chain(spoken = spoken, chain = m.group("chain")) + " " + noun
            for v, s in zip(values, spoken):
                ctx.num(spoken = s, written = str(v))
            ctx.kind(name = "罗马数字", surface = m.group())
            if sentence_end_after(text = text, end = m.end(), starters = STARTERS):
                out += "."
            return out
        if len(tokens) == 1:
            # 人名后的君主序号（Пётр I、Петра I、Екатерины II）优先：Фридриха II статус 中的序号与人名一致
            name_m = re.search(pattern = r"([А-ЯЁ][а-яё]+)\s$", string = text[:m.start()])
            if name_m:
                for pos, case, number, _, anim, score, proper, lemma in parses(word = name_m.group(1)):
                    gender = REGNAL_LEMMAS.get(lemma)
                    if pos == "NOUN" and gender and score >= 0.05 and number == "sing" and case in CASE_NAMES:
                        spoken = ord_form(
                            n = values[0],
                            case = case,
                            gender = gender,
                            animate = True,
                        )
                        options.add((spoken.capitalize(),))
        noun = self.noun_after(text = text, end = m.end())
        if not options and noun and noun[:1].islower():
            for case, number, gender, anim in noun_parses(word = noun):
                # 几个罗马数字并列时名词用复数，每个序数仍是单数（XIX и XX веков）
                plural = number == "plur" and len(tokens) == 1
                if number == "plur" and len(tokens) == 1:
                    continue
                options.add(tuple(
                    ord_form(
                        n = v,
                        case = case,
                        gender = gender,
                        plural = plural,
                        animate = anim,
                    )
                    for v in values
                ))
        if len(options) != 1:
            if single:
                return None
            return Skip(reason = f"罗马数字的读法取决于上下文「{m.group()}」")
        spoken = list(options.pop())
        for v, s in zip(values, spoken):
            ctx.num(spoken = s, written = str(v))
        ctx.kind(name = "罗马数字", surface = m.group())
        return self.join_chain(spoken = spoken, chain = m.group("chain"))

    def join_chain(self, spoken: list, chain: str) -> str:
        """把并列的罗马数字（XIX и XX）逐个换成读法，保留原文的连接词和标点。"""
        rest = iter(spoken[1:])
        return spoken[0] + re.sub(pattern = r"[IVXLCDM]+", repl = lambda _: next(rest), string = chain)

    # ------------------------------------------------------------------
    # 单位、百分号、数量缩写
    # ------------------------------------------------------------------

    def unit(self, m, ctx):
        """25 % → двадцать пять процентов，около 5 тыс. → около пяти тысяч，в 5 км → 丢弃（格不确定）。"""
        n = self.parse(written = m.group("num"))
        if n is None or not n.is_int:
            return Skip(reason = f"小数的格变化暂不处理「{m.group()}」")
        tokens = self.tokens_before(text = ctx.text, start = m.start())
        while tokens and tokens[0] in TRANSPARENT:
            tokens = tokens[1:]
        w = tokens[0] if tokens else ""
        if w in ("к", "ко"):
            case = "datv"
        elif w in ("о", "об", "при"):
            case = "loct"
        elif w in ("под", "над", "перед", "между"):
            case = "ablt"
        else:
            case = self.context_case(text = ctx.text, start = m.start(), strict = True)
        if case is None:
            return Skip(reason = f"俄语数词的格取决于上下文「{m.group()}」")
        lemma = UNIT_LEMMAS[m.group("unit")]
        parse = analyzer().parse(word = lemma)[0]
        cat = category(n = n.value)
        if case == "nomn":
            form = parse.make_agree_with_number(num = n.value)
        elif case == "gent" and cat != "one" and lemma == "человек":
            form = parse.inflect(required_grammemes = {"gent", "plur"})
            form = form._replace(word = "человек") if form is not None else None
        else:
            form = parse.inflect(required_grammemes = {case, "sing" if cat == "one" else "plur"})
        if form is None:
            return Skip(reason = f"单位的词形无法确定「{m.group()}」")
        gender = GENDERS.get(parse.tag.gender, "m")
        spoken = ctx.num(spoken = card_form(n = n.value, case = case, gender = gender), written = n.written)
        ctx.kind(name = "百分数" if m.group("unit") == "%" else "单位", surface = m.group())
        out = spoken + " " + form.word
        if m.group("unit").endswith(".") and sentence_end_after(text = ctx.text, end = m.end(), starters = STARTERS):
            out += "."
        return out

    # ------------------------------------------------------------------
    # 基数
    # ------------------------------------------------------------------

    def cardinal(self, m, ctx, n):
        if not n.is_int:
            return Skip(reason = f"小数的格变化暂不处理「{m.group()}」")
        noun = self.noun_after(text = ctx.text, end = m.end())
        if not noun:
            return Skip(reason = f"俄语数词的格取决于上下文「{m.group()}」")
        cat = category(n = n.value)
        options = set()
        undecided = False
        ordinal_like = False
        for case, number, gender, anim in noun_parses(word = noun.lower() if noun[0].islower() else noun):
            if cat == "one":
                if number == "sing" and case in CASE_NAMES:
                    options.add((case, gender, anim))
            elif cat == "few":
                if number == "sing" and case == "gent":
                    options.add(("nomn", gender, anim))
                elif number == "plur" and case in ("gent", "datv", "ablt", "loct"):
                    options.add((case, gender, anim))
            elif number == "plur" and case == "gent":
                undecided = True
            elif number == "sing" and case == "gent" and not options:
                # 「до 40 этажа」：5 以上的数后接单数属格，多为序数（сорокового）
                ordinal_like = True
            elif number == "plur" and case in ("datv", "ablt", "loct"):
                options.add((case, gender, anim))
        if ordinal_like and not options and not undecided:
            return Skip(reason = f"后面的名词是单数属格，可能是序数「{m.group()} {noun}」")
        if undecided:
            case = self.context_case(text = ctx.text, start = m.start())
            if case is None:
                return Skip(reason = f"俄语数词的格取决于上下文「{m.group()} {noun}」")
            options.add((case, "m", False))
        forms = {
            card_form(
                n = n.value,
                case = case,
                gender = gender,
                animate = anim,
            )
            for case, gender, anim in options
        }
        if len(forms) != 1:
            return Skip(reason = f"俄语数词的格取决于上下文「{m.group()} {noun}」")
        spoken = forms.pop()
        ctx.kind(name = "基数", surface = m.group())
        return ctx.num(spoken = spoken, written = n.written)


LANG = Ru
