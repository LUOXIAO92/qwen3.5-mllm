"""阿拉伯语：数词与被数名词的性、数、格一致。只在性、格都能确定（或读法与之无关）时改写，其余丢弃。

依据：FLEURS 阿拉伯语原文含数字的 91 条录音，用 Jais-ASR（带元音符号）转写后对照原文（T04c）：朗读者把年份
读作属格（في عام ألفٍ وتسعمئةٍ وسبعةٍ وستين、في سبتمبر ألفين وسبعة عشر），把日期中的日读作基数（في خمسة عشر
آب、في واحد وعشرين من كانون الثاني），百分数用阳性（ثمانية في المئة）。

读法规则（现代标准语）：
- 性：1、2 与名词同性（واحد/واحدة、اثنان/اثنتان）；3–10 与名词单数的性相反（ثلاثة أيام、ثلاث سنوات）；
  11–19 的个位按 3–10 的规则、عشر 与名词同性（خمسة عشر يومًا、خمس عشرة سنة）；21–99 的个位同上。
  名词的性来自本文件的被数名词表；「بين 18 و30 سنة」「من 3 إلى 5 أيام」中前一个数取后一个数所带名词的性；
  后面没有登记的被数名词时，只改写读法与性无关的数（整十、整百、整千）。
- 格：整十（عشرون/عشرين）、2（اثنان/اثنين）、12（اثنا/اثني）、200（مئتان/مئتين）、2000（ألفان/ألفين）随格变化。
  数字紧跟在介词（في、من、إلى、على、بين、حوالي、نحو 等，也包括附着的 بـ、لـ）、支配属格的名词（بنسبة、
  بمعدل、لمدة 等）、及物动词（يبلغ、يناهز、يتجاوز、تضم 等，或这些动词加 1–4 个词的主语）之后，或与这样的数用 و
  并列时为属格/宾格（عشرين）；紧跟在 هناك、يوجد、هو、هي 之后为主格（عشرون）；其余位置无法确定格，只改写读法
  与格无关的数。
- 被数名词的数：3–10 后用复数（أيام），11–99 后用单数宾格（يومًا，原文常省去 ا，也接受 يوم），整百整千后用
  单数属格（مئة يوم）。原文名词的形式与数不符时丢弃。1、2 以及末两位为 01、02 的数加名词丢弃。
- 千、百万：3–10 千读 ثلاثة آلاف；11–99 千只在后面紧跟被数名词时改写（خمسة وعشرين ألف قدم），单独出现时
  应读 ألفًا，比对模块不认这个词形，丢弃。
- 年份：عام、سنة、في、منذ 等之后，以及月份名之后（مارس 1974、رمضان 1445），读属格（ألفين وخمسة وعشرين）；
  其后的 م、هـ 读 للميلاد、للهجرة（与 عام、سنة 的性无关）；ق.م 读 قبل الميلاد。
- 日期：「15 نوفمبر 2010」读 خمسة عشر نوفمبر ألفين وعشرة，日按阳性基数，格同上。
- 百分数读 بالمئة，数用阳性；单位 كم、كلم、سم、ملم、كغ 读成 كيلومتر 等，按数选单数、复数或宾格；
  $、€ 读 دولار、يورو。
- 小数只用于百分数，并且小数部分只有一位且不是 2。
- 书写：介词加 tatweel 后接词（لـ اقتصاد）合写为 لاقتصاد、للشركة；词中拉长字形的 tatweel 删去；句号后缺少空格
  时补上空格。
"""

from __future__ import annotations

import re
from itertools import product

from ..rules import Rule, Skip, alternation
from .base import Lang
from .extras import common_rules, missing_space_rule

TASHKEEL = re.compile(pattern = "[ً-ْـ]")
ARABIC_LETTERS = f"{chr(0x0620)}-{chr(0x064A)}{chr(0x066E)}-{chr(0x06D3)}{chr(0x06FA)}-{chr(0x06FF)}"
# 含 tatweel 的整个词（جميــع、سـعادتـمـند）：词首前不是字母，tatweel 后面还有字母（不含 هـ、بـ 这类缩写和介词）
TATWEEL_WORD = (
    rf"(?<![\w\u0640\u064B-\u0652])(?=[{ARABIC_LETTERS}\u0640\u064B-\u0652]*\u0640+[{ARABIC_LETTERS}])"
    rf"[{ARABIC_LETTERS}\u0640\u064B-\u0652]+"
)
WORD = rf"[{ARABIC_LETTERS}ً-ْـ]+"

# ---------------------------------------------------------------------------
# 数词
# ---------------------------------------------------------------------------

_M = {1: "واحد", 3: "ثلاثة", 4: "أربعة", 5: "خمسة", 6: "ستة", 7: "سبعة", 8: "ثمانية", 9: "تسعة", 10: "عشرة"}
_F = {1: "واحدة", 3: "ثلاث", 4: "أربع", 5: "خمس", 6: "ست", 7: "سبع", 8: "ثماني", 9: "تسع", 10: "عشر"}
_TENS = {
    "nom": ["", "", "عشرون", "ثلاثون", "أربعون", "خمسون", "ستون", "سبعون", "ثمانون", "تسعون"],
    "obl": ["", "", "عشرين", "ثلاثين", "أربعين", "خمسين", "ستين", "سبعين", "ثمانين", "تسعين"],
}
_HUNDREDS = ["", "مئة", None, "ثلاثمئة", "أربعمئة", "خمسمئة", "ستمئة", "سبعمئة", "ثمانمئة", "تسعمئة"]
_TWO = {("m", "nom"): "اثنان", ("m", "obl"): "اثنين", ("f", "nom"): "اثنتان", ("f", "obl"): "اثنتين"}
# 数量级词：(单数, 双数主格, 双数宾属格, 复数)
_SCALES = {1000: ("ألف", "ألفان", "ألفين", "آلاف"), 10 ** 6: ("مليون", "مليونان", "مليونين", "ملايين")}


class Unknown(Exception):
    """读法取决于无法确定的性或格。"""


def _unit(u: int, gender: str) -> str:
    """3–10 与名词的性相反：阳性名词用带 ة 的形式。"""
    return _M[u] if gender == "m" else _F[u]


def _below_100(n: int, gender: str, case: str) -> str:
    if n == 1:
        return _M[1] if gender == "m" else _F[1]
    if n == 2:
        return _TWO[(gender, case)]
    if n <= 10:
        return _unit(u = n, gender = gender)
    if n == 11:
        return "أحد عشر" if gender == "m" else "إحدى عشرة"
    if n == 12:
        head = {("m", "nom"): "اثنا", ("m", "obl"): "اثني", ("f", "nom"): "اثنتا", ("f", "obl"): "اثنتي"}
        return head[(gender, case)] + (" عشر" if gender == "m" else " عشرة")
    if n < 20:
        return _unit(u = n - 10, gender = gender) + (" عشر" if gender == "m" else " عشرة")
    t, u = divmod(n, 10)
    tens = _TENS[case][t]
    if u == 0:
        return tens
    if u == 1:
        head = "واحد" if gender == "m" else "إحدى"
    elif u == 2:
        head = _TWO[(gender, case)]
    else:
        head = _unit(u = u, gender = gender)
    return head + " و" + tens


def _below_1000(n: int, gender: str, case: str, construct: bool) -> str:
    """construct 表示后面紧跟被数名词（此时 200 读 مئتا/مئتي）。"""
    h, rest = divmod(n, 100)
    parts = []
    if h == 2:
        if rest == 0 and construct:
            parts.append("مئتا" if case == "nom" else "مئتي")
        else:
            parts.append("مئتان" if case == "nom" else "مئتين")
    elif h:
        parts.append(_HUNDREDS[h])
    if rest:
        parts.append(_below_100(n = rest, gender = gender, case = case))
    return " و".join(parts)


def _scale_part(q: int, scale: int, case: str, construct: bool) -> str:
    """q 千或 q 百万。数量级词是阳性名词，系数一律按阳性读。"""
    one, two_nom, two_obl, plural = _SCALES[scale]
    if q == 1:
        return one
    if q == 2:
        if construct:
            # ألفي دولار；主格 ألفا 比对模块不认，不写
            if case != "obl":
                raise Unknown("主格的 ألفا 不处理")
            return two_obl[:-1]
        return two_nom if case == "nom" else two_obl
    rest = q % 100
    if rest in (1, 2) and q > 100:
        raise Unknown("末两位为 01、02 的千位系数")
    coef = _below_1000(n = q, gender = "m", case = case, construct = rest == 0)
    if 3 <= rest <= 10:
        return coef + " " + plural
    if rest == 0:
        return coef + " " + one
    if not construct:
        raise Unknown("11–99 千单独出现时读 ألفًا")
    return coef + " " + one


def ar_cardinal(n: int, gender: str, case: str, construct: bool) -> str:
    """阿拉伯语基数。gender 是被数名词的性（m/f），case 是数词的格（nom/obl），
    construct 表示后面紧跟被数名词。无法写出时抛出 Unknown。"""
    if not 1 <= n < 10 ** 9:
        raise Unknown("数值超出范围")
    parts = []
    remaining = n
    for scale in (10 ** 6, 1000):
        q, remaining = divmod(remaining, scale)
        if q:
            parts.append(_scale_part(q = q, scale = scale, case = case, construct = construct and remaining == 0))
    if remaining:
        parts.append(_below_1000(n = remaining, gender = gender, case = case, construct = construct))
    return " و".join(parts)


def gender_matters(n: int) -> bool:
    last = n % 100
    return 1 <= last <= 19 or (last > 20 and last % 10 != 0)


def resolve(n: int, gender, case, construct: bool) -> str:
    """在未知的性（None）和格（None）的所有取值下读法一致时返回读法，否则抛出 Unknown。"""
    genders = [gender] if gender else ["m", "f"]
    cases = [case] if case else ["nom", "obl"]
    outs = set()
    for g, c in product(genders, cases):
        outs.add(ar_cardinal(n = n, gender = g, case = c, construct = construct))
    if len(outs) != 1:
        missing = []
        if gender is None and len({ar_cardinal(n = n, gender = g, case = cases[0], construct = construct) for g in genders}) > 1:
            missing.append("性")
        if case is None and len({ar_cardinal(n = n, gender = genders[0], case = c, construct = construct) for c in cases}) > 1:
            missing.append("格")
        raise Unknown("、".join(missing or ["性、格"]) + "未知")
    return outs.pop()


# ---------------------------------------------------------------------------
# 被数名词表：(性, 单数, 复数写法)。单数宾格为单数加 ا（عامًا），以 ة、اء、ى 结尾的名词宾格与单数同形。
# ---------------------------------------------------------------------------

NOUNS = [
    ("m", "عام", ["أعوام"]), ("m", "يوم", ["أيام"]), ("m", "شهر", ["أشهر", "شهور"]), ("m", "أسبوع", ["أسابيع"]),
    ("m", "قرن", ["قرون"]), ("f", "سنة", ["سنوات", "سنين"]), ("f", "ساعة", ["ساعات"]), ("f", "دقيقة", ["دقائق"]),
    ("f", "ثانية", ["ثوان", "ثواني"]), ("f", "مرة", ["مرات"]), ("f", "ليلة", ["ليال", "ليالي"]),
    ("m", "شخص", ["أشخاص"]), ("m", "رجل", ["رجال"]), ("m", "طفل", ["أطفال"]), ("m", "جندي", ["جنود"]),
    ("m", "لاعب", ["لاعبين"]), ("m", "طالب", ["طلاب"]), ("m", "موظف", ["موظفين"]), ("m", "عضو", ["أعضاء"]),
    ("m", "قتيل", ["قتلى"]), ("m", "مقاتل", ["مقاتلين"]), ("m", "متفرج", ["متفرجين"]), ("m", "مشجع", ["مشجعين"]),
    ("m", "عامل", ["عمال"]), ("m", "فرد", ["أفراد"]), ("m", "راكب", ["ركاب"]), ("m", "باحث", ["باحثين"]),
    ("f", "نسمة", ["نسمات"]), ("f", "عائلة", ["عائلات"]), ("f", "أسرة", ["أسر"]),
    ("m", "دولار", ["دولارات"]), ("m", "جنيه", ["جنيهات"]), ("m", "دينار", ["دنانير"]), ("m", "درهم", ["دراهم"]),
    ("m", "ريال", ["ريالات"]), ("m", "فرنك", ["فرنكات"]), ("f", "ليرة", ["ليرات"]), ("f", "روبية", ["روبيات"]),
    ("m", "متر", ["أمتار"]), ("m", "كيلومتر", ["كيلومترات"]), ("m", "سنتيمتر", ["سنتيمترات"]),
    ("m", "مليمتر", ["مليمترات"]), ("m", "ميل", ["أميال"]), ("m", "طن", ["أطنان"]), ("m", "كيلوغرام", ["كيلوغرامات"]),
    ("m", "كيلوجرام", ["كيلوجرامات"]), ("m", "غرام", ["غرامات"]), ("m", "جرام", ["جرامات"]), ("m", "هكتار", ["هكتارات"]),
    ("m", "فدان", ["أفدنة"]), ("m", "دونم", ["دونمات"]), ("m", "رطل", ["أرطال"]), ("f", "درجة", ["درجات"]),
    ("f", "بوصة", ["بوصات"]), ("f", "ياردة", ["ياردات"]), ("m", "هدف", ["أهداف"]), ("m", "مقعد", ["مقاعد"]),
    ("m", "صوت", ["أصوات"]), ("m", "كتاب", ["كتب"]), ("m", "فيلم", ["أفلام"]), ("m", "ألبوم", ["ألبومات"]),
    ("m", "موسم", ["مواسم"]), ("m", "بلد", ["بلدان"]), ("m", "جزء", ["أجزاء"]), ("m", "نوع", ["أنواع"]),
    ("m", "فصل", ["فصول"]), ("m", "طابق", ["طوابق"]), ("m", "منزل", ["منازل"]), ("m", "صاروخ", ["صواريخ"]),
    ("m", "مدفع", ["مدافع"]), ("m", "حصان", ["أحصنة"]), ("m", "مقال", ["مقالات"]), ("m", "حرف", ["أحرف", "حروف"]),
    ("m", "مشروع", ["مشاريع"]), ("m", "مركز", ["مراكز"]), ("m", "مستشفى", ["مستشفيات"]), ("m", "مسجد", ["مساجد"]),
    ("m", "مصنع", ["مصانع"]), ("m", "حزب", ["أحزاب"]), ("m", "عنصر", ["عناصر"]), ("m", "جسر", ["جسور"]),
    ("f", "مباراة", ["مباريات"]), ("f", "نقطة", ["نقاط"]), ("f", "دولة", ["دول"]), ("f", "مدينة", ["مدن"]),
    ("f", "قرية", ["قرى"]), ("f", "طائرة", ["طائرات"]), ("f", "سيارة", ["سيارات"]), ("f", "سفينة", ["سفن"]),
    ("f", "دبابة", ["دبابات"]), ("f", "حلقة", ["حلقات"]), ("f", "أغنية", ["أغان", "أغاني"]), ("f", "رواية", ["روايات"]),
    ("f", "نسخة", ["نسخ"]), ("f", "صفحة", ["صفحات"]), ("f", "شركة", ["شركات"]), ("f", "جامعة", ["جامعات"]),
    ("f", "مدرسة", ["مدارس"]), ("f", "مقاطعة", ["مقاطعات"]), ("f", "ولاية", ["ولايات"]), ("f", "منطقة", ["مناطق"]),
    ("f", "جزيرة", ["جزر"]), ("f", "لغة", ["لغات"]), ("f", "حالة", ["حالات"]), ("f", "وحدة", ["وحدات"]),
    ("f", "مؤسسة", ["مؤسسات"]), ("f", "ميدالية", ["ميداليات"]), ("f", "جائزة", ["جوائز"]), ("f", "كتيبة", ["كتائب"]),
    ("f", "مرحلة", ["مراحل"]), ("f", "محطة", ["محطات"]), ("f", "آية", ["آيات"]), ("f", "قصيدة", ["قصائد"]),
    ("f", "عملية", ["عمليات"]), ("f", "غرفة", ["غرف"]), ("f", "كلية", ["كليات"]), ("f", "مجموعة", ["مجموعات"]),
    ("f", "بطولة", ["بطولات"]), ("f", "دورة", ["دورات"]), ("f", "قطعة", ["قطع"]), ("f", "صورة", ["صور"]),
    ("f", "كلمة", ["كلمات"]), ("f", "شجرة", ["أشجار"]), ("f", "سورة", ["سور"]), ("f", "بلدية", ["بلديات"]),
    ("f", "محافظة", ["محافظات"]), ("f", "طلعة", ["طلعات"]), ("f", "رحلة", ["رحلات"]), ("f", "مسرحية", ["مسرحيات"]),
    ("m", "فريق", []), ("f", "امرأة", ["نساء"]), ("m", "حاج", ["حجاج"]), ("m", "متسابق", ["متسابقين"]),
    ("m", "شهيد", ["شهداء"]), ("m", "مريض", ["مرضى"]), ("m", "خريج", ["خريجين"]), ("m", "ناخب", ["ناخبين"]),
    ("m", "مراقب", ["مراقبين"]), ("f", "مزرعة", ["مزارع"]), ("m", "موقع", ["مواقع"]), ("m", "برنامج", ["برامج"]),
    ("f", "شخصية", ["شخصيات"]), ("m", "مسار", ["مسارات"]), ("m", "اتحاد", ["اتحادات"]), ("f", "منظمة", ["منظمات"]),
    ("m", "متجر", ["متاجر"]), ("f", "حافلة", ["حافلات"]), ("f", "جملة", ["جمل"]), ("f", "رصاصة", ["رصاصات"]),
    ("m", "مشهد", ["مشاهد"]), ("m", "لوح", ["ألواح"]), ("f", "فتاة", ["فتيات"]), ("m", "زوج", ["أزواج"]),
    ("m", "نادي", ["أندية"]), ("m", "لاجئ", ["لاجئين"]), ("m", "مصاب", ["مصابين"]), ("m", "سجين", ["سجناء"]),
    ("m", "مواطن", ["مواطنين"]), ("m", "مدني", ["مدنيين"]), ("m", "ضابط", ["ضباط"]), 
    ("f", "قاعدة", ["قواعد"]), ("m", "مستوى", ["مستويات"]), ("m", "كوكب", ["كواكب"]), 
]
_FEMININE_ENDINGS = ("ة", "اء", "ى")


def _norm(word: str) -> str:
    return TASHKEEL.sub(repl = "", string = word)


def _build_lexicon() -> dict:
    """写法（去掉元音符号）→ (性, 形式)；形式为 sg（单数）、acc（单数宾格）、pl（复数）。
    同一写法对应不同的性或形式时（فرق 既是 فريق 也是 فرقة 的复数）不登记。"""
    table = {}
    conflicts = set()

    def add(form: str, value: tuple) -> None:
        if form in table and table[form] != value:
            conflicts.add(form)
        table[form] = value

    for gender, singular, plurals in NOUNS:
        add(form = singular, value = (gender, "sg"))
        if not singular.endswith(_FEMININE_ENDINGS):
            add(form = singular + "ا", value = (gender, "acc"))
        for plural in plurals:
            add(form = plural, value = (gender, "pl"))
    for form in conflicts:
        del table[form]
    return table


LEXICON = _build_lexicon()
# 单位缩写 → (单数, 复数, 宾格)
UNIT_ABBREV = {
    "كم": ("كيلومتر", "كيلومترات", "كيلومترًا"), "كلم": ("كيلومتر", "كيلومترات", "كيلومترًا"),
    "سم": ("سنتيمتر", "سنتيمترات", "سنتيمترًا"), "ملم": ("مليمتر", "مليمترات", "مليمترًا"),
    "كغ": ("كيلوغرام", "كيلوغرامات", "كيلوغرامًا"), "كغم": ("كيلوغرام", "كيلوغرامات", "كيلوغرامًا"),
    "كجم": ("كيلوجرام", "كيلوجرامات", "كيلوجرامًا"), "كـم": ("كيلومتر", "كيلومترات", "كيلومترًا"),
}
CURRENCY_SYMBOLS = {"$": ("دولار", "دولارات", "دولارًا"), "€": ("يورو", "يورو", "يورو")}
SCALE_WORDS = {"ألف": (1000, "sg"), "آلاف": (1000, "pl"), "مليون": (10 ** 6, "sg"), "ملايين": (10 ** 6, "pl"),
               "مليار": (10 ** 9, "sg"), "مليارات": (10 ** 9, "pl"), "ألفا": (1000, "acc"), "مليونا": (10 ** 6, "acc")}

# ---------------------------------------------------------------------------
# 上下文
# ---------------------------------------------------------------------------

# 数字紧跟其后时，数词为属格的介词和名词（可带连词 و、ف）
GOVERNORS = {
    "في", "من", "إلى", "الى", "على", "عن", "حتى", "منذ", "مذ", "مع", "عند", "لدى", "نحو", "حوالي", "حوالى", "قرابة",
    "زهاء", "بين", "خلال", "بعد", "قبل", "فوق", "تحت", "دون", "ضمن", "عبر", "طوال", "طيلة", "مقابل", "بحوالي", "بنحو",
    "بقرابة", "لحوالي", "لنحو", "بنسبة", "نسبة", "بمعدل", "بقيمة", "بمبلغ", "بمسافة", "بطول", "بعرض", "بعمق",
    "بارتفاع", "بمساحة", "بسعة", "بوزن", "بعدد", "لمدة", "مدة", "بتكلفة", "بسرعة", "بحجم", "بأكثر", "لأكثر",
    "يوم", "الموافق", "بتاريخ", "تاريخ", "مواليد", "بحلول", "غضون", "مرور", "أصل", "سن", "بعمر", "عمر", "مدى",
    "برصيد", "بسعر", "سعر", "مبلغ", "بمقدار", "البالغة", "البالغ",
}
# 数字紧跟其后时数词为主格：هناك、يوجد 之后的数是主语，هو、هي 之后的数是谓语
NOMINATIVE_BEFORE = {"هناك", "هنالك", "يوجد", "توجد", "هو", "هي"}
# 及物动词：紧跟其后、或隔着 1–4 个词的主语之后的数是宾语（يبلغ طوله 187 كم、عن عمر يناهز 70 عامًا）。
# 不收被动态与主动态同形、数可能是主语的动词（قتل、سجل）。
OBJECT_VERBS = {
    "بلغ", "بلغت", "يبلغ", "تبلغ", "يناهز", "ناهز", "تناهز", "يقارب", "تقارب", "قارب", "قاربت", "يتجاوز", "تتجاوز",
    "تجاوز", "تجاوزت", "يجاوز", "تجاوز", "يتعدى", "تتعدى", "تعدى", "يفوق", "تفوق", "فاق", "فاقت", "يعادل", "تعادل",
    "يساوي", "تساوي", "يضم", "تضم",
}
MONTHS = (
    "يناير", "فبراير", "مارس", "أبريل", "ابريل", "مايو", "يونيو", "يونيه", "يوليو", "يوليه", "أغسطس", "سبتمبر",
    "أكتوبر", "اكتوبر", "نوفمبر", "ديسمبر", "كانون الثاني", "شباط", "آذار", "نيسان", "أيار", "حزيران", "تموز", "آب",
    "أيلول", "تشرين الأول", "تشرين الثاني", "كانون الأول", "جانفي", "فيفري", "أفريل", "ماي", "جوان", "جويلية", "أوت",
    "محرم", "ربيع الأول", "ربيع الآخر", "ربيع الثاني", "جمادى الأولى", "جمادى الآخرة", "جمادى الثانية", "رجب",
    "شعبان", "رمضان", "شوال", "ذو القعدة", "ذو الحجة",
)
YEAR_NOUNS = {"عام": "m", "العام": "m", "لعام": "m", "بعام": "m", "وعام": "m", "عامي": "m",
              "سنة": "f", "السنة": "f", "لسنة": "f", "بسنة": "f", "وسنة": "f"}
# 年份后的纪元标记：م 读 للميلاد，هـ 读 للهجرة（与 عام、سنة 的性无关）
ERA_WORDS = {"م": "للميلاد", "هـ": "للهجرة", "ه": "للهجرة", "ھ": "للهجرة"}
FUNCTION_AFTER = {
    "و", "في", "من", "إلى", "الى", "حتى", "عندما", "حيث", "ثم", "بعد", "قبل", "على", "عن", "مع", "أو", "وحتى",
    "وفي", "ومن", "تم", "كان", "كانت", "أصبح", "أصبحت", "قام", "قامت", "بدأ", "بدأت", "لم", "قد", "وقد", "التي",
    "الذي", "حين", "بينما", "ولكن", "لكن", "ميلادي", "ميلادية", "للميلاد", "للهجرة", "هجري", "هجرية",
}


def _word_before(text: str, start: int) -> str:
    m = re.search(pattern = rf"({WORD})\s*$", string = text[:start])
    return _norm(word = m.group(1)) if m else ""


def _word_after(text: str, end: int) -> str:
    m = re.match(pattern = rf"\s*({WORD})", string = text[end:])
    return m.group(1) if m else ""


_NUM_BEFORE_WAW = re.compile(pattern = rf"([%٪]?\d[\d,.٫٬]*)(?:\s*[%٪]|\s+{WORD})?\s*و\s*$")
# 「X و Y 名词」「X إلى Y 名词」中 X 后面的部分，用来找共用的名词
_SHARED_NOUN = re.compile(pattern = rf"\s*(?:و|أو|إلى|الى)\s*\d[\d,٬]*\s+({WORD})")
# 附着在数字前的介词（بـ199、لـ 15、ل15）和连词 و（و150）
PREFIX = r"(?P<pre>(?<![\w\u0640])(?:و?[بلك]\u0640?\s?|و)(?=[\d$€%٪]))?"


def governed_case(text: str, start: int, depth: int = 0):
    """数字前面的成分决定其为属格/宾格时返回 obl，为主格时返回 nom，无法确定时返回 None。"""
    before = text[:start]
    # 「بين 18 و 30」：与前一个数并列，格相同
    m = _NUM_BEFORE_WAW.search(string = before)
    if m and depth < 3:
        return governed_case(text = text, start = m.start(1), depth = depth + 1)
    # 「يبلغ من العمر 71 عامًا」：من العمر 之后的年龄是 بلغ 的宾语
    if re.search(pattern = r"من\s+العمر\s*$", string = before):
        return "obl"
    word = _word_before(text = text, start = start)
    for w in (word, word[1:] if word[:1] in "وف" else ""):
        if w in GOVERNORS or w in OBJECT_VERBS:
            return "obl"
        if w in NOMINATIVE_BEFORE:
            return "nom"
    # 及物动词加 1–4 个词的主语
    tail = re.search(pattern = rf"(?:^|\s)({WORD})((?:\s+{WORD}){{1,4}})\s*$", string = before)
    if tail and _norm(word = tail.group(1)).lstrip("وف") in OBJECT_VERBS:
        words = [_norm(word = w) for w in tail.group(2).split()]
        if not any(w in GOVERNORS for w in words):
            return "obl"
    return None


class Ar(Lang):
    code = "ar"
    point = "فاصلة"

    def card(self, n: int) -> str:
        return ar_cardinal(n = n, gender = "m", case = "obl", construct = False)

    def rules(self) -> list:
        """年份的纪元标记（ق.م）先于公共保护规则运行；最后的普通数字规则把紧贴在前面的连词 و 一并替换。"""
        rules = super().rules()
        rules[-1] = Rule(
            pattern = re.compile(pattern = rf"(?P<waw>(?<![\w\u0640])و)?(?P<num>{self.num})"),
            handle = self.cardinal_with_waw,
        )
        return common_rules(lang = self, list_numbers = False) + [
            Rule(
                pattern = re.compile(pattern = rf"(?P<prev>(?<![{ARABIC_LETTERS}])(?:عام|العام|سنة|السنة|في|منذ)\s)?(?<![\d.,٫٬])(?P<num>\d{{3,4}})\s?(?P<era>ق\.?\s?م\.?)(?![{ARABIC_LETTERS}])"),
                handle = self.year_bc,
            ),
        ] + rules

    def extra_rules(self) -> list:
        months = alternation(words = MONTHS)
        year_nouns = alternation(words = YEAR_NOUNS)
        num = r"(?<![\d.,٫٬])\d{1,4}(?!\d|[.,٫٬]\d)"
        era = r"(?:\s?(?P<era>م|هـ|ه|ھ)(?![{0}]))?".format(ARABIC_LETTERS)
        return [
            # 日期：15 نوفمبر 2010、في 21 من كانون الثاني
            Rule(
                pattern = re.compile(pattern = rf"(?P<day>{num})\s(?P<min>من\s)?(?P<month>{months})(?![{ARABIC_LETTERS}])(?:\s(?P<year>\d{{3,4}})(?!\d|[.,٫٬]\d){era})?"),
                handle = self.date,
            ),
            # 月份后的年份：مارس 1974
            Rule(
                pattern = re.compile(pattern = rf"(?<=[{ARABIC_LETTERS}]\s)(?P<num>\d{{3,4}})(?!\d|[.,٫٬]\d){era}"),
                handle = self.year_after_month,
            ),
            # عام 2010م、عامي 2010 و2011、سنة 1517هـ
            Rule(
                pattern = re.compile(pattern = rf"(?<![{ARABIC_LETTERS}])(?P<prev>{year_nouns})\s(?P<num>\d{{1,4}})(?!\d|[.,٫٬]\d){era}"),
                handle = self.year,
            ),
            Rule(
                pattern = re.compile(pattern = rf"(?<![{ARABIC_LETTERS}])(?P<prev>(?:و|ف)?(?:في|منذ|حتى|من|إلى|الى|بين|بعد|قبل|خلال))\s(?P<num>\d{{4}})(?!\d|[.,٫٬]\d){era}"),
                handle = self.year,
            ),
            # 「عامي 2010 و2011」「بين 1990 و1995」中并列的第二个年份
            Rule(
                pattern = re.compile(pattern = rf"(?<=\d)(?P<waw>\s?و\s?)(?P<num>\d{{4}})(?!\d|[.,٫٬]\d){era}"),
                handle = self.coordinated_year,
            ),
            # 百分数：77%、%72、93.4%
            Rule(
                pattern = re.compile(pattern = PREFIX + r"(?:(?P<num>(?<![\d.,٫٬])\d+(?:[.٫]\d)?)(?:\s?[%٪]|\s(?P<word>(?:في\s|بال)(?:المائة|المئة)))|[%٪]\s?(?P<num2>\d+(?:[.٫]\d)?)(?!\d|[.,٫٬]\d))"),
                handle = self.percent_ar,
            ),
            # 货币符号：$5,000、5000 $
            Rule(
                pattern = re.compile(pattern = PREFIX + r"(?:(?P<cur>[$€])\s?(?P<num>\d{1,3}(?:[,٬]\d{3})+|\d+)(?![\d.,٫٬])|(?<![\d.,٫٬])(?P<num2>\d{1,3}(?:[,٬]\d{3})+|\d+)\s?(?P<cur2>[$€]))"),
                handle = self.currency_ar,
            ),
            # 单位缩写：187 كم、30 ملم
            Rule(
                pattern = re.compile(pattern = PREFIX + rf"(?<![\d.,٫٬])(?P<num>\d{{1,3}}(?:[,٬]\d{{3}})+|\d+)\s?(?P<unit>{alternation(words = UNIT_ABBREV)})(?![{ARABIC_LETTERS}ـ²])(?P<square>\s(?:مربع|مربعة))?"),
                handle = self.unit_ar,
            ),
            # 数字加被数名词或数量级词
            Rule(
                pattern = re.compile(pattern = PREFIX + rf"(?<![\d.,٫٬])(?P<num>\d{{1,3}}(?:[,٬]\d{{3}})+|\d+)(?!\d|[.,٫٬]\d)(?=\s(?P<noun>{WORD}))"),
                handle = self.counted,
            ),
            missing_space_rule(left = ARABIC_LETTERS, right = rf"[{ARABIC_LETTERS}]{{2,}}"),
            # 介词加 tatweel 后接词（لـ اقتصاد → لاقتصاد、بـ البروتودولوميت → بالبروتودولوميت）
            Rule(
                pattern = re.compile(pattern = rf"(?<![\w\u0640])(?P<p>[بلك])\u0640+\s?(?P<w>[{ARABIC_LETTERS}][\w\u064B-\u0652]*)"),
                handle = self.tatweel_preposition,
            ),
            # 词中用于拉长字形的 tatweel（تـتبع、مـا）
            Rule(
                pattern = re.compile(pattern = TATWEEL_WORD),
                handle = self.tatweel_word,
            ),
        ]

    # ------------------------------------------------------------------

    @staticmethod
    def _case_and_prefix(m, ctx):
        """附着的介词（بـ199）使数为属格，读法前接上这个介词字母；否则按前文判断格。"""
        pre = (m.groupdict().get("pre") or "").strip().replace("\u0640", "")
        if pre == "و":
            return governed_case(text = ctx.text, start = m.start() + 1), pre
        if pre:
            return "obl", pre
        return governed_case(text = ctx.text, start = m.start()), ""

    def _say(self, ctx, n: int, written: str, gender, case, construct: bool):
        try:
            spoken = resolve(n = n, gender = gender, case = case, construct = construct)
        except Unknown as exc:
            return Skip(reason = f"阿拉伯语数词的读法无法确定（{exc}）「{written}」")
        return ctx.num(spoken = spoken, written = written)

    def _year_words(self, ctx, written: str):
        """年份读属格的阳性基数（ألفين وخمسة وعشرين），后面没有被数名词。"""
        n = int(written)
        if not 1 <= n <= 2999 or written.startswith("0"):
            return None
        return ctx.num(spoken = ar_cardinal(n = n, gender = "m", case = "obl", construct = False), written = written)

    def _era(self, m):
        era = m.groupdict().get("era")
        return " " + ERA_WORDS[era] if era else ""

    def year(self, m, ctx):
        prev = m.group("prev")
        words = self._year_words(ctx = ctx, written = m.group("num"))
        if words is None:
            return None
        if prev not in YEAR_NOUNS:
            n = int(m.group("num"))
            if not 1000 <= n <= 2099:
                return None
            after = _norm(word = _word_after(text = ctx.text, end = m.end()))
            if after in LEXICON or after in SCALE_WORDS:
                return None
            if prev.lstrip("وف") not in ("في", "منذ") and after and after not in FUNCTION_AFTER and not m.group("era"):
                return None
        ctx.kind(name = "年份", surface = m.group())
        return prev + " " + words + self._era(m = m)

    def year_bc(self, m, ctx):
        words = self._year_words(ctx = ctx, written = m.group("num"))
        if words is None:
            return Skip(reason = f"年份写法无法确定「{m.group()}」")
        ctx.kind(name = "年份", surface = m.group())
        prev = m.group("prev") or ""
        return prev + words + " قبل الميلاد"

    def year_after_month(self, m, ctx):
        before = ctx.text[:m.start()].rstrip()
        if not any(before.endswith(month) for month in MONTHS):
            return None
        words = self._year_words(ctx = ctx, written = m.group("num"))
        if words is None:
            return None
        ctx.kind(name = "年份", surface = m.group())
        return words + self._era(m = m)

    def coordinated_year(self, m, ctx):
        before = ctx.text[:m.start()]
        prev = re.search(pattern = rf"(?:^|\s)(?P<word>{WORD})\s(?P<num>\d{{4}})$", string = before)
        if not prev:
            return None
        word = _norm(word = prev.group("word"))
        if word not in YEAR_NOUNS and word.lstrip("وف") not in ("في", "منذ", "بين", "من", "حتى"):
            return None
        words = self._year_words(ctx = ctx, written = m.group("num"))
        if words is None:
            return None
        ctx.kind(name = "年份", surface = m.group().strip())
        return m.group("waw").rstrip() + words + self._era(m = m)

    def date(self, m, ctx):
        day = int(m.group("day"))
        if not 1 <= day <= 31 or m.group("day").startswith("0"):
            return Skip(reason = f"日期写法无法确定「{m.group()}」")
        case = governed_case(text = ctx.text, start = m.start())
        spoken = self._say(ctx = ctx, n = day, written = m.group("day"), gender = "m", case = case, construct = False)
        if isinstance(spoken, Skip):
            return spoken
        out = spoken + " " + (m.group("min") or "") + m.group("month")
        if m.group("year"):
            words = self._year_words(ctx = ctx, written = m.group("year"))
            if words is None:
                return Skip(reason = f"年份写法无法确定「{m.group()}」")
            out += " " + words + self._era(m = m)
        ctx.kind(name = "日期", surface = m.group())
        return out

    def percent_ar(self, m, ctx):
        written = m.group("num") or m.group("num2")
        case, pre = self._case_and_prefix(m = m, ctx = ctx)
        integer, _, frac = written.replace("٫", ".").partition(".")
        n = int(integer)
        if n == 0 or integer.startswith("0") and len(integer) > 1:
            return Skip(reason = f"百分数写法无法确定「{m.group()}」")
        spoken = self._say(ctx = ctx, n = n, written = integer, gender = "m", case = case, construct = False)
        if isinstance(spoken, Skip):
            return spoken
        if frac:
            if frac == "2":
                return Skip(reason = f"小数部分 2 的格无法确定「{m.group()}」")
            digit = "صفر" if frac == "0" else _below_100(n = int(frac), gender = "m", case = "obl")
            spoken = spoken + " فاصلة " + digit
            ctx.checks[-1] = (spoken, written.replace("٫", "."))
        ctx.kind(name = "百分数", surface = m.group())
        return pre + spoken + " " + (m.group("word") or "بالمئة")

    def _with_noun(self, ctx, n: int, written: str, case, forms: tuple):
        """数加由本规则生成的名词（单位、货币）：按数选单数、复数或宾格。名词都是阳性。"""
        last = n % 100
        if n in (1, 2) or (n > 100 and last in (1, 2)):
            return Skip(reason = f"1、2 加名词的读法（名词单数、双数）不处理「{written}」")
        singular, plural, accusative = forms
        noun = plural if 3 <= last <= 10 else accusative if last else singular
        spoken = self._say(ctx = ctx, n = n, written = written, gender = "m", case = case, construct = True)
        if isinstance(spoken, Skip):
            return spoken
        return spoken, noun, last

    def currency_ar(self, m, ctx):
        written = m.group("num") or m.group("num2")
        symbol = m.group("cur") or m.group("cur2")
        n = int(re.sub(pattern = "[,٬]", repl = "", string = written))
        if n == 0:
            return Skip(reason = f"货币金额写法无法确定「{m.group()}」")
        after = _norm(word = _word_after(text = ctx.text, end = m.end()))
        if after in SCALE_WORDS:
            return Skip(reason = f"货币符号后的数量级词暂不处理「{m.group()} {after}」")
        case, pre = self._case_and_prefix(m = m, ctx = ctx)
        out = self._with_noun(ctx = ctx, n = n, written = written, case = case, forms = CURRENCY_SYMBOLS[symbol])
        if isinstance(out, Skip):
            return out
        spoken, noun, _ = out
        ctx.kind(name = "货币", surface = m.group())
        return pre + spoken + " " + noun

    def unit_ar(self, m, ctx):
        written = m.group("num")
        n = int(re.sub(pattern = "[,٬]", repl = "", string = written))
        if n == 0:
            return Skip(reason = f"单位前的数写法无法确定「{m.group()}」")
        case, pre = self._case_and_prefix(m = m, ctx = ctx)
        out = self._with_noun(ctx = ctx, n = n, written = written, case = case, forms = UNIT_ABBREV[m.group("unit")])
        if isinstance(out, Skip):
            return out
        spoken, noun, last = out
        if m.group("square"):
            noun += " " + ("مربعة" if 3 <= last <= 10 else "مربعًا" if last else "مربع")
        ctx.kind(name = "单位", surface = m.group())
        return pre + spoken + " " + noun

    def counted(self, m, ctx):
        written = m.group("num")
        noun = _norm(word = m.group("noun"))
        n = int(re.sub(pattern = "[,٬]", repl = "", string = written))
        last = n % 100
        if noun in SCALE_WORDS:
            _, form = SCALE_WORDS[noun]
            gender, expected = "m", form
        elif noun in LEXICON:
            gender, expected = LEXICON[noun]
        else:
            return None
        if n == 0 or written.startswith("0"):
            return Skip(reason = f"数字写法无法确定「{written}」")
        if n in (1, 2) or (n > 100 and last in (1, 2)):
            return Skip(reason = f"1、2 加名词的读法（名词单数、双数）不处理「{m.group()} {m.group('noun')}」")
        needed = ("pl",) if 3 <= last <= 10 else ("acc", "sg") if last else ("sg",)
        if expected not in needed:
            return Skip(reason = f"被数名词的形式与数不符「{m.group()} {m.group('noun')}」")
        case, pre = self._case_and_prefix(m = m, ctx = ctx)
        spoken = self._say(ctx = ctx, n = n, written = written, gender = gender, case = case, construct = True)
        if isinstance(spoken, Skip):
            return spoken
        ctx.kind(name = "数加名词", surface = m.group() + " " + m.group("noun"))
        return pre + spoken

    def tatweel_preposition(self, m, ctx):
        p, word = m.group("p"), m.group("w")
        if p == "ك" and _norm(word = word) == "م":
            return None
        ctx.kind(name = "删去拉长符", surface = m.group())
        # ل 加冠词 ال 写作 لل
        if p == "ل" and word.startswith("ال"):
            return "ل" + word[1:]
        return p + word

    def tatweel_word(self, m, ctx):
        ctx.kind(name = "删去拉长符", surface = m.group())
        return m.group().replace("\u0640", "")

    def cardinal_with_waw(self, m, ctx):
        n = self.parse(written = m.group("num"))
        if n is None:
            return Skip(reason = f"数字写法无法确定「{m.group('num')}」")
        out = self.cardinal(m = m, ctx = ctx, n = n)
        if isinstance(out, str):
            ctx.kind(name = "基数", surface = m.group("num"))
            return (m.group("waw") or "") + out
        return out

    def cardinal(self, m, ctx, n):
        """后面不是被数名词的数：读法与性、格无关（或格可由前文确定）时改写。"""
        if not n.is_int or n.value == 0:
            return Skip(reason = f"阿拉伯语小数、零的读法暂不处理「{m.group("num")}」")
        after = _norm(word = _word_after(text = ctx.text, end = m.end()))
        rest = ctx.text[m.end():].lstrip()
        if after and not rest.startswith(("،", ",", ".", "؛", ";", ")", "(", "»", "\"", ":", "؟", "!")) and after not in FUNCTION_AFTER:
            return Skip(reason = f"阿拉伯语数词与后面名词的性、数、格一致无法确定「{m.group("num")}」")
        case = governed_case(text = ctx.text, start = m.start("num"))
        gender = "m"
        if gender_matters(n = n.value):
            # 「بين 18 و 30 سنة」「من 3 إلى 5 أيام」：与后一个数共用名词，性取这个名词的性
            shared = _SHARED_NOUN.match(string = ctx.text[m.end():])
            noun = _norm(word = shared.group(1)) if shared else ""
            if noun not in LEXICON:
                return Skip(reason = f"阿拉伯语数词的性取决于所数的名词「{m.group("num")}」")
            gender = LEXICON[noun][0]
        return self._say(ctx = ctx, n = n.value, written = m.group("num"), gender = gender, case = case, construct = False)


LANG = Ar
