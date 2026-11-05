"""德语：基数、年份、年代（1980er）、序数与日期的变格、小数、百分数、单位、货币、时刻、常用缩写和君主序号。

德语数词只有「1」随名词变格（ein、eine、einen、einem），值为 1 或末两位为 01 的数在名词前丢弃，作年份
（2001 → zweitausendeins）和名词后的编号（Platz 1、§ 1 → eins）时照常读。

序数（3. Liga、am 10. August）按形容词弱变化和混合变化选词尾：前面是 am、im、vom、zum、dem、den、des、
einem、seinem 等时用 -en，是 das、ins 时用 -e；前面是 der、die、eine、ein、sein 等时要看名词的性和数，
名词在词表中（Jahrhundert 中性、Liga 阴性、月份阳性）才改写；前面是形容词时与形容词同词尾（im frühen
19. Jahrhundert）；并列的序数（am 3. und 4. Mai）沿用第一个的词尾。其余（没有冠词、名词不在词表中）丢弃。

1000–2099 的四位数：后面是大写名词时是数量（rund 1100 Einwohner），前面是表示数量的词也是数量，前面是
冠词、后面是形容词加名词时无法判断，丢弃；其余读作年份（neunzehnhundert…，2001 读 zweitausendeins）。

君主序号（Heinrich VIII.）按格读作「der Achte、den Achten、dem Achten、des Achten」：名字带属格 -s、
前面是支配与格或宾格的介词、或位于句首时能确定格，其余丢弃。
"""

from __future__ import annotations

import re

from ..numwords import n2w
from ..rules import Number, Rule, Skip, alternation
from .base import Lang, word_after, word_before
from .euro import (
    abbreviation_rules, initialism_rule, number_word_guard, number_words, roman_value, sentence_end_after,
    space_thousands_pattern, unit_follows,
)

MONTHS = (
    "Januar", "Jänner", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober",
    "November", "Dezember",
)
WEEKDAYS = ("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonnabend", "Sonntag")
# 后面是大写名词时，前面这些词表示数量
QUANTITY_BEFORE = {
    "mit", "über", "unter", "rund", "etwa", "circa", "fast", "knapp", "als", "insgesamt", "auf", "zu", "für",
    "gegen", "nur", "je", "pro", "mehr", "weniger", "alle", "allen", "sämtliche", "ungefähr", "annähernd",
    "nahezu", "mindestens", "höchstens", "weitere", "weiteren", "ca",
}
# 后面是大写名词时，前面这些词既可能引出年份也可能引出数量（bis 1998 Leiter、von 1500 Menschen）
YEAR_OR_QUANTITY = {"von", "bis", "zwischen", "ab", "seit", "um", "und", "oder"}
DETERMINERS = {"die", "der", "den", "dem", "des", "das", "diese", "dieser", "diesen", "diesem", "dieses"}
# 年份前常见的词
YEAR_WORDS = {
    "jahr", "jahre", "jahres", "jahrs", "seit", "anfang", "ende", "mitte", "herbst", "frühjahr", "frühling",
    "sommer", "winter", "um", "anno", "ab", "bis", "von", "im", "in", "nach", "vor", "zwischen", "und", "oder",
    "bzw", "sowie", "erst", "bereits", "schon", "noch", "spätestens", "frühestens", "saison", "stand",
} | {m.lower() for m in MONTHS}
STARTERS = {
    "Der", "Die", "Das", "Er", "Sie", "Es", "Im", "Am", "In", "Nach", "Mit", "Von", "Vom", "Bei", "Zum", "Zur",
    "Auf", "Aus", "Seit", "Ab", "Bis", "Ein", "Eine", "Dieser", "Diese", "Dieses", "Damit", "Dabei", "Danach",
    "Dort", "Hier", "Heute", "Später", "Auch", "Außerdem", "Zudem", "Allerdings", "Jedoch", "Doch", "Als",
    "Wie", "Wenn", "Während", "Nachdem", "Bevor", "Seine", "Sein", "Ihre", "Ihr", "Den", "Dem", "Des", "So",
    "Da", "Dann", "Dies", "Diesen", "Unter", "Über", "Für", "Gegen", "Zwischen", "Neben", "Vor", "Hinter",
    "Insgesamt", "Bereits", "Erst", "Noch", "Nur", "Weitere", "Zahlreiche", "Viele", "Einige", "Man", "Wir",
    "Ich", "Durch", "Um", "Trotz", "Wegen", "Laut", "Gemäß", "Anfang", "Ende", "Mitte",
}

# 序数前的限定词 → 词尾。"?" 表示要看名词的性和数（见 det_ending）
DET_ENDING = {
    "am": "en", "im": "en", "vom": "en", "zum": "en", "beim": "en", "zur": "en", "dem": "en", "den": "en",
    "des": "en", "einem": "en", "einen": "en", "eines": "en", "einer": "en", "diesem": "en", "diesen": "en",
    "jedem": "en", "jeden": "en", "das": "e", "ins": "e", "ans": "e", "aufs": "e", "fürs": "e", "ums": "e",
    "übers": "e", "durchs": "e", "vors": "e", "hinters": "e",
    "der": "?der", "die": "?die", "diese": "?die", "eine": "?die", "ein": "?ein",
}
for _stem in ("mein", "dein", "sein", "ihr", "unser", "euer", "kein"):
    for _suffix in ("em", "en", "es", "er"):
        DET_ENDING[_stem + _suffix] = "en"
    DET_ENDING[_stem + "e"] = "?die"
    DET_ENDING[_stem] = "?ein"
DET_ENDING["eure"] = "?die"
DET_ENDING["euren"] = DET_ENDING["eurem"] = DET_ENDING["eures"] = DET_ENDING["eurer"] = "en"
DET_ENDING["dieses"] = "?des"

# 序数后常见名词的性（m、f、n）。较长的写法也匹配复合词的末尾（Panzerdivision → division）。
NOUN_GENDER = {
    "jahrhundert": "n", "jahrtausend": "n", "jh.": "n", "jhd.": "n", "jhdt.": "n", "platz": "m", "minute": "f",
    "dynastie": "f", "klasse": "f", "geburtstag": "m", "lebensjahr": "n", "staffel": "f", "todestag": "m",
    "geschoss": "n", "obergeschoss": "n", "spieltag": "m", "episode": "f", "person": "f", "kompanie": "f",
    "staatsexamen": "n", "partie": "f", "division": "f", "rang": "m", "herzog": "m", "arrondissement": "n",
    "armee": "f", "auflage": "f", "konferenz": "f", "session": "f", "album": "n", "wahlperiode": "f",
    "jahrestag": "m", "satz": "m", "liga": "f", "runde": "f", "weltkrieg": "m", "regiment": "n", "stufe": "f",
    "mannschaft": "f", "hälfte": "f", "konzil": "n", "zeile": "f", "mal": "n", "wahl": "f", "bezirk": "m",
    "stelle": "f", "akt": "m", "instanz": "f", "etappe": "f", "lauf": "m", "kongress": "m", "parteitag": "m",
    "sinfonie": "f", "symphonie": "f", "kapitel": "n", "band": "m", "buch": "n", "sitzung": "f",
    "ausgabe": "f", "saison": "f", "generation": "f", "grad": "m", "jahrgang": "m", "kreuzzug": "m",
    "legislaturperiode": "f", "periode": "f", "stock": "m", "etage": "f", "brigade": "f", "bataillon": "n",
    "korps": "n", "flotte": "f", "republik": "f", "reich": "n", "woche": "f", "monat": "m", "quartal": "n",
    "semester": "n", "schuljahr": "n", "preis": "m", "sieg": "m", "titel": "m", "gruppe": "f", "kammer": "f",
    "senat": "m", "bundestag": "m", "landtag": "m", "kabinett": "n", "regierung": "f", "teil": "m",
    "abschnitt": "m", "paragraph": "m", "paragraf": "m", "artikel": "m", "absatz": "m", "strophe": "f",
    "szene": "f", "folge": "f", "rennen": "n", "spiel": "n", "turnier": "n", "finale": "n", "konzert": "n",
    "festival": "n", "gipfel": "m", "versuch": "m", "durchgang": "m", "wettbewerb": "m", "meisterschaft": "f",
    "olympiade": "f", "amtszeit": "f", "ehe": "f", "sohn": "m", "tochter": "f", "kind": "n", "bruder": "m",
    "schwester": "f", "präsident": "m", "könig": "m", "kaiser": "m", "graf": "m", "fürst": "m", "bischof": "m",
    "papst": "m", "kanzler": "m", "ebene": "f", "zone": "f", "fc": "m", "ffc": "m", "fsv": "m", "sv": "m",
    "tag": "m", "jahr": "n", "advent": "m", "sonntag": "m", "studio": "n", "station": "f", "streich": "m",
    "parlament": "n", "legislatur": "f", "infanteriedivision": "f", "panzerdivision": "f", "armeekorps": "n",
    "ordnung": "f", "größe": "f", "größenklasse": "f", "jahrgangsstufe": "f", "qualifikationsrunde": "f",
    "tabellenplatz": "m", "startplatz": "m", "gesamtrang": "m", "kreuz": "n", "zug": "m", "position": "f",
    "dan": "m", "hauptstadt": "f", "stadt": "f", "film": "m", "roman": "m", "staffelfinale": "n",
    "bundesliga": "f", "oberliga": "f", "regionalliga": "f", "landesliga": "f", "kreisliga": "f",
    "verbandsliga": "f", "bezirksliga": "f", "heimspiel": "n", "auswärtsspiel": "n", "länderspiel": "n",
    "tor": "n", "treffer": "m", "punkt": "m", "sekunde": "f", "stunde": "f", "gebot": "n", "brief": "m",
} | {m.lower(): "m" for m in MONTHS}
# 没有冠词时，介词决定的格（序数按强变化）；Stand 1. Januar 读主格
STRONG_PREPS = {
    "seit": "dat", "von": "dat", "ab": "dat", "mit": "dat", "nach": "dat", "aus": "dat", "bei": "dat",
    "zu": "dat", "bis": "acc", "für": "acc", "durch": "acc", "ohne": "acc", "gegen": "acc", "stand": "nom",
}
STRONG_ENDING = {
    ("nom", "m"): "er", ("nom", "f"): "e", ("nom", "n"): "es", ("acc", "m"): "en", ("acc", "f"): "e",
    ("acc", "n"): "es", ("dat", "m"): "em", ("dat", "f"): "er", ("dat", "n"): "em",
}
# 名词后的固定属格短语：Orden 2. Klasse → zweiter Klasse，Cousine 2. Grades → zweiten Grades
GENITIVE_PHRASES = {"Klasse": "er", "Ordnung": "er", "Grades": "en", "Güte": "er", "Wahl": "er"}
# 年份后常见的分词（das 1412 verfasste Werk、der 1959 gegründeten Gesellschaft）
YEAR_PARTICIPLES = (
    "gegründet", "errichtet", "erbaut", "gebaut", "verfasst", "eröffnet", "entstanden", "geboren", "gestorben",
    "erschienen", "veröffentlicht", "angelegt", "gestiftet", "eingeweiht", "fertiggestellt", "geschlossen",
    "abgerissen", "renoviert", "umgebaut", "entdeckt", "beschrieben", "gedreht", "uraufgeführt", "verliehen",
    "eingeführt", "gegossen", "geweiht", "zerstört", "aufgelöst", "begonnen", "begründet", "gemalt",
    "geschaffen", "komponiert", "verstorben", "erworben", "erlassen", "unterzeichnet", "abgeschlossen",
)
# 名词后读编号的词（Platz 1、Nummer 1 → eins）
ID_NOUNS = {
    "nummer", "platz", "kapitel", "folge", "staffel", "teil", "band", "abschnitt", "artikel", "absatz",
    "paragraf", "paragraph", "gleis", "formel", "bundesstraße", "straße", "weg", "abteilung", "zelle", "ebene",
    "stufe", "klasse", "gruppe", "runde", "spieltag", "tor", "halle", "saal", "linie", "bahnsteig", "version",
    "serie", "episode", "kanal", "kategorie", "level", "typ", "modell", "seite", "tabelle", "abbildung",
    "landesstraße", "kreisstraße", "autobahn", "route", "staatsstraße", "buslinie", "haus", "block", "sektor",
    "zimmer", "raum", "heft", "lektion", "übung", "aufgabe", "strophe", "akt", "szene", "satz", "liga", "rang",
}
ABBREVIATIONS = {
    "z. B.": ("zum Beispiel", "any"), "z.B.": ("zum Beispiel", "any"), "d. h.": ("das heißt", "any"),
    "d.h.": ("das heißt", "any"), "usw.": ("und so weiter", "any"), "etc.": ("et cetera", "any"),
    "bzw.": ("beziehungsweise", "any"), "ca.": ("circa", "any"), "vgl.": ("vergleiche", "any"),
    "ggf.": ("gegebenenfalls", "any"), "evtl.": ("eventuell", "any"), "inkl.": ("inklusive", "any"),
    "insb.": ("insbesondere", "any"), "bspw.": ("beispielsweise", "any"), "z. T.": ("zum Teil", "any"),
    "v. a.": ("vor allem", "any"), "u. U.": ("unter Umständen", "any"), "i. d. R.": ("in der Regel", "any"),
    "s. o.": ("siehe oben", "any"), "s. u.": ("siehe unten", "any"), "u. ä.": ("und Ähnliches", "any"),
    "o. Ä.": ("oder Ähnliches", "any"), "u. Ä.": ("und Ähnliches", "any"), "o. ä.": ("oder Ähnliches", "any"),
    "v. Chr.": ("vor Christus", "any"), "n. Chr.": ("nach Christus", "any"), "Dr.": ("Doktor", "name"),
    "Prof.": ("Professor", "name"), "St.": ("Sankt", "name"), "Mr.": ("Mister", "name"),
    "z. Zt.": ("zurzeit", "any"), "u. v. a.": ("und viele andere", "any"), "u. v. m.": ("und vieles mehr", "any"),
}
# 编号缩写后接数字（Nr. 5 → Nummer fünf）
NUMBER_ABBREVIATIONS = {
    "Nr": "Nummer", "Abs": "Absatz", "Art": "Artikel", "Abb": "Abbildung", "Bd": "Band", "Kap": "Kapitel",
    "S": "Seite",
}
# 数量缩写（5 Mio. → fünf Millionen）：(单数, 复数)
SCALE_ABBREVIATIONS = {"Mio": ("Million", "Millionen"), "Mrd": ("Milliarde", "Milliarden"), "Tsd": ("Tausend", "Tausend")}
# 君主、教皇名（阳性；FEMALE_REGNAL 为阴性）
REGNAL_NAMES = {
    "Heinrich", "Ludwig", "Friedrich", "Wilhelm", "Karl", "Otto", "Konrad", "Rudolf", "Albrecht", "Leopold",
    "Maximilian", "Joseph", "Josef", "Franz", "Ferdinand", "Philipp", "Johann", "Johannes", "Paul", "Pius",
    "Benedikt", "Gregor", "Innozenz", "Leo", "Clemens", "Klemens", "Alexander", "Nikolaus", "Peter", "Iwan",
    "Napoleon", "Eduard", "Georg", "Richard", "Jakob", "Gustav", "Christian", "Harald", "Olav", "Haakon",
    "Waldemar", "Erich", "Magnus", "Sigismund", "Kasimir", "Stephan", "Andreas", "Ladislaus", "Matthias",
    "Sixtus", "Urban", "Julius", "Hadrian", "Bonifatius", "Martin", "Eugen", "Felix", "Viktor", "Emanuel",
    "Umberto", "Alfons", "Sancho", "Pedro", "Juan", "Carlos", "Felipe", "Manuel", "Michael", "Konstantin",
    "Theodosius", "Valentinian", "Justinian", "Alarich", "Theoderich", "Theuderich", "Chlodwig", "Childebert",
    "Chlothar", "Dagobert", "Pippin", "Karlmann", "Lothar", "Arnulf", "Berengar", "Ottokar", "Wenzel",
    "Albert", "Amadeus", "Mehmed", "Murad", "Selim", "Bayezid", "Mustafa", "Ahmed", "Schapur", "Chosrau",
    "Darius", "Xerxes", "Artaxerxes", "Antiochos", "Seleukos", "Ptolemaios", "Amenophis", "Thutmosis",
    "Ramses", "Sesostris", "Psammetich", "Bernhard", "Ernst", "Eberhard", "Ulrich", "Hermann", "Adolf",
    "Wladislaw", "Boleslaw", "Mieszko", "Béla", "Géza", "Andronikos", "Manuel", "Basileios", "Romanos",
    "Leopold", "Kaspar", "Balduin", "Gottfried", "Amalrich", "Bohemund", "Raimund", "Hugo", "Robert",
}
FEMALE_REGNAL = {
    "Katharina", "Elisabeth", "Maria", "Kleopatra", "Isabella", "Johanna", "Margarethe", "Anna", "Christina",
    "Viktoria", "Wilhelmina", "Juliana", "Beatrix", "Ulrike", "Eleonore", "Theodora", "Irene", "Zoe",
}
REGNAL_TITLES = {
    "Cousin", "Sohn", "Bruder", "Vater", "Onkel", "Neffe", "Enkel", "Großvater", "Nachfolger", "Vorgänger",
    "Halbbruder", "Schwiegersohn", "Schwiegervater", "Gemahl", "Gatte", "Ehemann", "Mutter", "Tochter",
    "Schwester", "Witwe", "Gemahlin", "Ehefrau", "Nichte", "Cousine", "Großmutter",
    "König", "Kaiser", "Papst", "Herzog", "Graf", "Fürst", "Zar", "Pharao", "Königin", "Kaiserin", "Kurfürst",
    "Markgraf", "Landgraf", "Pfalzgraf", "Erzherzog", "Großherzog", "Prinz", "Prinzessin", "Patriarch",
    "Bischof", "Erzbischof", "Herzogin", "Gräfin", "Fürstin", "Zarin", "Sultan", "Schah", "Großfürst",
}
DATIVE_PREPS = {"mit", "von", "bei", "nach", "zu", "aus", "seit", "unter", "gegenüber", "nebst", "samt", "vor", "neben"}
ACCUSATIVE_PREPS = {"für", "gegen", "durch", "ohne", "um", "wider"}
REGNAL_ARTICLE = {
    ("m", "nom"): ("der", "e"), ("m", "acc"): ("den", "en"), ("m", "dat"): ("dem", "en"), ("m", "gen"): ("des", "en"),
    ("f", "nom"): ("die", "e"), ("f", "acc"): ("die", "e"), ("f", "dat"): ("der", "en"), ("f", "gen"): ("der", "en"),
}
# 复合形容词、复合名词中数字后的部分（30-jährig、100-prozentig、20-Minuten-Takt）
COMPOUND_TAIL = (
    "jährig", "prozentig", "fach", "stündig", "tägig", "wöchig", "monatig", "minütig", "seitig", "köpfig",
    "teilig", "stöckig", "geschossig", "spurig", "gleisig", "malig", "stellig", "sitzig", "achsig", "zylindrig",
    "bändig", "sätzig", "eckig", "stufig", "polig", "farbig", "zeilig", "wertig", "reihig", "kanalig",
)


def needs_inflection(n: int) -> bool:
    """1 和末两位为 01 的数在名词前要变格（ein Mann、hundertein Jahre）。"""
    return n == 1 or n % 100 == 1


def noun_gender(word: str) -> str | None:
    """名词的性：完整匹配词表，或以词表中 5 个字母以上的词结尾（复合词）。"""
    w = word.lower().split("-")[-1]
    if w in NOUN_GENDER:
        return NOUN_GENDER[w]
    best = None
    for key, gender in NOUN_GENDER.items():
        if len(key) >= 5 and w.endswith(key) and (best is None or len(key) > len(best[0])):
            best = (key, gender)
    return best[1] if best else None


class De(Lang):
    code = "de"
    point = "Komma"
    percent = "Prozent"
    units = {
        "km/h": ("Kilometer pro Stunde", "Kilometer pro Stunde"), "km²": ("Quadratkilometer", "Quadratkilometer"),
        "m²": ("Quadratmeter", "Quadratmeter"), "km": ("Kilometer", "Kilometer"), "cm": ("Zentimeter", "Zentimeter"),
        "mm": ("Millimeter", "Millimeter"), "kg": ("Kilogramm", "Kilogramm"), "m": ("Meter", "Meter"),
        "g": ("Gramm", "Gramm"), "°C": ("Grad Celsius", "Grad Celsius"), "℃": ("Grad Celsius", "Grad Celsius"),
        "°F": ("Grad Fahrenheit", "Grad Fahrenheit"), "℉": ("Grad Fahrenheit", "Grad Fahrenheit"),
    }
    currencies = {"€": ("Euro", "Euro"), "$": ("Dollar", "Dollar"), "£": ("Pfund", "Pfund")}

    def card(self, n: int) -> str:
        return n2w(n = n, lang = "de")

    def digits(self, digits: str) -> str:
        return " ".join(n2w(n = int(d), lang = "de") for d in digits)

    def ordinal(self, n: int, ending: str) -> str:
        """序数加词尾：dritte + n → dritten。"""
        return n2w(n = n, lang = "de", to = "ordinal")[:-1] + ending

    def unit_number(self, m, ctx, n):
        if n.is_int and needs_inflection(n = n.value):
            return Skip(reason = f"「1」要随格变化「{m.group()}」")
        return self.say(n = n, ctx = ctx)

    currency_number = unit_number

    # ------------------------------------------------------------------
    # 规则
    # ------------------------------------------------------------------

    def rules(self) -> list:
        words = number_words(card = self.card, ordinal = lambda n: n2w(n = n, lang = "de", to = "ordinal"))
        words |= {"ein", "eine", "einen", "einem", "einer", "eines", "erster", "ersten", "erste", "erstes"}
        early = [number_word_guard(words = words, conjunctions = ("und",))]
        early += abbreviation_rules(table = ABBREVIATIONS, starters = STARTERS) + [
            Rule(
                pattern = re.compile(
                    pattern = rf"(?<![\w.])(?P<abbr>{alternation(words = NUMBER_ABBREVIATIONS)})\.\s?(?P<num>\d+)(?![\d]|[.,]\d)",
                ),
                handle = self.numbered,
            ),
            Rule(
                pattern = re.compile(pattern = r"(?P<sym>§§?)\s?(?P<num>\d+)(?![\d]|[.,]\d)(?:\s(?P<to>bis|und)\s(?P<num2>\d+)(?![\d]|[.,]\d))?"),
                handle = self.paragraph,
            ),
            Rule(pattern = re.compile(pattern = r"(?<![\w.])u\. ?a\.(?![^\W\d_])"), handle = self.unter_anderem),
            initialism_rule(starters = STARTERS),
            Rule(pattern = re.compile(pattern = space_thousands_pattern()), handle = self.space_thousands),
            Rule(
                pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>\d{1,4})\.(?=\s)(?P<chain>(?:\s(?:und|bis|oder|sowie|bzw\.)\s\d{1,4}\.)*)\s(?P<noun>[^\W\d_][\w-]*\.?)"),
                handle = self.ordinal_phrase,
            ),
            Rule(
                pattern = re.compile(pattern = r"(?<![\w])(?P<roman>[IVXLCDM]+)\.(?=\s|$)"),
                handle = self.regnal,
            ),
            Rule(
                pattern = re.compile(pattern = r"(?<![\w])(?P<roman>[IVXLCDM]{2,})(?![\w.])"),
                handle = self.roman_cardinal,
            ),
            Rule(
                pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>\d{1,4}|1\d{3}|20\d{2})er(?P<tail>-Jahre\w*|-|\b)"),
                handle = self.decade,
            ),
            Rule(
                pattern = re.compile(pattern = rf"(?<![\d.,\w-])(?P<num>\d+)-(?P<tail>(?:{alternation(words = COMPOUND_TAIL)})\w*|[A-ZÄÖÜ][\w-]*)"),
                handle = self.compound,
            ),
        ]
        return early + super().rules()

    def extra_rules(self) -> list:
        return [
            Rule(
                pattern = re.compile(pattern = r"(?<![\d.,])(?P<h>\d{1,2})[:.](?P<mm>\d{2})\s?Uhr\b"),
                handle = self.time,
            ),
            Rule(pattern = re.compile(pattern = r"(?<![\d.,])(?P<h>\d{1,2})\s?Uhr\b"), handle = self.time),
            Rule(
                pattern = re.compile(
                    pattern = r"(?P<num>\d+(?:,\d+)?)\s?(?P<scale>Mio|Mrd|Tsd)\.(?:\s?(?P<cur>[€$£]))?",
                ),
                handle = self.scale_abbreviation,
            ),
            Rule(pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>\d{1,4})\.(?=\s+\S|[^\W\d_]|\s*$)"), handle = self.dotted),
            Rule(pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>1\d{3}|20\d{2})(?![\d]|[.,]\d)"), handle = self.year),
        ]

    # ------------------------------------------------------------------
    # 缩写与符号
    # ------------------------------------------------------------------

    def numbered(self, m, ctx):
        if m.group("num").startswith("0"):
            return Skip(reason = f"编号写法无法确定「{m.group()}」")
        ctx.kind(name = "缩写", surface = m.group("abbr") + ".")
        n = int(m.group("num"))
        return NUMBER_ABBREVIATIONS[m.group("abbr")] + " " + ctx.num(spoken = self.card(n = n), written = m.group("num"))

    def paragraph(self, m, ctx):
        """§ 5 → Paragraf fünf，§§ 121 bis 150 → Paragrafen hunderteinundzwanzig bis hundertfünfzig。"""
        ctx.kind(name = "符号", surface = m.group("sym"))
        word = "Paragrafen" if m.group("sym") == "§§" else "Paragraf"
        out = word + " " + ctx.num(spoken = self.card(n = int(m.group("num"))), written = m.group("num"))
        if m.group("num2"):
            out += " " + m.group("to") + " " + ctx.num(spoken = self.card(n = int(m.group("num2"))), written = m.group("num2"))
        return out

    def unter_anderem(self, m, ctx):
        """u. a. 在句中作状语时读 unter anderem；在名单末尾（Müller, Schmidt u. a.）是 und andere，丢弃。"""
        before = ctx.text[:m.start()].rstrip()
        rest = ctx.text[m.end():].lstrip()
        if not rest or rest[0] in ".)],;" or re.search(pattern = r",\s*[A-ZÄÖÜ][^\W\d_]*$", string = before):
            return Skip(reason = f"缩写读法取决于上下文「{m.group()}」")
        ctx.kind(name = "缩写", surface = m.group())
        return "unter anderem"

    def scale_abbreviation(self, m, ctx):
        n = self.parse(written = m.group("num"))
        if n is None:
            return Skip(reason = f"数字写法无法确定「{m.group()}」")
        if n.is_int and n.value == 1:
            return Skip(reason = f"「1」要随格变化「{m.group()}」")
        singular, plural = SCALE_ABBREVIATIONS[m.group("scale")]
        out = self.join(self.say(n = n, ctx = ctx), plural)
        if m.group("cur"):
            out += " " + self.currencies[m.group("cur")][1]
        elif sentence_end_after(text = ctx.text, end = m.end(), starters = STARTERS):
            out += "."
        ctx.kind(name = "缩写", surface = m.group("scale") + ".")
        return out

    def space_thousands(self, m, ctx):
        """10 000、299 792 458：以空格分千位的数按一个数读（后面是单位时丢弃）。"""
        if unit_follows(text = ctx.text, end = m.end(), units = self.units):
            return Skip(reason = f"空格分隔的数字后接单位「{m.group()}」")
        digits = m.group().replace(" ", "")
        n = Number(
            written = digits,
            value = int(digits),
            frac = None,
            int_digits = digits,
            grouped = True,
        )
        out = self.cardinal(m = m, ctx = ctx, n = n)
        if isinstance(out, str):
            ctx.kind(name = "空格分千位", surface = m.group())
        return out

    # ------------------------------------------------------------------
    # 序数、日期
    # ------------------------------------------------------------------

    def det_ending(self, text: str, start: int, noun: str) -> str | None:
        """由序数前的限定词（或与它同词尾的形容词）和名词决定序数词尾，无法确定时返回 None。"""
        head = text[:start]
        prev_m = re.search(pattern = r"([^\W\d_]+)(\s*[,:]?\s*)$", string = head)
        if not prev_m:
            return None
        prev, sep = prev_m.group(1), prev_m.group(2).strip()
        low = prev.lower()
        words = re.findall(pattern = r"[^\W\d_]+", string = head[:prev_m.start(1)][-40:])
        gender = noun_gender(word = noun) if re.fullmatch(pattern = r"[^\W\d_][\w-]*\.?", string = noun) else None
        if sep == ",":
            # 日期前的星期：am Dienstag, 20. Juni
            if prev in WEEKDAYS and words and words[-1].lower() == "am":
                return "en"
            return None
        if sep == ":":
            return "er" if low == "stand" and gender == "m" else None
        # 形容词与序数同词尾（im frühen 19. Jahrhundert），形容词前必须是限定词
        if low not in DET_ENDING and prev[:1].islower() and words and words[-1].lower() in DET_ENDING:
            m = re.search(pattern = r"(en|em|er|es|e)$", string = low)
            return m.group(1) if m else None
        # 没有冠词：介词决定格，按强变化（seit 25. August → seit fünfundzwanzigstem August）
        if low in STRONG_PREPS and gender is not None:
            return STRONG_ENDING.get((STRONG_PREPS[low], gender))
        # 名词后的固定属格短语（Orden 2. Klasse、Cousine 2. Grades）
        if prev[:1].isupper() and low not in DET_ENDING and noun in GENITIVE_PHRASES:
            return GENITIVE_PHRASES[noun]
        rule = DET_ENDING.get(low)
        if rule is None:
            if noun.upper() in ("FC", "FFC", "FSV", "SV") and (not head.strip() or head.rstrip()[-1] in ".(,:"):
                return "er"
            return None
        if not rule.startswith("?"):
            return rule
        if gender is None:
            return None
        if rule == "?der":
            # der + 阳性单数：主格 -e；der + 阴性单数：与格、属格 -en
            return {"m": "e", "f": "en"}.get(gender)
        if rule == "?die":
            return "e" if gender == "f" else None
        if rule == "?ein":
            return {"m": "er", "n": "es"}.get(gender)
        if rule == "?des":
            return "en" if noun.lower().endswith("s") and gender in ("m", "n") else None
        return None

    def ordinal_phrase(self, m, ctx):
        """序数（含并列的序数）加名词：am 3. und 4. Mai → am dritten und vierten Mai。"""
        noun = m.group("noun")
        text = ctx.text
        if len(m.group("num")) == 4 and not m.group("chain"):
            # 四位数后的点是句号（1968. Anschließend …）
            return None
        if noun in STARTERS or noun[:1].islower() and noun not in ("und", "bis"):
            # 后面是句首词：数字后的点是句号；后面是小写词：可能是 die 3. größte Stadt，丢弃
            if noun in STARTERS:
                return None
            return Skip(reason = f"数字后的句点可能表示序数「{m.group()}」")
        if not noun[:1].isupper():
            return Skip(reason = f"数字后的句点可能表示序数「{m.group()}」")
        abbrev = noun in ("Jh.", "Jhd.", "Jhdt.", "Jhdts.", "Jahrh.")
        if noun.endswith(".") and not abbrev:
            noun = noun[:-1]
        ending = self.det_ending(text = text, start = m.start(), noun = noun)
        if ending is None:
            return Skip(reason = f"序数的词尾取决于冠词和名词「{m.group().split(' ')[0]} {noun}」")
        numbers = [m.group("num")] + re.findall(pattern = r"(\d{1,4})\.", string = m.group("chain"))
        if any(n.startswith("0") or int(n) == 0 for n in numbers):
            return Skip(reason = f"序数写法无法确定「{m.group()}」")
        spoken = [ctx.num(spoken = self.ordinal(n = int(n), ending = ending), written = n + ".") for n in numbers]
        chain = m.group("chain")
        out = spoken[0]
        for i, link in enumerate(re.findall(pattern = r"\s(und|bis|oder|sowie|bzw\.)\s\d{1,4}\.", string = chain)):
            link = "beziehungsweise" if link == "bzw." else link
            out += f" {link} {spoken[i + 1]}"
        ctx.kind(name = "日期" if noun in MONTHS else "序数", surface = m.group())
        if abbrev:
            genitive = text[:m.start()].rstrip().split(" ")[-1].lower() in ("des", "eines", "dieses")
            word = "Jahrhunderts" if genitive else "Jahrhundert"
            if sentence_end_after(text = text, end = m.end(), starters = STARTERS):
                word += "."
            ctx.kind(name = "缩写", surface = m.group("noun"))
            return out + " " + word
        return out + " " + m.group("noun")

    def dotted(self, m, ctx):
        """数字后有句点：后面是句首词时是句号（按普通数字读），四位数也视为句末，其余可能是序数，丢弃。"""
        rest = ctx.text[m.end():]
        if not rest.strip():
            # 句末的「am 3.」是序数（am dritten），其余是句号
            if word_before(text = ctx.text, start = m.start()) in DET_ENDING or word_before(text = ctx.text, start = m.start()) in STRONG_PREPS:
                return Skip(reason = f"数字后的句点可能表示序数「{m.group()}」")
            return None
        nxt = re.match(pattern = r"\s+([^\W\d_]+)", string = rest)
        if nxt and (nxt.group(1) in STARTERS or (len(m.group("num")) == 4 and nxt.group(1)[:1].isupper())):
            return None
        return Skip(reason = f"数字后的句点可能表示序数「{m.group()}」")

    def time(self, m, ctx):
        h = int(m.group("h"))
        mm = int(m.group("mm")) if m.groupdict().get("mm") else 0
        if h > 24 or mm > 59:
            return Skip(reason = f"时刻写法无法确定「{m.group()}」")
        ctx.kind(name = "时刻", surface = m.group())
        hour = "ein" if h == 1 else self.card(n = h)
        out = ctx.num(spoken = hour, written = m.group("h")) + " Uhr"
        if mm:
            out += " " + ctx.num(spoken = self.card(n = mm), written = m.group("mm"))
        return out

    # ------------------------------------------------------------------
    # 年份、年代、复合词
    # ------------------------------------------------------------------

    def year_words(self, n: int) -> str:
        return n2w(n = n, lang = "de", to = "year") if 1100 <= n <= 1999 else self.card(n = n)

    def decade(self, m, ctx):
        """1980er Jahre → neunzehnhundertachtziger Jahre，20er-Jahre → zwanziger Jahre。"""
        n = int(m.group("num"))
        if n % 10 or n == 0 or (n >= 100 and not 1000 <= n <= 2090):
            return Skip(reason = f"年代写法无法确定「{m.group()}」")
        words = ctx.num(spoken = self.year_words(n = n), written = m.group("num"))
        ctx.kind(name = "年代", surface = m.group())
        tail = m.group("tail")
        if tail.startswith("-Jahre"):
            return words + "er " + tail[1:]
        return words + "er" + tail

    def compound(self, m, ctx):
        """30-jährige → dreißig-jährige，20-Minuten-Takt → zwanzig-Minuten-Takt（保留连字符）。"""
        n = int(m.group("num"))
        if m.group("num").startswith("0") or needs_inflection(n = n) or 1000 <= n <= 2099 and not m.group("tail")[:1].islower():
            return None
        ctx.kind(name = "复合词", surface = m.group())
        return ctx.num(spoken = self.card(n = n), written = m.group("num")) + "-" + m.group("tail")

    def year(self, m, ctx):
        n = int(m.group("num"))
        text = ctx.text
        rest = text[m.end():]
        if re.match(pattern = r"\s?(?:[%‰]|(?:km|cm|mm|kg|m|g|t|ha|h|l)\b)", string = rest):
            return None
        before = word_before(text = text, start = m.start())
        after = word_after(text = text, end = m.end())
        lead = text[:m.start()].rstrip()
        if lead[-1:] and lead[-1] in "$€£":
            return None
        # 月份后、「von 1992 bis 1998」「zwischen 1933 und 1939」中是年份
        if before in {mo.lower() for mo in MONTHS} or re.search(pattern = r"\b(?:von|zwischen)\s(?:1\d{3}|20\d{2})\s(?:bis|und)\s$", string = text[:m.start()]):
            ctx.kind(name = "年份", surface = m.group())
            return ctx.num(spoken = self.year_words(n = n), written = m.group())
        # 后面是大写名词（不是月份、句首词）：数量
        if after[:1].isupper() and after not in MONTHS and after not in STARTERS:
            if (before in QUANTITY_BEFORE or before in DETERMINERS) and before not in YEAR_OR_QUANTITY:
                if needs_inflection(n = n):
                    return Skip(reason = f"「1」要随名词变格「{m.group()}」")
                ctx.kind(name = "基数", surface = m.group())
                return ctx.num(spoken = self.card(n = n), written = m.group())
            return Skip(reason = f"四位数是年份还是数量需要上下文「{m.group()} {after}」")
        # 后面是小写形容词加名词：die 1500 verletzten Menschen 与 dem 1976 angelegten Garten 无法区分
        nxt = re.match(pattern = r"\s+([a-zäöüß]+)\s+([A-ZÄÖÜ]\w*)", string = rest)
        if nxt and nxt.group(1).startswith(YEAR_PARTICIPLES) and 1100 <= n <= 2099:
            ctx.kind(name = "年份", surface = m.group())
            return ctx.num(spoken = self.year_words(n = n), written = m.group())
        if nxt and re.search(pattern = r"(?:e|en|er|es|em)$", string = nxt.group(1)) and before in (DETERMINERS | QUANTITY_BEFORE):
            return Skip(reason = f"四位数是年份还是数量需要上下文「{m.group()} {nxt.group(1)} {nxt.group(2)}」")
        if before in QUANTITY_BEFORE - {"etwa", "circa", "ca", "gegen", "ungefähr"}:
            return Skip(reason = f"四位数是年份还是数量需要上下文「{before} {m.group()}」")
        ctx.kind(name = "年份", surface = m.group())
        return ctx.num(spoken = self.year_words(n = n), written = m.group())

    # ------------------------------------------------------------------
    # 罗马数字
    # ------------------------------------------------------------------

    def regnal(self, m, ctx):
        """Heinrich VIII. → Heinrich der Achte（格由名字的属格 -s、前面的介词或句首位置决定）。"""
        token = m.group("roman")
        value = roman_value(text = token)
        if value is None or (len(token) == 1 and token not in "IVX"):
            return None
        # 后面是大写的姓：这是名字的首字母（Peter V. Smith），不是序号
        after = word_after(text = ctx.text, end = m.end())
        if after[:1].isupper() and after not in STARTERS:
            return None if len(token) == 1 else Skip(reason = f"罗马数字的读法取决于上下文「{m.group()} {after}」")
        head = ctx.text[:m.start()]
        name_m = re.search(pattern = r"([A-ZÄÖÜ][a-zäöüé]+)\s$", string = head)
        if not name_m:
            return Skip(reason = f"罗马数字的读法取决于上下文「{m.group()}」")
        name = name_m.group(1)
        case = None
        if name in REGNAL_NAMES or name in FEMALE_REGNAL:
            base = name
        elif name.endswith("s") and (name[:-1] in REGNAL_NAMES or name[:-1] in FEMALE_REGNAL):
            base, case = name[:-1], "gen"
        else:
            return Skip(reason = f"罗马数字的读法取决于上下文「{name} {m.group()}」")
        gender = "f" if base in FEMALE_REGNAL else "m"
        if case is None:
            words = re.findall(pattern = r"[^\W\d_]+|[.!?(]", string = head[:name_m.start()])
            # 跳过称号（König、Papst）和并列的名字（Johannes Paul II.）
            while words and (words[-1] in REGNAL_TITLES or words[-1] in REGNAL_NAMES):
                words.pop()
            prev = words[-1] if words else ""
            article_case = {"der": "nom", "die": "nom", "dem": "dat", "den": "acc", "des": "gen"}
            for stem in ("sein", "ihr", "ein"):
                article_case.update({stem + "em": "dat", stem + "en": "acc", stem + "es": "gen"})
            # 冠词与名字之间的形容词（Der wirkliche Ludwig XVI.）
            if prev[:1].islower() and re.search(pattern = r"(?:e|en)$", string = prev) and len(words) >= 2 and words[-2].lower() in article_case:
                words.pop()
                prev = words[-1]
            # 从句开头的连词后，名字是主语（dass König Wenzel IV. …）
            if prev.lower() in ("dass", "weil", "nachdem", "obwohl", "bevor", "ob") or (prev == "Als" and len(words) == 1):
                prev = "."
            if prev.lower() in article_case and len(words) >= 2 and words[-2].lower() in DATIVE_PREPS | ACCUSATIVE_PREPS:
                return Skip(reason = f"君主序号的格取决于上下文「{name} {m.group()}」")
            if prev.lower() in article_case and (prev.lower() != "der" or gender == "m") and (prev.lower() != "die" or gender == "f"):
                case = article_case[prev.lower()]
            elif prev.lower() in DATIVE_PREPS:
                case = "dat"
            elif prev.lower() in ACCUSATIVE_PREPS:
                case = "acc"
            elif prev in ("", ".", "!", "?", "("):
                case = "nom"
            else:
                return Skip(reason = f"君主序号的格取决于上下文「{name} {m.group()}」")
        article, ending = REGNAL_ARTICLE[(gender, case)]
        ordinal = ctx.num(spoken = self.ordinal(n = value, ending = ending), written = f"{value}.")
        ctx.kind(name = "罗马数字", surface = f"{name} {m.group()}")
        out = article + " " + ordinal[:1].upper() + ordinal[1:]
        if sentence_end_after(text = ctx.text, end = m.end(), starters = STARTERS):
            out += "."
        return out

    def roman_cardinal(self, m, ctx):
        """不带点的罗马数字只在 Teil、Band 这类词后读作基数（Teil II → Teil zwei）。"""
        token = m.group("roman")
        value = roman_value(text = token)
        # 与 detect 一致：两个字母的 CD、DC、MC 等不算罗马数字
        if value is None or (len(token) < 3 and not set(token) <= set("IVXL")):
            return None
        prev = word_before(text = ctx.text, start = m.start())
        if prev in ("teil", "band", "akt", "kapitel", "buch", "phase", "stufe", "typ", "klasse", "staffel", "folge"):
            ctx.kind(name = "罗马数字", surface = m.group())
            return ctx.num(spoken = self.card(n = value), written = str(value))
        return Skip(reason = f"罗马数字的读法取决于上下文「{m.group()}」")

    # ------------------------------------------------------------------
    # 普通数字
    # ------------------------------------------------------------------

    def say(self, n, ctx, card = None):
        """eine Million、eine Milliarde 中的 eine 随格变化（mit einer Million …），丢弃。"""
        words = (card or self.card)(n = n.value)
        if re.search(pattern = r"\beine (?:Million|Milliarde|Billion)\b", string = words):
            return Skip(reason = f"「eine Million」要随名词变格「{n.written}」")
        return super().say(n = n, ctx = ctx, card = card)

    def cardinal(self, m, ctx, n):
        prev = word_before(text = ctx.text, start = m.start())
        if n.is_int and needs_inflection(n = n.value):
            if prev in ID_NOUNS and not word_after(text = ctx.text, end = m.end())[:1].isupper():
                return ctx.num(spoken = self.card(n = n.value), written = n.written)
            return Skip(reason = f"「1」要随名词变格「{m.group()}」")
        return self.say(n = n, ctx = ctx)


LANG = De
