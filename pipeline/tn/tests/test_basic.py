"""基础读法的逐语言测试：每种语言列出改写成功的例子和必须丢弃的例子。

    python -m unittest pipeline.tn.tests.test_basic

需要 num2words（英语、德语、法语、西班牙语、葡萄牙语、意大利语、俄语）。
"""

import unittest

from pipeline.compare import compare
from pipeline.tn import clean, normalize_entry

ZWNJ = chr(0x200C)

# 改写成功：(原文, 朗读稿)
NORMALIZED = {
    "zh": [
        ("並於1937年徹底禁止", "并于一九三七年彻底禁止"),
        ("入圍第22屆金馬獎及第5屆", "入围第二十二届金马奖及第五届"),
        ("我有2个苹果，2月2日到", "我有两个苹果，二月二日到"),
        ("增长了10%，约3.5公里", "增长了百分之十，约三点五公里"),
        ("人口20000人，花了2026元", "人口两万人，花了两千零二十六元"),
        ("14:30开会", "十四点三十分开会"),
        ("温度30℃，价格$5", "温度三十摄氏度，价格五美元"),
        ("1050年代", "一零五零年代"),
    ],
    "ja": [
        ("短大で4年間演劇を勉強した。", "短大で四年間演劇を勉強した。"),
        ("1950年代までは週2回以上", "千九百五十年代までは週二回以上"),
        ("約4万1千人分の死者名簿", "約四万一千人分の死者名簿"),
        ("毎年4月1日の人口。", "毎年四月一日の人口。"),
        ("¥500です。10%増えた。", "五百円です。十パーセント増えた。"),
    ],
    "ko": [
        ("2년 후, 르네", "이년 후, 르네"),
        ("그 계수는 2이다.", "그 계수는 이이다."),
        ("3개 가도, 1개 진을 관할한다.", "세개 가도, 한개 진을 관할한다."),
        ("1944년, 미연방", "천구백사십사년, 미연방"),
        ("제56회 그래미 어워드", "제오십육회 그래미 어워드"),
        ("6월 10일", "유월 십일"),
        ("새벽 5시에 20개", "새벽 다섯시에 스무개"),
    ],
    "en": [
        ("In 1886, Peebles was incorporated.", "In eighteen eighty-six, Peebles was incorporated."),
        ("On October 13, 1960, portions were exchanged.", "On October thirteenth, nineteen sixty, portions were exchanged."),
        ("Lake Tahoe is fed by 63 tributaries.", "Lake Tahoe is fed by sixty-three tributaries."),
        ("The 1990s were great.", "The nineteen nineties were great."),
        ("She came 21st with 10% of 600,000 votes.", "She came twenty-first with ten percent of six hundred thousand votes."),
        ("It costs $5 million.", "It costs five million dollars."),
        ("a $70 million impact", "a seventy million dollar impact"),
        ("At 9:05 it was 25°C.", "At nine oh five it was twenty-five degrees Celsius."),
    ],
    "de": [
        ("Ende 1989 vergab er den Preis.", "Ende neunzehnhundertneunundachtzig vergab er den Preis."),
        ("1920 soll sie verstorben sein.", "Neunzehnhundertzwanzig soll sie verstorben sein."),
        ("Avitus war von 455 bis 456 Kaiser.", "Avitus war von vierhundertfünfundfünfzig bis vierhundertsechsundfünfzig Kaiser."),
        ("Im Jahr 2010 gab es 21 Spieler.", "Im Jahr zweitausendzehn gab es einundzwanzig Spieler."),
        ("Um 14:30 Uhr kamen 25 % der Gäste.", "Um vierzehn Uhr dreißig kamen fünfundzwanzig Prozent der Gäste."),
        ("Es kostet 12 €.", "Es kostet zwölf Euro."),
    ],
    "fr": [
        ("Au 1er octobre 2023, le nombre est de 37.", "Au premier octobre deux mille vingt-trois, le nombre est de trente-sept."),
        ("lancé en 1991 avec 600 collaborateurs sur les 2600", "lancé en mille neuf cent quatre-vingt-onze avec six cents collaborateurs sur les deux mille six cents"),
        ("En 1900, Redon expose.", "En mille neuf cent, Redon expose."),
        ("le 3e album, 3,14 km et 10 %", "le troisième album, trois virgule quatorze kilomètres et dix pour cent"),
        ("Il a 21 ans.", "Il a vingt et un ans."),
    ],
    "es": [
        ("El rodaje finalizó en agosto de 2024.", "El rodaje finalizó en agosto de dos mil veinticuatro."),
        ("apenas 2 o 3 segundos", "apenas dos o tres segundos"),
        ("el 11 de mayo de 2025", "el once de mayo de dos mil veinticinco"),
        ("en 1940 había 21 km", "en mil novecientos cuarenta había veintiún kilómetros"),
        ("unos 200 hombres", "unos doscientos hombres"),
        ("la 3.ª edición", "la tercera edición"),
    ],
    "pt": [
        ("aniversário de 13 anos", "aniversário de treze anos"),
        ("limitado a 3,6°F (2°C).", "limitado a três vírgula seis graus Fahrenheit (dois graus Celsius)."),
        ("contribuir com 10% dos rendimentos", "contribuir com dez por cento dos rendimentos"),
        ("entre 1942 e 1949.", "entre mil novecentos e quarenta e dois e mil novecentos e quarenta e nove."),
        ("em 21 de maio de 1974", "em vinte e um de maio de mil novecentos e setenta e quatro"),
        ("Terminou em 3º lugar", "Terminou em terceiro lugar"),
    ],
    "it": [
        ("Nel 1908 è presente", "Nel millenovecentotto è presente"),
        ("di 749 abitanti", "di settecentoquarantanove abitanti"),
        ("distanza di 7,6 pollici (15,433 m).", "distanza di sette virgola sei pollici (quindici virgola quattro tre tre metri)."),
        ("il 25 marzo 2023", "il venticinque marzo duemilaventitré"),
        ("della 36ª Brigata", "della trentaseiesima Brigata"),
        ("21 anni", "Ventuno anni"),
    ],
    "ru": [
        ("В 1950-х годах стандарты были пересмотрены.", "В тысяча девятьсот пятидесятых годах стандарты были пересмотрены."),
        ("В 2025 году территория застроена.", "В две тысячи двадцать пятом году территория застроена."),
        ("в комедии 2008 года", "в комедии две тысячи восьмого года"),
        ("было продано более 600 000 копий.", "было продано более шестисот тысяч копий."),
        ("к 2025 году", "к две тысячи двадцать пятому году"),
    ],
    "id": [
        ("Saya meminjam 1.000 yen.", "Saya meminjam seribu yen."),
        ("pada tahun 1908.", "pada tahun seribu sembilan ratus delapan."),
        ("abad ke-19 dan 3,5 persen", "abad kesembilan belas dan tiga koma lima persen"),
        ("hingga $1.730.", "hingga seribu tujuh ratus tiga puluh dolar."),
    ],
    "ms": [
        ("Pada 21 Jun, 1974, José datang.", "Pada dua puluh satu Jun, seribu sembilan ratus tujuh puluh empat, José datang."),
        ("pada pukul 6.", "pada pukul enam."),
        ("dalam 300 batu dan 1.5 km", "dalam tiga ratus batu dan satu perpuluhan lima kilometer"),
        ("membayar RM50", "membayar lima puluh ringgit"),
    ],
    "th": [
        ("ไปกับอังกฤษ 99 ปี", "ไปกับอังกฤษ เก้าสิบเก้า ปี"),
        ("เขาเกิดในศตวรรษที่ 19", "เขาเกิดในศตวรรษที่ สิบเก้า"),
        ("น้ำเดือดที่ 100 องศาเซลเซียส", "น้ำเดือดที่ หนึ่งร้อย องศาเซลเซียส"),
        ("มี ๒๑ คน", "มี ยี่สิบเอ็ด คน"),
    ],
    "vi": [
        ("FIFA World Cup 2026 là lần thứ 23", "FIFA World Cup hai nghìn không trăm hai mươi sáu là lần thứ hai mươi ba"),
        ("ngày 11 tháng 4", "ngày mười một tháng tư"),
        ("khoảng 700 tỉ đô la", "khoảng bảy trăm tỉ đô la"),
        ("giá 1.500 đồng, tăng 3,5%", "giá một nghìn năm trăm đồng, tăng ba phẩy năm phần trăm"),
        ("thứ 1 và 105", "Thứ nhất và một trăm linh năm"),
    ],
    "tr": [
        ("Termometre 30 dereceyi gösteriyor.", "Termometre otuz dereceyi gösteriyor."),
        ("kazanan rakamlar 7, 18, 3, 41 ve 6 idi.", "kazanan rakamlar yedi, on sekiz, üç, kırk bir ve altı idi."),
        ("İspanya, 2010 FIFA Dünya Kupası'nın galibidir.", "İspanya, iki bin on FIFA Dünya Kupası'nın galibidir."),
        ("4. sınıf ve %25 indirim", "Dördüncü sınıf ve yüzde yirmi beş indirim"),
        ("2005 kötü bir yıldı.", "İki bin beş kötü bir yıldı."),
    ],
    "fa": [
        ("حداقل 3000 زبان", "حداقل سه هزار زبان"),
        ("در سال ۱۸۹۴ منتشر شد", "در سال هزار و هشتصد و نود و چهار منتشر شد"),
        ("تا ساعت 11 بر گردی", "تا ساعت یازده بر گردی"),
        ("امروز ۱۲ اوت است", "امروز دوازدهم اوت است"),
        ("در ۳۰ ژانویه", f"در سی{ZWNJ}ام ژانویه"),
        ("۹۹ درصد", "نود و نه درصد"),
    ],
    "ar": [
        ("في نهائي عام 2025 الذي", "في نهائي عام ألفين وخمسة وعشرين الذي"),
        ("منذ 2006.", "منذ ألفين وستة."),
        ("في عام 1987،", "في عام ألف وتسعمئة وسبعة وثمانين،"),
    ],
}

# 必须丢弃：(原文, tn_note 中应出现的片段)
DROPPED = {
    "zh": [("WIN350的实验", "数字与字母相连"), ("宣告成立:29。", "冒号"), ("1-2个", "范围"), ("在2楼", "两"),
           ("西元前8000年左右", "年数"), ("编号IVPP V 11821）", "编号"), ("他是2", "两")],
    "ja": [("300km/hで", "单位"), ("番号12345。", "编号"), ("3〜5日", "范围")],
    "ko": [("자신의 3 포인트 슈팅", "量词"), ("120명", "1–99"), ("2-3개", "范围")],
    "en": [("in 1500 cases", "四位数"), ("on 21 December", "the … of"), ("1,5 people", "无法确定"),
           ("PA 225 leads north", "编号"), ("at 260 nm", "单位"), ("3-5 days", "范围"), ("by Rev.W.Awdry", "缩写")],
    "de": [("Es kamen 1500 Menschen.", "四位数"), ("am 3. Mai 2020", "变格"), ("Die 3. Liga", "序数"),
           ("Es sind 1 km.", "1"), ("bis auf 34 400 km", "空格"), ("von 15,94 h", "单位"), ("Ramses III. war", "罗马数字")],
    "fr": [("Il a 21 voitures.", "阴阳性"), ("Le XXe siècle", "罗马数字"), ("3-5 jours", "范围")],
    "es": [("21 casas", "阴阳性"), ("unos 200 mujeres", "阴阳性"), ("la 3 Arts Entertainment", "专名"),
           ("el 1.º piso", "primer")],
    "pt": [("22 pessoas", "阴阳性"), ("Segundo a sra. Ambrose", "缩写"), ("por M1 e M5", "字母")],
    "it": [("La legge n. 10", "缩写"), ("di 2 880 abitanti", "空格"), ("21 case", "阴阳性")],
    "ru": [("Впереди ограничение 40, а следом", "格"), ("до 40 этажа", "序数"), ("более 21 копии", "1"),
           ("После 24-летней паузы", "连字符"), ("533 Мбайт/с", None)],
    "id": [("COVID-19 tetapi", "编号"), ("pukul 09.30", "无法确定"), ("tahun 2000-an", "连字符")],
    "ms": [("Veronique AGI No. 47.", "缩写")],
    "th": [("เวลา 14.30 น.", "时刻")],
    "vi": [("số 10-15", "范围")],
    "fil": [("noong mga 2100.", "菲律宾语"), ("sa ika-23 ng Marso", "编号")],
    "ar": [("في 12 أغسطس،", "阿拉伯语"), ("وعاش بها 31 سنة", "阿拉伯语"), ("16.667 %", "百分数")],
    "tr": [("2.30'a kadar", "撇号"), ("Fanta'yı 2. Dünya Savaşının ortasında", "序数"), ("Burroughs B5500 gibi", "字母")],
    "fa": [("برادم جک 14-ساله است.", "连字符"), ("(1616-1564)", "范围"), ("3ام", "相连")],
}


class NormalizedTest(unittest.TestCase):
    def test_every_language_covered(self):
        from pipeline.tn.core import LANGS

        for lang in LANGS:
            self.assertIn(lang, DROPPED)
            if lang != "fil":
                self.assertIn(lang, NORMALIZED)

    def test_normalized(self):
        for lang, cases in NORMALIZED.items():
            for original, script in cases:
                with self.subTest(lang = lang, original = original):
                    result = normalize_entry(original = original, lang = lang)
                    self.assertEqual(result.status, "normalized", msg = result.note)
                    self.assertEqual(result.script, script)
                    self.assertTrue(result.note.startswith("改写："))

    def test_dropped(self):
        for lang, cases in DROPPED.items():
            for original, reason in cases:
                with self.subTest(lang = lang, original = original):
                    result = normalize_entry(original = original, lang = lang)
                    self.assertEqual(result.status, "dropped", msg = result.script)
                    self.assertIsNone(result.script)
                    if reason:
                        self.assertIn(reason, result.note)

    def test_scripts_have_no_digits(self):
        for lang, cases in NORMALIZED.items():
            for original, _ in cases:
                result = normalize_entry(original = original, lang = lang)
                self.assertFalse(any(ch.isdigit() for ch in result.script), msg = result.script)

    def test_numbers_match_original(self):
        """不含单位、货币、时刻的改写，朗读稿与原文经 pipeline.compare 数词统一后一致。

        「유월 십일」中的「십일」被 compare 读成 11，属于比对模块的已知局限，不在这里检查。
        """
        skip = ("单位", "货币", "时刻", "日期")
        known = {("ko", "6월 10일")}
        for lang, cases in NORMALIZED.items():
            for original, _ in cases:
                result = normalize_entry(original = original, lang = lang)
                if any(word in result.note for word in skip) or (lang, original) in known:
                    continue
                with self.subTest(lang = lang, original = original):
                    check = compare(script = result.script, hypothesis = clean(text = original, lang = lang), lang = lang)
                    self.assertTrue(check.match, msg = check.detail)


class NumberWordsTest(unittest.TestCase):
    def test_zh(self):
        from pipeline.tn.numwords import zh_cardinal, zh_digits

        cases = {0: "零", 2: "二", 10: "十", 15: "十五", 110: "一百一十", 1005: "一千零五", 1050: "一千零五十",
                 2000: "两千", 12000: "一万两千", 20000: "两万", 100010: "十万零一十", 105000: "十万五千",
                 100000005: "一亿零五"}
        for n, words in cases.items():
            self.assertEqual(zh_cardinal(n = n), words)
        self.assertEqual(zh_digits(digits = "2026"), "二零二六")

    def test_ja(self):
        from pipeline.tn.numwords import ja_cardinal

        cases = {10: "十", 100: "百", 1000: "千", 1950: "千九百五十", 10000: "一万", 41000: "四万千",
                 10000000: "一千万", 1005: "千五"}
        for n, words in cases.items():
            self.assertEqual(ja_cardinal(n = n), words)

    def test_ko(self):
        from pipeline.tn.numwords import ko_native_attr, ko_sino

        self.assertEqual(ko_sino(n = 1944), "천구백사십사")
        self.assertEqual(ko_sino(n = 10000), "만")
        self.assertEqual(ko_sino(n = 56), "오십육")
        self.assertEqual([ko_native_attr(n = n) for n in (1, 2, 3, 4, 20, 21, 99)],
                         ["한", "두", "세", "네", "스무", "스물한", "아흔아홉"])

    def test_th(self):
        from pipeline.tn.numwords import th_cardinal

        cases = {1: "หนึ่ง", 10: "สิบ", 11: "สิบเอ็ด", 20: "ยี่สิบ", 21: "ยี่สิบเอ็ด", 100: "หนึ่งร้อย",
                 101: "หนึ่งร้อยเอ็ด", 1636: "หนึ่งพันหกร้อยสามสิบหก", 1000000: "หนึ่งล้าน"}
        for n, words in cases.items():
            self.assertEqual(th_cardinal(n = n), words)

    def test_vi(self):
        from pipeline.tn.numwords import vi_cardinal

        cases = {14: "mười bốn", 15: "mười lăm", 21: "hai mươi mốt", 24: "hai mươi tư", 25: "hai mươi lăm",
                 105: "một trăm linh năm", 2026: "hai nghìn không trăm hai mươi sáu", 1000000: "một triệu"}
        for n, words in cases.items():
            self.assertEqual(vi_cardinal(n = n), words)

    def test_id_ms(self):
        from pipeline.tn.numwords import idms_cardinal

        self.assertEqual(idms_cardinal(n = 1730, lang = "id"), "seribu tujuh ratus tiga puluh")
        self.assertEqual(idms_cardinal(n = 18, lang = "ms"), "lapan belas")
        self.assertEqual(idms_cardinal(n = 2000000000, lang = "ms"), "dua bilion")
        self.assertEqual(idms_cardinal(n = 2000000000, lang = "id"), "dua miliar")

    def test_tr(self):
        from pipeline.tn.numwords import tr_cardinal, tr_ordinal

        self.assertEqual(tr_cardinal(n = 2010), "iki bin on")
        self.assertEqual(tr_cardinal(n = 1100), "bin yüz")
        self.assertEqual(tr_ordinal(n = 4), "dördüncü")
        self.assertEqual(tr_ordinal(n = 21), "yirmi birinci")

    def test_fa(self):
        from pipeline.tn.numwords import fa_cardinal

        self.assertEqual(fa_cardinal(n = 1894), "هزار و هشتصد و نود و چهار")
        self.assertEqual(fa_cardinal(n = 18), "هجده")
        self.assertEqual(fa_cardinal(n = 2000), "دو هزار")

    def test_ar_year(self):
        from pipeline.tn.numwords import ar_year_genitive

        self.assertEqual(ar_year_genitive(n = 2025), "ألفين وخمسة وعشرين")
        self.assertEqual(ar_year_genitive(n = 631), "ستمئة وواحد وثلاثين")
        self.assertEqual(ar_year_genitive(n = 1910), "ألف وتسعمئة وعشرة")

    def test_number_words_round_trip(self):
        """自己实现的读法经 pipeline.compare 转回数值，与原数一致。"""
        from pipeline.tn.numwords import (
            fa_cardinal, idms_cardinal, ja_cardinal, ko_sino, th_cardinal, tr_cardinal, vi_cardinal, zh_cardinal,
        )

        funcs = {
            "zh": zh_cardinal, "ja": ja_cardinal, "ko": ko_sino, "th": th_cardinal, "vi": vi_cardinal,
            "tr": tr_cardinal, "fa": fa_cardinal,
            "id": lambda n: idms_cardinal(n = n, lang = "id"), "ms": lambda n: idms_cardinal(n = n, lang = "ms"),
        }
        values = list(range(0, 130)) + [199, 200, 205, 999, 1000, 1001, 1010, 1100, 1900, 1999, 2000, 2026, 9999,
                                         10000, 10005, 12345, 100000, 1000000, 1234567]
        for lang, func in funcs.items():
            for n in values:
                with self.subTest(lang = lang, n = n):
                    words = func(n = n)
                    self.assertTrue(compare(script = words, hypothesis = str(n), lang = lang).match, msg = words)


if __name__ == "__main__":
    unittest.main()
