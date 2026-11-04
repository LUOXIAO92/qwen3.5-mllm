"""独立审阅后补充的规则和切句修复的单元测试。每条规则至少一个应丢弃、一个应保留的例子。"""

from __future__ import annotations

import pytest

from pipeline.text.dedup import Deduper
from pipeline.text.exclusion import load_blocklist
from pipeline.text.filters import check_text
from pipeline.text.rules import check_language_rules
from pipeline.text.normalize import clean_text, fix_persian, norm_key
from pipeline.text.split import split_sentences

# ---------------------------------------------------------------- 切句器


@pytest.mark.parametrize(
    ("text", "lang", "count"),
    [
        ("Er kam am 18. April 1990 nach Berlin. Dann ging er.", "de", 2),  # 序数
        ("Nach dem Tod Mieszkos III. Wurde das Reich geteilt. Er starb 1202.", "de", 2),  # 罗马数字
        ("Der Dichter (geb. 1900 in Wien) starb 1980. Er war bekannt.", "de", 2),  # geb.
        ("Das war vor Chr. Geburt bekannt. Viele wussten das.", "de", 2),
        ("Es gibt z. B. Äpfel und Birnen. Alle schmecken gut.", "de", 2),
        ("He moved to the U.S. Army camp in 1990. Then he left.", "en", 2),
        ("They live on St. John Street. It is quiet.", "en", 2),
        ("Mr. Smith Jr. Went home. He slept.", "en", 2),
        ("Он родился в селе св. Николая. Потом уехал.", "ru", 2),
        ("En 44 av. J.-C. César mourut. La Rome changea.", "fr", 2),
        ("Des des XXXXVIII. Armeekorps kam an. Er starb.", "de", 2),  # 不规范罗马数字
        ("Er wohnte im District No. 2 pursuant to Law No. 5. It ended.", "en", 2),  # No. 后接数字
        ("Марквардт издал том (1849) и тт. 3 и 4. Потом умер.", "ru", 2),
    ],
)
def test_splitter_does_not_cut_at_abbreviations(text, lang, count):
    sentences = split_sentences(text, lang = lang)
    assert len(sentences) == count, sentences


def test_splitter_still_cuts_real_sentence_ends():
    assert len(split_sentences("Er kam 1990 an. Er ging 2005.", lang = "de")) == 2
    assert len(split_sentences("The film was released in 1990. It was a hit.", lang = "en")) == 2
    assert len(split_sentences("Он умер. Она осталась.", lang = "ru")) == 2


# ---------------------------------------------------------------- 通用规则


def test_punct_residue_and_double_dot():
    assert check_language_rules("Mérens-les-Vals,,, est une commune française du département.", lang = "fr") == "punct_residue"
    assert check_language_rules("La vitesse atteint 83 km/h),. Puis elle diminue.", lang = "fr") == "punct_residue"
    assert check_text("Ce village est situé dans la vallée du Rhône, près de la ville.", lang = "fr") is None
    assert check_text("Albert Guiod est né en 1852 à Paris puis il est parti..", lang = "fr") == "double_dot"
    assert check_text("Il hésita un instant... puis il ouvrit enfin la porte.", lang = "fr") is None


def test_slash_separator_poetry_and_dual_calendar():
    assert check_text('Lorsque levant ses yeux, / tendant au ciel ses mains, il dit adieu.', lang = "fr") == "slash_separator"
    assert check_text("They ordered and/or cancelled the whole thing before nine o'clock.", lang = "en") is None
    assert check_language_rules("وقع الحدث سنة ١٣٣٠ هـ / ١٩١٢ م في المدينة القديمة.", lang = "ar") == "slash_separator"


def test_stats_line_and_numbered_list_and_brackets():
    assert check_language_rules("Нация выиграла 4—2, потом 3—1, потом 5—0 и наконец 2—2 в финале.", lang = "ru") == "stats_line"
    assert check_text("Maasé Hasham sacrificio 141. Maasé Zevaj paz 142. No dejar carne 143. Quemar restos.", lang = "es") == "numbered_list"
    assert check_text("Er kam am 18. April und am 3. Mai nach Berlin zurück.", lang = "de") is None
    assert check_text("Der Ort (Kreis A) liegt an der Bahn (Linie B) und am Fluss (der Rhein).", lang = "de") == "many_brackets"
    assert check_text("Der Ort (Kreis A) liegt an der Bahnlinie nach der großen Stadt im Westen.", lang = "de") is None
    assert check_text("Das ist (die ganze lange Antwort auf alle Fragen) so.", lang = "de") == "bracket_heavy"


def test_repeated_words_sentences():
    assert check_text("Il y a un problème de de longueur dans cette phrase.", lang = "fr") == "repeated_function_word"
    assert check_text("La charge a le potentiel de produire de l'énergie.", lang = "fr") is None
    assert check_text("Se dice que le plus grand de de Gaulle est parti.", lang = "fr") == "repeated_function_word"
    assert check_text("Aku tidak tahu karena terdapat terdapat gas beracun di sini.", lang = "id") == "repeated_word"
    assert check_text("Dia bilang itu itu bisa-bisa saja terjadi lagi.", lang = "id") == "repeated_word"
    assert check_text("Ich will gehen. Ich will gehen. Du bleibst hier.", lang = "de") == "repeated_sentence"


# ---------------------------------------------------------------- 各语言规则


def test_fr_it_elided_word_ending():
    assert check_text("Dans son message, le Conseil fédéral reconnait que l'.", lang = "fr") == "elided_end"
    assert check_text("Il suo discorso è stato lungo e noioso per l'.", lang = "it") == "elided_end"
    assert check_text("Il parle de l'histoire de France avec passion.", lang = "fr") is None


def test_pt_lowercase_after_question():
    assert check_text("No dia seguinte? o menino voltou ao poço de manhã.", lang = "pt") == "lowercase_after_question"
    assert check_text("Quando esses livros foram escritos? O menino perguntou ao pai.", lang = "pt") is None


def test_ko_hanja_gloss_and_dependent_ending():
    assert check_text("안남(安南)은 당나라가 하노이에 설치한 안남도호부에서 유래했다.", lang = "ko") == "ko_foreign_script"
    assert check_language_rules("이 개는 몰타(이탈리아어: Malta) 섬이 원산지여서 붙여졌다고 한다.", lang = "ko", max_bracket_ratio = 1.0) == "ko_gloss"
    assert check_text("켄터키주의 지역은 다음과 같다.", lang = "ko") == "dependent_ending"
    assert check_text("신철이가 웃는 것을 보니 좀더 성은 나면서도 따라 웃지 않았다.", lang = "ko") is None


def test_ko_dependent_with_quote_intro():
    assert check_text("그러나 그는 다음과 같이 경고했다.", lang = "ko") == "dependent_ending"


def test_th_fragments():
    assert check_text("แต่ผมไม่รู้เรื่องนี้เลยสักนิด", lang = "th") == "th_fragment"  # 连词开头
    assert check_text("เขาเดินไปที่ตลาดเพื่อซื้อผลไม้และ", lang = "th") == "th_fragment"  # 连词结尾
    assert check_text("เพิ่งรู้ว่าเขาเรียกว่า ริมน้ำ - ริมราง", lang = "th") == "th_fragment"  # 标题、署名
    assert check_text("ห้องสมุดเก่าตรงหัวมุมเปิดทุกเช้าตอนเก้าโมง", lang = "th") is None
    assert check_text("เขาค่อนข้างชัดเจนว่าเขาไม่ได้ดีกว่า", lang = "th") is None  # 比较句式 กว่า 结尾是完整句


def test_ar_rules():
    assert check_text("هذه قائمة بالأطعمة التي يحبها الناس في المنطقة.", lang = "ar") == "list_intro"
    assert check_text("قائمة حكام ليبيا تضم جميع الرؤساء الذين حكموا البلاد.", lang = "ar") == "list_intro"
    assert check_text("هناك من يرى أصل التسمية قديما..فالقادة كانوا هناك.", lang = "ar") == "double_dot"
    assert check_text("تفتح المكتبة القديمة عند الزاوية كل صباح في الساعة التاسعة.", lang = "ar") is None


def test_ja_dependent_ending():
    assert check_text("各部の名称は以下の通り。", lang = "ja") == "dependent_ending"
    assert check_text("角にある古い図書館は毎朝九時に開きます。", lang = "ja") is None


def test_ru_notation_and_stats():
    assert check_text("Белые играют Лf5 и выигрывают пешку в этой партии.", lang = "ru") == "notation"
    assert check_text("С 2016 года Суон представляет Великобританию, её рекорд W/L: 4—2.", lang = "ru") == "notation"
    assert check_text("Старая библиотека на углу открывается каждое утро в девять часов.", lang = "ru") is None


# ---------------------------------------------------------------- 波斯语规范化、近重复、人工排除


def test_persian_normalization():
    assert fix_persian("كتاب علي") == "کتاب علی"  # ك→ک ي→ی
    assert fix_persian("‌می‌شود‌") == "می‌شود"  # 词首词尾的 ZWNJ 去掉，词中保留
    assert fix_persian("ها‌‌ی") == "ها‌ی"  # 连续的合并
    assert fix_persian("خانه ‌، کتاب") == "خانه ، کتاب"  # 紧邻空格和标点的去掉
    assert clean_text("علي‌", lang = "fa") == "علی"
    assert norm_key("كتاب", lang = "fa") == norm_key("کتاب", lang = "fa")


def test_near_duplicate_prefix():
    deduper = Deduper(lang = "id", max_per_prefix = 1, prefix_len = 28)
    first = 'Aku tidak mengerti bagaimana bisa disaat mereka bersedia menyimpan barangmu, bisa-bisanya salah seorang petugasnya cuma berkata "Maaf".'
    second = 'Aku tidak mengerti bagaimana bisa mereka mengambil barangmu untuk disimpan, tapi petugasnya cuma berkata "Maaf".'
    assert deduper.add(first)
    assert not deduper.add(second)  # 开头 28 个键字符相同，按近重复丢弃
    assert deduper.add("Dia pergi ke pasar untuk membeli buah dan sayur kemarin pagi.")


def test_blocklist_loaded():
    refs = load_blocklist()
    assert "tatoeba:4993432" in refs
    assert all(ref.count(":") == 1 for ref in refs)
