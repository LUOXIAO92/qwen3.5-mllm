"""T03b 返工：整条网页规则、Wikinews 链接模板、缩写切句、黑名单和字符模型的单元测试。"""

from __future__ import annotations

import pytest

from pipeline.text.exclusion import load_blocklist
from pipeline.text.mtscore import CharLM
from pipeline.text.sources.web import entry_options, wikitext_to_text
from pipeline.text.split import split_sentences
from pipeline.text.web_entry_rules import check_web_entry


@pytest.mark.parametrize(
    ("text", "lang", "reason"),
    [
        ("Apa itu ekonomi syariah menurut para ahli?", "id", "question_heading"),
        ("So dachten wir.", "de", "too_few_words"),
        ("Ya.", "ms", "too_few_words"),
        ("今天很好。", "zh", "too_short_entry"),
        ("Pengantar: Pada kesempatan ini kami akan membahas tentang pasar saham dunia hari ini.", "id", "label_prefix"),
        ("1/12 Pemerintah daerah menyatakan bahwa jalan akan segera diperbaiki tahun depan.", "id", "numbering"),
        ("第三条 本办法适用于所有在本市注册的企业和个人用户的相关事项。", "zh", "numbering"),
        ("BNews–MAGELANG–Pemerintah kabupaten menyatakan bahwa jalan akan segera diperbaiki.", "id", "byline"),
        ("بقلم أحمد - كتب المحلل السياسي مقالا طويلا عن الأوضاع في المنطقة هذا الأسبوع.", "ar", "byline"),
        ("3Ang mga tao ay nagtipon sa plaza upang makinig sa mga salita ng kanilang pinuno.", "fil", "section_number"),
        ("Majlis perasmian itu telah dirasmikan secara rasmi oleh Ybhg.", "ms", "abbreviation_end"),
        ("Harga tiket konser itu mencapai Rp.", "id", "abbreviation_end"),
        ("Forecasters at the have issued a warning for the coast of Florida on Monday.", "en", "link_gap"),
        ("는 2일 블로그를 통해 새로운 계획을 발표하고 지지자들에게 감사의 뜻을 전했다.", "ko", "link_gap"),
        ("علیه ، تاکنون هیچ اقدامی از سوی دولت برای حل این مشکل انجام نشده است.", "fa", "link_gap"),
        ("Bu İmparatoru?nun saraylarında yaşayan insanlar için çok önemli bir olaydı hep.", "tr", "encoding"),
        ("Dia berkata bahwa ' s kebijakan baru itu akan mulai berlaku bulan depan nanti.", "id", "encoding"),
        ("Der Text enthält ein kaputtes Zeichen � mitten im Satz und geht noch weiter.", "de", "encoding"),
    ],
)
def test_entry_rules_drop(text, lang, reason):
    assert check_web_entry(text, lang = lang) == reason


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Pemerintah daerah menyatakan bahwa jalan akan segera diperbaiki pada tahun depan.", "id"),
        ("Rapat dimulai pukul 10:30 WIB di ruang utama dan dihadiri oleh seluruh anggota.", "id"),
        ("Kebijakan anti-korupsi yang baru itu akan mulai berlaku pada bulan depan nanti.", "id"),
        ("Die Bundesregierung hat gestern ein neues Gesetz über die Besteuerung beschlossen.", "de"),
        ("市政府昨天宣布了新的交通管理办法，将从下月开始在全市范围内实施。", "zh"),
    ],
)
def test_entry_rules_keep(text, lang):
    assert check_web_entry(text, lang = lang) is None


def test_spam_keywords_only_for_flagged_sources():
    text = "Situs judi online terpercaya menawarkan bonus besar untuk semua pemain baru setiap hari."
    assert check_web_entry(text, lang = "id") is None
    assert check_web_entry(text, lang = "id", spam = True) == "spam"
    assert entry_options("id", "Wikinews").spam is False
    assert entry_options("id", "HPLT 3.0").spam is True
    assert entry_options("id", "Tatoeba").entry is False  # part1 来源不受影响


def test_wikinews_link_templates_keep_display_text():
    text = wikitext_to_text("Forecasters at the {{w|National Hurricane Center|NHC}} and {{w|Miami}} have issued a warning.")
    assert text == "Forecasters at the NHC and Miami have issued a warning."
    assert wikitext_to_text("{{v|Gazze}} şeridinde olay oldu.").startswith("Gazze şeridinde")
    assert "{{date" not in wikitext_to_text("{{date|2008}} Metin burada.")


@pytest.mark.parametrize(
    ("text", "lang", "count"),
    [
        ("Ybhg. Datuk Seri hadir di majlis itu. Majlis bermula.", "ms", 2),
        ("Harganya Rp. 5 juta per unit. Selesai.", "id", 2),
        ("Hj. Ahmad dan Dr. Siti hadir. Mereka pulang.", "ms", 2),
    ],
)
def test_splitter_malay_indonesian_abbreviations(text, lang, count):
    assert len(split_sentences(text, lang = lang)) == count


def test_blocklist_keeps_urls_with_hash():
    refs = load_blocklist()
    assert any(ref.startswith("http") and "#" in ref for ref in refs)
    assert all(" " not in ref for ref in refs)


def test_char_lm_prefers_in_domain_text():
    model = CharLM()
    for _ in range(30):
        model.train("Ang bahay ay malaki at maganda ang tanawin.")
        model.train("Pumunta kami sa palengke upang bumili ng isda.")
    natural = model.cross_entropy("Ang bahay namin ay maganda.")
    noisy = model.cross_entropy("Zxqv wkjh TAP PHP100 voucher!!")
    assert noisy > natural
