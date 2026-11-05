"""网页来源规则（T03b）的单元测试：每条规则至少一个应丢弃、一个应保留的例子。"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import pytest

from pipeline.text.build import SOURCE_INFO
from pipeline.text.extract import extract_candidates
from pipeline.text.filters import check_text, default_options
from pipeline.text.merge import merge_language
from pipeline.text.sources.web import (
    hplt_doc_ok,
    hplt_main_register,
    read_hplt,
    text_to_paragraphs,
    wikitext_to_text,
)
from pipeline.text.split import split_sentences
from pipeline.text.web_rules import check_web_rules


def web_check(text: str, lang: str) -> str | None:
    return check_text(text, lang = lang, options = dataclasses.replace(default_options(lang), web = True))


# ---------------------------------------------------------------- 样板、号召语


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Klik di sini untuk membaca selengkapnya tentang berita ini.", "id"),
        ("Copyright 2020 semua hak dilindungi oleh undang-undang yang berlaku.", "id"),
        ("Subscribe to our newsletter to get the latest news every week.", "en"),
        ("Alle Rechte vorbehalten, Datenschutz und Impressum beachten Sie bitte.", "de"),
        ("Tous droits réservés, lisez notre politique de confidentialité ici.", "fr"),
        ("版权所有，未经许可不得转载本站任何内容。", "zh"),
        ("无断転載を禁じます、お問い合わせはこちらからどうぞ。", "ja"),
        ("Hubungi kami sekarang untuk mendapatkan promo menarik bulan ini.", "ms"),
        ("Bạn đang xem bài viết này trên trang của chúng tôi hôm nay.", "vi"),
        ("Tüm hakları saklıdır, devamını okumak için tıklayın lütfen.", "tr"),
        ("جميع الحقوق محفوظة لدى الموقع ولا يجوز النسخ بدون إذن.", "ar"),
    ],
)
def test_boilerplate_dropped(text, lang):
    assert check_web_rules(text, lang = lang) == "boilerplate"


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Mereka menghargai pekerjaan itu dengan sangat baik sekali.", "id"),  # harga 不命中 menghargai
        ("The committee met on Tuesday to discuss the proposed budget.", "en"),
        ("Die Regierung hat am Dienstag den neuen Haushalt beschlossen.", "de"),
        ("市政府昨天宣布了新的交通管理办法，将从下月开始实施。", "zh"),
    ],
)
def test_normal_sentences_kept(text, lang):
    assert web_check(text, lang) is None


def test_boilerplate_only_active_for_web_entries():
    text = "Subscribe to the newsletter, said the organizer of the event."
    assert check_text(text, lang = "en") is None  # part1 的检查不受影响
    assert web_check(text, "en") == "boilerplate"


# ---------------------------------------------------------------- 其他模式


@pytest.mark.parametrize(
    ("text", "lang", "reason"),
    [
        ("Follow the team on #worldcup #football every day of the tournament.", "en", "social"),
        ("Nomor kantor pusat kami adalah 0812-3456-7890 dan masih aktif sampai sekarang.", "id", "phone"),
        ("The new phone costs $299 and will be sold in stores next month.", "en", "price"),
        ("Pembayaran sebesar 150000 rupiah harus diselesaikan sebelum akhir bulan ini.", "id", "price"),
        ("Das Treffen fand am 12/03/2019 in der Stadthalle von Köln statt.", "de", "numeric_date"),
        ("This is absolutely amazing news for everyone in the whole country!!", "en", "emphasis"),
        ("Home » News » Politics » Local government meets on Tuesday afternoon.", "en", "breadcrumb"),
        ("Photo: Reuters, the crowd gathered outside of the main building yesterday.", "en", "caption"),
        ("Top 10 Tips For Better Sleep And Healthy Living This Year.", "en", "title_case"),
        ("THE BEST OFFER YOU WILL SEE THIS WEEK IN THE CITY FOR SURE.", "en", "shouting"),
        ("Para saintis merekrut 88 peserta untuk penyelidikan.Mereka membahagikan peserta kepada dua kumpulan.", "ms", "glued_sentence"),
        ("Jumlah pembiayaan peribadi ditingkatkan kepada sehingga lima ribu ringgit susulan kenyataan terkini …", "ms", "ellipsis_end"),
        ("P: Que tipo de mercado é adequado para usar o Grid Trading hoje em dia?", "pt", "caption"),
        ("R.: O diagnóstico é feito pelo médico a partir dos dados da história contada.", "pt", "caption"),
        ("A lesão dos flexores profundos causa uma inca- pacidade de flexão das articulações.", "pt", "broken_hyphen"),
        ("ピルスナービールのおすすめ10選!", "ja", "cjk_ascii_punct"),
        ("千百年来,在构成人们实用需求方面充当着重要角色，同时也表明了它的价值。", "zh", "cjk_ascii_punct"),
        ("【特集】試したくなる新しいスマートフォンの選び方を紹介します。", "ja", "tag_bracket"),
        ("의학의 발전 2.", "ko", "heading_number"),
        ("다음은 20세기를 흔든 인도의 정치 위인의 명언입니다.", "ko", "dependent_start"),
        ("Here are the best ways to save money on your next holiday trip.", "en", "dependent_start"),
        ("Bạn có biết nằm mơ thấy chim tu hú đánh số con gì để trúng lớn hôm nay?", "vi", "boilerplate"),
        ("Buy cheap shoes buy cheap shoes buy cheap shoes buy cheap shoes online now.", "en", "repetitive"),
    ],
)
def test_pattern_rules_drop(text, lang, reason):
    assert check_web_rules(text, lang = lang) == reason
    assert web_check(text, lang) is not None  # 整个过滤链也会丢弃（原因可能是更早命中的通用规则）


def test_glued_and_ellipsis_keep_normal_text():
    assert web_check("Dr. Smith holds a doctorate from the university in Boston.", "en") is None
    assert web_check("Он сказал, что придёт позже... Но так и не пришёл.", "ru") is None  # 句中省略号不算


def test_cjk_ascii_comma_between_digits_is_fine():
    assert web_check("该公司去年的营业收入达到了三千万元，其中海外市场的贡献约有1,200万元。", "zh") is None


def test_phone_rule_keeps_ordinary_numbers():
    assert web_check("The company hired 250 new workers during the last quarter.", "en") is None
    assert web_check("Pada tahun 2019 jumlah penduduk kota ini mencapai tiga juta jiwa.", "id") is None


def test_title_case_not_applied_to_german_nouns():
    assert web_check("Die Bundesregierung hat gestern das Gesetz über Steuern beschlossen.", "de") is None


# ---------------------------------------------------------------- 读取器


def test_text_to_paragraphs_splits_lines():
    assert text_to_paragraphs("Title\n\nFirst paragraph.\nSecond one.", lang = "en") == ["Title", "First paragraph.", "Second one."]


def test_japanese_short_lines_are_merged():
    text = "今日は晴れでした。\n公園に行きました。\n楽しかったです。"
    assert text_to_paragraphs(text, lang = "ja") == ["今日は晴れでした。公園に行きました。楽しかったです。"]


def _doc(**register):
    base = {"MT": 0.05, "LY": 0.05, "SP": 0.05, "ID": 0.05, "NA": 0.05, "HI": 0.05, "IN": 0.05, "OP": 0.05, "IP": 0.05}
    base.update(register)
    return {"web-register": base, "filter": "keep", "pii": [], "cluster_size": 1, "lang": ["vie_Latn"], "prob": [1.0], "seg_langs": ["vie_Latn"] * 5}


def test_hplt_main_register_and_genre_filter():
    assert hplt_main_register(_doc(NA = 0.9)["web-register"]) == ("NA", 0.9)
    assert hplt_doc_ok(_doc(NA = 0.9), genres = ("NA",)) == "na"
    assert hplt_doc_ok(_doc(OP = 0.9), genres = ("NA",)) is None
    assert hplt_doc_ok(_doc(ID = 0.9), genres = ("NA", "OP", "IN")) is None  # 讨论区


def test_hplt_doc_filters():
    assert hplt_doc_ok(_doc(NA = 0.9, MT = 0.5), genres = ("NA",)) is None  # 机翻概率高
    doc = _doc(NA = 0.9)
    doc["cluster_size"] = 500
    assert hplt_doc_ok(doc, genres = ("NA",)) is None  # 模板页
    doc = _doc(NA = 0.9)
    doc["pii"] = [[1, 2]]
    assert hplt_doc_ok(doc, genres = ("NA",)) is None
    doc = _doc(NA = 0.9)
    doc["seg_langs"] = ["vie_Latn", "cmn_Hans", "eng_Latn", "vie_Latn", "vie_Latn"]
    assert hplt_doc_ok(doc, genres = ("NA",), glotlid_code = "vie_Latn") is None  # 混有其他语种
    assert hplt_doc_ok(_doc(NA = 0.9), genres = ("NA",), glotlid_code = "vie_Latn") == "na"


def test_read_hplt_streams_zst(tmp_path):
    import subprocess

    docs = [
        {**_doc(NA = 0.9), "u": "https://example.org/a", "text": "Dòng một.\nDòng hai."},
        {**_doc(ID = 0.9), "u": "https://example.org/b", "text": "Bình luận."},
    ]
    raw = tmp_path / "x.jsonl"
    raw.write_text("\n".join(json.dumps(doc, ensure_ascii = False) for doc in docs) + "\n", encoding = "utf-8")
    subprocess.run(["zstd", "-q", str(raw), "-o", str(tmp_path / "x.jsonl.zst")], check = True)
    rows = list(read_hplt(files = [str(tmp_path / "x.jsonl.zst")], lang = "vi", genre = "na", glotlid_code = "vie_Latn"))
    assert rows == [("Dòng một.", "https://example.org/a#0"), ("Dòng hai.", "https://example.org/a#1")]


def test_wikitext_cleaning_drops_templates_categories_and_sources():
    wikitext = (
        "{{date|March 3, 2008}}\n'''Paris''' - The mayor said [[city]] taxes will rise.<ref>x</ref>\n"
        "[[Category:France]]\n[[File:Map.png|thumb|A map]]\n== Sources ==\n* [http://a.example a]\n"
    )
    text = wikitext_to_text(wikitext)
    assert "mayor said city taxes will rise." in text
    assert "Category" not in text and "Map.png" not in text and "example" not in text


# ---------------------------------------------------------------- 抽取与合并


def test_doc_and_domain_caps():
    records = [
        (f"Sentence number {word} is a perfectly fine sentence to read aloud.", f"https://one.example/doc{doc}#{n}")
        for doc, n, word in [(1, 0, "alpha"), (1, 1, "bravo"), (1, 2, "charlie"), (2, 0, "delta"), (2, 1, "echo"), (3, 0, "foxtrot")]
    ]
    stats = Counter()
    candidates = extract_candidates(
        records,
        lang = "en",
        exclusion = {},
        identifier = None,
        max_per_doc = 1,
        max_per_domain = 2,
        stats = stats,
    )
    assert len(candidates) == 2  # 同一域名最多 2 条；同一文档最多 1 条
    assert stats["drop_doc_cap"] + stats["drop_domain_cap"] == 4


def test_web_sources_registered_with_distinct_slugs():
    slugs = {SOURCE_INFO[name][1] for name in ("hplt_na", "hplt_op", "hplt_in", "fw2hq", "fwedu", "wikinews")}
    assert slugs == {"hplt", "fw2hq", "fwedu", "wikinews"}
    assert not slugs & {SOURCE_INFO[name][1] for name in ("common_voice", "tatoeba", "wikipedia")}


def _write(path, rows):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text("".join(json.dumps(row, ensure_ascii = False) + "\n" for row in rows), encoding = "utf-8")


def _entry(i, bin_name, text):
    return {
        "entry_id": f"en-tatoeba-{i:06d}", "lang": "en", "source": "Tatoeba", "source_license": "CC-BY-2.0-FR",
        "source_ref": str(i), "original": text, "n_chars": len(text), "length_bin": bin_name, "est_duration_s": 3.0,
    }


def _cand(text, ref, bin_name):
    return {"lang": "en", "text": text, "ref": ref, "length_bin": bin_name, "est_duration_s": 8.0}


def test_merge_keeps_part1_dedups_and_fills_gaps(tmp_path):
    part1 = [_entry(1, "short", "I like to read books at night."), _entry(2, "medium", "She walked slowly along the river bank today.")]
    _write(tmp_path / "part1" / "en.jsonl", part1)
    # 候选里有一条与 part1 重复（只差大小写和标点）、一条含样板句、两条合格
    _write(
        tmp_path / "cand" / "hplt_na" / "en.jsonl",
        [
            _cand("She walked slowly along the river bank, today!", "https://a.example/1#0", "medium"),
            _cand("Subscribe to our newsletter for the latest local news.", "https://a.example/2#0", "medium"),
            _cand("The council approved the new library budget on Monday evening.", "https://a.example/3#0", "medium"),
        ],
    )
    _write(tmp_path / "cand" / "wikinews" / "en.jsonl", [_cand("Heavy rain flooded several streets in the northern district.", "wikinews:en:Rain#0", "medium")])
    entries, new_entries, info = merge_language(
        "en",
        part1_dir = tmp_path / "part1",
        cand_dir = tmp_path / "cand",
        sources = ["hplt_na", "wikinews"],
        exclusion = {},
        targets = {"short": 1, "medium": 5, "long": 2},
        seed = 0,
    )
    assert entries[:2] == part1  # part1 原样保留
    assert [entry["entry_id"] for entry in entries[:2]] == ["en-tatoeba-000001", "en-tatoeba-000002"]
    assert len(new_entries) == 2
    assert {entry["source"] for entry in new_entries} == {"HPLT 3.0", "Wikinews"}
    assert {entry["entry_id"] for entry in new_entries} == {"en-hplt-000001", "en-wikinews-000001"}
    assert info["gaps"] == {"medium": 5 - 1 - 2, "long": 2}  # 来源不足时如实报告缺口
    assert info["filter_stats"]["drop_rules"] == 1 and info["filter_stats"]["drop_duplicate"] == 1


def test_thai_sentence_split_when_pythainlp_available():
    pytest.importorskip("pythainlp")
    sentences = split_sentences("สำหรับคุณภรรยาเมื่อทราบว่าตัวเองกำลังตั้งครรภ์ คุณสามีเองก็เช่นเดียวกัน คงรู้สึกดีใจไม่น้อยไปกว่าคุณภรรยาเป็นแน่", lang = "th")
    assert len(sentences) >= 2



def test_splitter_does_not_cut_abbreviation_in_brackets():
    sentences = split_sentences("Bazı tabirciler, rüyada Nuh (A.S.) gören kişiye müjde verir. İkinci cümle burada.", lang = "tr")
    assert len(sentences) == 2 and sentences[0].endswith("verir.")


def test_hplt_persian_segment_code_accepted():
    doc = _doc(NA = 0.9)
    doc["lang"], doc["prob"], doc["seg_langs"] = ["pes_Arab"], [0.98], ["fas_Arab"] * 4
    assert hplt_doc_ok(doc, genres = ("NA",), glotlid_code = "pes_Arab") == "na"
