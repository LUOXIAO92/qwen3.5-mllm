"""泰语数词（无空格，按最长匹配扫描）。เอ็ด 与 หนึ่ง 在比较时合并。"""

from ..numerals import LangSpec, add, big, place, with_fold, zero

UNITS = ["หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า"]
PLACES = {"สิบ": 1, "ร้อย": 2, "พัน": 3, "หมื่น": 4, "แสน": 5}


def build() -> LangSpec:
    spec = LangSpec(code = "th", scan = True)
    spec.add(words = "ศูนย์", morphs = zero())
    for v, w in enumerate(UNITS, start = 1):
        spec.add(words = w, morphs = add(value = v))
    # เอ็ด：十位或更高位之后的「一」；ยี่：只用在 สิบ 前的「二」
    spec.add(words = "เอ็ด", morphs = with_fold(morph = add(value = 1), fold = "หนึ่ง", after = "any"))
    spec.add(words = "ยี่", morphs = add(value = 2, needs = frozenset({"place"})))
    for w, exp in PLACES.items():
        spec.add(words = w, morphs = place(exp = exp))
    spec.add(words = "ล้าน", morphs = big(value = 10 ** 6))
    spec.add_sep(words = "จุด")
    spec.add_pct(words = ["เปอร์เซ็นต์", "เปอร์เซนต์"])
    spec.add_pct(words = "ร้อยละ", position = "pre")
    spec.add_minus(words = "ลบ")
    return spec
