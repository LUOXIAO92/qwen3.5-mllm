"""网页来源专用的过滤规则（T03b）。

网页和新闻文本里有大量维基、句库里没有的内容：页首页尾样板、版权声明、订阅与社交媒体号召、
电话和价格、图片说明、关键词堆砌、赌博和成人广告。这些规则只对网页来源开启
（`FilterOptions.web = True`），不改变 part1 条目的检查结果。

`check_web_rules` 返回丢弃原因或 None。规则按「文档级质量分无法保证干净」的原则，在句子级逐条过滤。
"""

from __future__ import annotations

import re
import unicodedata

from .langs import CASED_LANGS, LANGS as LANG_CODES, NO_SPACE_LANGS

# ---------------------------------------------------------------- 样板与号召语
# 每种语言列出常见的页首页尾、版权、订阅、点击、联系方式、推广用语（小写比较）。
_BOILERPLATE_WORDS = {
    "all": [
        "cookie", "copyright", "©", "all rights reserved", "privacy policy", "terms of use", "terms of service",
        "terms and conditions", "subscribe", "newsletter", "click here", "read more", "read also", "sign up",
        "log in", "login", "share this", "follow us", "like us", "whatsapp", "telegram", "facebook", "instagram",
        "twitter", "youtube", "tiktok", "linkedin", "pinterest", "advertisement", "sponsored", "affiliate",
        "free shipping", "add to cart", "buy now", "order now", "download now", "hotline", "wechat", "weibo",
        "http", "www", "e-mail", "email", "whats app", "line id",
        "casino", "slot gacor", "togel", "bokep", "xnxx", "porno", "sex video", "judi online", "betting", "viagra",
    ],
    "en": ["comments", "leave a reply", "posted by", "posted on", "filed under", "tags:", "related posts", "you may also like", "skip to", "menu", "search for"],
    "de": ["datenschutz", "impressum", "alle rechte vorbehalten", "newsletter abonnieren", "jetzt kaufen", "warenkorb", "kommentar", "mehr erfahren", "weiterlesen", "mehr lesen", "zum inhalt"],
    "fr": ["tous droits réservés", "politique de confidentialité", "mentions légales", "s'abonner", "lire la suite", "cliquez", "panier", "commentaire", "en savoir plus", "abonnez-vous", "partager"],
    "es": ["todos los derechos reservados", "política de privacidad", "suscríbete", "suscríbase", "leer más", "haz clic", "haga clic", "comentario", "carrito", "aviso legal", "compartir"],
    "pt": ["todos os direitos reservados", "política de privacidade", "leia mais", "clique", "comentário", "carrinho", "inscreva-se", "saiba mais", "compartilhe"],
    "it": ["tutti i diritti riservati", "leggi di più", "leggi anche", "clicca", "iscriviti", "carrello", "commento", "condividi", "informativa", "privacy"],
    "ru": ["все права защищены", "читать далее", "читайте также", "подписаться", "политика конфиденциальности", "нажмите", "комментарий", "в корзину", "подробнее", "поделиться", "куки"],
    "zh": ["版权所有", "版權所有", "免责声明", "免責聲明", "点击", "點擊", "关注我们", "關注我們", "扫码", "掃碼", "微信", "联系电话", "聯繫電話", "在线客服", "上一篇", "下一篇", "责任编辑", "责编", "转载", "轉載", "本站", "本网", "客服", "加盟", "订阅", "訂閱", "二维码", "公众号", "热线", "热销", "优惠", "促销", "包邮", ],
    "ja": ["著作権", "無断転載", "お問い合わせ", "プライバシーポリシー", "クリック", "送料無料", "カート", "会員登録", "ログイン", "シェア", "フォロー", "メルマガ", "詳しくはこちら", "続きを読む", "ご注文", "お申し込み", "運営者", "サイトマップ", "ブログランキング", "キャンペーン"],
    "ko": ["저작권", "무단 전재", "무단전재", "구독", "클릭", "문의", "개인정보", "댓글", "팔로우", "광고", "배송", "구매", "회원가입", "로그인", "사진=", "카카오톡"],
    "id": ["hak cipta", "baca juga", "klik", "hubungi kami", "berlangganan", "kebijakan privasi", "komentar", "bagikan", "selengkapnya", "baca selengkapnya", "simak juga", "pesan sekarang", "promo", "diskon", "gratis ongkir", "tag:"],
    "ms": ["hak cipta", "baca juga", "klik", "hubungi kami", "langgan", "dasar privasi", "komen", "kongsi", "selanjutnya", "baca lagi", "promosi", "diskaun", "harga"],
    "th": ["ลิขสิทธิ์", "อ่านต่อ", "คลิก", "ติดต่อ", "สมัครสมาชิก", "ราคา", "สั่งซื้อ", "แชร์", "ติดตาม", "ไลน์", "คอมเมนต์", "ความคิดเห็น", "ลงทะเบียน", "เข้าสู่ระบบ", "โปรโมชั่น", "ส่วนลด", "สล็อต", "บาคาร่า", "แทงบอล", "คาสิโน", ],
    "vi": ["bản quyền", "đọc thêm", "xem thêm", "liên hệ", "đăng ký", "hotline", "bạn đang xem", "bình luận", "chia sẻ", "đặt hàng", "khuyến mãi", "giảm giá", "mua ngay", "nhà cái", "cờ bạc", "lô đề", "sòng bài", "kèo", "đăng nhập", "tin liên quan"],
    "fil": ["karapatang-ari", "basahin", "i-click", "mag-subscribe", "mag-ugnay", "komento", "ibahagi", "presyo", "diskwento", "mag-log in", "pindutin", "sundan", "bisitahin"],
    "ar": ["جميع الحقوق محفوظة", "اقرأ أيضا", "اقرأ أيضاً", "اقرأ المزيد", "اشترك", "اتصل بنا", "اضغط", "تابعنا", "تعليق", "شارك", "السعر", "اطلب", "تسجيل الدخول", "للتواصل", "واتس", "إعلان", "تحميل"],
    "tr": ["tüm hakları saklıdır", "devamını oku", "abone ol", "tıklayın", "tıkla", "iletişim", "gizlilik", "yorum", "paylaş", "fiyat", "sipariş", "indirim", "bahis", "canlı", "kayıt ol", "giriş yap", "kampanya", "sepete"],
    "fa": ["کلیه حقوق", "ادامه مطلب", "بیشتر بخوانید", "عضویت", "تماس با ما", "کلیک", "نظرات", "اشتراک", "قیمت", "سفارش", "خرید", "ثبت نام", "ورود", "تخفیف", "دانلود", "شرط بندی", ],
}
# 太短或太通用的词用词边界匹配，避免命中普通词的一部分（如 "line id" 命中 "online identity"）。
def _compile_boilerplate(words: list[str], *, lang: str) -> re.Pattern:
    body = "|".join(re.escape(word) for word in sorted(set(words), key = len, reverse = True))
    # 有空格分词的语言要求词首对齐（"harga" 不命中 "menghargai"）；中日泰等不分词的语言按子串匹配。
    prefix = "" if lang in NO_SPACE_LANGS else r"(?<!\w)"
    return re.compile(prefix + "(?:" + body + ")", re.IGNORECASE)


_BOILERPLATE_RE = {lang: _compile_boilerplate(_BOILERPLATE_WORDS["all"] + _BOILERPLATE_WORDS.get(lang, []), lang = lang) for lang in LANG_CODES}
_BOILERPLATE_FALLBACK = _compile_boilerplate(_BOILERPLATE_WORDS["all"], lang = "en")
# 英语里 "comments"、"menu"、"search for" 这类词在正常句子里也会出现，只在句子很短时当样板。
_WEAK_EN = re.compile(r"\b(comments|menu|search for|skip to)\b", re.IGNORECASE)

# ---------------------------------------------------------------- 其他模式
_SOCIAL = re.compile(r"(^|\s)[#@][\w]{2,}|#\w+\s+#\w+")
_PHONE = re.compile(r"(\+\d[\d\s().-]{7,}\d|\b\d{2,4}[\s.-]\d{3,4}[\s.-]\d{3,4}\b|\(\d{2,4}\)\s?\d{3,4}[\s.-]?\d{3,4})")
_PRICE = re.compile(
    r"[$€£¥₩฿₫₺₽₹]\s?\d|\d\s?(usd|eur|vnd|idr|rupiah|rp\b|rm\b|ringgit|baht|บาท|元|円|원|руб|₽|đồng|dong|tl\b|₺|lira|php|peso|ريال|درهم|تومان)",
    re.IGNORECASE,
)
_NUMERIC_DATE = re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b")
_EMPHASIS = re.compile(r"[!?]{2,}|[!?][!?.]+[!?]|\.{4,}")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️]")
_BREADCRUMB = re.compile(r"\s[»›«‹|>→▶►]\s|[»›]\s*$|^\s*[«‹»›]")
_CAPTION = re.compile(
    r"^\s*\(?(photo|image|source|foto|fotos|fuente|fonte|quelle|crédit|credit|ảnh|nguồn|sumber|gambar|kaynak|fotoğraf|фото|источник|"
    r"图片|圖片|来源|來源|出典|写真|사진|출처|صورة|المصدر|عکس|منبع|ภาพ|ที่มา|larawan|pinagmulan|bild|video|by|tags?|tag|"
    r"posted|updated|published|diterbitkan|ditulis oleh|penulis|reporter|editor|admin|oleh|ngày|date|tác giả)\s*[:：]",
    re.IGNORECASE,
)
# 标点前多出空格（"lagi ?"、"OPD , organisasi"）、括号内侧空格：来源排版粗糙，朗读稿里的停顿会被 ASR 对错。
# 法语的 ? ! : ; 前面本来就有空格，只检查逗号和句点。
_SPACED_PUNCT = re.compile(r"\s[,;:?!]|\s\.(?!\.)|\(\s|\s\)")
_SPACED_PUNCT_FR = re.compile(r"\s[,.]|\(\s|\s\)")
# 站点地址里带赌博、成人、药品推广的词，整篇文档丢弃（在读取器里调用）。
_SPAM_URL = re.compile(
    r"slot|togel|judi|bokep|porn|porno|xnxx|xvideos|sex|casino|betting|\bbet\b|bet365|poker|gacor|viagra|cialis|penis|escort|"
    r"bandar|sbobet|maxwin|obat-kuat|hentai|nude|erotic|lottery|lotto|bahis|iddaa|canli|nhacai|danhbai|"
    r"赌|賭|博彩|彩票|娱乐城|娛樂城|棋牌|色情|成人|av[-_]|kasino|sabung",
    re.IGNORECASE,
)
_QUESTION_ANSWER = re.compile(r"^\s*(q|a|q&a|hỏi|đáp|soal|jawab|soru|cevap|سؤال|جواب)\s*[:：]", re.IGNORECASE)


def url_is_spam(url: str) -> bool:
    """站点地址带赌博、成人、药品推广词。"""
    return bool(_SPAM_URL.search(url))


def _words(text: str, *, lang: str) -> list[str]:
    return text.split() if lang not in NO_SPACE_LANGS else []


def _title_case(text: str, *, lang: str) -> bool:
    """每个词都大写开头的标题式句子（"Top 10 Tips For Better Sleep."）。"""
    if lang not in CASED_LANGS or lang == "de":
        return False
    words = [word for word in text.split() if word[:1].isalpha()]
    if len(words) < 5:
        return False
    capitalized = sum(word[0].isupper() for word in words)
    return capitalized / len(words) >= 0.75


def _shouting(text: str, *, lang: str) -> bool:
    if lang not in CASED_LANGS:
        return False
    words = [word for word in text.split() if len(word) >= 2 and word.isalpha()]
    if len(words) < 4:
        return False
    return sum(word.isupper() for word in words) / len(words) >= 0.3


def _repetitive(text: str, *, lang: str) -> bool:
    words = [word.lower().strip(".,;:!?\"'") for word in _words(text, lang = lang)]
    if len(words) < 10:
        return False
    return len(set(words)) / len(words) < 0.55


def check_web_rules(text: str, *, lang: str) -> str | None:
    """网页来源的附加规则；命中返回原因，通过返回 None。"""
    pattern = _BOILERPLATE_RE.get(lang, _BOILERPLATE_FALLBACK)
    if pattern.search(text):
        return "boilerplate"
    if lang == "en" and len(text) < 90 and _WEAK_EN.search(text):
        return "boilerplate"
    if _SOCIAL.search(text):
        return "social"
    if _PHONE.search(text):
        return "phone"
    if _PRICE.search(text):
        return "price"
    if _NUMERIC_DATE.search(text):
        return "numeric_date"
    if (_SPACED_PUNCT_FR if lang == "fr" else _SPACED_PUNCT).search(text):
        return "spaced_punct"
    if _EMPHASIS.search(text):
        return "emphasis"
    if _EMOJI.search(text):
        return "emoji"
    if _BREADCRUMB.search(text):
        return "breadcrumb"
    if _CAPTION.search(text) or _QUESTION_ANSWER.search(text):
        return "caption"
    if _shouting(text, lang = lang):
        return "shouting"
    if _title_case(text, lang = lang):
        return "title_case"
    if _repetitive(text, lang = lang):
        return "repetitive"
    if sum(unicodedata.category(ch) == "Sk" for ch in text):
        return "modifier_symbol"
    return None
