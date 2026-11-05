"""印尼语、马来语、泰语、越南语、菲律宾语、阿拉伯语、土耳其语、波斯语读法规则的测试（T04c）。

    python -m unittest pipeline.tn.tests.test_t04c

每种语言列出改写成功的例子（原文与朗读稿逐字对照）和必须丢弃的例子（tn_note 中应出现的说明），
另有菲律宾语、阿拉伯语、土耳其语自写读法函数的测试，以及读法经 pipeline.compare 转回数值的回环测试。
需要 num2words（pipeline.tn 的基础读法依赖）。
"""

import unittest

from pipeline.compare import compare
from pipeline.tn import clean, normalize_entry


# 改写成功：(原文, 朗读稿)
NORMALIZED = {
    "id": [
        ("Saya meminjam 1.000 yen.", "Saya meminjam seribu yen."),
        ("pada tahun 1908.", "pada tahun seribu sembilan ratus delapan."),
        ("abad ke-19 dan 3,5 persen", "abad kesembilan belas dan tiga koma lima persen"),
        ("hingga $1.730.", "hingga seribu tujuh ratus tiga puluh dolar."),
        ("Pesawat lepas landas pada pukul 17.30.", "Pesawat lepas landas pada pukul tujuh belas tiga puluh."),
        ("Aku tiba di Boston pukul 2.30.", "Aku tiba di Boston pukul dua tiga puluh."),
        ("Akan kutunggu sampai pukul 2:30.", "Akan kutunggu sampai pukul dua tiga puluh."),
        ("Di Jepang, sekarang 3:00 pagi.", "Di Jepang, sekarang tiga pagi."),
        ("Kegiatan dimulai pada pukul 8:10 pagi.", "Kegiatan dimulai pada pukul delapan sepuluh pagi."),
        ("pukul 09.30 WIB", "Pukul sembilan tiga puluh WIB"),
        ("Sejak tahun 70-an, NASA", "Sejak tahun tujuh puluhan, NASA"),
        ("sejak pertengahan 1970-an.", "sejak pertengahan seribu sembilan ratus tujuh puluhan."),
        ("Harga kamar RM160 per malam.", "Harga kamar seratus enam puluh ringgit per malam."),
        ("Jumlah sudut adalah 1080°.", "Jumlah sudut adalah seribu delapan puluh derajat."),
        ("Suhu turun ke -10 derajat.", "Suhu turun ke minus sepuluh derajat."),
        ("Ini menarik 10000 orang.", "Ini menarik sepuluh ribu orang."),
        ("pada tanggal 26 Juli 1887.", "pada tanggal dua puluh enam Juli seribu delapan ratus delapan puluh tujuh."),
        ("itu mencapai 250 juta", "itu mencapai dua ratus lima puluh juta"),
        ("Rp 50 ribu", "Lima puluh ribu rupiah"),
        ("tahun 2000-an", "tahun dua ribuan"),
        ("kecepatan 100 km/h", "kecepatan seratus kilometer per jam"),
        ("berat 5 kg", "berat lima kilogram"),
        ("suhu 30°C", "suhu tiga puluh derajat Celsius"),
        ("jam 10.15 malam", "Jam sepuluh lima belas malam"),
        ("abad ke-21", "abad kedua puluh satu"),
        ("50% siswa", "Lima puluh persen siswa"),
        ("3.000 pelajar", "Tiga ribu pelajar"),
        ("1,5 juta orang", "Satu koma lima juta orang"),
        ("pukul 23.59 WIB", "Pukul dua puluh tiga lima puluh sembilan WIB"),
    ],
    "ms": [
        ("Pada 21 Jun, 1974, José datang.", "Pada dua puluh satu Jun, seribu sembilan ratus tujuh puluh empat, José datang."),
        ("pada pukul 6.", "pada pukul enam."),
        ("dalam 300 batu dan 1.5 km", "dalam tiga ratus batu dan satu perpuluhan lima kilometer"),
        ("membayar RM50", "membayar lima puluh ringgit"),
        ("Aku akan kembali pada pukul 2:30.", "Aku akan kembali pada pukul dua tiga puluh."),
        ("pergi pada pukul 7.30?", "pergi pada pukul tujuh tiga puluh?"),
        ("harga 7.30 ringgit", "harga tujuh perpuluhan tiga kosong ringgit"),
        ("Dia lebih 80 kilo.", "Dia lebih lapan puluh kilo."),
        ("tidur selama 11 jam", "tidur selama sebelas jam"),
        ("seramai 1,000,000 orang", "seramai satu juta orang"),
        ("pada 18 Oktober 1963", "pada lapan belas Oktober seribu sembilan ratus enam puluh tiga"),
        ("8 Bilion", "Lapan Bilion"),
        ("Hadiah Nobel 1967 dalam Fizik", "Hadiah Nobel seribu sembilan ratus enam puluh tujuh dalam Fizik"),
        ("ke-2 kali", "Kedua kali"),
        ("1980-an", "Seribu sembilan ratus lapan puluhan"),
        ("30%", "Tiga puluh peratus"),
        ("pukul 6 petang", "pukul enam petang"),
        ("jarak 12 km", "jarak dua belas kilometer"),
        ("berat 2 kg", "berat dua kilogram"),
        ("RM1,500", "Seribu lima ratus ringgit"),
        ("pada tahun 2008", "pada tahun dua ribu lapan"),
        ("berumur 25 tahun", "berumur dua puluh lima tahun"),
        ("kira-kira 100 orang", "kira-kira seratus orang"),
        ("abad ke-21", "abad kedua puluh satu"),
        ("pukul 10.45 pagi", "Pukul sepuluh empat puluh lima pagi"),
        ("dalam 5 minit", "dalam lima minit"),
        ("50% pelajar", "Lima puluh peratus pelajar"),
        ("3,000 pelajar", "Tiga ribu pelajar"),
        ("1.5 juta orang", "Satu perpuluhan lima juta orang"),
        ("suhu 30°C", "suhu tiga puluh darjah Celsius"),
        ("ke-3 kalinya", "Ketiga kalinya"),
    ],
    "th": [
        ("ไปกับอังกฤษ 99 ปี", "ไปกับอังกฤษ เก้าสิบเก้า ปี"),
        ("เขาเกิดในศตวรรษที่ 19", "เขาเกิดในศตวรรษที่ สิบเก้า"),
        ("มี ๒๑ คน", "มี ยี่สิบเอ็ด คน"),
        ("เวลา 14.30 น. เราไป", "เวลา สิบสี่นาฬิกาสามสิบนาที เราไป"),
        ("เวลา 08.00 น.", "เวลา แปดนาฬิกา"),
        ("นัดกันตอน 9:45", "นัดกันตอน เก้านาฬิกาสี่สิบห้านาที"),
        ("ราคา 14.30 บาท", "ราคา สิบสี่บาทสามสิบสตางค์"),
        ("ราคา 99.50 บาท", "ราคา เก้าสิบเก้าบาทห้าสิบสตางค์"),
        ("ราคา 100.00 บาท", "ราคา หนึ่งร้อยบาท"),
        ("ถูกหยุดไปเมื่อปีค.ศ.1993", "ถูกหยุดไปเมื่อปีคริสต์ศักราชหนึ่งพันเก้าร้อยเก้าสิบสาม"),
        ("ในปี พ.ศ. 2500 มี", "ในปี พุทธศักราช สองพันห้าร้อย มี"),
        ("อุณหภูมิ -10 องศา", "อุณหภูมิ ลบสิบ องศา"),
        ("อุณหภูมิ -10°C", "อุณหภูมิ ลบสิบองศาเซลเซียส"),
        ("มาสายดีกว่าไม่มาเลย.", "มาสายดีกว่าไม่มาเลย"),
        ("80 เปอร์เซ็นต์", "แปดสิบ เปอร์เซ็นต์"),
        ("น้ำเดือดที่ 100 องศาเซลเซียส", "น้ำเดือดที่ หนึ่งร้อย องศาเซลเซียส"),
        ("ลดราคา 5%", "ลดราคา ห้าเปอร์เซ็นต์"),
        ("ยาว 3 km", "ยาว สามกิโลเมตร"),
        ("1,000,000 คน", "หนึ่งล้าน คน"),
        ("ค่า 3.14", "ค่า สามจุดหนึ่งสี่"),
        ("ปี 2023", "ปี สองพันยี่สิบสาม"),
        ("11 คน", "สิบเอ็ด คน"),
        ("ราคา 20 บาท", "ราคา ยี่สิบ บาท"),
        ("ปี พ.ศ. 2567", "ปี พุทธศักราช สองพันห้าร้อยหกสิบเจ็ด"),
        ("ค.ศ. 2020", "คริสต์ศักราช สองพันยี่สิบ"),
        ("เวลา 9.05 น.", "เวลา เก้านาฬิกาห้านาที"),
        ("ลด 10%", "ลด สิบเปอร์เซ็นต์"),
        ("ห้อง 101", "ห้อง หนึ่งร้อยเอ็ด"),
        ("1,500 คน", "หนึ่งพันห้าร้อย คน"),
        ("-5 องศา", "ลบห้า องศา"),
        ("เวลา 07.15 น.", "เวลา เจ็ดนาฬิกาสิบห้านาที"),
        ("หนัก 15.5 กิโลกรัม", "หนัก สิบห้าจุดห้า กิโลกรัม"),
    ],
    "vi": [
        ("FIFA World Cup 2026 là lần thứ 23", "FIFA World Cup hai nghìn không trăm hai mươi sáu là lần thứ hai mươi ba"),
        ("ngày 11 tháng 4", "ngày mười một tháng tư"),
        ("khoảng 700 tỉ đô la", "khoảng bảy trăm tỉ đô la"),
        ("giá 1.500 đồng, tăng 3,5%", "giá một nghìn năm trăm đồng, tăng ba phẩy năm phần trăm"),
        ("thứ 1 và 105", "Thứ nhất và một trăm linh năm"),
        ("Gặp tôi vào lúc 6:30 tối.", "Gặp tôi vào lúc sáu giờ ba mươi phút tối."),
        ("Hạn chót là chiều mai lúc 2:30.", "Hạn chót là chiều mai lúc hai giờ ba mươi phút."),
        ("lúc 14:00", "Lúc mười bốn giờ"),
        ("Máy bay bay với tốc độ 400 km một giờ.", "Máy bay bay với tốc độ bốn trăm ki lô mét một giờ."),
        ("Đi thẳng 200m, sau đó rẽ phải.", "Đi thẳng hai trăm mét, sau đó rẽ phải."),
        ("nặng 5 kg", "nặng năm ki lô gam"),
        ("Hiện tại là -10°C.", "Hiện tại là âm mười độ C."),
        ("thu hút hơn 10000 người dùng", "thu hút hơn mười nghìn người dùng"),
        ("Được, tốt rồi.Tạm biệt nhé.", "Được, tốt rồi. Tạm biệt nhé."),
        ("năm 2008", "năm hai nghìn không trăm linh tám"),
        ("21 tuổi", "Hai mươi mốt tuổi"),
        ("24 giờ", "Hai mươi tư giờ"),
        ("15 phút", "Mười lăm phút"),
        ("1.000.000 người", "Một triệu người"),
        ("nhiệt độ 30°C", "nhiệt độ ba mươi độ C"),
        ("lúc 8:05 sáng", "Lúc tám giờ năm phút sáng"),
        ("khoảng 15%", "khoảng mười lăm phần trăm"),
        ("năm 1975", "năm một nghìn chín trăm bảy mươi lăm"),
        ("thứ 4", "Thứ tư"),
        ("tháng 1", "Tháng một"),
        ("tầng 10", "tầng mười"),
        ("số pi là 3,14", "số pi là ba phẩy mười bốn"),
        ("tốc độ 50 km/h", "tốc độ năm mươi ki lô mét trên giờ"),
        ("lúc 23:59 đêm", "Lúc hai mươi ba giờ năm mươi chín phút đêm"),
        ("giá 10.000 đồng", "giá mười nghìn đồng"),
    ],
    "fil": [
        ("Isinilang po ako noong ika-14 ng Pebrero, taong 1960.", "Isinilang po ako noong ikalabing-apat ng Pebrero, taong isang libo siyam na raan at animnapu."),
        ("noong 1983, at hindi na", "noong isang libo siyam na raan at walumpu't tatlo, at hindi na"),
        ("120 nang 160 sentimetro", "Isang daan at dalawampu nang isang daan at animnapung sentimetro"),
        ("nitong ika-18 ng Disyembre ng 2022 sa kapihan.", "nitong ikalabingwalo ng Disyembre ng dalawang libo at dalawampu't dalawa sa kapihan."),
        ("higit sa 400 gradong Celsius", "higit sa apat na raang gradong Celsius"),
        ("30 taon na siyang nagtatrabaho.", "Tatlumpung taon na siyang nagtatrabaho."),
        ("Higit nang 100 milyon na", "Higit nang isang daang milyon na"),
        ("eksaktong alas 6 ng gabi.", "eksaktong alas sais ng gabi."),
        ("nahuli ng 10 minuto.", "nahuli ng sampung minuto."),
        ("Mukhang 19 taong gulang", "Mukhang labinsiyam na taong gulang"),
        ("Marso 1841, naging", "Marso isang libo walong daan at apatnapu't isa, naging"),
        ("mahigit nang 70%.", "mahigit nang pitumpung porsiyento."),
        ("ang ika-5 planetang", "ang ikalimang planetang"),
        ("alas 3 at alas 5 ng madaling araw", "Alas tres at alas singko ng madaling araw"),
        ("ang 136 na taon", "ang isang daan at tatlumpu't anim na taon"),
        ("mga 1300 Tiyera", "mga isang libo tatlong daang Tiyera"),
        ("mga 5000 o mahigit", "mga limang libo o mahigit"),
        ("May 4 na libro ako.", "May apat na libro ako."),
        ("hanggang 100 dolyar.", "hanggang isang daang dolyar."),
        ("bawang, atbp.", "bawang, at iba pa."),
        ("atbp. na mga", "At iba pa na mga"),
        ("10000 yen", "Sampung libong yen"),
        ("noong 2005 bilang kasapi", "noong dalawang libo at lima bilang kasapi"),
        ("21 sa 20", "Dalawampu't isa sa dalawampu"),
        ("dekadang 1970 nang", "dekadang isang libo siyam na raan at pitumpu nang"),
        ("11 taong", "Labing-isang taong"),
        ("ika-23 ng Marso", "Ikadalawampu't tatlo ng Marso"),
        ("Gradong 7 ako", "Gradong pito ako"),
        ("ika-2 at ika-3", "Ikalawa at ikatlo"),
    ],
    "ar": [
        ("في نهائي عام 2025 الذي", "في نهائي عام ألفين وخمسة وعشرين الذي"),
        ("منذ 2006.", "منذ ألفين وستة."),
        ("في عام 1987،", "في عام ألف وتسعمئة وسبعة وثمانين،"),
        ("توفي رجل يبلغ من العمر 71 عامًا بعد ذلك.", "توفي رجل يبلغ من العمر واحد وسبعين عامًا بعد ذلك."),
        ("في أبريل 1919. وقد انضمت", "في أبريل ألف وتسعمئة وتسعة عشر. وقد انضمت"),
        ("حتى عام 1849 م، عندما", "حتى عام ألف وثمانمئة وتسعة وأربعين للميلاد، عندما"),
        ("في عام 2021م رعى", "في عام ألفين وواحد وعشرين للميلاد رعى"),
        ("بلغ معدل الإلمام بالقراءة 93.4%.", "بلغ معدل الإلمام بالقراءة ثلاثة وتسعين فاصلة أربعة بالمئة."),
        ("تصل إلى €5,000 لشراء سيارة.", "تصل إلى خمسة آلاف يورو لشراء سيارة."),
        ("تقع على بعد 187 كم إلى الشرق", "تقع على بعد مئة وسبعة وثمانين كيلومترًا إلى الشرق"),
        ("بقي حوالي 40 من السجناء", "بقي حوالي أربعين من السجناء"),
        ("وفاز بنتيجة 3 أهداف.", "وفاز بنتيجة ثلاثة أهداف."),
        ("استمرت الحرب 5 سنوات.", "استمرت الحرب خمس سنوات."),
        ("تضم المدينة 15 مدرسة.", "تضم المدينة خمس عشرة مدرسة."),
        ("في 15 مباراة", "في خمس عشرة مباراة"),
        ("خلال 12 سنة", "خلال اثنتي عشرة سنة"),
        ("عاش 100 عام", "عاش مئة عام"),
        ("بين عامي 1990 و1995", "بين عامي ألف وتسعمئة وتسعين وألف وتسعمئة وخمسة وتسعين"),
        ("ما بين 75 و150 سنتيمتراً", "ما بين خمسة وسبعين ومئة وخمسين سنتيمتراً"),
        ("بين 18 و30 سنة", "بين ثماني عشرة وثلاثين سنة"),
        ("بـ199 دولار", "بمئة وتسعة وتسعين دولار"),
        ("لـ اقتصاد رومانيا", "لاقتصاد رومانيا"),
        ("التابع لـ الشركة", "التابع للشركة"),
        ("عام 430 ق م", "عام أربعمئة وثلاثين قبل الميلاد"),
        ("في 21 من كانون الثاني", "في واحد وعشرين من كانون الثاني"),
        ("في 30 مارس 1974", "في ثلاثين مارس ألف وتسعمئة وأربعة وسبعين"),
        ("يقدر بنحو 2000 شخص.", "يقدر بنحو ألفي شخص."),
        ("مكتوبة في 22 دولة", "مكتوبة في اثنتين وعشرين دولة"),
        ("عن عمر يناهز 70 عامًا", "عن عمر يناهز سبعين عامًا"),
        ("هناك 25 طالبًا", "هناك خمسة وعشرون طالبًا"),
        ("في 1 كانون الثاني", "في واحد كانون الثاني"),
        ("بنسبة 8 في المائة", "بنسبة ثمانية في المائة"),
        ("يبلغ طوله 120 كم مربع", "يبلغ طوله مئة وعشرين كيلومترًا مربعًا"),
        ("في عام 2012م", "في عام ألفين واثني عشر للميلاد"),
        ("سنة 1517هـ", "سنة ألف وخمسمئة وسبعة عشر للهجرة"),
        ("في 9 رجب 1395هـ", "في تسعة رجب ألف وثلاثمئة وخمسة وتسعين للهجرة"),
        ("تراوحت بين %72 و%80 من", "تراوحت بين اثنين وسبعين بالمئة وثمانين بالمئة من"),
        ("حصل على 3 جوائز", "حصل على ثلاث جوائز"),
    ],
    "tr": [
        ("7 Eylül 2006'da Amsterdam'da okudum.", "Yedi Eylül iki bin altıda Amsterdam'da okudum."),
        ("Saat 6'ya kadar vegan", "Saat altıya kadar vegan"),
        ("yetişkinlerin yaklaşık % 65'i kilolu.", "yetişkinlerin yaklaşık yüzde altmış beşi kilolu."),
        ("Ada ihracatının %82'si tarımsal.", "Ada ihracatının yüzde seksen ikisi tarımsal."),
        ("4'ü de geldi.", "Dördü de geldi."),
        ("7'nci sırada", "Yedinci sırada"),
        ("1990'lı yıllarda ve 1980'lerde", "Bin dokuz yüz doksanlı yıllarda ve bin dokuz yüz seksenlerde"),
        ("100'lük banknot", "Yüzlük banknot"),
        ("Akşam saat 06:30'da buluşalım.", "Akşam saat altı otuzda buluşalım."),
        ("tren 4:35'te geldi", "tren dört otuz beşte geldi"),
        ("Tom saat 2.30'a kadar çalıştı.", "Tom saat iki otuza kadar çalıştı."),
        ("Canım, saat 11.00'de iş için", "Canım, saat on birde iş için"),
        ("Neredeyse -40 dereceydi.", "Neredeyse eksi kırk dereceydi."),
        ("II. Dünya Savaşı hakkında", "İkinci Dünya Savaşı hakkında"),
        ("23. Tümen, bulunduğu yeri", "Yirmi üçüncü Tümen, bulunduğu yeri"),
        ("2. Dünya Savaşının ortasında", "İkinci Dünya Savaşının ortasında"),
        ("1919'da geldi", "Bin dokuz yüz on dokuzda geldi"),
        ("15'te bitti", "On beşte bitti"),
        ("3'ü aldı", "Üçü aldı"),
        ("5'er elma", "Beşer elma"),
        ("1000'den fazla", "Binden fazla"),
        ("2009'dan beri", "İki bin dokuzdan beri"),
        ("Kapılar açar.Eğer biri gelirse", "Kapılar açar. Eğer biri gelirse"),
        ("4. sınıf", "Dördüncü sınıf"),
        ("%25 indirim", "Yüzde yirmi beş indirim"),
        ("5,5 milyonluk", "Beş virgül beş milyonluk"),
        ("saat 14:30", "Saat on dört otuz"),
        ("1940'larda", "Bin dokuz yüz kırklarda"),
        ("dört 4'te", "dört dörtte"),
    ],
    "fa": [
        ("حداقل 3000 زبان", "حداقل سه هزار زبان"),
        ("در سال ۱۸۹۴ منتشر شد", "در سال هزار و هشتصد و نود و چهار منتشر شد"),
        ("تا ساعت 11 بر گردی", "تا ساعت یازده بر گردی"),
        ("امروز ۱۲ اوت است", "امروز دوازدهم اوت است"),
        ("در ۳۰ ژانویه", "در سی‌ام ژانویه"),
        ("۹۹ درصد", "نود و نه درصد"),
        ("نسبت به قیمت عادی ٪۱۰ تخفیف دارد.", "نسبت به قیمت عادی ده درصد تخفیف دارد."),
        ("15ام مارس 20امین سالگرد", "پانزدهم مارس بیستمین سالگرد"),
        ("او تا 90سالگی عمر", "او تا نود‌سالگی عمر"),
        ("برادم جک 14-ساله است.", "برادم جک چهارده‌ساله است."),
        ("در ساعت 2:30 پرواز", "در ساعت دو و سی دقیقه پرواز"),
        ("ساعت 13:00 برویم", "ساعت سیزده برویم"),
        ("فقط ۱۰۰۰۰ راه پیدا کردم", "فقط ده هزار راه پیدا کردم"),
        ("فـقط اندکی می‌دانند", "فقط اندکی می‌دانند"),
        ("دمای هوا -5 درجه است", "دمای هوا منفی پنج درجه است"),
        ("3ام", "سوم"),
        ("۲٫۵ درصد", "دو ممیز پنج درصد"),
        ("رفته تو.کسی نیست", "رفته تو. کسی نیست"),
        ("12.5٪ کل", "دوازده ممیز پنج درصد کل"),
        ("۱٬۰۰۰ تومان", "هزار تومان"),
        ("۵ km", "پنج کیلومتر"),
        ("در ابتدای ۲۰ سالگی", "در ابتدای بیست سالگی"),
        ("با شماره 110 تماس", "با شماره صد و ده تماس"),
        ("۲۳ مرداد", "بیست و سوم مرداد"),
        ("0 درجه", "صفر درجه"),
        ("۳۲ دندان", "سی و دو دندان"),
        ("۲۰۰۰۰۰ نفر", "دویست هزار نفر"),
        ("1۳ سال", "سیزده سال"),
        ("سال ۱۴۰۲ شمسی", "سال هزار و چهارصد و دو شمسی"),
        ("۳٬۵۰۰ نفر", "سه هزار و پانصد نفر"),
        ("21امین سالگرد", "بیست و یکمین سالگرد"),
        ("-10 درجه", "منفی ده درجه"),
        ("۷ km", "هفت کیلومتر"),
    ],
}

# 必须丢弃：(原文, tn_note 中应出现的片段)
DROPPED = {
    "id": [
        ("Kadar CO2 naik", "数字与字母相连"),
        ("pukul 08.05 pagi", "分钟以 0 开头"),
        ("gempa 8.8 skala richter", "数字写法无法确定"),
        ("hubungi 119.", "电话号码"),
        ("Nomor 5 adalah", "电话号码"),
        ("orang 12.5% dari", "数字写法无法确定"),
        ("selama 2-3 menit", "范围"),
        ("COVID-19 tetapi", "型号或编号"),
        ("PLTN No.1 di Fukushima", "缩写"),
        ("Dr. Baker", "缩写"),
        ("pukul 25.00", "时刻写法无法确定"),
    ],
    "ms": [
        ("suhu -5 darjah", "负号"),
        ("Langkah 1: Jadilah menarik.", "冒号"),
        ("Veronique AGI No. 47.", "缩写"),
        ("Gambar baru 15GB", "单位读法未登记"),
        ("nombor 5", "电话号码"),
        ("harga RM2.50", "货币金额暂不处理"),
        ("COVID-19 melanda", "型号或编号"),
        ("papan 8x8", "数字与字母相连"),
        ("10-20 orang", "范围"),
    ],
    "th": [
        ("จอมพล ป. พิบูลสงคราม", "缩写"),
        ("ซี. เอส. ลิวอิส", "缩写"),
        ("ที่รพ.บำรุงราษฎร์", "缩写"),
        ("กสพท. ประกาศ", "缩写"),
        ("เวลา 25.00 น.", "时刻写法无法确定"),
        ("โควิด-19 เป็นโรค", "连字符"),
        ("ม.6 เทอม2", "缩写"),
        ("นัด 10-12 คน", "范围"),
        ("ตัดสินใจครั้งนี้อีกด้วยดร.", "缩写"),
    ],
    "vi": [
        ("Nó lỡ chuyến xe 8:30.", "是否为时刻"),
        ("tỉ số 2:1", "冒号"),
        ("tỷ lệ là 3.4%", "数字写法无法确定"),
        ("tệp đối tượng db.o.", "缩写"),
        ("Vua Charles III", "罗马数字"),
        ("lưới 8x8", "数字与字母相连"),
        ("độ tuổi 11-12", "范围"),
        ("lúc 25:00", "时刻写法无法确定"),
        ("tháng 13", "月份写法无法确定"),
        ("Covid-19", "型号或编号"),
    ],
    "fil": [
        ("noong 410 AD.", "纪元标记"),
        ("96.5 porsiyentong karbon", "小数"),
        ("pulong alas 2:30.", "分钟"),
        ("ang ika-1 ng Enero", "序数"),
        ("ika-1,000 selyo", "序数"),
        ("sa COVID-19", "型号或编号"),
        ("nang 5:00 ng hapon", "冒号"),
        ("ang 18 000 daigdig", "空格分隔"),
        ("sa 7-11", "范围"),
        ("0 grado", "零"),
        ("alas 13", "时刻写法无法确定"),
    ],
    "ar": [
        ("كانت الرحلة تقل 246 راكبًا.", "格未知"),
        ("قتل 25 شخصًا", "格未知"),
        ("2 مباراتي الذهاب", "性、数、格"),
        ("في 5.55 مليون", "小数"),
        ("بين 25,000 و 30,000 قدم", "ألفًا"),
        ("فاز تشيلسي 3–1.", "范围"),
        ("ذات الـ11 عاماً", "缩写"),
        ("خلال 101 عام", "1、2 加名词"),
        ("خلال 13 أيام", "形式与数不符"),
        ("حصل على 19 جائزة كبرى منها 4 أخرى", "性、数、格"),
    ],
    "tr": [
        ("öğleden sonra 3:00 gibi", "是否为时刻"),
        ("Yaklaşık 1:58'lik o akordun", "时长"),
        ("450.000' in üzerinde", "撇号与后缀分开"),
        ("Kameri ay 29.53 gündür.", "数字写法无法确定"),
        ("1999-2006'da oldu", "范围"),
        ("1935'de geldi", "不和谐"),
        ("Puanı 2. Sonra geldi", "序数"),
        ("1.250,50 TL", "货币金额暂不处理"),
        ("10-15 öğrenci", "范围"),
        ("CDMA2000'dır", "数字与字母相连"),
        ("B-52 tek", "型号或编号"),
    ],
    "fa": [
        ("ساعت ۳ ب.ظ.", "缩写"),
        ("بین 1 و -1 است", "负号"),
        ("(1616-1564)", "范围"),
        ("N8 اولین دستگاه", "数字与字母相连"),
        ("۱۶۳۳: اینکه", "冒号"),
        ("۱/۲ کیلو", "分数或斜线"),
        ("حدود ۲-۳ روز", "范围"),
    ],
}

# 读法与原文写法语序或成分不同、比对模块不转换的类别：单位、货币、时刻、纪年和纪元词、年代后缀、缩写、
# 罗马数字、菲律宾语连接成分 na。这些改写只核对每处数词的数值（normalize_entry 内部已做），不比对整句。
COMPARE_SKIP = ("单位", "货币", "时刻", "年代", "缩写", "序数「II.」", "句末句点")
COMPARE_SKIP_SCRIPTS = (
    "للميلاد", "للهجرة", "قبل الميلاد", "พุทธศักราช", "คริสต์ศักราช", " na ", "at iba pa",
)
# 比对模块不认的写法：「7'nci」（撇号加序数后缀）、「لـ الشركة」（ل 与冠词合写为 لل）
COMPARE_KNOWN = {("tr", "7'nci sırada"), ("ar", "التابع لـ الشركة")}


class T04cNormalizedTest(unittest.TestCase):
    def test_counts(self):
        for lang in NORMALIZED:
            with self.subTest(lang = lang):
                self.assertGreaterEqual(len(NORMALIZED[lang]) + len(DROPPED[lang]), 40)

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
                    self.assertIn(reason, result.note)

    def test_scripts_have_no_digits(self):
        for lang, cases in NORMALIZED.items():
            for _, script in cases:
                self.assertFalse(any(ch.isdigit() for ch in script), msg = script)

    def test_numbers_match_original(self):
        for lang, cases in NORMALIZED.items():
            for original, script in cases:
                note = normalize_entry(original = original, lang = lang).note
                if any(kind in note for kind in COMPARE_SKIP) or any(w in script for w in COMPARE_SKIP_SCRIPTS):
                    continue
                if (lang, original) in COMPARE_KNOWN:
                    continue
                with self.subTest(lang = lang, original = original):
                    check = compare(script = script, hypothesis = clean(text = original, lang = lang), lang = lang)
                    self.assertTrue(check.match, msg = check.detail)


class ArabicNumberTest(unittest.TestCase):
    def test_gender_and_case(self):
        from pipeline.tn.langs.ar import ar_cardinal

        cases = [
            # (数, 名词的性, 格, 后接名词, 读法)
            (3, "m", "nom", True, "ثلاثة"), (3, "f", "nom", True, "ثلاث"), (8, "f", "obl", True, "ثماني"),
            (10, "m", "obl", True, "عشرة"), (10, "f", "obl", True, "عشر"), (11, "m", "nom", True, "أحد عشر"),
            (11, "f", "nom", True, "إحدى عشرة"), (12, "m", "nom", True, "اثنا عشر"), (12, "f", "obl", True, "اثنتي عشرة"),
            (15, "m", "nom", True, "خمسة عشر"), (15, "f", "obl", True, "خمس عشرة"), (18, "f", "obl", True, "ثماني عشرة"),
            (20, "m", "nom", True, "عشرون"), (20, "f", "obl", True, "عشرين"), (21, "m", "nom", True, "واحد وعشرون"),
            (21, "f", "obl", True, "إحدى وعشرين"), (22, "f", "obl", True, "اثنتين وعشرين"), (25, "f", "nom", True, "خمس وعشرون"),
            (100, "m", "nom", True, "مئة"), (200, "m", "nom", True, "مئتا"), (200, "m", "obl", False, "مئتين"),
            (354, "f", "obl", True, "ثلاثمئة وأربع وخمسين"), (1000, "m", "nom", True, "ألف"), (2000, "m", "obl", True, "ألفي"),
            (2000, "m", "nom", False, "ألفان"), (3000, "f", "nom", True, "ثلاثة آلاف"), (25000, "m", "obl", True, "خمسة وعشرين ألف"),
            (100000, "m", "nom", True, "مئة ألف"), (200000, "m", "obl", True, "مئتي ألف"), (1517, "m", "obl", False, "ألف وخمسمئة وسبعة عشر"),
            (2025, "m", "obl", False, "ألفين وخمسة وعشرين"), (3000000, "m", "nom", True, "ثلاثة ملايين"),
        ]
        for n, gender, case, construct, words in cases:
            with self.subTest(n = n, gender = gender, case = case):
                self.assertEqual(ar_cardinal(n = n, gender = gender, case = case, construct = construct), words)

    def test_unknown(self):
        from pipeline.tn.langs.ar import Unknown, resolve

        self.assertEqual(resolve(n = 300, gender = None, case = None, construct = False), "ثلاثمئة")
        self.assertEqual(resolve(n = 13, gender = "f", case = None, construct = True), "ثلاث عشرة")
        for n, gender, case in ((15, None, "obl"), (25, "m", None), (2000, "m", None)):
            with self.subTest(n = n):
                with self.assertRaises(Unknown):
                    resolve(n = n, gender = gender, case = case, construct = True)
        with self.assertRaises(Unknown):
            resolve(n = 25000, gender = "m", case = "obl", construct = False)

    def test_round_trip(self):
        """阳性属格读法（年份、整数）经 pipeline.compare 转回数值，与原数一致。"""
        from pipeline.tn.langs.ar import ar_cardinal

        for n in list(range(1, 130)) + [200, 300, 999, 1000, 1001, 1100, 1517, 1919, 1999, 2000, 2012, 2025, 2999]:
            with self.subTest(n = n):
                words = ar_cardinal(n = n, gender = "m", case = "obl", construct = False)
                self.assertTrue(compare(script = words, hypothesis = str(n), lang = "ar").match, msg = words)


class FilipinoNumberTest(unittest.TestCase):
    def test_cardinal(self):
        from pipeline.tn.langs.fil import fil_cardinal

        cases = {
            1: "isa", 10: "sampu", 11: "labing-isa", 14: "labing-apat", 17: "labimpito", 21: "dalawampu't isa",
            40: "apatnapu", 100: "isang daan", 108: "isang daan at walo", 409: "apat na raan at siyam",
            600: "anim na raan", 1300: "isang libo tatlong daan", 1755: "isang libo pitong daan at limampu't lima",
            1966: "isang libo siyam na raan at animnapu't anim", 2005: "dalawang libo at lima",
            2011: "dalawang libo at labing-isa", 10000: "sampung libo", 400000: "apat na raang libo",
            1000000: "isang milyon",
        }
        for n, words in cases.items():
            with self.subTest(n = n):
                self.assertEqual(fil_cardinal(n = n), words)

    def test_ordinal_and_linker(self):
        from pipeline.tn.langs.fil import fil_ordinal, linked

        self.assertEqual([fil_ordinal(n = n) for n in (2, 3, 4, 7, 18, 21, 30)],
                         ["ikalawa", "ikatlo", "ikaapat", "ikapito", "ikalabingwalo", "ikadalawampu't isa", "ikatatlumpu"])
        self.assertEqual([linked(words = w) for w in ("sampu", "isang daan", "apat", "dalawampu't anim", "milyon")],
                         ["sampung", "isang daang", "apat na", "dalawampu't anim na", "milyong"])

    def test_round_trip(self):
        from pipeline.tn.langs.fil import fil_cardinal, fil_ordinal

        for n in list(range(1, 200)) + [999, 1000, 1010, 1755, 1966, 2005, 2022, 10000, 30000, 1234567]:
            with self.subTest(n = n):
                words = fil_cardinal(n = n)
                self.assertTrue(compare(script = words, hypothesis = str(n), lang = "fil").match, msg = words)
        for n in range(2, 100):
            with self.subTest(ordinal = n):
                words = fil_ordinal(n = n)
                self.assertTrue(compare(script = words, hypothesis = f"ika-{n}", lang = "fil").match, msg = words)


class TurkishSuffixTest(unittest.TestCase):
    def test_suffix_forms(self):
        from pipeline.tn.langs.tr import with_suffix

        cases = [
            ("iki bin altı", "da", "iki bin altıda"), ("bin dokuz yüz on dokuz", "da", "bin dokuz yüz on dokuzda"),
            ("on beş", "te", "on beşte"), ("dört", "ü", "dördü"), ("dört", "e", "dörde"), ("üç", "ü", "üçü"),
            ("altı", "ya", "altıya"), ("yüz", "lük", "yüzlük"), ("doksan", "lı", "doksanlı"), ("seksen", "lerde", "seksenlerde"),
            ("yedi", "nci", "yedinci"), ("bir", "inci", "birinci"), ("beş", "er", "beşer"), ("iki", "şer", "ikişer"),
            ("altmış beş", "i", "altmış beşi"), ("seksen iki", "si", "seksen ikisi"), ("bin", "den", "binden"),
        ]
        for spoken, suffix, out in cases:
            with self.subTest(spoken = spoken, suffix = suffix):
                self.assertEqual(with_suffix(spoken = spoken, suffix = suffix), out)

    def test_disharmony(self):
        from pipeline.tn.langs.tr import with_suffix

        for spoken, suffix in (("beş", "de"), ("altı", "a"), ("on", "de"), ("dört", "ncü")):
            with self.subTest(spoken = spoken, suffix = suffix):
                self.assertIsNone(with_suffix(spoken = spoken, suffix = suffix))


if __name__ == "__main__":
    unittest.main()
