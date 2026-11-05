"""英语、德语、法语、俄语、葡萄牙语、西班牙语、意大利语的完整读法测试（任务 T04b）。

    python -m unittest pipeline.tn.tests.test_european

每种语言列出改写成功的例子（原文、朗读稿）和必须丢弃的例子（原文、tn_note 中应出现的说明），合计不少于
40 条；另检查读法函数中改正 num2words 输出的部分。需要 num2words 和 pymorphy3（俄语）。
"""

import unittest

from pipeline.compare import compare
from pipeline.tn import clean, normalize_entry

# tn_note 中出现这些类别时，朗读稿与原文的比对不一致属于比对模块的已知局限（缩写、罗马数字、单位等不转回原写法）
COMPARE_GAPS = ("缩写", "罗马数字", "单位", "货币", "时刻", "符号", "日期", "复合词", "年代", "空格分千位", "省音")

NORMALIZED = {
    'en': [
        ('In 1886, Peebles was incorporated.', 'In eighteen eighty-six, Peebles was incorporated.'),
        ('The 1998 season was great.', 'The nineteen ninety-eight season was great.'),
        ('About 1500 people died in 1942.', 'About one thousand five hundred people died in nineteen forty-two.'),
        ('On 11 December 1942, Jews arrived.', 'On the eleventh of December nineteen forty-two, Jews arrived.'),
        ('He moved to the U.S. The next year he died.', 'He moved to the U S. The next year he died.'),
        ('The U.S. Army fought, i.e. it won, etc.', 'The U S Army fought, that is it won, et cetera.'),
        ('Dr. Smith met Mr. Jones in St. Louis.', 'Doctor Smith met Mister Jones in Saint Louis.'),
        ('Martin Luther King Jr. was born.', 'Martin Luther King Junior was born.'),
        ('Henry VIII married six times during World War II.', 'Henry the Eighth married six times during World War Two.'),
        ("The company Johnson & Johnson grew in the '90s and mid-1960s.", 'The company Johnson and Johnson grew in the nineties and mid-nineteen sixties.'),
        ('It cost $1.1 billion in the 60s.', 'It cost one point one billion dollars in the sixties.'),
        ('In June 2005, he was inducted.', 'In June two thousand and five, he was inducted.'),
        ('The 2000 U.S. census.', 'The two thousand U S census.'),
        ('c. 1500 the town grew.', 'Circa fifteen hundred the town grew.'),
        ('It was released in 1998 by No. 5 records at 7 p.m.', 'It was released in nineteen ninety-eight by number five records at seven P M.'),
        ('He killed 1542 soldiers.', 'He killed one thousand five hundred and forty-two soldiers.'),
        ('He was born 4 July 1990.', 'He was born the fourth of July nineteen ninety.'),
        ("The government's 1936 creation of Trans-Canada Air Lines.", "The government's nineteen thirty-six creation of Trans-Canada Air Lines."),
        ('The house was grade II listed in 1954.', 'The house was grade Two listed in nineteen fifty-four.'),
        ('Elizabeth II visited Mt. Wilson.', 'Elizabeth the Second visited Mount Wilson.'),
        ('the III Corps was to hold its ground', 'the Third Corps was to hold its ground'),
        ('Pope Pius XII spoke vs. the crowd.', 'Pope Pius the Twelfth spoke versus the crowd.'),
        ('It was founded about 1715, while the front was older.', 'It was founded about seventeen fifteen, while the front was older.'),
        ('Lake Tahoe is fed by 63 tributaries.', 'Lake Tahoe is fed by sixty-three tributaries.'),
        ('The 1990s were great.', 'The nineteen nineties were great.'),
        ('She came 21st with 10% of 600,000 votes.', 'She came twenty-first with ten percent of six hundred thousand votes.'),
        ('a $70 million impact', 'a seventy million dollar impact'),
        ('At 9:05 it was 25°C.', 'At nine oh five it was twenty-five degrees Celsius.'),
        ("Credits are adapted from the album's 1976 liner notes.", "Credits are adapted from the album's nineteen seventy-six liner notes."),
        ('It was dated 1796 by experts.', 'It was dated seventeen ninety-six by experts.'),
        ('Warner Bros. and Ltd. companies.', 'Warner Brothers and Limited companies.'),
        ('He arrived on the 13 October.', 'He arrived on the thirteenth of October.'),
        ('It happened in 1999 and 2000.', 'It happened in nineteen ninety-nine and two thousand.'),
        ('Super Bowl XLIV was watched.', 'Super Bowl Forty-four was watched.'),
        ('The Ph.D. program started in 2010.', 'The P H D program started in twenty ten.'),
        ('Type IV civilizations exist.', 'Type Four civilizations exist.'),
        ("Baldwin IV's deathbed.", "Baldwin the Fourth's deathbed."),
        ('He had 1500 dollars in the 1998 elections.', 'He had one thousand five hundred dollars in the nineteen ninety-eight elections.'),
        ('Between 1999 and 2000 it grew.', 'Between nineteen ninety-nine and two thousand it grew.'),
    ],
    'de': [
        ('Am 10. August 1990 wurde er im 19. Jahrhundert geboren.', 'Am zehnten August neunzehnhundertneunzig wurde er im neunzehnten Jahrhundert geboren.'),
        ('Der 3. Oktober ist der Tag der Deutschen Einheit.', 'Der dritte Oktober ist der Tag der Deutschen Einheit.'),
        ('Er spielte in der 3. Liga und die 2. Liga war besser.', 'Er spielte in der dritten Liga und die zweite Liga war besser.'),
        ('Das 1980 für Touristen erbaute Haus wurde im frühen 20. Jh. umgebaut.', 'Das neunzehnhundertachtzig für Touristen erbaute Haus wurde im frühen zwanzigsten Jahrhundert umgebaut.'),
        ('In den 1980er Jahren und den 1990er-Jahren gab es 20er Jahre.', 'In den neunzehnhundertachtziger Jahren und den neunzehnhundertneunziger Jahren gab es zwanziger Jahre.'),
        ('Am 3. und 4. Mai 2001 spielte der 1. FC Köln.', 'Am dritten und vierten Mai zweitausendeins spielte der erste FC Köln.'),
        ('Heinrich VIII. heiratete sechs Frauen. Unter Heinrich VIII. wurde England protestantisch.', 'Heinrich der Achte heiratete sechs Frauen. Unter Heinrich dem Achten wurde England protestantisch.'),
        ('Er war z. B. Mitglied der Partei bzw. Kandidat, u. a. in Berlin.', 'Er war zum Beispiel Mitglied der Partei beziehungsweise Kandidat, unter anderem in Berlin.'),
        ('Die Stadt hat rund 1100 Einwohner und 10 000 Häuser, d. h. viele.', 'Die Stadt hat rund eintausendeinhundert Einwohner und zehntausend Häuser, das heißt viele.'),
        ('Der 39-Jährige hatte eine 100-prozentige Tochter im 30-Minuten-Takt.', 'Der neununddreißig-Jährige hatte eine einhundert-prozentige Tochter im dreißig-Minuten-Takt.'),
        ('Sie kam um 1 Uhr und ging um 4.40 Uhr.', 'Sie kam um ein Uhr und ging um vier Uhr vierzig.'),
        ('Nach § 114 Abs. 4 FamFG gilt Nr. 1 nicht.', 'Nach Paragraf einhundertvierzehn Absatz vier FamFG gilt Nummer eins nicht.'),
        ('Er verdiente 5 Mio. Euro und 2,5 Mrd. €.', 'Er verdiente fünf Millionen Euro und zwei Komma fünf Milliarden Euro.'),
        ('Der Riksvei 22. Der Riksvei kreuzt.', 'Der Riksvei zweiundzwanzig. Der Riksvei kreuzt.'),
        ('Am Dienstag, 20. Juni 1944, wurde er verhaftet.', 'Am Dienstag, zwanzigsten Juni neunzehnhundertvierundvierzig, wurde er verhaftet.'),
        ('Seit 25. August 2024 ist er Minister.', 'Seit fünfundzwanzigstem August zweitausendvierundzwanzig ist er Minister.'),
        ('Er war Ritter des Ordens 2. Klasse.', 'Er war Ritter des Ordens zweiter Klasse.'),
        ('Er wurde 1955 geehrt. 1890 folgten weitere.', 'Er wurde neunzehnhundertfünfundfünfzig geehrt. Achtzehnhundertneunzig folgten weitere.'),
        ('Er war von 1992 bis 1998 Parlamentarischer Staatssekretär.', 'Er war von neunzehnhundertzweiundneunzig bis neunzehnhundertachtundneunzig Parlamentarischer Staatssekretär.'),
        ('Das 1412 verfasste Werk ist alt.', 'Das vierzehnhundertzwölf verfasste Werk ist alt.'),
        ('Den Worten Wilhelms II. zufolge war er krank.', 'Den Worten Wilhelms des Zweiten zufolge war er krank.'),
        ('Der wirkliche Ludwig XVI. war belesen.', 'Der wirkliche Ludwig der Sechzehnte war belesen.'),
        ('Das Kloster entstand im 8. Jh. und wurde 1211 erwähnt.', 'Das Kloster entstand im achten Jahrhundert und wurde zwölfhundertelf erwähnt.'),
        ('Er spielt auf Platz 1 und kam zum 5. Mal.', 'Er spielt auf Platz eins und kam zum fünften Mal.'),
        ('Im Dezember 2014 hatte er Aussichten.', 'Im Dezember zweitausendvierzehn hatte er Aussichten.'),
        ('Das geschah 2001 in Berlin, vgl. Kapitel 3.', 'Das geschah zweitausendeins in Berlin, vergleiche Kapitel drei.'),
        ('Der Film Teil II erschien.', 'Der Film Teil zwei erschien.'),
        ('Peter V. Smith kam 1998.', 'Peter V. Smith kam neunzehnhundertachtundneunzig.'),
        ('Im Jahr 2010 gab es 21 Spieler.', 'Im Jahr zweitausendzehn gab es einundzwanzig Spieler.'),
        ('Er wurde am 1. Januar 2020 Präsident.', 'Er wurde am ersten Januar zweitausendzwanzig Präsident.'),
        ('Die 1990er waren wild.', 'Die neunzehnhundertneunziger waren wild.'),
        ('Ab dem 9. Jh. waren Händler hier.', 'Ab dem neunten Jahrhundert waren Händler hier.'),
        ('Er belegte den 87. Platz.', 'Er belegte den siebenundachtzigsten Platz.'),
        ('Die 18. Dynastie herrschte lange.', 'Die achtzehnte Dynastie herrschte lange.'),
        ('Bis zum 20. September 2009 war er Chef.', 'Bis zum zwanzigsten September zweitausendneun war er Chef.'),
        ('Er hatte 200 Millionen Euro.', 'Er hatte zweihundert Millionen Euro.'),
    ],
    'fr': [
        ('Au XVIIe siècle, Louis XIV et François Ier régnaient. Charles V aussi.', 'Au dix-septième siècle, Louis quatorze et François premier régnaient. Charles cinq aussi.'),
        ('Il a obtenu 21 voix contre 20 et 81 sélections.', 'Il a obtenu vingt et une voix contre vingt et quatre-vingt-une sélections.'),
        ('En mai 2021, le premier ministre est venu à 21 h 30.', 'En mai deux mille vingt et un, le premier ministre est venu à vingt et une heures trente.'),
        ('M. Dupont et Mme Martin habitent au no 3, etc.', 'Monsieur Dupont et Madame Martin habitent au numéro trois, et cetera.'),
        ('Le Ve siècle av. J.-C. fut marqué par la IIIe République.', 'Le cinquième siècle avant Jésus-Christ fut marqué par la troisième République.'),
        ('Poésies de Guillaume de Machaut (1831), en 901, Formule 1.', 'Poésies de Guillaume de Machaut (mille huit cent trente et un), en neuf cent un, Formule un.'),
        ("Le tome II et l'acte III.", "Le tome deux et l'acte trois."),
        ('Il a 21 ans et une fille de 1 an.', "Il a vingt et un ans et une fille d'un an."),
        ('Elle a 21 voitures et 101 maisons.', 'Elle a vingt et une voitures et cent une maisons.'),
        ('Le 2e album est sorti le 1er octobre 2023.', 'Le deuxième album est sorti le premier octobre deux mille vingt-trois.'),
        ('Il y a 41 communes et 31 villages.', 'Il y a quarante et une communes et trente et un villages.'),
        ('Le film dure 1 h.', 'Le film dure une heure.'),
        ('La population est de 3,14 km² et 10 %.', 'La population est de trois virgule quatorze kilomètres carrés et dix pour cent.'),
        ('Il travaille 31 heures par semaine.', 'Il travaille trente et une heures par semaine.'),
        ('En 1900, Redon expose 600 tableaux.', 'En mille neuf cent, Redon expose six cents tableaux.'),
        ('Le pape Jean-Paul II et Benoît XVI.', 'Le pape Jean-Paul deux et Benoît seize.'),
        ('Il est né vers 1650 et mort en 1715.', 'Il est né vers mille six cent cinquante et mort en mille sept cent quinze.'),
        ('Le XXIe siècle commence en 2001.', 'Le vingt et unième siècle commence en deux mille un.'),
        ('Il y avait 1 million de francs.', 'Il y avait un million de francs.'),
        ('Le concile Vatican II eut lieu.', 'Le concile Vatican deux eut lieu.'),
        ('Elle a remporté 61 médailles.', 'Elle a remporté soixante et une médailles.'),
        ("C'était le 81e Rapport annuel.", "C'était le quatre-vingt-unième Rapport annuel."),
        ('Il a marqué 21 buts en 41 matchs.', 'Il a marqué vingt et un buts en quarante et un matchs.'),
        ('Le chapitre IV commence à la page 21.', 'Le chapitre quatre commence à la page vingt et un.'),
        ('Il est arrivé à 6 h et reparti à 1 h 15.', 'Il est arrivé à six heures et reparti à une heure quinze.'),
        ('Mme Curie et Mgr Dupuis, p. ex. Pierre.', 'Madame Curie et Monseigneur Dupuis, par exemple Pierre.'),
        ('Il y a 400 000 habitants.', 'Il y a quatre cent mille habitants.'),
        ('Le 14 juillet 1789 fut célèbre.', 'Le quatorze juillet mille sept cent quatre-vingt-neuf fut célèbre.'),
        ('Mlle Dupont a 31 ans.', 'Mademoiselle Dupont a trente et un ans.'),
        ('Il fut élu en 1991 avec 51 % des voix.', 'Il fut élu en mille neuf cent quatre-vingt-onze avec cinquante et un pour cent des voix.'),
        ('Les années 1980 furent difficiles.', 'Les années mille neuf cent quatre-vingt furent difficiles.'),
        ('Il est né le 3 mai 1990.', 'Il est né le trois mai mille neuf cent quatre-vingt-dix.'),
        ('Elle a 1 fille.', 'Elle a une fille.'),
        ('Il y a 21 % de femmes.', 'Il y a vingt et un pour cent de femmes.'),
        ('La IIe République.', 'La deuxième République.'),
        ('Henri IV fut roi en 1589.', 'Henri quatre fut roi en mille cinq cent quatre-vingt-neuf.'),
        ('Dans le 5ᵉ arrondissement de Paris au XIXᵉ siècle.', 'Dans le cinquième arrondissement de Paris au dix-neuvième siècle.'),
        ('Il a 21 ans et 3 enfants.', 'Il a vingt et un ans et trois enfants.'),
    ],
    'ru': [
        ('В 1950-х годах стандарты были пересмотрены.', 'В тысяча девятьсот пятидесятых годах стандарты были пересмотрены.'),
        ('было продано более 600 000 копий.', 'было продано более шестисот тысяч копий.'),
        ('В городе проживает 1500 человек, а погибли 25 человек.', 'В городе проживает тысяча пятьсот человек, а погибли двадцать пять человек.'),
        ('Он жил в 5 домах и 2 книги прочитал за 21 день.', 'Он жил в пяти домах и две книги прочитал за двадцать один день.'),
        ('В XIX веке и в начале XX в. жили люди.', 'В девятнадцатом веке и в начале двадцатого века жили люди.'),
        ('13 сентября 1990 года он родился, а к 15 октября уехал.', 'Тринадцатого сентября тысяча девятьсот девяностого года он родился, а к пятнадцатому октября уехал.'),
        ('В 1990 г. население выросло на 25 % до 5 тыс. человек.', 'В тысяча девятьсот девяностом году население выросло на двадцать пять процентов до пяти тысяч человек.'),
        ('Он занял 5-е место на 23-м этапе в 1990-х.', 'Он занял пятое место на двадцать третьем этапе в тысяча девятьсот девяностых.'),
        ('Это 20-летние люди, т. е. молодые, и т. д.', 'Это двадцатилетние люди, то есть молодые, и так далее.'),
        ('Орден Ленина III степени получили 2 студента.', 'Орден Ленина третьей степени получили два студента.'),
        ('В V веке до н. э. жили люди.', 'В пятом веке до нашей эры жили люди.'),
        ('Площадь составляет 25 км.', 'Площадь составляет двадцать пять километров.'),
        ('с 13 по 15 сентября 2001 года', 'с тринадцатого по пятнадцатое сентября две тысячи первого года'),
        ('в XIX и XX веках', 'в девятнадцатом и двадцатом веках'),
        ('Пётр I и Екатерина II правили. Сын Петра I умер.', 'Пётр Первый и Екатерина Вторая правили. Сын Петра Первого умер.'),
        ('В 59 году до нашей эры.', 'В пятьдесят девятом году до нашей эры.'),
        ('Школа № 1 открылась.', 'Школа номер один открылась.'),
        ('С 1836 по 1839 год он учился.', 'С тысяча восемьсот тридцать шестого по тысяча восемьсот тридцать девятый год он учился.'),
        ('В возрасте 26 лет он уехал.', 'В возрасте двадцати шести лет он уехал.'),
        ('Он прожил 86 лет.', 'Он прожил восемьдесят шесть лет.'),
        ('Матч посетили больше 2000 болельщиков.', 'Матч посетили больше двух тысяч болельщиков.'),
        ('Он был избран на 5 лет.', 'Он был избран на пять лет.'),
        ('Это случилось в 1881 году, а в 2025 году всё изменилось.', 'Это случилось в тысяча восемьсот восемьдесят первом году, а в две тысячи двадцать пятом году всё изменилось.'),
        ('Батарея из 16 орудий стреляла.', 'Батарея из шестнадцати орудий стреляла.'),
        ('В 2 раза больше, через 2 недели.', 'В два раза больше, через две недели.'),
        ('Сеть охватывает 31 центр.', 'Сеть охватывает тридцать один центр.'),
        ('Он работал с 21 человеком.', 'Он работал с двадцатью одним человеком.'),
        ('Музей им. Пушкина.', 'Музей имени Пушкина.'),
        ('Около 113 года до н. э. бойи отбили.', 'Около сто тринадцатого года до нашей эры бойи отбили.'),
        ('Доля составляет 18 %.', 'Доля составляет восемнадцать процентов.'),
        ('В итоге около 70 млн тонн угля.', 'В итоге около семидесяти миллионов тонн угля.'),
        ('Задержаны 6 афганцев.', 'Задержаны шесть афганцев.'),
        ('Сегодня исполняется 54 года.', 'Сегодня исполняется пятьдесят четыре года.'),
        ('Было 25 тысяч человек.', 'Было двадцать пять тысяч человек.'),
        ('Город получил от Фридриха II статус вольного.', 'Город получил от Фридриха Второго статус вольного.'),
        ('Ему исполнилось 22 года.', 'Ему исполнилось двадцать два года.'),
        ('С 622 по 750 год правили арабы.', 'С шестьсот двадцать второго по семьсот пятидесятый год правили арабы.'),
    ],
    'pt': [
        ('Havia 22 pessoas e 2 mil pessoas no século XX, com D. Pedro I e Isabel II.', 'Havia vinte e duas pessoas e duas mil pessoas no século vinte, com Dom Pedro primeiro e Isabel segunda.'),
        ('O sr. Silva tem 200.000 habitantes e 21 casas desde 1991, na II Guerra Mundial.', 'O senhor Silva tem duzentos mil habitantes e vinte e uma casas desde mil novecentos e noventa e um, na segunda Guerra Mundial.'),
        ('Em 917, o exilarca chegou ao séc. XV.', 'Em novecentos e dezessete, o exilarca chegou ao século quinze.'),
        ('Foi o 3º lugar e a 36ª edição.', 'Foi o terceiro lugar e a trigésima sexta edição.'),
        ('Ao longo de 1 430 páginas conhecidas.', 'Ao longo de mil quatrocentas e trinta páginas conhecidas.'),
        ('Deixou mais de 300.000 pessoas sem casa.', 'Deixou mais de trezentas mil pessoas sem casa.'),
        ('Com 31 acusações nomeando 31 documentos.', 'Com trinta e uma acusações nomeando trinta e um documentos.'),
        ('Gastou 580.690 cópias.', 'Gastou quinhentas e oitenta mil seiscentas e noventa cópias.'),
        ('Morreu em 21 de maio de 1974.', 'Morreu em vinte e um de maio de mil novecentos e setenta e quatro.'),
        ('Do rei Carlos II e do Papa Gregório XV.', 'Do rei Carlos segundo e do Papa Gregório quinze.'),
        ('A diabetes tipo 2 e a Fórmula 1.', 'A diabetes tipo dois e a Fórmula um.'),
        ('Foi número 1 em simples.', 'Foi número um em simples.'),
        ('O Dr. Souza e a Dra. Lima, etc.', 'O doutor Souza e a doutora Lima, etcétera.'),
        ('Tinha 2 filhos e 2 filhas.', 'Tinha dois filhos e duas filhas.'),
        ('Havia 22 escolas e 32 universidades.', 'Havia vinte e duas escolas e trinta e duas universidades.'),
        ('Comprou 1 casa e 1 carro.', 'Comprou uma casa e um carro.'),
        ('Durante o século VI a.C. estes povoados existiam.', 'Durante o século sexto antes de Cristo estes povoados existiam.'),
        ('A Igreja não estabelece a obrigação de contribuir com 10% dos rendimentos.', 'A Igreja não estabelece a obrigação de contribuir com dez por cento dos rendimentos.'),
        ('Desagua no Tejo cerca de 12 km a jusante.', 'Desagua no Tejo cerca de doze quilômetros a jusante.'),
        ('Em 2015, cresceu 30% e 475.000 toneladas.', 'Em dois mil e quinze, cresceu trinta por cento e quatrocentas e setenta e cinco mil toneladas.'),
        ('Foi a XXV edição do festival.', 'Foi a vigésima quinta edição do festival.'),
        ('O XII duque de Bragança.', 'O décimo segundo duque de Bragança.'),
        ('Ele tem 21 anos e 2 meses.', 'Ele tem vinte e um anos e dois meses.'),
        ('Viveu entre 1942 e 1949.', 'Viveu entre mil novecentos e quarenta e dois e mil novecentos e quarenta e nove.'),
        ('A Afonso XIII de Espanha.', 'A Afonso treze de Espanha.'),
        ('Tem 222 222 pessoas.', 'Tem duzentas e vinte e duas mil duzentas e vinte e duas pessoas.'),
        ('Tem 2 filhas e 200 casas.', 'Tem duas filhas e duzentas casas.'),
        ('O século XXI começou.', 'O século vinte e um começou.'),
        ('Em 1500 chegaram 13 navios.', 'Em mil e quinhentos chegaram treze navios.'),
        ('Ganhou 1 milhão de reais.', 'Ganhou um milhão de reais.'),
        ('O Papa Bento XVI renunciou em 2013.', 'O Papa Bento dezesseis renunciou em dois mil e treze.'),
        ('D. Maria II reinou.', 'Dona Maria segunda reinou.'),
        ('No dia 4 de outubro de 2020.', 'No dia quatro de outubro de dois mil e vinte.'),
        ('Foram 2 mil pessoas.', 'Foram duas mil pessoas.'),
        ('Tinha 1 irmã e 2 irmãos.', 'Tinha uma irmã e dois irmãos.'),
        ('Pesava 2 quilos e media 1 metro.', 'Pesava dois quilos e media um metro.'),
        ('Havia 200 mil pessoas e 2 milhões de casas.', 'Havia duzentas mil pessoas e dois milhões de casas.'),
        ('Entre 1942 e 1949 viveu.', 'Entre mil novecentos e quarenta e dois e mil novecentos e quarenta e nove viveu.'),
    ],
    'es': [
        ('Había 200.000 personas y 21 casas en 1991.', 'Había doscientas mil personas y veintiuna casas en mil novecientos noventa y uno.'),
        ('En el siglo XX, Felipe II e Isabel II reinaron durante la II Guerra Mundial y el III Reich.', 'En el siglo veinte, Felipe segundo e Isabel segunda reinaron durante la segunda Guerra Mundial y el tercer Reich.'),
        ('Tiene 21 años y vive en el piso 1 del edificio.', 'Tiene veintiún años y vive en el piso uno del edificio.'),
        ('El Dr. García, etc., vivió en EE. UU. desde el año 604 d. C.', 'El doctor García, etcétera, vivió en Estados Unidos desde el año seiscientos cuatro después de Cristo.'),
        ('Disputó 112 carreras en Fórmula 1 entre las temporadas 2001 y 2002.', 'Disputó ciento doce carreras en Fórmula uno entre las temporadas dos mil uno y dos mil dos.'),
        ('En 884 en Kaumberg prestó juramento.', 'En ochocientos ochenta y cuatro en Kaumberg prestó juramento.'),
        ('La población es de 4 561 habitantes, con 29,61 km².', 'La población es de cuatro mil quinientos sesenta y un habitantes, con veintinueve coma seis uno kilómetros cuadrados.'),
        ('Los años 1950 se caracterizan por la canción n.º 5.', 'Los años mil novecientos cincuenta se caracterizan por la canción número cinco.'),
        ('En mayo 2001 llegaron 31 soldados y 124 heridos.', 'En mayo dos mil uno llegaron treinta y un soldados y ciento veinticuatro heridos.'),
        ('Se ubica a 900 m s. n. m. desde el s. XV.', 'Se ubica a novecientos metros sobre el nivel del mar desde el siglo quince.'),
        ('Sebastián de Covarrubias (1673), incluye posada.', 'Sebastián de Covarrubias (mil seiscientos setenta y tres), incluye posada.'),
        ('Fue la 3.ª edición y el 1.º piso.', 'Fue la tercera edición y el primer piso.'),
        ('Hubo 501 víctimas y 201 muertos.', 'Hubo quinientas una víctimas y doscientos un muertos.'),
        ('Ganó 2 mil personas y 200 mil votos.', 'Ganó dos mil personas y doscientos mil votos.'),
        ('Los siglos XIX y XX fueron importantes.', 'Los siglos diecinueve y veinte fueron importantes.'),
        ('El capítulo III del tomo II.', 'El capítulo tres del tomo dos.'),
        ('El 11.º congreso.', 'El undécimo congreso.'),
        ('El evento va por su XIII edición.', 'El evento va por su decimotercera edición.'),
        ('Fue el XII duque del Infantado.', 'Fue el duodécimo duque del Infantado.'),
        ('A finales del siglo XVIII y principios del XIX se generaliza.', 'A finales del siglo dieciocho y principios del diecinueve se generaliza.'),
        ('El Concilio Vaticano II cambió todo.', 'El Concilio Vaticano segundo cambió todo.'),
        ('Vendió 350.000 copias y 245 veces más.', 'Vendió trescientas cincuenta mil copias y doscientas cuarenta y cinco veces más.'),
        ('Había 21 mujeres y 31 hombres.', 'Había veintiuna mujeres y treinta y un hombres.'),
        ('Tenía 1 hija y 1 hijo.', 'Tenía una hija y un hijo.'),
        ('El 11 de mayo de 2025 llegó.', 'El once de mayo de dos mil veinticinco llegó.'),
        ('Desde el año 2010 al 2017.', 'Desde el año dos mil diez al dos mil diecisiete.'),
        ('En 1940 había 21 km.', 'En mil novecientos cuarenta había veintiún kilómetros.'),
        ('Unos 200 hombres llegaron.', 'Unos doscientos hombres llegaron.'),
        ('El 25 % de los 600 000 votos.', 'El veinticinco por ciento de los seiscientos mil votos.'),
        ('Hubo 21 000 personas.', 'Hubo veintiuna mil personas.'),
        ('En 1990 había 101 casas.', 'En mil novecientos noventa había ciento una casas.'),
        ('El rey Alfonso XIII murió.', 'El rey Alfonso trece murió.'),
        ('Isabel I nació en 1451.', 'Isabel primera nació en mil cuatrocientos cincuenta y uno.'),
        ('Ganó 1 millón de dólares.', 'Ganó un millón de dólares.'),
        ('Hubo 200 mil votos y 2 millones de personas.', 'Hubo doscientos mil votos y dos millones de personas.'),
        ('Entre 2001 y 2002 creció.', 'Entre dos mil uno y dos mil dos creció.'),
    ],
    'it': [
        ("Negli anni '80 Pio XII ed Elisabetta II vissero nel XX secolo.", 'Negli anni ottanta Pio dodicesimo ed Elisabetta seconda vissero nel ventesimo secolo.'),
        ('Il comune ha 4 561 abitanti e 1 scuola, a 300 m s.l.m.', 'Il comune ha quattromilacinquecentosessantuno abitanti e una scuola, a trecento metri sul livello del mare.'),
        ('Nel marzo 2021 la Rai 1 trasmise la II guerra mondiale ecc.', 'Nel marzo duemilaventuno la Rai uno trasmise la seconda guerra mondiale eccetera.'),
        ('Mercoledì 21 la partita a Roseto, atto III e parte II.', 'Mercoledì ventuno la partita a Roseto, atto terzo e parte seconda.'),
        ("Nacque l'8 dicembre 1851 e morì l'11 febbraio.", "Nacque l' otto dicembre milleottocentocinquantuno e morì l' undici febbraio."),
        ("Si manifesta nell'80% dei pazienti.", "Si manifesta nell' ottanta per cento dei pazienti."),
        ("Fu l'80ª edizione e arrivò all'8° posto.", "Fu l' ottantesima edizione e arrivò all' ottavo posto."),
        ('Nel 1908 è presente nel 1318.', 'Nel millenovecentotto è presente nel milletrecentodiciotto.'),
        ('Il comune ha 749 abitanti.', 'Il comune ha settecentoquarantanove abitanti.'),
        ('Ha 21 anni e 1 figlia.', 'Ha ventuno anni e una figlia.'),
        ('Il 25 marzo 2023 arrivò.', 'Il venticinque marzo duemilaventitré arrivò.'),
        ('La 36ª Brigata.', 'La trentaseiesima Brigata.'),
        ('La n. 10 del Dott. Rossi.', 'La numero dieci del dottor Rossi.'),
        ('Fu pubblicato nel sec. XV.', 'Fu pubblicato nel secolo quindicesimo.'),
        ('Erano 31 uomini.', 'Erano trentuno uomini.'),
        ('Il Sig. Bianchi e il Prof. Verdi.', 'Il signor Bianchi e il professor Verdi.'),
        ('Vittorio Emanuele II era re.', 'Vittorio Emanuele secondo era re.'),
        ('Dal 2008 al 2017 crebbe.', 'Dal duemilaotto al duemiladiciassette crebbe.'),
        ('Distrusse 321 carri armati.', 'Distrusse trecentoventuno carri armati.'),
        ('Ci sono 2 880 abitanti.', 'Ci sono duemilaottocentottanta abitanti.'),
        ('Il 1° maggio 2020.', 'Il primo maggio duemilaventi.'),
        ("La legge del '45 e la band degli anni '90.", 'La legge del quarantacinque e la band degli anni novanta.'),
        ('Giovanni XXIII fu papa.', 'Giovanni ventitreesimo fu papa.'),
        ('Il capitolo IV e il libro II.', 'Il capitolo quarto e il libro secondo.'),
        ('Il prezzo è di 21 euro.', 'Il prezzo è di ventuno euro.'),
        ('Pietro II e Carlo I regnarono.', 'Pietro secondo e Carlo primo regnarono.'),
        ('Ha 2 figlie.', 'Ha due figlie.'),
        ("Nel '68 ci fu la rivolta.", 'Nel sessantotto ci fu la rivolta.'),
        ('Morì nel 1976 a 81 anni.', 'Morì nel millenovecentosettantasei a ottantuno anni.'),
        ('Sono 1 milione di persone.', 'Sono un milione di persone.'),
        ('Fu eletto con 51 voti.', 'Fu eletto con cinquantuno voti.'),
        ('Il XIX secolo fu lungo.', 'Il diciannovesimo secolo fu lungo.'),
        ('Ha 1 figlio e 3 nipoti.', 'Ha un figlio e tre nipoti.'),
        ('Nel 2008 e 2017 crebbe.', 'Nel duemilaotto e duemiladiciassette crebbe.'),
    ],
}
DROPPED = {
    'en': [
        ('The population was 1500.', '四位数是年份还是数量需要上下文'),
        ('Rocky III was an enormous hit.', '罗马数字的读法取决于上下文'),
        ('from 1950–1953 there', '范围或编号'),
        ('never much more than 2000, will', '四位数是年份还是数量需要上下文'),
        ('Apple Inc. was founded.', '含缩写'),
        ('in 1500 cases', '四位数是年份还是数量需要上下文'),
        ('between 3 and 5 May', '并列或范围的日期'),
        ('Mr. smith arrived.', '含缩写'),
        ('In 2020 one person died.', '数字后紧跟数词，读出来会连成另一个数'),
        ('It cost 20 two-seat cars.', '数字后紧跟数词，读出来会连成另一个数'),
    ],
    'de': [
        ('Er hatte etwa 1740 die Straße gebaut.', '四位数是年份还是数量需要上下文'),
        ('Mit 1.830.000 Besuchern war es voll.', ''),
        ('Es kamen 1500 Menschen.', '四位数是年份还是数量需要上下文'),
        ('Er kam am 3.', '数字后的句点可能表示序数'),
        ('Die 3. größte Stadt', '数字后的句点可能表示序数'),
        ('bis auf 34 400 km', '空格分隔的数字后接单位'),
        ('Sie heiratete Ludwig XIV.', '君主序号的格取决于上下文'),
        ('Er wurde 1 Mal Erster.', ''),
        ('Die Formel-1 fährt.', '型号或编号'),
        ('Er war bis 1998 Leiter der Abteilung.', '四位数是年份还是数量需要上下文'),
        ('1880 ein Haus wurde gebaut.', '数字后紧跟数词，读出来会连成另一个数'),
    ],
    'fr': [
        ('Il en a 21.', '数词的阴阳性取决于名词'),
        ('Le film Rocky II', '罗马数字的读法取决于上下文'),
        ('3-5 jours', '范围或编号'),
        ('Jean M. Dupont a 1 enfant.', '含缩写'),
        ('Le XIXe-XXe siècles', '罗马数字的范围'),
        ('Il compte 21 000 femmes.', '阴性数词的读法不固定'),
        ('Elle a 41 dioptries.', '数词的阴阳性取决于名词'),
        ('La version courante est la 3.8. Il existe aussi iOS 4.0.', '数字中的点读法不固定（版本号或外来写法）'),
        ('En 2020 une loi fut votée.', '数字后紧跟数词，读出来会连成另一个数'),
    ],
    'ru': [
        ('Впереди ограничение 40, а следом', '俄语数词的格取决于上下文'),
        ('до 40 этажа', '后面的名词是单数属格，可能是序数'),
        ('увидел 21 студента', '俄语数词的格取决于上下文'),
        ('После 124-летней паузы', '复合词的读法不固定'),
        ('533 Мбайт/с', '俄语数词的格取决于上下文'),
        ('Царь Аретой IV правил.', '罗马数字的读法取决于上下文'),
        ('В 1920 годах.', '复数 год 前的年份写法不规范'),
        ('Поможет им.', '含缩写'),
        ('1,5 км', '小数的格变化暂不处理'),
        ('Версия 3.8 вышла.', '数字中的点读法不固定（版本号或外来写法）'),
        ('В 1990 один из них умер.', '数字后紧跟数词，读出来会连成另一个数'),
        ('Сегодня исполняется 191 год Льюису.', '数后接 год 可能是年数而不是年份'),
        ('Телефон вышел 19 декабря 2007 г .', '俄语数词的格取决于上下文'),
    ],
    'pt': [
        ('Havia 22.', '数词的阴阳性取决于名词'),
        ('Segundo a sra. de idade', '含缩写'),
        ('por M1 e M5', '数字与字母相连'),
        ('Eram 2 a 2.', '数词的阴阳性取决于名词'),
        ('Ele tem 22 caixinhas.', '数词的阴阳性取决于名词'),
        ('Foi substituído por 120 homens.', 'por 后的数读起来像 por ciento、por cento'),
    ],
    'es': [
        ('Tenía 21.', '数词的阴阳性取决于名词'),
        ('unos 200 de ellas', '数词的阴阳性取决于名词'),
        ('la 3 Arts Entertainment', '数字后接专名，读法无法确定'),
        ('el 1.º lugar', '名词前 primero、tercero 要缩略为 primer、tercer'),
        ('La película Rocky II', '罗马数字的读法取决于上下文'),
        ('Se vendieron 21 casitas.', '数词的阴阳性取决于名词'),
        ('En 2030 y una más.', '数字后紧跟数词，读出来会连成另一个数'),
        ('Fue sustituido por 120 hombres.', 'por 后的数读起来像 por ciento、por cento'),
    ],
    'it': [
        ("Il '500-'600 fu ricco.", '撇号省略的年份'),
        ('21 case', '阴性数词的读法不固定'),
        ('Lo disse il Sig. rossi', '含缩写'),
        ('di 2 880 km', '空格分隔的数字后接单位'),
        ("È il 40' del primo tempo.", '数字后的撇号（分钟、英尺）'),
        ('Il racconto Centaurus II', '罗马数字的读法取决于上下文'),
        ('Ci sono 31 persone.', '阴性数词的读法不固定'),
        ('Vinse 21 partite.', '阴性数词的读法不固定'),
    ],
}


class EuropeanCasesTest(unittest.TestCase):
    def test_at_least_40_cases(self):
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
                    self.assertFalse(any(ch.isdigit() for ch in result.script), msg = result.script)

    def test_dropped(self):
        for lang, cases in DROPPED.items():
            for original, reason in cases:
                with self.subTest(lang = lang, original = original):
                    result = normalize_entry(original = original, lang = lang)
                    self.assertEqual(result.status, "dropped", msg = result.script)
                    self.assertIsNone(result.script)
                    self.assertIn(reason, result.note)

    def test_numbers_match_original(self):
        """不含已知比对局限类别的改写，朗读稿与原文经 pipeline.compare 数词统一后一致。"""
        for lang, cases in NORMALIZED.items():
            for original, _ in cases:
                result = normalize_entry(original = original, lang = lang)
                if any(kind in result.note for kind in COMPARE_GAPS):
                    continue
                with self.subTest(lang = lang, original = original):
                    check = compare(script = result.script, hypothesis = clean(text = original, lang = lang), lang = lang)
                    self.assertTrue(check.match, msg = check.detail)


class ReadingFunctionsTest(unittest.TestCase):
    def test_roman_value(self):
        from pipeline.tn.langs.euro import roman_value

        self.assertEqual(roman_value(text = "XIV"), 14)
        self.assertEqual(roman_value(text = "MCMXC"), 1990)
        self.assertIsNone(roman_value(text = "IIII"))
        self.assertIsNone(roman_value(text = "VX"))

    def test_romance_feminine(self):
        from pipeline.tn.langs import get_lang

        es, pt, fr, it = (get_lang(lang = code) for code in ("es", "pt", "fr", "it"))
        self.assertEqual(es.fem(n = 221000), "doscientas veintiuna mil")
        self.assertEqual(es.fem(n = 2000000), "dos millones")
        self.assertEqual(es.fem(n = 2500000), "dos millones quinientas mil")
        self.assertEqual(pt.fem(n = 22), "vinte e duas")
        self.assertEqual(pt.fem(n = 2200000), "dois milhões e duzentas mil")
        self.assertEqual(fr.fem(n = 81), "quatre-vingt-une")
        self.assertEqual(fr.fem(n = 71), "soixante et onze")
        self.assertIsNone(fr.fem(n = 21000))
        self.assertEqual(it.fem(n = 1), "una")
        self.assertIsNone(it.fem(n = 21))

    def test_ordinal_spelling(self):
        from pipeline.tn.langs import get_lang

        es, pt, it, fr = (get_lang(lang = code) for code in ("es", "pt", "it", "fr"))
        self.assertEqual(es.ordinal_words(n = 11), "undécimo")
        self.assertEqual(es.ordinal_words(n = 19, gender = "f"), "decimonovena")
        self.assertEqual(es.ordinal_words(n = 40), "cuadragésimo")
        self.assertIsNone(es.ordinal_words(n = 400))
        self.assertIsNone(pt.ordinal_words(n = 400))
        self.assertEqual(it.card(n = 1318), "milletrecentodiciotto")
        self.assertEqual(it.card(n = 2023), "duemilaventitré")
        self.assertEqual(fr.ordinal_words(n = 200), "deux centième")

    def test_russian_forms(self):
        from pipeline.tn.langs.ru import card_form, category

        self.assertEqual([category(n = n) for n in (1, 21, 11, 2, 24, 12, 5, 100)],
                         ["one", "one", "many", "few", "few", "many", "many", "many"])
        self.assertEqual(card_form(n = 1500, case = "nomn"), "тысяча пятьсот")
        self.assertEqual(card_form(n = 1500, case = "gent"), "тысячи пятисот")
        self.assertEqual(card_form(n = 2, case = "nomn", gender = "f"), "две")
        self.assertEqual(card_form(n = 21, case = "accs", gender = "f"), "двадцать одну")
        self.assertEqual(card_form(n = 5, case = "loct"), "пяти")


if __name__ == "__main__":
    unittest.main()
