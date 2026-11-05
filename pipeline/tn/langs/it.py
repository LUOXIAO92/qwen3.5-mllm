"""意大利语：基数、年份（含 anni '80 → anni ottanta）、「21 maggio 1974」日期、序数（3º、36ª）、小数（virgola，
逐位读）、百分数、单位、货币、常用缩写、世纪与君主序号（XX secolo → ventesimo secolo，Pio XII → Pio
dodicesimo）。

以 1 结尾的数（uno/una、ventuno）随名词变性：阳性名词前 1 读 un，阴性单数名词前读 una；ventuno、trentuno
这类复合数在阴性名词前的写法不统一（ventuno、ventun、ventuna），丢弃。复合数词末尾的 tre 带重音（ventitré）。
罗马数字一律读作序数，并与名词或人名的性一致（Elisabetta II → seconda）。
"""

from __future__ import annotations

import re

from ..numwords import n2w
from ..rules import Rule, Skip
from .euro import sentence_end_after
from .romance import Romance, roman_ok


def _words(text: dict) -> dict:
    return {w: g for g, words in text.items() for w in words.split()}


NOUNS = _words(text = {
    "m": """
        anni anno giorni giorno mesi mese secoli secolo gradi grado chilometri chilometro metri metro minuti
        minuto secondi secondo punti punto uomini uomo euro dollari dollaro milioni milione miliardi miliardo
        gol voti voto paesi paese chili chilo centimetri centimetro millimetri millimetro soldati soldato membri
        membro giocatori giocatore studenti studente alunni alunno bambini bambino figli figlio libri libro
        dischi disco album episodi episodio capitoli capitolo articoli articolo posti posto comuni comune
        quartieri quartiere distretti distretto stati stato chilogrammi chilogrammo grammi grammo litri litro
        piedi piede cavalli cavallo aerei aereo velivoli velivolo cannoni cannone carri carro veicoli veicolo
        treni treno autobus edifici edificio templi tempio monumenti monumento musei museo ospedali ospedale
        centri centro ponti ponte fiumi fiume laghi lago abitanti abitante feriti ferito prigionieri prigioniero
        profughi lavoratori lavoratore dipendenti dipendente spettatori spettatore tifosi passeggeri passeggero
        visitatori partecipanti esemplari esemplare volumi volume titoli titolo premi premio trofei trofeo tornei
        torneo campionati campionato incontri incontro giochi gioco concerti concerto programmi programma
        problemi problema sistemi sistema temi tema pianeti pianeta poemi poema centesimi centesimo seggi seggio
        deputati deputato senatori senatore consiglieri consigliere elettori elettore caratteri carattere colori
        colore attacchi attacco incidenti incidente incendi incendio terremoti terremoto casi caso pazienti
        medici medico professori professore vescovi vescovo sacerdoti sacerdote monaci monaco re principi
        principe fratelli fratello padri padre amici amico cittadini cittadino ebrei cristiani italiani francesi
        inglesi tedeschi spagnoli territori territorio regni regno piani piano livelli livello motori motore
        cilindri cilindro voli volo passi passo salti salto tiri tiro colpi colpo eserciti esercito reggimenti
        reggimento battaglioni battaglione corpi corpo congressi congresso concili concilio festival saloni
        salone simposi imperi impero governi governo duchi duca conti conte anniversari anniversario quadri
        quadro animali animale alberi albero strumenti strumento documenti documento progetti progetto studi
        studio gruppi gruppo club servizi servizio prodotti prodotto modelli modello tipi tipo mondiali pareggi
        pareggio porti porto aeroporti aeroporto capitani capitano ufficiali ufficiale generali marinai
        marinaio schiavi schiavo coloni colono guerrieri guerriero cavalieri cavaliere film romanzi romanzo
        giornali giornale satelliti satellite ettari ettaro sterlini kuruş dinaro dinari maschi maschio
        battelli battello negozi negozio pezzi pezzo testi testo brani brano singoli singolo sacerdoti uccelli
        uccello fiori fiore semi seme serpenti serpente pesci pesce cani cane gatti gatto lupi lupo animali
        tentativi tentativo successi successo scudetti scudetto record gradini atto atti tomo tomi canto canti
        """,
    "f": """
        persone persona donne donna ore ora settimane settimana volte volta case casa tonnellate tonnellata
        libbre libbra miglia copie copia unità specie vittime vittima famiglie famiglia città località province
        provincia regioni regione comunità nazioni nazione lingue lingua parole parola pagine pagina canzoni
        canzone opere opera novelle novella scene scena stagioni stagione edizioni edizione corse corsa vittorie
        vittoria sconfitte sconfitta medaglie medaglia stelle stella piazze piazza camere camera abitazioni
        abitazione strade strada chiese chiesa scuole scuola università aziende azienda compagnie compagnia
        fabbriche fabbrica teste testa armi arma bombe bomba mine mina navi nave imbarcazioni imbarcazione
        macchine macchina locomotive locomotiva linee linea stazioni stazione fermate fermata rotte rotta vie
        via isole isola montagne montagna spiagge spiaggia colline collina parrocchie parrocchia diocesi
        lettere lettera firme firma monete moneta lire lira foto fotografie fotografia immagini immagine
        colonne colonna torri torre porte porta finestre finestra sale sala piante pianta truppe divisioni
        divisione brigate brigata entrate entrata vendite vendita azioni azione parti parte tappe tappa fasi
        fase giornate giornata calorie caloria vite notti notte generazioni generazione dinastie dinastia
        tribù razze classi classe categorie categoria leghe lega coppe coppa semifinali leggi legge norme
        norma regole regola misure misura proposte proposta elezioni elezione domande domanda risposte
        risposta opzioni opzione versioni versione forme forma cause causa idee idea teorie teoria prove prova
        cellule cellula molecole molecola proteine proteina foglie foglia api ape formiche balene balena
        mucche mucca pecore pecora capre capra galline gallina suore suora sorelle sorella figlie figlia madri
        madre mogli moglie regine regina principesse principessa ragazze ragazza bambine bambina guerre guerra
        repubbliche repubblica crociate crociata legislature legislatura conferenze conferenza assemblee
        assemblea olimpiadi olimpiade mostre mostra tavole tavola sedie sedia lezioni lezione missioni missione
        spedizioni spedizione battaglie battaglia campagne campagna operazioni operazione squadre squadra
        flotte flotta occasioni occasione frazioni frazione cappelle cappella cattedrali cattedrale sedi sede
        filiali filiale marche marca discipline disciplina gare gara partite partita reti rete presenze
        presenza puntate puntata tracce traccia serie sterline sterlina chiavi chiave medaglia cifre cifra
        stanze stanza ville villa aree area zone zona sezioni sezione voci voce nomination candidature
        """,
})
REGNAL_NAMES = set("""
    Luigi Carlo Filippo Ferdinando Alfonso Enrico Giovanni Pietro Giacomo Pio Benedetto Gregorio Leone
    Innocenzo Clemente Urbano Bonifacio Alessandro Niccolò Nicola Martino Paolo Sisto Giulio Adriano Eugenio
    Federico Guglielmo Edoardo Giorgio Riccardo Napoleone Ramesse Tolomeo Amenofi Thutmose Costantino Michele
    Manuele Ivan Vittorio Emanuele Umberto Amedeo Rodolfo Ottone Leopoldo Giuseppe Francesco Gustavo Cristiano
    Selim Maometto Murad Solimano Dario Serse Antioco Seleuco Corrado Baldovino Raimondo Boemondo Andronico
    Basilio Teodosio Giustiniano Alarico Teodorico Lotario Berengario Ludovico Callisto Stefano Ladislao
    Casimiro Sigismondo Venceslao Haakon Olaf Harald Sancho Pedro Juan Felipe Fernando Carlos Ranieri Alberto
    Celestino Felice Onorio Anastasio Sergio Lucio Marcello Agapito Vitaliano Norton Vaticano Mehmed Ercole
    Roberto Bosone Teodoro Ramses Sargon Ottone Guido Tancredi Ruggero Goffredo Raimondo Manfredi Azzone
    Obizzo Borso Cosimo Lorenzo Giangaleazzo Galeazzo Ferrante Federigo Guidobaldo
""".split())
REGNAL_FEMALE = set("""
    Elisabetta Isabella Giovanna Maria Caterina Cleopatra Margherita Anna Cristina Vittoria Eleonora
    Teodora Irene Elena Sofia Beatrice Matilde Bianca
""".split())


class It(Romance):
    code = "it"
    point = "virgola"
    percent = "per cento"
    units = {
        "km/h": ("chilometro orario", "chilometri orari"), "km²": ("chilometro quadrato", "chilometri quadrati"),
        "m²": ("metro quadrato", "metri quadrati"), "km": ("chilometro", "chilometri"), "cm": ("centimetro", "centimetri"),
        "mm": ("millimetro", "millimetri"), "kg": ("chilogrammo", "chilogrammi"), "m": ("metro", "metri"),
        "g": ("grammo", "grammi"), "°C": ("grado Celsius", "gradi Celsius"), "℃": ("grado Celsius", "gradi Celsius"),
        "°F": ("grado Fahrenheit", "gradi Fahrenheit"), "℉": ("grado Fahrenheit", "gradi Fahrenheit"),
    }
    currencies = {"€": ("euro", "euro"), "$": ("dollaro", "dollari")}
    scales = ("mila", "milioni", "milione", "miliardi", "miliardo")
    scale_links = {"milioni": "di", "milione": "di", "miliardi": "di", "miliardo": "di"}
    months = (
        "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre",
        "novembre", "dicembre",
    )
    year_before = {
        "nel", "dal", "al", "del", "fino", "anno", "anni", "entro", "tra", "fra", "e", "il", "dell", "sul", "verso",
        "prima", "dopo", "nell", "circa", "estate", "inverno", "autunno", "primavera",
    }
    year_before_short = {"nel", "anno", "dal", "al", "verso", "circa"}
    nouns = NOUNS
    fem_suffixes = ("zioni", "sioni", "zione", "sione", "tà", "trici", "trice")
    masc_suffixes = ("menti", "mento", "ismi", "ismo", "tori", "tore")
    id_nouns = {
        "tipo", "numero", "capitolo", "articolo", "pagina", "stagione", "episodio", "formula", "gruppo",
        "categoria", "livello", "divisione", "lega", "strada", "linea", "porto", "sala", "piano", "volume", "tomo",
        "parte", "fase", "tappa", "edizione", "versione", "serie", "classe", "canale", "cromosoma", "distretto",
        "zona", "settore", "porta", "binario", "via", "tabella", "figura", "legge", "decreto", "sezione",
        "paragrafo", "atto", "scena", "canto", "salmo", "autostrada", "modello", "missione", "rai", "vault",
        "urna", "lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica", "grado", "blocco",
        "padiglione", "girone", "giornata", "lotto", "regione",
    }
    first_day = "primo"
    starters = {
        "Il", "La", "Lo", "I", "Gli", "Le", "Nel", "Nella", "Un", "Una", "Questo", "Questa", "Suo", "Sua", "Si",
        "Fu", "È", "Con", "Per", "Da", "Dal", "Dopo", "Durante", "Senza", "Inoltre", "Anche", "Oggi", "Già",
    }
    abbreviations = {
        "ecc.": ("eccetera", "any"), "etc.": ("eccetera", "any"), "Mons.": ("monsignor", "name"),
        "Dott.": ("dottor", "name"), "Sig.": ("signor", "name"), "Sig.ra": ("signora", "name"),
        "Avv.": ("avvocato", "name"), "Ing.": ("ingegner", "name"), "Prof.": ("professor", "name"),
        "Dr.": ("dottor", "name"), "St.": ("Saint", "name"), "Mr.": ("mister", "name"), "Jr.": ("junior", "any"),
        "a.C.": ("avanti Cristo", "any"), "d.C.": ("dopo Cristo", "any"), "a. C.": ("avanti Cristo", "any"),
        "d. C.": ("dopo Cristo", "any"),
    }
    number_abbreviations = {
        "n.": "numero", "N.": "numero", "pag.": "pagina", "vol.": "volume", "art.": "articolo",
        "cap.": "capitolo", "No.": "numero", "Nr.": "numero", "c.": "circa",
    }
    regnal_names = REGNAL_NAMES
    regnal_female = REGNAL_FEMALE
    century_words = {"secolo", "secoli"}
    # 意语的君主序号、世纪一律读序数
    roman_ordinal_max = 10 ** 4

    def card(self, n: int) -> str:
        # num2words 对 223、2023 输出 duecentoventitre、duemilaventitre，词尾的 tre 应写作 tré；
        # 对 118、1318 输出 centodicotto，应为 centodiciotto
        text = n2w(n = n, lang = "it").replace("dicotto", "diciotto")
        return re.sub(pattern = r"(?<=[a-z])tre$", repl = "tré", string = text)

    def ordinal_words(self, n: int, gender: str = "m") -> str | None:
        words = n2w(n = n, lang = "it", to = "ordinal")
        if gender == "f":
            words = re.sub(pattern = r"o\b", repl = "a", string = words)
        return words

    def ordinal(self, n: int) -> str:
        return self.ordinal_words(n = n)

    def roman_id_nouns(self) -> set:
        return {"atto", "capitolo", "parte", "libro", "volume", "tomo", "canto", "scena", "fase", "titolo", "sezione"}

    def roman_after_noun(self, value: int, noun: str) -> str | None:
        """意语名词后的罗马数字读序数，与名词同性（atto terzo、parte seconda）。"""
        gender = self.gender_of(word = noun)
        return self.ordinal_words(n = value, gender = gender) if gender else None

    def roman_before_noun_ok(self, noun: str) -> bool:
        """意语名词前的罗马数字（II guerra mondiale、XX secolo）读序数，名词可以小写。"""
        return True

    def digits(self, digits: str) -> str:
        return " ".join(n2w(n = int(d), lang = "it") for d in digits)

    def masc(self, n: int) -> str:
        return "un" if n == 1 else self.card(n = n)

    def fem(self, n: int) -> str | None:
        return "una" if n == 1 else None

    def gender_sensitive(self, n: int) -> bool:
        return n % 10 == 1 and n % 100 != 11

    def function_words(self) -> set:
        return {
            "e", "ed", "o", "in", "di", "del", "della", "con", "per", "che", "a", "al", "alla", "si", "è", "fu",
            "da", "dal", "dalla", "tra", "fra", "su", "come", "dove", "quando", "il", "la", "lo", "i", "gli", "le",
            "insieme", "contro", "nel", "nella",
        }

    def early_rules(self) -> list:
        return [
            Rule(
                pattern = re.compile(pattern = r"(?<![\w'’])(?P<art>[A-Za-z]{1,4}['’])(?P<num>\d+)(?P<suf>[ªº°]|\s?%)?(?![\w]|[.,]\d)"),
                handle = self.elided,
            ),
            Rule(
                pattern = re.compile(pattern = r"(?<![\w'’])['’](?P<d>\d{2})(?![\w'’]|[.,]\d)"),
                handle = self.short_year_apostrophe,
            ),
            Rule(
                pattern = re.compile(pattern = r"(?<![\w.])sec\.\s?(?P<roman>[IVXLC]+)\b"),
                handle = self.century_abbreviation,
            ),
            Rule(
                pattern = re.compile(pattern = rf"(?P<num>{self.num})\s?m\s?s\.\s?l\.\s?m\.(?![\w])"),
                handle = self.altitude,
            ),
        ]

    def elided(self, m, ctx):
        """冠词省音后的数：l'8 dicembre → l' otto dicembre，nell'80% → nell' ottanta per cento，l'80ª → l' ottantesima。

        撇号后留一个空格，读音不变，比对模块据此把冠词和数词分开（连写的 l'otto 会被当成一个词）。
        """
        art = m.group("art") + " "
        if m.group("art")[:-1].lower() not in ("l", "dell", "all", "dall", "nell", "sull", "coll"):
            return None
        n = int(m.group("num"))
        suf = (m.group("suf") or "").strip()
        if m.group("num").startswith("0") or n == 0:
            return Skip(reason = f"数字写法无法确定「{m.group()}」")
        if suf in ("ª", "º", "°"):
            spoken = self.ordinal_words(n = n, gender = "f" if suf == "ª" else "m")
            ctx.kind(name = "序数", surface = m.group())
            return art + ctx.num(spoken = spoken, written = str(n))
        if self.gender_sensitive(n = n):
            return Skip(reason = f"数词的阴阳性取决于名词「{m.group()}」")
        spoken = ctx.num(spoken = self.card(n = n), written = m.group("num"))
        if suf == "%":
            ctx.kind(name = "百分数", surface = m.group())
            return art + spoken + " per cento"
        ctx.kind(name = "基数", surface = m.group())
        return art + spoken

    def short_year_apostrophe(self, m, ctx):
        """anni '80 → anni ottanta，nel '68 → nel sessantotto。"""
        ctx.kind(name = "年份", surface = m.group())
        return ctx.num(spoken = self.card(n = int(m.group("d"))), written = m.group("d"))

    def century_abbreviation(self, m, ctx):
        """sec. XV → secolo quindicesimo。"""
        value = roman_ok(token = m.group("roman"))
        if value is None:
            return None
        ctx.kind(name = "罗马数字", surface = m.group())
        return "secolo " + ctx.num(spoken = self.ordinal_words(n = value), written = str(value))

    def altitude(self, m, ctx):
        """300 m s.l.m. → trecento metri sul livello del mare。"""
        n = self.parse(written = m.group("num"))
        if n is None:
            return None
        out = self.say(n = n, ctx = ctx, card = self.masc if n.is_int else None)
        ctx.kind(name = "缩写", surface = "s.l.m.")
        word = "metro" if n.is_int and n.value == 1 else "metri"
        end = "." if sentence_end_after(text = ctx.text, end = m.end(), starters = self.starters) else ""
        return f"{out} {word} sul livello del mare{end}"


LANG = It
