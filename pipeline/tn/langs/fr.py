"""法语：基数、年份、「1er octobre 2023」日期、序数（1er、2e）、小数（virgule）、百分数、单位、货币（€、$）、
时刻（6 h 30）、常用缩写、带序数后缀的罗马数字（XVIIe siècle → dix-septième siècle）和君主序号
（Louis XIV → Louis quatorze，François Ier → François premier）。

以 1 结尾的数（un/une、vingt et un/une、quatre-vingt-un/une）按后面名词的性选择词形，onze、soixante et onze、
quatre-vingt-onze 不变；千位组以 1 结尾（vingt et un mille）在阴性名词前的写法不统一，丢弃。年份中的 cent、
quatre-vingt 不加 s。英镑（la livre）为阴性，不收录。
"""

from __future__ import annotations

import re

from ..numwords import n2w
from ..rules import Rule, Skip
from .base import word_after
from .romance import Romance, roman_ok


def _words(text: dict) -> dict:
    return {w: g for g, words in text.items() for w in words.split()}


NOUNS = _words(text = {
    "m": """
        ans an jours jour mois siècles siècle degrés degré kilomètres kilomètre mètres mètre points point hommes
        homme habitants habitant euros euro dollars dollar millions million milliards milliard pays membres
        membre joueurs joueur sièges siège étudiants étudiant élèves élève enfants enfant buts but matchs match
        matches exemplaires exemplaire épisodes épisode titres titre albums album bâtiments bâtiment logements
        logement emplois emploi véhicules véhicule chevaux cheval navires navire avions avion canons canon
        soldats soldat prisonniers prisonnier officiers officier députés député élus élu salariés salarié
        employés employé passagers passager visiteurs visiteur spectateurs spectateur participants participant
        détenus détenu réfugiés réfugié juges juge cavaliers cavalier marins marin clubs club établissements
        établissement départements département cantons canton arrondissements arrondissement diocèses diocèse
        villages village quartiers quartier sommets sommet tirs tir trains train bateaux bateau kilos kilo
        kilogrammes kilogramme grammes gramme litres litre hectares hectare milles pieds pied pouces pouce
        nœuds nœud bits bit octets octet pixels pixel francs franc centimes centime morceaux morceau chapitres
        chapitre articles article volumes volume tomes tome numéros numéro films film romans roman poèmes poème
        ouvrages ouvrage programmes programme projets projet prix records record trophées trophée tournois
        tournoi championnats championnat jeux jeu concerts concert spectacles spectacle festivals festival
        ponts pont fleuves fleuve lacs lac musées musée hôpitaux hôpital collèges collège lycées lycée centres
        centre temples temple monuments monument châteaux château sites site parcs parc jardins jardin étages
        étage niveaux niveau moteurs moteur cylindres cylindre vols vol coups coup essais essai animaux animal
        arbres arbre oiseaux oiseau insectes insecte poissons poisson chiens chien chats chat fils frères frère
        pères père rois roi princes prince évêques évêque prêtres prêtre moines moine papes pape ministres
        ministre conseillers conseiller maires maire sénateurs sénateur électeurs électeur citoyens citoyen
        résidents résident ouvriers ouvrier travailleurs travailleur militaires militaire policiers policier
        médecins médecin professeurs professeur chercheurs chercheur auteurs auteur écrivains écrivain peintres
        peintre musiciens musicien chanteurs chanteur acteurs acteur coureurs coureur pilotes pilote morts
        actifs inscrits votants suffrages mandats mandat sièges matchs postes poste dossiers dossier
        """,
    "f": """
        personnes personne femmes femme heures heure minutes minute secondes seconde semaines semaine années
        année journées journée fois maisons maison tonnes tonne unités unité espèces espèce victimes victime
        familles famille villes ville communes commune provinces province régions région nations nation
        langues langue pages page chansons chanson œuvres œuvre pièces pièce scènes scène saisons saison
        éditions édition courses course victoires victoire défaites défaite médailles médaille étoiles étoile
        places place chambres chambre rues rue églises église écoles école universités université entreprises
        entreprise sociétés société usines usine têtes tête armes arme bombes bombe mines mine machines machine
        locomotives locomotive lignes ligne stations station gares gare routes route voies voie îles île
        montagnes montagne plages plage collines colline paroisses paroisse lettres lettre cartes carte
        signatures signature monnaies monnaie photos photo photographies photographie images image colonnes
        colonne portes porte fenêtres fenêtre salles salle plantes plante troupes troupe divisions division
        brigades brigade entrées entrée ventes vente actions action parties partie étapes étape phases phase
        manches manche calories calorie vies vie nuits nuit décennies décennie générations génération dynasties
        dynastie cultures culture tribus tribu races race classes classe catégories catégorie ligues ligue
        coupes coupe finales finale amendes amende lois loi règles règle mesures mesure propositions proposition
        élections élection questions question réponses réponse options option versions version variantes
        variante formes forme causes cause idées idée théories théorie preuves preuve cellules cellule
        molécules molécule protéines protéine fleurs fleur feuilles feuille graines graine abeilles abeille
        fourmis fourmi baleines baleine vaches vache chèvres chèvre poules poule religieuses religieuse sœurs
        sœur filles fille mères mère épouses épouse reines reine princesses princesse dames dame guerres guerre
        républiques république croisades croisade législatures législature conférences conférence assemblées
        assemblée olympiades olympiade expositions exposition tables table chaises chaise leçons leçon missions
        mission expéditions expédition batailles bataille campagnes campagne opérations opération escadres
        escadre flottes flotte occasions occasion chapelles chapelle cathédrales cathédrale agences agence
        filiales filiale marques marque disciplines discipline épreuves épreuve voitures voiture voix matières
        matière associations association sélections sélection plumes plume estampes estampe lieues lieue
        paires paire résidences résidence rencontres rencontre cases case dates date sections section
        galaxies galaxie rames rame acres acre passes passe chaînes chaîne émissions émission séries série
        listes liste circonscriptions circonscription délégations délégation équipes équipe nouvelles
        """,
})
REGNAL_NAMES = set("""
    Louis Henri Charles François Philippe Jean Pierre Paul Pie Benoît Grégoire Léon Innocent Clément Urbain
    Boniface Alexandre Nicolas Martin Sixte Jules Adrien Eugène Frédéric Guillaume Édouard Georges Richard
    Napoléon Ramsès Ptolémée Aménophis Thoutmôsis Constantin Michel Manuel Ivan Victor Humbert Amédée Rodolphe
    Othon Léopold Joseph Gustave Christian Haakon Olaf Harald Sélim Mehmed Mourad Soliman Darius Xerxès
    Antiochos Séleucos Conrad Baudouin Raymond Bohémond Andronic Basile Théodose Justinien Alaric Théodoric
    Lothaire Pépin Carloman Childéric Clotaire Dagobert Sigismond Casimir Ladislas Étienne Venceslas Alphonse
    Ferdinand Sanche Jacques Robert Hugues Raoul Eudes Rainier Albert Boleslas Thibaut Amaury Guy Godefroy
    Tancrède Roger Bertrand Calixte Célestin Félix Honorius Anastase Serge Lucius Marcel Damase Agapet
    Jean-Paul Pedro Juan Carlos Felipe Fernando Alfonso Juan-Carlos Abbas Hussein Farouk Fouad Mohammed
    Vatican Sargon George Otton Wilfrid Enguerrand Mehmed Ahmed Séthi Amenemhat Sésostris Mentouhotep
    Thoutmosis Psammétique Nectanébo Antiochus Séleucus Mithridate Tigrane Hérode Agrippa Juba
""".split())
REGNAL_FEMALE = set("""
    Élisabeth Marie Catherine Cléopâtre Isabelle Jeanne Marguerite Anne Christine Victoria Éléonore Théodora
    Irène Béatrice Mathilde Blanche
""".split())
STARTERS = {
    "Le", "La", "Les", "Il", "Elle", "Ils", "Elles", "En", "Un", "Une", "Ce", "Cette", "Son", "Sa", "Ses", "Au",
    "Aux", "Dans", "Pour", "Par", "Depuis", "Après", "Avant", "Pendant", "Sans", "De", "Des", "Du", "On", "Je",
    "Nous", "Vous", "Mais", "Puis", "Ensuite", "Enfin", "Cependant", "Aujourd'hui",
}


class Fr(Romance):
    code = "fr"
    point = "virgule"
    frac_as_number = True
    percent = "pour cent"
    units = {
        "km/h": ("kilomètre par heure", "kilomètres par heure"), "km²": ("kilomètre carré", "kilomètres carrés"),
        "m²": ("mètre carré", "mètres carrés"), "km": ("kilomètre", "kilomètres"), "cm": ("centimètre", "centimètres"),
        "mm": ("millimètre", "millimètres"), "kg": ("kilogramme", "kilogrammes"), "m": ("mètre", "mètres"),
        "g": ("gramme", "grammes"), "°C": ("degré Celsius", "degrés Celsius"), "℃": ("degré Celsius", "degrés Celsius"),
        "°F": ("degré Fahrenheit", "degrés Fahrenheit"), "℉": ("degré Fahrenheit", "degrés Fahrenheit"),
    }
    currencies = {"€": ("euro", "euros"), "$": ("dollar", "dollars")}
    months = (
        "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre",
        "décembre",
    )
    year_before = {
        "en", "de", "depuis", "dès", "vers", "jusqu'en", "avant", "après", "année", "an", "fin", "début",
        "printemps", "été", "automne", "hiver", "entre", "et", "à", "l'an", "l'année", "mi", "années", "circa",
    }
    year_before_short = {"en", "an", "vers", "l'an", "depuis", "dès"}
    nouns = NOUNS
    fem_suffixes = ("tions", "sions", "tion", "sion", "ités", "ité")
    masc_suffixes = ("ments", "ismes", "isme")
    id_nouns = {
        "type", "numéro", "chapitre", "page", "saison", "épisode", "formule", "groupe", "catégorie", "niveau",
        "division", "ligue", "route", "ligne", "porte", "salle", "étage", "volume", "tome", "partie", "phase",
        "étape", "édition", "version", "série", "classe", "chaîne", "chromosome", "zone", "secteur", "quai",
        "voie", "tableau", "figure", "loi", "section", "paragraphe", "scène", "acte", "chant", "psaume", "opus",
        "autoroute", "modèle", "mission", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
        "bloc", "pavillon", "poule", "stade", "lot", "région",
    }
    first_day = "premier"
    starters = STARTERS
    abbreviations = {
        "etc.": ("et cetera", "any"), "Mme": ("Madame", "name"), "Mmes": ("Mesdames", "name"),
        "Mlle": ("Mademoiselle", "name"), "Mgr": ("Monseigneur", "name"), "Dr": ("Docteur", "name"),
        "Dr.": ("Docteur", "name"), "St": ("Saint", "name"), "St.": ("Saint", "name"), "Ste": ("Sainte", "name"),
        "av. J.-C.": ("avant Jésus-Christ", "any"), "apr. J.-C.": ("après Jésus-Christ", "any"),
        "av.": ("avenue", "name"), "bd": ("boulevard", "name"), "env.": ("environ", "any"),
        "p. ex.": ("par exemple", "any"), "c.-à-d.": ("c'est-à-dire", "any"), "Jr.": ("Junior", "any"),
        "Mr.": ("Mister", "name"),
    }
    number_abbreviations = {"no": "numéro", "n°": "numéro", "N°": "numéro", "p.": "page", "vol.": "volume", "art.": "article", "c.": "circa"}
    regnal_names = REGNAL_NAMES
    regnal_female = REGNAL_FEMALE
    century_words = set()
    # 法语君主序号读基数（Louis quatorze），Ier 由序数后缀规则读 premier
    roman_ordinal_max = 0

    def card(self, n: int) -> str:
        return n2w(n = n, lang = "fr")

    def digits(self, digits: str) -> str:
        return " ".join(n2w(n = int(d), lang = "fr") for d in digits)

    def year_words(self, n: int) -> str:
        """年份中的 cent、quatre-vingt 不加 s（en mille neuf cent、les années mille neuf cent quatre-vingt）。"""
        return re.sub(pattern = r"\b(cent|vingt)s\b", repl = r"\1", string = self.card(n = n))

    def plural(self, n) -> bool:
        """法语从 2 起用复数：1,5 kilomètre。"""
        return n.value >= 2

    def gender_sensitive(self, n: int) -> bool:
        for group in (n % 1000, n // 1000 % 1000):
            if group != 1 and group % 10 == 1 and group % 100 not in (11, 71, 91):
                return True
        return n % 1000 == 1

    def fem(self, n: int) -> str | None:
        """阴性：词末的 un 改为 une（vingt et une、quatre-vingt-une、cent une）。"""
        thousands = n // 1000 % 1000
        if thousands % 10 == 1 and thousands % 100 not in (11, 71, 91) and thousands != 1:
            return None
        if n % 1000 != 1 and (n % 10 != 1 or n % 100 in (11, 71, 91)):
            return self.card(n = n)
        return re.sub(pattern = r"un$", repl = "une", string = self.card(n = n))

    def ordinal_words(self, n: int, gender: str = "m") -> str | None:
        if n == 1:
            return "première" if gender == "f" else "premier"
        return re.sub(pattern = r"cents?ième", repl = "centième", string = n2w(n = n, lang = "fr", to = "ordinal"))

    def roman_id_nouns(self) -> set:
        return {
            "tome", "acte", "chapitre", "livre", "volume", "partie", "chant", "titre", "article", "scène",
            "section", "annexe", "tableau", "phase", "type", "classe", "groupe",
        }

    def roman_before_noun_ok(self, noun: str) -> bool:
        """法语名词前的罗马数字都带序数后缀（IIIe République），不带后缀的不改写。"""
        return False

    def function_words(self) -> set:
        return {
            "et", "ou", "en", "de", "du", "des", "avec", "par", "pour", "que", "qui", "à", "au", "aux", "se", "est",
            "fut", "a", "depuis", "jusqu'à", "entre", "sur", "comme", "où", "quand", "le", "la", "les", "un", "une",
            "dans", "contre", "sous",
        }

    def early_rules(self) -> list:
        return [
            Rule(pattern = re.compile(pattern = r"(?<![\w.])(?:M|MM)\.(?=\s[A-ZÀ-Ý])"), handle = self.monsieur),
            Rule(pattern = re.compile(pattern = r"(?<![\w'’])(?P<de>[Dd]e)\s1(?![\d.,]|\s\d|\s?[%€$]|\s?h\b|er\b|re\b)"), handle = self.de_un),
            Rule(
                pattern = re.compile(pattern = r"(?<![\w'’-])(?P<roman>[IVXLCDM]+)(?P<suf>er|re|ère|e|ème|è)(?![\w])"),
                handle = self.roman_ordinal,
            ),
            Rule(
                pattern = re.compile(pattern = r"(?<![\d.,])(?P<h>\d{1,2})\s?h\s?(?P<mm>\d{2})?(?![\w])"),
                handle = self.time,
            ),
        ]

    def de_un(self, m, ctx):
        """de 1 an → d'un an，de 1 heure → d'une heure（un、une 前的 de 省音）。"""
        noun = word_after(text = ctx.text, end = m.end())
        gender = self.gender_of(word = noun) if noun else None
        if gender is None:
            return Skip(reason = f"数词的阴阳性取决于名词「{m.group()}」")
        ctx.kind(name = "省音", surface = m.group())
        spoken = ctx.num(spoken = "une" if gender == "f" else "un", written = "1")
        return m.group("de")[0] + "'" + spoken

    def monsieur(self, m, ctx):
        """M. Dupont → Monsieur Dupont；前面是大写的名字时（Jean M. Dupont）是首字母，不处理。"""
        if re.search(pattern = r"[A-ZÀ-Ý][\w-]*\s$", string = ctx.text[:m.start()]):
            return None
        ctx.kind(name = "缩写", surface = m.group())
        return "Messieurs" if m.group() == "MM." else "Monsieur"

    def roman_ordinal(self, m, ctx):
        """XVIIe → dix-septième，Ier → premier，Ire → première。"""
        token, suf = m.group("roman"), m.group("suf")
        value = roman_ok(token = token)
        if value is None or (len(token) == 1 and token not in "IVX"):
            return None
        if ctx.text[m.end():m.end() + 1] in "-–":
            return Skip(reason = f"罗马数字的范围「{m.group()}」")
        if value == 1 and suf == "er":
            spoken = "premier"
        elif value == 1 and suf in ("re", "ère"):
            spoken = "première"
        elif value >= 2 and suf in ("e", "ème", "è"):
            spoken = self.ordinal_words(n = value)
        else:
            return Skip(reason = f"序数写法无法确定「{m.group()}」")
        ctx.kind(name = "罗马数字", surface = m.group())
        return ctx.num(spoken = spoken, written = str(value))

    def time(self, m, ctx):
        """6 h 30 → six heures trente，21 h → vingt et une heures。"""
        h = int(m.group("h"))
        mm = int(m.group("mm")) if m.group("mm") else 0
        if h > 24 or mm > 59:
            return Skip(reason = f"时刻写法无法确定「{m.group()}」")
        hour = self.fem(n = h) if self.gender_sensitive(n = h) else self.card(n = h)
        out = ctx.num(spoken = hour, written = m.group("h")) + (" heure" if h <= 1 else " heures")
        if m.group("mm"):
            out += " " + ctx.num(spoken = self.card(n = mm) if mm else "zéro zéro", written = m.group("mm"))
        ctx.kind(name = "时刻", surface = m.group())
        return out

    def ordinal_rules(self) -> list:
        return [
            Rule(
                pattern = re.compile(pattern = r"(?<![\d.,])(?P<num>\d{1,4})(?P<suf>er|re|ère|e|ème|è)(?![\w])"),
                handle = self.ordinal,
            ),
        ]

    def ordinal(self, m, ctx):
        n = int(m.group("num"))
        suf = m.group("suf")
        if n == 1 and suf == "er":
            spoken = "premier"
        elif n == 1 and suf in ("re", "ère"):
            spoken = "première"
        elif n >= 2 and suf in ("e", "ème", "è"):
            spoken = self.ordinal_words(n = n)
        else:
            return Skip(reason = f"序数写法无法确定「{m.group()}」")
        ctx.kind(name = "序数", surface = m.group())
        return ctx.num(spoken = spoken, written = m.group())


LANG = Fr
