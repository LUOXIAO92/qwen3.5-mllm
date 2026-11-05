"""葡萄牙语：基数、年份、「21 de maio de 1974」日期、序数（3º、36ª）、小数（vírgula）、百分数、单位、货币、
常用缩写、世纪与君主序号（século XX → século vinte，D. Pedro I → Pedro primeiro）。

数词用巴西拼写（dezesseis、quilômetro）。um/uma、dois/duas 和 duzentos–novecentos（千位组同样）随名词
变性：后面的名词在词表中或带有 -ção、-dade 等阴性词尾时选择词形（duzentas pessoas、vinte e duas casas）；
milhões（阳性）之前的部分不变。十亿以上巴西（bilhão）与葡萄牙（bilião）数值不同，丢弃。
"""

from __future__ import annotations

import re

from ..numwords import n2w
from ..rules import Rule, Skip
from .romance import Romance, roman_ok

_SCALE = re.compile(pattern = r"\b(?:milhão|milhões|bilhão|bilhões)\b")


def _words(text: dict) -> dict:
    return {w: g for g, words in text.items() for w in words.split()}


NOUNS = _words(text = {
    "m": """
        ano anos dia dias mês meses século séculos grau graus quilômetro quilômetros quilómetro quilómetros
        metro metros minuto minutos segundo segundos ponto pontos homem homens euro euros dólar dólares real
        reais milhão milhões bilhão bilhões gol gols golo golos jogo jogos voto votos país países quilo quilos
        centímetro centímetros milímetro milímetros soldado soldados membro membros time times jogador
        jogadores estudante estudantes aluno alunos filho filhos livro livros disco discos álbum álbuns
        episódio episódios capítulo capítulos artigo artigos posto postos município municípios bairro bairros
        distrito distritos departamento departamentos estado estados quilograma quilogramas grama gramas litro
        litros pé pés acre acres cavalo cavalos avião aviões barco barcos navio navios canhão canhões carro
        carros veículo veículos trem trens comboio comboios ônibus edifício edifícios templo templos
        monumento monumentos museu museus hospital hospitais colégio colégios centro centros rio rios lago
        lagos habitante habitantes morto mortos ferido feridos prisioneiro prisioneiros refugiado refugiados
        trabalhador trabalhadores empregado empregados funcionário funcionários espectador espectadores
        torcedor torcedores passageiro passageiros tripulante tripulantes visitante visitantes participante
        participantes exemplar exemplares volume volumes título títulos prêmio prêmios prémio prémios troféu
        troféus torneio torneios campeonato campeonatos encontro encontros concerto concertos programa
        programas problema problemas sistema sistemas tema temas idioma idiomas mapa mapas planeta planetas
        poema poemas centavo centavos cêntimo cêntimos escudo escudos assento assentos deputado deputados
        senador senadores vereador vereadores eleitor eleitores bit bits byte bytes pixel pixels caractere
        caracteres ataque ataques acidente acidentes incêndio incêndios terremoto terremotos furacão furacões
        caso casos paciente pacientes médico médicos professor professores bispo bispos padre padres monge
        monges rei reis príncipe príncipes irmão irmãos neto netos pai pais avô avós amigo amigos cidadão
        cidadãos vizinho vizinhos judeu judeus cristão cristãos português portugueses brasileiro brasileiros
        francês franceses inglês ingleses alemão alemães italiano italianos espanhol espanhóis território
        territórios reino reinos andar andares nível níveis motor motores cilindro cilindros voo voos passo
        passos salto saltos lançamento lançamentos tiro tiros set sets golpe golpes buraco buracos exército
        exércitos regimento regimentos batalhão batalhões corpo corpos congresso congressos concílio concílios
        festival festivais salão salões simpósio simpósios império impérios governo governos plano planos
        duque duques conde condes aniversário aniversários quadro quadros animal animais hectare hectares instrumento
        instrumentos documento documentos projeto projetos estudo estudos grupo grupos clube clubes serviço
        serviços produto produtos modelo modelos tipo tipos mundial mundiais empate empates quarto quartos
        porto portos aeroporto aeroportos capitão capitães oficial oficiais general generais marinheiro
        marinheiros índio índios escravo escravos colono colonos guerreiro guerreiros cavaleiro cavaleiros
        dólar milhar milhares conto contos romance romances filme filmes jornal jornais satélite satélites
        bebé bebés bebê bebês atracadouro atracadouros teste testes parque parques agente agentes pastor pastores
        registo registos registro registros felino felinos militar militares infiltrado infiltrados duque
        marquês conde senhor barão visconde
        """,
    "f": """
        pessoa pessoas mulher mulheres hora horas semana semanas vez vezes casa casas tonelada toneladas
        libra libras milha milhas polegada polegadas jarda jardas cópia cópias unidade unidades espécie
        espécies vítima vítimas família famílias cidade cidades localidade localidades província províncias
        região regiões comunidade comunidades nação nações língua línguas palavra palavras página páginas
        canção canções música músicas película películas obra obras novela novelas peça peças cena cenas
        temporada temporadas edição edições corrida corridas vitória vitórias derrota derrotas medalha
        medalhas estrela estrelas praça praças cama camas habitação habitações moradia moradias rua ruas
        igreja igrejas escola escolas universidade universidades empresa empresas companhia companhias loja
        lojas fábrica fábricas cabeça cabeças arma armas bomba bombas mina minas nave naves embarcação
        embarcações aeronave aeronaves máquina máquinas locomotiva locomotivas linha linhas estação estações
        parada paradas rota rotas estrada estradas via vias ilha ilhas montanha montanhas praia praias colina
        colinas paróquia paróquias diocese dioceses letra letras carta cartas assinatura assinaturas moeda
        moedas rupia rupias coroa coroas lira liras foto fotos fotografia fotografias imagem imagens coluna
        colunas torre torres porta portas janela janelas sala salas planta plantas tropa tropas divisão
        divisões brigada brigadas entrada entradas venda vendas ação ações parte partes etapa etapas fase
        fases rodada rodadas jornada jornadas volta voltas légua léguas caloria calorias vida vidas morte
        mortes baixa baixas noite noites década décadas geração gerações dinastia dinastias cultura culturas
        tribo tribos raça raças classe classes categoria categorias liga ligas copa copas taça taças
        semifinal semifinais assistência assistências cesta cestas falta faltas multa multas lei leis norma
        normas regra regras medida medidas proposta propostas emenda emendas votação votações eleição
        eleições pergunta perguntas resposta respostas opção opções versão versões variante variantes forma
        formas causa causas ideia ideias teoria teorias prova provas amostra amostras célula células molécula
        moléculas proteína proteínas flor flores folha folhas semente sementes fruta frutas ave aves abelha
        abelhas formiga formigas serpente serpentes cobra cobras baleia baleias vaca vacas ovelha ovelhas
        cabra cabras galinha galinhas freira freiras irmã irmãs filha filhas mãe mães esposa esposas rainha
        rainhas princesa princesas dama damas senhora senhoras menina meninas criança crianças guerra guerras
        república repúblicas cruzada cruzadas legislatura legislaturas cúpula cúpulas feira feiras bienal
        bienais conferência conferências assembleia assembleias olimpíada olimpíadas exposição exposições
        mesa mesas cadeira cadeiras lição lições missão missões expedição expedições batalha batalhas campanha
        campanhas operação operações esquadra esquadras frota frotas ocasião ocasiões aldeia aldeias vila vilas
        fazenda fazendas comarca comarcas mesquita mesquitas capela capelas catedral catedrais sede sedes
        agência agências filial filiais marca marcas disciplina disciplinas modalidade modalidades máscara
        máscaras estrutura estruturas característica características laranja laranjas diretriz diretrizes
        cerimônia cerimônias cerimónia cerimónias duquesa condessa marquesa baronesa viscondessa
        árvore árvores estrofe estrofes canção faixa faixas série séries emissora emissoras
        """,
})
REGNAL_NAMES = set("""
    Pedro João Afonso Sancho Dinis Fernando Duarte Manuel Sebastião Henrique Filipe Carlos Luís Miguel José
    Pio Bento Gregório Leão Inocêncio Clemente Urbano Bonifácio Alexandre Nicolau Martinho Paulo Sisto Júlio
    Adriano Eugénio Eugênio Frederico Guilherme Eduardo Jorge Ricardo Napoleão Ramsés Ptolomeu Amenófis
    Tutemés Constantino Ivã Ivan Vítor Humberto Amadeu Rodolfo Otão Leopoldo Francisco Gustavo Cristiano
    Selim Mehmed Murade Solimão Dario Xerxes Antíoco Seleuco Conrado Balduíno Raimundo Boemundo Andrónico
    Andrônico Basílio Teodósio Justiniano Alarico Teodorico Lotário Calisto Estêvão Ladislau Casimiro
    Sigismundo Venceslau Haakon Olavo Haroldo Afonso Ramiro Bermudo Ordonho Garcia Jaime Alfonso Juan Luis
    Floris Guilherme Cosroes Sapor Inácio Vaticano Jaime Roberto Ricardo Teodoro Hércules Godofredo Raimundo
    Mehmed Ahmed Maomé Amenemés Sesóstris Seti Ramessés Tutancâmon Mentuhotep
""".split())
REGNAL_FEMALE = set("""
    Isabel Joana Maria Catarina Cleópatra Margarida Ana Cristina Vitória Leonor Urraca Teodora Irene Helena
    Sofia Beatriz Teresa Amélia
""".split())


class Pt(Romance):
    code = "pt"
    point = "vírgula"
    frac_as_number = True
    percent = "por cento"
    units = {
        "km/h": ("quilômetro por hora", "quilômetros por hora"),
        "km²": ("quilômetro quadrado", "quilômetros quadrados"), "m²": ("metro quadrado", "metros quadrados"),
        "km": ("quilômetro", "quilômetros"), "cm": ("centímetro", "centímetros"), "mm": ("milímetro", "milímetros"),
        "kg": ("quilograma", "quilogramas"), "m": ("metro", "metros"), "g": ("grama", "gramas"),
        "°C": ("grau Celsius", "graus Celsius"), "℃": ("grau Celsius", "graus Celsius"),
        "°F": ("grau Fahrenheit", "graus Fahrenheit"), "℉": ("grau Fahrenheit", "graus Fahrenheit"),
    }
    currencies = {"€": ("euro", "euros"), "$": ("dólar", "dólares")}
    scales = ("mil", "milhões", "milhão")
    scale_links = {"milhões": "de", "milhão": "de"}
    months = (
        "janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro",
        "novembro", "dezembro",
    )
    year_before = {
        "em", "de", "desde", "até", "ano", "entre", "e", "a", "para", "durante", "antes", "após", "depois", "o",
        "final", "início", "meados", "anos", "circa", "verão", "inverno", "outono", "primavera",
    }
    year_before_short = {"em", "ano", "desde", "até", "cerca"}
    nouns = NOUNS
    fem_suffixes = ("ções", "sões", "dades", "tudes", "agens", "ção", "são", "dade", "tude", "agem")
    masc_suffixes = ("mentos", "mento", "ismos", "ismo")
    id_nouns = {
        "tipo", "número", "capítulo", "artigo", "página", "temporada", "episódio", "fórmula", "grupo",
        "categoria", "nível", "divisão", "liga", "rota", "estrada", "linha", "porto", "sala", "andar", "volume",
        "tomo", "parte", "fase", "etapa", "edição", "versão", "série", "classe", "canal", "cromossomo",
        "cromossoma", "distrito", "zona", "setor", "sector", "porta", "plataforma", "via", "tabela", "figura",
        "lei", "decreto", "seção", "secção", "parágrafo", "passo", "rodada", "jornada", "ato", "acto", "cena",
        "canto", "salmo", "autoestrada", "avenida", "rua", "modelo", "missão", "dimensão", "grau", "nível",
        "domingo", "sábado", "bloco", "pavilhão", "sonda",
    }
    first_day = "primeiro"
    date_link = r"\s(?:de\s)?"
    starters = {
        "O", "A", "Os", "As", "Em", "Um", "Uma", "Este", "Esta", "Seu", "Sua", "Foi", "É", "No", "Na", "Com",
        "Por", "Para", "Desde", "Após", "Depois", "Durante", "Sem", "Além", "Também", "Hoje", "Já",
    }
    abbreviations = {
        "etc.": ("etcétera", "any"), "Dr.": ("doutor", "name"), "Dra.": ("doutora", "name"),
        "dr.": ("doutor", "name"), "dra.": ("doutora", "name"), "Sr.": ("senhor", "name"),
        "Sra.": ("senhora", "name"), "sr.": ("senhor", "name"), "sra.": ("senhora", "name"),
        "Srs.": ("senhores", "name"), "Prof.": ("professor", "name"), "St.": ("Saint", "name"),
        "Mr.": ("míster", "name"), "a.C.": ("antes de Cristo", "any"), "d.C.": ("depois de Cristo", "any"),
        "a. C.": ("antes de Cristo", "any"), "d. C.": ("depois de Cristo", "any"), "Ltda.": ("limitada", "any"),
        "Jr.": ("júnior", "any"),
    }
    number_abbreviations = {
        "n.º": "número", "N.º": "número", "nº": "número", "Nº": "número", "pág.": "página", "vol.": "volume",
        "art.": "artigo", "cap.": "capítulo", "No.": "número", "c.": "cerca de",
    }
    regnal_names = REGNAL_NAMES
    regnal_female = REGNAL_FEMALE
    century_words = {"século", "séculos"}
    thousand_word = "mil"
    conjunctions = ("e",)

    def card(self, n: int) -> str:
        return n2w(n = n, lang = "pt")

    def ordinal_words(self, n: int, gender: str = "m") -> str | None:
        """序数：num2words 的 400 以上拼写有误（quadrigentésimo），不处理。"""
        if n >= 400:
            return None
        words = n2w(n = n, lang = "pt", to = "ordinal")
        if gender == "f":
            words = re.sub(pattern = r"o\b", repl = "a", string = words)
        return words

    def ordinal(self, n: int) -> str:
        return self.ordinal_words(n = n) or n2w(n = n, lang = "pt", to = "ordinal")

    def roman_id_nouns(self) -> set:
        return {
            "tomo", "capítulo", "parte", "ato", "acto", "livro", "volume", "fase", "tipo", "classe", "canto",
            "título", "artigo", "cena", "seção", "secção", "anexo", "quadro", "livros", "capítulos", "dinastia",
        }

    def determiners(self) -> set:
        return {"o", "a", "os", "as", "seu", "sua", "do", "da", "no", "na", "ao", "à", "este", "esta", "um", "uma"}

    def title_nouns(self) -> set:
        return {
            "duque", "duquesa", "conde", "condessa", "marquês", "marquesa", "senhor", "senhora", "barão",
            "baronesa", "visconde", "viscondessa",
        }

    def digits(self, digits: str) -> str:
        return " ".join(n2w(n = int(d), lang = "pt") for d in digits)

    def say(self, n, ctx, card = None):
        if n.value >= 10 ** 9:
            return Skip(reason = f"十亿以上的数在巴西和葡萄牙读法不同「{n.written}」")
        return super().say(n = n, ctx = ctx, card = card)

    def fem(self, n: int) -> str | None:
        """阴性：最后一个 milhões 之后的 -entos 改为 -entas，um、dois 改为 uma、duas。"""
        if n >= 10 ** 9:
            return None
        text = self.card(n = n)
        cut = 0
        for m in _SCALE.finditer(string = text):
            cut = m.end()
        head, tail = text[:cut], text[cut:]
        tail = re.sub(pattern = r"entos\b", repl = "entas", string = tail)
        tail = re.sub(pattern = r"\bum\b", repl = "uma", string = tail)
        tail = re.sub(pattern = r"\bdois\b", repl = "duas", string = tail)
        return head + tail

    def gender_sensitive(self, n: int) -> bool:
        units, thousands = n % 1000, n // 1000 % 1000
        if units % 10 in (1, 2) and units % 100 not in (11, 12):
            return True
        if units // 100 >= 2 or thousands // 100 >= 2:
            return True
        return thousands != 1 and thousands % 10 in (1, 2) and thousands % 100 not in (11, 12)

    def function_words(self) -> set:
        return {
            "e", "ou", "em", "de", "do", "da", "com", "por", "para", "que", "a", "o", "ao", "à", "se", "é", "foi",
            "desde", "até", "entre", "sobre", "como", "onde", "quando", "no", "na", "os", "as", "um", "uma",
        }

    def early_rules(self) -> list:
        names = "|".join(sorted(self.regnal_names | self.regnal_female, key = len, reverse = True))
        return [
            Rule(pattern = re.compile(pattern = rf"(?<![\w.])D\.\s(?=(?:{names})\b)"), handle = self.dom),
            self.por_guard(),
            Rule(
                pattern = re.compile(pattern = r"(?<![\w.])séc\.\s?(?P<roman>[IVXLC]+)\b"),
                handle = self.century_abbreviation,
            ),
        ]

    def dom(self, m, ctx):
        """D. Pedro → Dom Pedro，D. Maria → Dona Maria（只在君主名前）。"""
        name = re.match(pattern = r"\w+", string = ctx.text[m.end():]).group()
        ctx.kind(name = "缩写", surface = "D.")
        return "Dona " if name in self.regnal_female else "Dom "

    def century_abbreviation(self, m, ctx):
        """séc. XX → século vinte。"""
        value = roman_ok(token = m.group("roman"))
        if value is None:
            return None
        spoken = self.roman_reading(value = value, gender = "m", ordinal_max = self.roman_ordinal_max)
        if spoken is None:
            return None
        ctx.kind(name = "罗马数字", surface = m.group())
        return "século " + ctx.num(spoken = spoken, written = str(value))


LANG = Pt
