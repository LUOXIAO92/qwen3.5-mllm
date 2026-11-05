"""西班牙语：基数、年份、「11 de mayo de 2025」日期、序数（3.º、1.ª）、小数（coma）、百分数、单位、货币、
常用缩写、世纪与君主序号（siglo XX → siglo veinte，Felipe II → Felipe segundo）。

uno/una、veintiuno/veintiuna 和 doscientos–novecientos（千位组同样）随名词变性：后面的名词在词表中
或带有 -ción、-dad 等阴性词尾时选择词形（doscientas personas、veintiuna casas），阳性名词前 uno 缩为 un
（un kilómetro、veintiún años）。百万以上的部分随 millones（阳性）不变。序数 primero、tercero 在阳性
单数名词前缩为 primer、tercer。
"""

from __future__ import annotations

import re

from ..numwords import n2w
from ..rules import Rule
from .euro import sentence_end_after
from .romance import Romance

_APOCOPE_BEFORE_SCALE = re.compile(pattern = r"\b(veinti)?uno(?= (?:mil|millón|millones|billón|billones)\b)")
_SCALE = re.compile(pattern = r"\b(?:millón|millones|billón|billones)\b")
ORDINAL_TEENS = {
    11: "undécimo", 12: "duodécimo", 13: "decimotercero", 14: "decimocuarto", 15: "decimoquinto",
    16: "decimosexto", 17: "decimoséptimo", 18: "decimoctavo", 19: "decimonoveno",
}


def apocope(text: str) -> str:
    """uno → un、veintiuno → veintiún（用在名词或 mil、millones 之前）。"""
    def repl(m):
        return "veintiún" if m.group(1) else "un"

    return _APOCOPE_BEFORE_SCALE.sub(repl = repl, string = text)


def _words(text: str) -> dict:
    return {w: g for g, words in text.items() for w in words.split()}


NOUNS = _words(text = {
    "m": """
        año años día días mes meses siglo siglos grado grados kilómetro kilómetros metro metros minuto minutos
        segundo segundos punto puntos hombre hombres euro euros dólar dólares millón millones gol goles partido
        partidos voto votos país países kilo kilos peso pesos centímetro centímetros milímetro milímetros
        soldado soldados miembro miembros equipo equipos jugador jugadores estudiante estudiantes alumno alumnos
        niño niños hijo hijos libro libros disco discos álbum álbumes episodio episodios capítulo capítulos
        artículo artículos puesto puestos municipio municipios pueblo pueblos barrio barrios distrito distritos
        departamento departamentos estado estados kilogramo kilogramos gramo gramos litro litros pie pies acre
        acres caballo caballos avión aviones barco barcos buque buques cañón cañones carro carros coche coches
        vehículo vehículos tren trenes autobús autobuses edificio edificios templo templos monumento monumentos
        museo museos hospital hospitales colegio colegios centro centros puente puentes río ríos lago lagos
        habitante habitantes muerto muertos herido heridos prisionero prisioneros refugiado refugiados
        trabajador trabajadores empleado empleados espectador espectadores aficionado aficionados pasajero
        pasajeros tripulante tripulantes visitante visitantes participante participantes ejemplar ejemplares
        volumen volúmenes título títulos premio premios trofeo trofeos torneo torneos campeonato campeonatos
        encuentro encuentros juego juegos triunfo triunfos concierto conciertos programa programas problema
        problemas sistema sistemas tema temas idioma idiomas mapa mapas planeta planetas poema poemas centavo
        centavos céntimo céntimos real reales franco francos marco marcos rublo rublos escaño escaños diputado
        diputados senador senadores concejal concejales votante votantes asiento asientos megavatio megavatios
        kilovatio kilovatios voltio voltios vatio vatios bit bits byte bytes píxel píxeles carácter caracteres
        color colores rebote rebotes cuadrangular cuadrangulares ataque ataques accidente accidentes incendio
        incendios terremoto terremotos huracán huracanes bombardeo bombardeos combate combates caso casos
        fallecido fallecidos infectado infectados contagio contagios paciente pacientes médico médicos profesor
        profesores obispo obispos sacerdote sacerdotes monje monjes rey reyes príncipe príncipes hermano
        hermanos nieto nietos padre padres abuelo abuelos amigo amigos ciudadano ciudadanos vecino vecinos
        judío judíos musulmán musulmanes cristiano cristianos español españoles francés franceses inglés
        ingleses alemán alemanes italiano italianos territorio territorios reino reinos piso pisos nivel
        niveles motor motores cilindro cilindros viaje viajes vuelo vuelos paso pasos salto saltos intento
        intentos lanzamiento lanzamientos disparo disparos tiro tiros set sets golpe golpes hoyo hoyos ejército
        ejércitos regimiento regimientos batallón batallones cuerpo cuerpos congreso congresos concilio
        concilios festival festivales salón salones simposio simposios imperio imperios gobierno gobiernos plan
        planes duque duques conde condes centenario reich aniversario aniversarios cuadro cuadros cartucho
        cartuchos animal animales árbol árboles ejemplar instrumento instrumentos documento documentos
        proyecto proyectos estudio estudios grupo grupos club clubes servicio servicios producto productos
        modelo modelos tipo tipos mundial mundiales encuentro empate empates cuarto cuartos litro byte puerto
        puertos aeropuerto aeropuertos capitán capitanes oficial oficiales general generales marinero
        marineros indio indios esclavo esclavos colono colonos guerrero guerreros caballero caballeros efectivo
        efectivos hexámetro hexámetros defensor defensores barril barriles cocodrilo cocodrilos meteoro meteoros
        piano pianos envío envíos refrán refranes cine cines duque marqués conde señor barón vizconde
        """,
    "f": """
        persona personas mujer mujeres hora horas semana semanas vez veces casa casas hectárea hectáreas
        tonelada toneladas libra libras milla millas pulgada pulgadas yarda yardas copia copias unidad unidades
        especie especies víctima víctimas familia familias ciudad ciudades localidad localidades provincia
        provincias región regiones comunidad comunidades nación naciones lengua lenguas palabra palabras
        página páginas canción canciones película películas obra obras novela novelas pieza piezas escena
        escenas temporada temporadas edición ediciones carrera carreras victoria victorias derrota derrotas
        medalla medallas estrella estrellas plaza plazas cama camas habitación habitaciones vivienda viviendas
        calle calles iglesia iglesias escuela escuelas universidad universidades empresa empresas compañía
        compañías tienda tiendas fábrica fábricas cabeza cabezas arma armas bomba bombas mina minas nave naves
        embarcación embarcaciones aeronave aeronaves máquina máquinas locomotora locomotoras línea líneas
        estación estaciones parada paradas ruta rutas carretera carreteras vía vías isla islas montaña
        montañas playa playas colina colinas parroquia parroquias diócesis letra letras carta cartas firma
        firmas moneda monedas peseta pesetas rupia rupias corona coronas lira liras foto fotos fotografía
        fotografías imagen imágenes columna columnas torre torres puerta puertas ventana ventanas sala salas
        planta plantas tropa tropas división divisiones brigada brigadas entrada entradas venta ventas acción
        acciones parte partes etapa etapas fase fases ronda rondas jornada jornadas vuelta vueltas legua
        leguas cuadra cuadras manzana manzanas caloría calorías vida vidas muerte muertes baja bajas noche
        noches década décadas generación generaciones dinastía dinastías cultura culturas tribu tribus raza
        razas clase clases categoría categorías liga ligas copa copas semifinal semifinales asistencia
        asistencias canasta canastas falta faltas tarjeta tarjetas sanción sanciones multa multas ley leyes
        norma normas regla reglas medida medidas propuesta propuestas enmienda enmiendas votación votaciones
        elección elecciones encuesta encuestas pregunta preguntas respuesta respuestas opción opciones versión
        versiones variante variantes forma formas causa causas idea ideas teoría teorías prueba pruebas
        muestra muestras célula células molécula moléculas proteína proteínas flor flores hoja hojas semilla
        semillas fruta frutas ave aves abeja abejas hormiga hormigas serpiente serpientes ballena ballenas
        vaca vacas oveja ovejas cabra cabras gallina gallinas monja monjas hermana hermanas hija hijas madre
        madres esposa esposas reina reinas princesa princesas dama damas señora señoras chica chicas niña
        niñas guerra guerras república repúblicas cruzada cruzadas legislatura legislaturas cumbre cumbres
        feria ferias bienal bienales conferencia conferencias asamblea asambleas olimpiada olimpiadas
        exposición exposiciones mesa mesas silla sillas butaca butacas obra lección lecciones misión misiones
        parcela parcelas lanza lanzas estatua estatuas ceremonia ceremonias cosecha cosechas duquesa condesa
        marquesa baronesa vizcondesa princesa
        expedición expediciones batalla batallas campaña campañas operación operaciones escuadra escuadras
        flota flotas división ocasión ocasiones vez parroquia aldea aldeas villa villas finca fincas granja
        granjas comarca comarcas mezquita mezquitas capilla capillas catedral catedrales sede sedes oficina
        oficinas sucursal sucursales filial filiales tienda marca marcas especialidad especialidades
        asignatura asignaturas materia materias disciplina disciplinas prueba modalidad modalidades
        """,
})
REGNAL_NAMES = set("""
    Felipe Carlos Fernando Alfonso Enrique Juan Pedro Jaime Sancho Ramiro García Ordoño Bermudo Luis Pío
    Benedicto Gregorio León Inocencio Clemente Urbano Bonifacio Alejandro Nicolás Martín Pablo Sixto Julio
    Adriano Eugenio Federico Guillermo Eduardo Jorge Ricardo Napoleón Ramsés Ptolomeo Amenofis Tutmosis
    Constantino Miguel Manuel Iván Víctor Humberto Amadeo Rodolfo Otón Leopoldo José Francisco Gustavo
    Cristián Selim Mehmed Murad Solimán Darío Jerjes Antíoco Seleuco Conrado Balduino Raimundo Bohemundo
    Andrónico Basilio Teodosio Justiniano Alarico Teodorico Lotario Berengario Ludovico Calixto Paulo Esteban
    Ladislao Casimiro Segismundo Wenceslao Haakon Olaf Harald Cristian Ramón Berenguer Sapor Cosroes
    Vaticano Jacobo Miró Herberto Landulfo Nicéforo Roberto Ricardo Teodoro Hércules Sargón Otón Godofredo
    Guido Tancredo Rogelio Mehmed Ahmed Abdalá Abderramán Hisham Muhammad Mohamed Hassan Husein Faisal
    Amenemhat Sesostris Psamético Seti Ramses Tutankamón Mentuhotep Nectanebo
""".split())
REGNAL_FEMALE = set("""
    Isabel Juana María Catalina Cleopatra Margarita Ana Cristina Victoria Leonor Urraca Berenguela Teodora
    Irene Elena Sofía Petronila Blanca
""".split())


class Es(Romance):
    code = "es"
    point = "coma"
    frac_as_number = True
    percent = "por ciento"
    units = {
        "km/h": ("kilómetro por hora", "kilómetros por hora"), "km²": ("kilómetro cuadrado", "kilómetros cuadrados"),
        "m²": ("metro cuadrado", "metros cuadrados"), "km": ("kilómetro", "kilómetros"), "cm": ("centímetro", "centímetros"),
        "mm": ("milímetro", "milímetros"), "kg": ("kilogramo", "kilogramos"), "m": ("metro", "metros"),
        "g": ("gramo", "gramos"), "°C": ("grado Celsius", "grados Celsius"), "℃": ("grado Celsius", "grados Celsius"),
        "°F": ("grado Fahrenheit", "grados Fahrenheit"), "℉": ("grado Fahrenheit", "grados Fahrenheit"),
    }
    currencies = {"€": ("euro", "euros"), "$": ("dólar", "dólares")}
    scales = ("mil", "millones", "millón")
    scale_links = {"millones": "de", "millón": "de"}
    months = (
        "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "setiembre",
        "octubre", "noviembre", "diciembre",
    )
    year_before = {
        "en", "de", "desde", "hasta", "año", "del", "entre", "y", "hacia", "para", "durante", "antes", "después",
        "a", "o", "el", "finales", "principios", "mediados", "años", "circa",
    }
    year_before_short = {"en", "año", "hacia", "desde", "hasta"}
    nouns = NOUNS
    fem_suffixes = ("ciones", "siones", "dades", "tades", "tudes", "umbres", "ción", "sión", "dad", "tad", "tud", "umbre")
    masc_suffixes = ("mientos", "miento", "ismos", "ismo")
    id_nouns = {
        "tipo", "número", "capítulo", "artículo", "página", "temporada", "episodio", "fórmula", "grupo",
        "categoría", "nivel", "división", "liga", "ruta", "carretera", "línea", "puerto", "sala", "planta", "piso",
        "volumen", "tomo", "parte", "fase", "etapa", "edición", "versión", "serie", "clase", "canal", "cromosoma",
        "distrito", "zona", "sector", "puerta", "andén", "vía", "tabla", "figura", "ley", "decreto", "apartado",
        "sección", "párrafo", "paso", "hoyo", "ronda", "jornada", "acto", "escena", "canto", "salmo", "autopista",
        "avenida", "calle", "modelo", "misión", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado",
        "domingo", "grado", "nivel", "dorsal", "habitación", "cuarto", "bloque", "hangar", "pabellón",
    }
    first_day = "primero"
    date_link = r"\s(?:de\s)?"
    starters = {
        "El", "La", "Los", "Las", "En", "Un", "Una", "Este", "Esta", "Su", "Sus", "Se", "Fue", "Es", "Al", "Con",
        "Por", "Para", "Desde", "Tras", "Después", "Durante", "Sin", "Además", "También", "Sin", "Hoy", "Ya",
    }
    abbreviations = {
        "etc.": ("etcétera", "any"), "Dr.": ("doctor", "name"), "Dra.": ("doctora", "name"),
        "Sr.": ("señor", "name"), "Sra.": ("señora", "name"), "Srta.": ("señorita", "name"),
        "Sres.": ("señores", "name"), "Lic.": ("licenciado", "name"), "Prof.": ("profesor", "name"),
        "St.": ("Saint", "name"), "Mr.": ("míster", "name"), "Ud.": ("usted", "any"), "Uds.": ("ustedes", "any"),
        "a. C.": ("antes de Cristo", "any"), "d. C.": ("después de Cristo", "any"), "a.C.": ("antes de Cristo", "any"),
        "d.C.": ("después de Cristo", "any"), "EE. UU.": ("Estados Unidos", "any"), "EE.UU.": ("Estados Unidos", "any"),
        "aprox.": ("aproximadamente", "any"), "Av.": ("avenida", "name"), "Avda.": ("avenida", "name"),
        "Jr.": ("júnior", "any"),
    }
    number_abbreviations = {
        "n.º": "número", "N.º": "número", "nº": "número", "Nº": "número", "núm.": "número", "pág.": "página",
        "vol.": "volumen", "art.": "artículo", "cap.": "capítulo", "No.": "número", "c.": "circa",
    }
    regnal_names = REGNAL_NAMES
    regnal_female = REGNAL_FEMALE
    century_words = {"siglo", "siglos"}
    thousand_word = "mil"
    conjunctions = ("y", "e")

    def card(self, n: int) -> str:
        return apocope(text = n2w(n = n, lang = "es"))

    def ordinal_words(self, n: int, gender: str = "m") -> str | None:
        """序数：num2words 的 11–19（décimoprimero）和 cuadragésimo 拼写有误，这里改正；400 以上不处理。"""
        if n >= 400 or (n > 100 and 11 <= n % 100 <= 19):
            return None
        words = ORDINAL_TEENS.get(n) or n2w(n = n, lang = "es", to = "ordinal").replace("quadra", "cuadra")
        if gender == "f":
            words = re.sub(pattern = r"o\b", repl = "a", string = words)
        return words

    def ordinal(self, n: int) -> str:
        return self.ordinal_words(n = n) or n2w(n = n, lang = "es", to = "ordinal")

    def apocope_ordinal(self, words: str, noun: str) -> str:
        """阳性单数名词前 primero、tercero 缩为 primer、tercer（III Reich → Tercer Reich）。"""
        if self.gender_of(word = noun) == "m" and not noun.lower().endswith("s") and words.endswith(("primero", "tercero")):
            return words[:-1]
        return words

    def roman_id_nouns(self) -> set:
        return {
            "tomo", "capítulo", "parte", "acto", "libro", "volumen", "fase", "tipo", "clase", "canto", "título",
            "artículo", "escena", "sección", "apartado", "anexo", "cuadro", "tomos", "capítulos", "libros",
            "volúmenes", "partes", "actos", "dinastía",
        }

    def determiners(self) -> set:
        return {"el", "la", "los", "las", "su", "sus", "del", "al", "este", "esta", "un", "una", "nuestra", "nuestro"}

    def title_nouns(self) -> set:
        return {
            "duque", "duquesa", "conde", "condesa", "marqués", "marquesa", "señor", "señora", "barón", "baronesa",
            "vizconde", "vizcondesa", "príncipe", "princesa",
        }

    def digits(self, digits: str) -> str:
        return " ".join(n2w(n = int(d), lang = "es") for d in digits)

    def masc(self, n: int) -> str:
        text = self.card(n = n)
        if text.endswith("veintiuno"):
            return text[:-len("veintiuno")] + "veintiún"
        if text == "uno" or text.endswith(" uno"):
            return text[:-len("uno")] + "un"
        return text

    def fem(self, n: int) -> str | None:
        """阴性：最后一个 millones 之后的 -ientos 改为 -ientas，uno、veintiún 改为 una、veintiuna。"""
        text = self.card(n = n)
        cut = 0
        for m in _SCALE.finditer(string = text):
            cut = m.end()
        head, tail = text[:cut], text[cut:]
        tail = re.sub(pattern = r"ientos\b", repl = "ientas", string = tail)
        tail = re.sub(pattern = r"\bveinti(?:uno|ún)\b", repl = "veintiuna", string = tail)
        tail = re.sub(pattern = r"\b(?:uno|un)\b", repl = "una", string = tail)
        return head + tail

    def gender_sensitive(self, n: int) -> bool:
        units, thousands = n % 1000, n // 1000 % 1000
        if units % 10 == 1 and units % 100 != 11:
            return True
        if units // 100 >= 2 or thousands // 100 >= 2:
            return True
        return thousands != 1 and thousands % 10 == 1 and thousands % 100 != 11

    def function_words(self) -> set:
        return {
            "y", "e", "o", "u", "en", "de", "del", "con", "por", "para", "que", "a", "al", "se", "es", "fue", "ha",
            "desde", "hasta", "entre", "sobre", "como", "donde", "cuando", "la", "el", "los", "las", "un", "una",
        }

    def early_rules(self) -> list:
        return [
            self.por_guard(),
            Rule(
                pattern = re.compile(pattern = rf"(?P<num>{self.num})\s?(?:m\s?s\.\s?n\.\s?m\.|msnm)(?![\w])"),
                handle = self.altitude,
            ),
            Rule(
                pattern = re.compile(pattern = r"(?<![\w.])s\.\s?(?P<roman>[IVXLC]+)\b"),
                handle = self.century_abbreviation,
            ),
        ]

    def altitude(self, m, ctx):
        """900 m s. n. m. → novecientos metros sobre el nivel del mar。"""
        n = self.parse(written = m.group("num"))
        if n is None:
            return None
        out = self.say(n = n, ctx = ctx, card = self.masc if n.is_int else None)
        ctx.kind(name = "缩写", surface = "s. n. m.")
        word = "metro" if n.is_int and n.value == 1 else "metros"
        end = "." if sentence_end_after(text = ctx.text, end = m.end(), starters = self.starters) else ""
        return f"{out} {word} sobre el nivel del mar{end}"

    def century_abbreviation(self, m, ctx):
        """s. XX → siglo veinte。"""
        from .romance import roman_ok

        value = roman_ok(token = m.group("roman"))
        if value is None:
            return None
        spoken = self.roman_reading(value = value, gender = "m", ordinal_max = self.roman_ordinal_max)
        if spoken is None:
            return None
        ctx.kind(name = "罗马数字", surface = m.group())
        return "siglo " + ctx.num(spoken = spoken, written = str(value))


LANG = Es
