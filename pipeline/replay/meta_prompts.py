"""带系统提示的回放组（#71，free_sys_chat、free_sys_chat_think）的元指令：先让参考模型按一句手写的元指令写出系统提示，
再把写出的系统提示放进系统轮，接着由模型自己写用户消息和回答（用户 2026-11-28 决定）。

从空的 `<|im_start|>system\\n` 直接写系统提示时，试运行（#71）中 42% 只有 1–3 个词，20% 的用户消息为空。改成元指令后，
系统提示完整、多样，带有各种约束，但几乎都是英语，约三成以「You are an expert」开头，话题集中在软件上。这里的元指令因此：

- 用项目的 18 种语言分别写成，并要求用该语言写系统提示；
- 轮换 8 种措辞（KINDS），分别要求不同类型的系统提示：角色或人物、任务助手、可检查的格式规则、语气与风格、安全与拒绝
  规则、分步辅导、具体机构的助手、很短的系统提示；
- 每条元指令配一个主题（DOMAINS，34 个日常与专业领域，编程只是其中之一），写在第一行「主题：…」中，避免语法变格；
- 明确要求不以「你是一位专家」开头，只输出系统提示本身，不加标题和解释。

元指令只是一小段手写的提示，系统提示、用户消息和回答都由参考模型写出，不用任何外部数据。元指令这一轮（用户轮为元指令，
助手轮按 `enable_thinking = False` 打开）只用于生成，不进入保存的训练序列：保存的样本就是
`<|im_start|>system\\n{系统}<|im_end|>\\n<|im_start|>user\\n{用户}<|im_end|>\\n<|im_start|>assistant\\n…`。

生成时的筛选（`generate --resample-system/--resample-user`，finalize 再检查一遍）：

- 系统提示为空、含 `<think>` / `</think>`、以代码围栏开头，或第一行写着「系统提示」之类的标签（多是「这是一段系统提示：」
  一类的说明），见 `system_problem`；
- 用户消息为空，或抄写系统提示（用户消息至少 20 个字符，并且是系统提示的一部分，或其中 60% 以上的 5 字符片段出现在系统
  提示中），见 `user_problem`。

出现这些情况时重新采样这一段（同样的上下文，次数有上限），用完次数仍不合格的样本整条不用。
"""

import random
import re
import unicodedata

META_VERSION = "1"
LANGS = ("zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "id", "ms", "th", "vi", "fil", "ar", "tr", "fa")
KINDS = ("persona", "task", "format", "style", "safety", "tutor", "organization", "short")
DOMAINS = (
    "cooking",
    "travel",
    "money",
    "fitness",
    "nutrition",
    "gardening",
    "pets",
    "home_repair",
    "parenting",
    "law",
    "health",
    "wellbeing",
    "languages",
    "math",
    "history",
    "literature",
    "music",
    "art",
    "sports",
    "farming",
    "housing",
    "careers",
    "business",
    "shopping",
    "banking",
    "tourism",
    "transport",
    "environment",
    "science",
    "government",
    "cars",
    "events",
    "devices",
    "programming",
)

# 每种语言：主题行（{domain} 处填主题）、8 种措辞、结尾要求、34 个主题（与 DOMAINS 同序）
TEXT = {
    "zh": {
        "topic": "主题：{domain}",
        "kinds": {
            "persona": "请写一段系统提示词，给 AI 助手设定一个与这个主题有关的具体角色或人物，包括名字、性格和说话方式。",
            "task": "请为一个在这个主题上帮助用户的 AI 助手写一段系统提示词，说明它的任务以及应该怎样回答。",
            "format": "请为这个主题的 AI 助手写一段系统提示词，对回答的形式定下明确、可以检查的规则，例如字数限制、使用列表或表格、固定的结构，或者规定开头或结尾的句子。",
            "style": "请写一段系统提示词，规定 AI 助手在这个主题上回答的语气和风格，例如正式或随意、用孩子能懂的简单词语、回答非常简短，或者指定回答所用的语言。",
            "safety": "请为这个主题的 AI 助手写一段系统提示词，说明它必须拒绝或避免哪些事情，以及遇到这些情况时应该怎样回复。",
            "tutor": "请为这个主题的 AI 辅导老师或教练写一段系统提示词，要求它一步一步地引导学习者、多向学习者提问，不直接给出答案。",
            "organization": "请为与这个主题相关的某个具体机构（例如商店、诊所、学校、俱乐部或办事处）的 AI 助手写一段系统提示词，写明它必须知道的具体信息和必须遵守的规则。",
            "short": "请为这个主题的 AI 助手写一段很短的系统提示词，只有一到三句话，包含一条明确的规则。",
        },
        "suffix": "系统提示词用中文写。不要以“你是一位专家”开头。只输出系统提示词本身，不要标题，也不要任何解释。",
        "domains": (
            "烹饪与食谱", "旅行", "个人理财与储蓄", "健身与运动", "营养与健康饮食", "园艺", "宠物护理", "家居维修与手工",
            "育儿", "日常法律问题", "健康与医疗", "压力、睡眠与心理健康", "学习外语", "数学", "历史", "图书与文学", "音乐",
            "艺术与设计", "体育", "农业", "租房与买房", "求职与职业发展", "经营小生意", "网上购物", "银行与保险", "酒店与旅游业",
            "公共交通", "环境保护与垃圾回收", "科学", "政府服务与办事手续", "汽车与驾驶", "婚礼与聚会策划", "手机和电脑的日常使用",
            "编程",
        ),
    },
    "en": {
        "topic": "Topic: {domain}",
        "kinds": {
            "persona": "Write a system prompt that gives an AI assistant a specific character or role related to this topic, with a name, a personality and a way of speaking.",
            "task": "Write a system prompt for an AI assistant that helps people with this topic. Describe its task and how it should answer.",
            "format": "Write a system prompt for an AI assistant on this topic that sets clear, checkable rules for the form of its answers, such as a length limit, a list or a table, a fixed structure, or a required first or last sentence.",
            "style": "Write a system prompt that sets the tone and style of an AI assistant's answers on this topic, for example formal or casual, simple words for children, very short replies, or a fixed answer language.",
            "safety": "Write a system prompt for an AI assistant on this topic that says what it must refuse or avoid, and how it should reply in those cases.",
            "tutor": "Write a system prompt for an AI tutor or coach on this topic who guides the learner step by step, asks questions and does not simply give away the answers.",
            "organization": "Write a system prompt for the AI assistant of a particular organization connected with this topic (such as a shop, a clinic, a school, a club or an office), with concrete facts it must know and rules it must follow.",
            "short": "Write a very short system prompt, one to three sentences long, for an AI assistant on this topic, with one clear rule.",
        },
        "suffix": "Write the system prompt in English. Do not start with \"You are an expert\". Reply with the system prompt only, without a title or any explanation.",
        "domains": (
            "cooking and recipes", "travel", "personal finance and saving money", "fitness and exercise", "nutrition and healthy eating",
            "gardening", "pet care", "home repair and DIY", "parenting", "everyday legal questions", "health and medicine",
            "stress, sleep and mental well-being", "learning a foreign language", "mathematics", "history", "books and literature", "music",
            "art and design", "sports", "farming", "renting and buying a home", "job hunting and careers", "running a small business",
            "online shopping", "banking and insurance", "hotels and tourism", "public transport", "the environment and recycling", "science",
            "government services and paperwork", "cars and driving", "planning weddings and parties", "everyday use of phones and computers",
            "programming",
        ),
    },
    "ja": {
        "topic": "テーマ：{domain}",
        "kinds": {
            "persona": "このテーマに関わる具体的なキャラクターや役割（名前、性格、話し方を含む）を AI アシスタントに与えるシステムプロンプトを書いてください。",
            "task": "このテーマについて人々を手助けする AI アシスタントのためのシステムプロンプトを書いてください。その役割と、どのように答えるべきかを説明してください。",
            "format": "このテーマの AI アシスタントのためのシステムプロンプトを書き、回答の形式について明確で確認できるルール（文字数の上限、箇条書きや表、決まった構成、最初または最後に入れる文など）を定めてください。",
            "style": "このテーマについての AI アシスタントの回答の口調とスタイル（丁寧かくだけたか、子ども向けのやさしい言葉、ごく短い返事、回答に使う言語など）を決めるシステムプロンプトを書いてください。",
            "safety": "このテーマの AI アシスタントのためのシステムプロンプトを書き、断るべきことや避けるべきこと、そしてその場合にどう返答するかを定めてください。",
            "tutor": "このテーマの AI 家庭教師またはコーチのためのシステムプロンプトを書いてください。学習者を一歩ずつ導き、質問を投げかけ、答えをすぐには教えないようにさせてください。",
            "organization": "このテーマに関係する特定の組織（店、診療所、学校、クラブ、役所など）の AI アシスタントのためのシステムプロンプトを書き、知っておくべき具体的な情報と守るべきルールを入れてください。",
            "short": "このテーマの AI アシスタントのための、1〜3文のとても短いシステムプロンプトを書いてください。明確なルールを一つ入れてください。",
        },
        "suffix": "システムプロンプトは日本語で書いてください。「あなたは専門家です」で始めないでください。タイトルや説明は付けず、システムプロンプトだけを出力してください。",
        "domains": (
            "料理とレシピ", "旅行", "家計管理と貯金", "フィットネスと運動", "栄養と健康的な食事", "ガーデニング", "ペットの世話", "家の修理とDIY",
            "子育て", "身近な法律問題", "健康と医療", "ストレス・睡眠・心の健康", "外国語の学習", "数学", "歴史", "本と文学", "音楽",
            "アートとデザイン", "スポーツ", "農業", "賃貸と住宅購入", "就職活動とキャリア", "小さな店や会社の経営", "ネットショッピング",
            "銀行と保険", "ホテルと観光", "公共交通機関", "環境とリサイクル", "科学", "行政サービスと手続き", "車と運転",
            "結婚式やパーティーの準備", "スマホとパソコンの日常的な使い方", "プログラミング",
        ),
    },
    "ko": {
        "topic": "주제: {domain}",
        "kinds": {
            "persona": "이 주제와 관련된 구체적인 캐릭터나 역할(이름, 성격, 말투 포함)을 AI 어시스턴트에게 부여하는 시스템 프롬프트를 작성하세요.",
            "task": "이 주제에 대해 사람들을 돕는 AI 어시스턴트를 위한 시스템 프롬프트를 작성하세요. 어시스턴트의 임무와 답변하는 방식을 설명하세요.",
            "format": "이 주제의 AI 어시스턴트를 위한 시스템 프롬프트를 작성하고, 답변 형식에 대해 분명하고 확인 가능한 규칙(글자 수 제한, 목록이나 표, 정해진 구조, 반드시 넣어야 하는 첫 문장이나 마지막 문장 등)을 정하세요.",
            "style": "이 주제에 대한 AI 어시스턴트 답변의 어조와 스타일(격식체 또는 친근한 말투, 어린이용 쉬운 말, 아주 짧은 답변, 답변에 사용할 언어 등)을 정하는 시스템 프롬프트를 작성하세요.",
            "safety": "이 주제의 AI 어시스턴트를 위한 시스템 프롬프트를 작성하고, 거절하거나 피해야 할 것과 그런 경우 어떻게 답해야 하는지를 정하세요.",
            "tutor": "이 주제의 AI 튜터나 코치를 위한 시스템 프롬프트를 작성하세요. 학습자를 한 단계씩 이끌고, 질문을 던지며, 답을 바로 알려 주지 않도록 하세요.",
            "organization": "이 주제와 관련된 특정 기관(가게, 병원, 학교, 동호회, 관공서 등)의 AI 어시스턴트를 위한 시스템 프롬프트를 작성하고, 알아야 할 구체적인 정보와 지켜야 할 규칙을 넣으세요.",
            "short": "이 주제의 AI 어시스턴트를 위한 아주 짧은 시스템 프롬프트를 한 문장에서 세 문장으로 작성하고, 분명한 규칙을 하나 넣으세요.",
        },
        "suffix": "시스템 프롬프트는 한국어로 작성하세요. \"당신은 전문가입니다\"로 시작하지 마세요. 제목이나 설명 없이 시스템 프롬프트만 출력하세요.",
        "domains": (
            "요리와 레시피", "여행", "개인 재무와 저축", "피트니스와 운동", "영양과 건강한 식사", "원예", "반려동물 돌보기", "집수리와 DIY",
            "육아", "생활 법률 문제", "건강과 의료", "스트레스, 수면과 정신 건강", "외국어 학습", "수학", "역사", "책과 문학", "음악",
            "미술과 디자인", "스포츠", "농업", "전월세와 집 구매", "구직과 경력 개발", "작은 가게 운영", "온라인 쇼핑", "은행과 보험",
            "호텔과 관광", "대중교통", "환경과 재활용", "과학", "행정 서비스와 서류 처리", "자동차와 운전", "결혼식과 파티 준비",
            "휴대폰과 컴퓨터의 일상적인 사용", "프로그래밍",
        ),
    },
    "de": {
        "topic": "Thema: {domain}",
        "kinds": {
            "persona": "Schreibe einen Systemprompt, der einem KI-Assistenten eine bestimmte Figur oder Rolle zu diesem Thema gibt, mit einem Namen, einer Persönlichkeit und einer eigenen Sprechweise.",
            "task": "Schreibe einen Systemprompt für einen KI-Assistenten, der Menschen bei diesem Thema hilft. Beschreibe seine Aufgabe und wie er antworten soll.",
            "format": "Schreibe einen Systemprompt für einen KI-Assistenten zu diesem Thema, der klare, überprüfbare Regeln für die Form der Antworten festlegt, zum Beispiel eine Längenbegrenzung, eine Liste oder Tabelle, eine feste Struktur oder einen vorgeschriebenen ersten oder letzten Satz.",
            "style": "Schreibe einen Systemprompt, der Ton und Stil der Antworten eines KI-Assistenten zu diesem Thema festlegt, zum Beispiel förmlich oder locker, einfache Wörter für Kinder, sehr kurze Antworten oder eine feste Antwortsprache.",
            "safety": "Schreibe einen Systemprompt für einen KI-Assistenten zu diesem Thema, der festlegt, was der Assistent ablehnen oder vermeiden muss und wie er in diesen Fällen antworten soll.",
            "tutor": "Schreibe einen Systemprompt für einen KI-Tutor oder Coach zu diesem Thema, der die lernende Person Schritt für Schritt anleitet, Fragen stellt und die Lösungen nicht einfach verrät.",
            "organization": "Schreibe einen Systemprompt für den KI-Assistenten einer bestimmten Einrichtung, die mit diesem Thema zu tun hat (zum Beispiel ein Geschäft, eine Praxis, eine Schule, ein Verein oder ein Amt), mit konkreten Fakten, die er kennen muss, und Regeln, die er befolgen muss.",
            "short": "Schreibe einen sehr kurzen Systemprompt von ein bis drei Sätzen für einen KI-Assistenten zu diesem Thema, mit einer klaren Regel.",
        },
        "suffix": "Schreibe den Systemprompt auf Deutsch. Beginne nicht mit „Du bist ein Experte“. Gib nur den Systemprompt aus, ohne Überschrift und ohne Erklärung.",
        "domains": (
            "Kochen und Rezepte", "Reisen", "private Finanzen und Sparen", "Fitness und Bewegung", "Ernährung und gesundes Essen",
            "Gartenarbeit", "Haustierpflege", "Reparaturen im Haus und Heimwerken", "Kindererziehung", "alltägliche Rechtsfragen",
            "Gesundheit und Medizin", "Stress, Schlaf und psychisches Wohlbefinden", "eine Fremdsprache lernen", "Mathematik", "Geschichte",
            "Bücher und Literatur", "Musik", "Kunst und Design", "Sport", "Landwirtschaft", "Mieten und Wohnungskauf", "Jobsuche und Karriere",
            "ein kleines Unternehmen führen", "Online-Shopping", "Bankgeschäfte und Versicherungen", "Hotels und Tourismus",
            "öffentlicher Nahverkehr", "Umwelt und Recycling", "Naturwissenschaften", "Behördengänge und Formulare", "Autos und Autofahren",
            "Hochzeits- und Partyplanung", "alltägliche Nutzung von Handy und Computer", "Programmieren",
        ),
    },
    "fr": {
        "topic": "Thème : {domain}",
        "kinds": {
            "persona": "Écris un prompt système qui donne à un assistant IA un personnage ou un rôle précis lié à ce thème, avec un nom, une personnalité et une manière de parler.",
            "task": "Écris un prompt système pour un assistant IA qui aide les gens sur ce thème. Décris sa tâche et la façon dont il doit répondre.",
            "format": "Écris un prompt système pour un assistant IA sur ce thème qui fixe des règles claires et vérifiables sur la forme de ses réponses, par exemple une longueur maximale, une liste ou un tableau, une structure fixe, ou une première ou dernière phrase imposée.",
            "style": "Écris un prompt système qui fixe le ton et le style des réponses d'un assistant IA sur ce thème, par exemple formel ou familier, des mots simples pour les enfants, des réponses très courtes, ou une langue de réponse imposée.",
            "safety": "Écris un prompt système pour un assistant IA sur ce thème qui précise ce qu'il doit refuser ou éviter, et comment il doit répondre dans ces cas.",
            "tutor": "Écris un prompt système pour un tuteur ou un coach IA sur ce thème, qui guide l'apprenant étape par étape, lui pose des questions et ne donne pas directement les réponses.",
            "organization": "Écris un prompt système pour l'assistant IA d'un organisme précis lié à ce thème (par exemple un magasin, un cabinet médical, une école, un club ou un bureau), avec des informations concrètes qu'il doit connaître et des règles qu'il doit suivre.",
            "short": "Écris un prompt système très court, d'une à trois phrases, pour un assistant IA sur ce thème, avec une règle claire.",
        },
        "suffix": "Écris le prompt système en français. Ne commence pas par « Vous êtes un expert ». Réponds uniquement avec le prompt système, sans titre ni explication.",
        "domains": (
            "la cuisine et les recettes", "les voyages", "les finances personnelles et l'épargne", "la remise en forme et l'exercice physique",
            "la nutrition et l'alimentation saine", "le jardinage", "les soins aux animaux de compagnie", "les réparations à la maison et le bricolage",
            "l'éducation des enfants", "les questions juridiques du quotidien", "la santé et la médecine", "le stress, le sommeil et la santé mentale",
            "l'apprentissage d'une langue étrangère", "les mathématiques", "l'histoire", "les livres et la littérature", "la musique",
            "l'art et le design", "le sport", "l'agriculture", "la location et l'achat d'un logement", "la recherche d'emploi et la carrière",
            "la gestion d'une petite entreprise", "les achats en ligne", "la banque et l'assurance", "l'hôtellerie et le tourisme",
            "les transports en commun", "l'environnement et le recyclage", "les sciences", "les démarches administratives",
            "les voitures et la conduite", "l'organisation de mariages et de fêtes", "l'utilisation quotidienne du téléphone et de l'ordinateur",
            "la programmation",
        ),
    },
    "ru": {
        "topic": "Тема: {domain}",
        "kinds": {
            "persona": "Напиши системный промпт, который задаёт ИИ-ассистенту конкретного персонажа или роль, связанную с этой темой: имя, характер и манеру речи.",
            "task": "Напиши системный промпт для ИИ-ассистента, который помогает людям по этой теме. Опиши его задачу и то, как он должен отвечать.",
            "format": "Напиши системный промпт для ИИ-ассистента по этой теме, который задаёт чёткие, проверяемые правила оформления ответов: например, ограничение длины, список или таблицу, фиксированную структуру или обязательное первое или последнее предложение.",
            "style": "Напиши системный промпт, который задаёт тон и стиль ответов ИИ-ассистента по этой теме: например, официальный или разговорный, простые слова для детей, очень короткие ответы или определённый язык ответа.",
            "safety": "Напиши системный промпт для ИИ-ассистента по этой теме, в котором сказано, от чего он должен отказываться или чего избегать и как отвечать в таких случаях.",
            "tutor": "Напиши системный промпт для ИИ-репетитора или наставника по этой теме, который ведёт ученика шаг за шагом, задаёт вопросы и не выдаёт готовые ответы сразу.",
            "organization": "Напиши системный промпт для ИИ-ассистента конкретной организации, связанной с этой темой (например, магазина, клиники, школы, клуба или учреждения), с конкретными сведениями, которые он должен знать, и правилами, которые он должен соблюдать.",
            "short": "Напиши очень короткий системный промпт из одного–трёх предложений для ИИ-ассистента по этой теме, с одним чётким правилом.",
        },
        "suffix": "Напиши системный промпт на русском языке. Не начинай с «Ты эксперт». Выведи только сам системный промпт, без заголовка и без пояснений.",
        "domains": (
            "кулинария и рецепты", "путешествия", "личные финансы и сбережения", "фитнес и физические упражнения", "питание и здоровый рацион",
            "садоводство", "уход за домашними животными", "ремонт дома и работа своими руками", "воспитание детей", "бытовые юридические вопросы",
            "здоровье и медицина", "стресс, сон и психическое здоровье", "изучение иностранного языка", "математика", "история",
            "книги и литература", "музыка", "искусство и дизайн", "спорт", "сельское хозяйство", "аренда и покупка жилья",
            "поиск работы и карьера", "ведение малого бизнеса", "онлайн-покупки", "банки и страхование", "гостиницы и туризм",
            "общественный транспорт", "экология и переработка отходов", "наука", "госуслуги и оформление документов",
            "автомобили и вождение", "организация свадеб и праздников", "повседневное использование телефона и компьютера", "программирование",
        ),
    },
    "pt": {
        "topic": "Tema: {domain}",
        "kinds": {
            "persona": "Escreva um prompt de sistema que dê a um assistente de IA um personagem ou papel específico ligado a este tema, com um nome, uma personalidade e uma maneira de falar.",
            "task": "Escreva um prompt de sistema para um assistente de IA que ajuda as pessoas com este tema. Descreva a tarefa dele e como ele deve responder.",
            "format": "Escreva um prompt de sistema para um assistente de IA sobre este tema que defina regras claras e verificáveis para a forma das respostas, como um limite de tamanho, uma lista ou tabela, uma estrutura fixa ou uma primeira ou última frase obrigatória.",
            "style": "Escreva um prompt de sistema que defina o tom e o estilo das respostas de um assistente de IA sobre este tema, por exemplo formal ou descontraído, palavras simples para crianças, respostas bem curtas ou um idioma fixo de resposta.",
            "safety": "Escreva um prompt de sistema para um assistente de IA sobre este tema que diga o que ele deve recusar ou evitar e como deve responder nesses casos.",
            "tutor": "Escreva um prompt de sistema para um tutor ou treinador de IA sobre este tema que guie o aluno passo a passo, faça perguntas e não entregue as respostas de imediato.",
            "organization": "Escreva um prompt de sistema para o assistente de IA de uma organização específica ligada a este tema (como uma loja, uma clínica, uma escola, um clube ou uma repartição), com informações concretas que ele precisa saber e regras que deve seguir.",
            "short": "Escreva um prompt de sistema muito curto, de uma a três frases, para um assistente de IA sobre este tema, com uma regra clara.",
        },
        "suffix": "Escreva o prompt de sistema em português. Não comece com \"Você é um especialista\". Responda apenas com o prompt de sistema, sem título nem explicação.",
        "domains": (
            "culinária e receitas", "viagens", "finanças pessoais e poupança", "condicionamento físico e exercícios", "nutrição e alimentação saudável",
            "jardinagem", "cuidados com animais de estimação", "reparos domésticos e faça você mesmo", "criação dos filhos",
            "questões jurídicas do dia a dia", "saúde e medicina", "estresse, sono e saúde mental", "aprender uma língua estrangeira", "matemática",
            "história", "livros e literatura", "música", "arte e design", "esportes", "agricultura", "aluguel e compra de imóveis",
            "busca de emprego e carreira", "gestão de um pequeno negócio", "compras online", "bancos e seguros", "hotelaria e turismo",
            "transporte público", "meio ambiente e reciclagem", "ciência", "serviços públicos e documentação", "carros e direção",
            "planejamento de casamentos e festas", "uso diário de celular e computador", "programação",
        ),
    },
    "es": {
        "topic": "Tema: {domain}",
        "kinds": {
            "persona": "Escribe un prompt de sistema que dé a un asistente de IA un personaje o un papel concreto relacionado con este tema, con un nombre, una personalidad y una forma de hablar.",
            "task": "Escribe un prompt de sistema para un asistente de IA que ayuda a las personas con este tema. Describe su tarea y cómo debe responder.",
            "format": "Escribe un prompt de sistema para un asistente de IA sobre este tema que fije reglas claras y comprobables para la forma de sus respuestas, como un límite de extensión, una lista o una tabla, una estructura fija o una primera o última frase obligatoria.",
            "style": "Escribe un prompt de sistema que fije el tono y el estilo de las respuestas de un asistente de IA sobre este tema, por ejemplo formal o informal, palabras sencillas para niños, respuestas muy breves o un idioma de respuesta fijo.",
            "safety": "Escribe un prompt de sistema para un asistente de IA sobre este tema que diga qué debe rechazar o evitar y cómo debe responder en esos casos.",
            "tutor": "Escribe un prompt de sistema para un tutor o entrenador de IA sobre este tema que guíe al estudiante paso a paso, le haga preguntas y no le dé las respuestas directamente.",
            "organization": "Escribe un prompt de sistema para el asistente de IA de una organización concreta relacionada con este tema (por ejemplo, una tienda, una clínica, una escuela, un club o una oficina), con datos concretos que debe conocer y normas que debe seguir.",
            "short": "Escribe un prompt de sistema muy breve, de una a tres frases, para un asistente de IA sobre este tema, con una regla clara.",
        },
        "suffix": "Escribe el prompt de sistema en español. No empieces con \"Eres un experto\". Responde solo con el prompt de sistema, sin título ni explicaciones.",
        "domains": (
            "cocina y recetas", "viajes", "finanzas personales y ahorro", "actividad física y ejercicio", "nutrición y alimentación saludable",
            "jardinería", "cuidado de mascotas", "reparaciones en casa y bricolaje", "crianza de los hijos", "cuestiones legales cotidianas",
            "salud y medicina", "estrés, sueño y salud mental", "aprender un idioma extranjero", "matemáticas", "historia", "libros y literatura",
            "música", "arte y diseño", "deportes", "agricultura", "alquiler y compra de vivienda", "búsqueda de empleo y carrera profesional",
            "gestión de un pequeño negocio", "compras en línea", "banca y seguros", "hoteles y turismo", "transporte público",
            "medio ambiente y reciclaje", "ciencia", "trámites y servicios públicos", "coches y conducción", "organización de bodas y fiestas",
            "uso diario del móvil y el ordenador", "programación",
        ),
    },
    "it": {
        "topic": "Tema: {domain}",
        "kinds": {
            "persona": "Scrivi un prompt di sistema che dia a un assistente IA un personaggio o un ruolo preciso legato a questo tema, con un nome, una personalità e un modo di parlare.",
            "task": "Scrivi un prompt di sistema per un assistente IA che aiuta le persone su questo tema. Descrivi il suo compito e come deve rispondere.",
            "format": "Scrivi un prompt di sistema per un assistente IA su questo tema che stabilisca regole chiare e verificabili sulla forma delle risposte, per esempio un limite di lunghezza, un elenco o una tabella, una struttura fissa o una prima o ultima frase obbligatoria.",
            "style": "Scrivi un prompt di sistema che stabilisca il tono e lo stile delle risposte di un assistente IA su questo tema, per esempio formale o informale, parole semplici per bambini, risposte molto brevi o una lingua di risposta fissa.",
            "safety": "Scrivi un prompt di sistema per un assistente IA su questo tema che indichi che cosa deve rifiutare o evitare e come deve rispondere in quei casi.",
            "tutor": "Scrivi un prompt di sistema per un tutor o un coach IA su questo tema che guidi chi impara passo dopo passo, faccia domande e non dia subito le risposte.",
            "organization": "Scrivi un prompt di sistema per l'assistente IA di un'organizzazione specifica legata a questo tema (per esempio un negozio, un ambulatorio, una scuola, un circolo o un ufficio), con informazioni concrete che deve conoscere e regole che deve seguire.",
            "short": "Scrivi un prompt di sistema molto breve, da una a tre frasi, per un assistente IA su questo tema, con una regola chiara.",
        },
        "suffix": "Scrivi il prompt di sistema in italiano. Non iniziare con \"Sei un esperto\". Rispondi solo con il prompt di sistema, senza titolo né spiegazioni.",
        "domains": (
            "cucina e ricette", "viaggi", "finanze personali e risparmio", "fitness ed esercizio fisico", "nutrizione e alimentazione sana",
            "giardinaggio", "cura degli animali domestici", "riparazioni domestiche e fai da te", "educazione dei figli", "questioni legali quotidiane",
            "salute e medicina", "stress, sonno e benessere mentale", "imparare una lingua straniera", "matematica", "storia", "libri e letteratura",
            "musica", "arte e design", "sport", "agricoltura", "affitto e acquisto della casa", "ricerca di lavoro e carriera",
            "gestione di una piccola impresa", "acquisti online", "banche e assicurazioni", "alberghi e turismo", "trasporti pubblici",
            "ambiente e riciclo", "scienza", "servizi pubblici e pratiche burocratiche", "automobili e guida", "organizzazione di matrimoni e feste",
            "uso quotidiano di telefono e computer", "programmazione",
        ),
    },
    "id": {
        "topic": "Topik: {domain}",
        "kinds": {
            "persona": "Tulislah prompt sistem yang memberi asisten AI sebuah karakter atau peran tertentu yang berkaitan dengan topik ini, lengkap dengan nama, kepribadian, dan gaya bicara.",
            "task": "Tulislah prompt sistem untuk asisten AI yang membantu orang dalam topik ini. Jelaskan tugasnya dan bagaimana ia harus menjawab.",
            "format": "Tulislah prompt sistem untuk asisten AI tentang topik ini yang menetapkan aturan yang jelas dan dapat diperiksa untuk bentuk jawabannya, misalnya batas panjang, daftar atau tabel, struktur tetap, atau kalimat pertama atau terakhir yang wajib.",
            "style": "Tulislah prompt sistem yang menetapkan nada dan gaya jawaban asisten AI tentang topik ini, misalnya formal atau santai, kata-kata sederhana untuk anak-anak, jawaban yang sangat singkat, atau bahasa jawaban tertentu.",
            "safety": "Tulislah prompt sistem untuk asisten AI tentang topik ini yang menyebutkan apa yang harus ditolak atau dihindari, dan bagaimana ia harus menjawab dalam kasus tersebut.",
            "tutor": "Tulislah prompt sistem untuk tutor atau pelatih AI tentang topik ini yang membimbing pelajar langkah demi langkah, mengajukan pertanyaan, dan tidak langsung memberikan jawabannya.",
            "organization": "Tulislah prompt sistem untuk asisten AI milik sebuah organisasi tertentu yang berkaitan dengan topik ini (misalnya toko, klinik, sekolah, klub, atau kantor), berisi informasi konkret yang harus diketahuinya dan aturan yang harus diikutinya.",
            "short": "Tulislah prompt sistem yang sangat singkat, satu sampai tiga kalimat, untuk asisten AI tentang topik ini, dengan satu aturan yang jelas.",
        },
        "suffix": "Tulis prompt sistem dalam bahasa Indonesia. Jangan dimulai dengan \"Anda adalah seorang ahli\". Balas hanya dengan prompt sistem itu, tanpa judul atau penjelasan apa pun.",
        "domains": (
            "memasak dan resep", "perjalanan", "keuangan pribadi dan menabung", "kebugaran dan latihan fisik", "gizi dan pola makan sehat",
            "berkebun", "merawat hewan peliharaan", "perbaikan rumah dan DIY", "mengasuh anak", "masalah hukum sehari-hari",
            "kesehatan dan pengobatan", "stres, tidur, dan kesehatan mental", "belajar bahasa asing", "matematika", "sejarah", "buku dan sastra",
            "musik", "seni dan desain", "olahraga", "pertanian", "menyewa dan membeli rumah", "mencari kerja dan karier", "menjalankan usaha kecil",
            "belanja daring", "perbankan dan asuransi", "hotel dan pariwisata", "transportasi umum", "lingkungan dan daur ulang",
            "ilmu pengetahuan", "layanan pemerintah dan pengurusan dokumen", "mobil dan mengemudi", "merencanakan pernikahan dan pesta",
            "penggunaan ponsel dan komputer sehari-hari", "pemrograman",
        ),
    },
    "ms": {
        "topic": "Topik: {domain}",
        "kinds": {
            "persona": "Tulis prompt sistem yang memberikan pembantu AI satu watak atau peranan tertentu yang berkaitan dengan topik ini, lengkap dengan nama, personaliti dan cara bercakap.",
            "task": "Tulis prompt sistem untuk pembantu AI yang membantu orang ramai dalam topik ini. Terangkan tugasnya dan cara ia patut menjawab.",
            "format": "Tulis prompt sistem untuk pembantu AI tentang topik ini yang menetapkan peraturan yang jelas dan boleh disemak bagi bentuk jawapannya, contohnya had panjang, senarai atau jadual, struktur tetap, atau ayat pertama atau terakhir yang wajib.",
            "style": "Tulis prompt sistem yang menetapkan nada dan gaya jawapan pembantu AI tentang topik ini, contohnya rasmi atau santai, perkataan mudah untuk kanak-kanak, jawapan yang sangat ringkas, atau bahasa jawapan yang tetap.",
            "safety": "Tulis prompt sistem untuk pembantu AI tentang topik ini yang menyatakan perkara yang mesti ditolak atau dielakkan, dan cara ia patut membalas dalam keadaan tersebut.",
            "tutor": "Tulis prompt sistem untuk tutor atau jurulatih AI tentang topik ini yang membimbing pelajar langkah demi langkah, bertanya soalan dan tidak terus memberikan jawapan.",
            "organization": "Tulis prompt sistem untuk pembantu AI sebuah organisasi tertentu yang berkaitan dengan topik ini (contohnya kedai, klinik, sekolah, kelab atau pejabat), dengan maklumat konkrit yang mesti diketahuinya dan peraturan yang mesti dipatuhinya.",
            "short": "Tulis prompt sistem yang sangat pendek, satu hingga tiga ayat, untuk pembantu AI tentang topik ini, dengan satu peraturan yang jelas.",
        },
        "suffix": "Tulis prompt sistem dalam bahasa Melayu (Malaysia). Jangan mulakan dengan \"Anda seorang pakar\". Balas dengan prompt sistem itu sahaja, tanpa tajuk atau sebarang penjelasan.",
        "domains": (
            "memasak dan resipi", "melancong", "kewangan peribadi dan menabung", "kecergasan dan senaman", "pemakanan dan diet sihat",
            "berkebun", "penjagaan haiwan peliharaan", "membaiki rumah dan kerja tangan sendiri", "keibubapaan", "isu undang-undang harian",
            "kesihatan dan perubatan", "tekanan, tidur dan kesihatan mental", "belajar bahasa asing", "matematik", "sejarah",
            "buku dan kesusasteraan", "muzik", "seni dan reka bentuk", "sukan", "pertanian", "menyewa dan membeli rumah",
            "mencari kerja dan kerjaya", "menguruskan perniagaan kecil", "membeli-belah dalam talian", "perbankan dan insurans",
            "hotel dan pelancongan", "pengangkutan awam", "alam sekitar dan kitar semula", "sains", "perkhidmatan kerajaan dan urusan dokumen",
            "kereta dan memandu", "merancang majlis perkahwinan dan parti", "penggunaan telefon dan komputer harian", "pengaturcaraan",
        ),
    },
    "th": {
        "topic": "หัวข้อ: {domain}",
        "kinds": {
            "persona": "เขียนพรอมต์ระบบที่กำหนดตัวละครหรือบทบาทเฉพาะที่เกี่ยวข้องกับหัวข้อนี้ให้ผู้ช่วย AI โดยมีชื่อ บุคลิก และวิธีพูด",
            "task": "เขียนพรอมต์ระบบสำหรับผู้ช่วย AI ที่ช่วยเหลือผู้คนในหัวข้อนี้ อธิบายหน้าที่ของผู้ช่วยและวิธีที่ควรตอบ",
            "format": "เขียนพรอมต์ระบบสำหรับผู้ช่วย AI ในหัวข้อนี้ ที่กำหนดกฎที่ชัดเจนและตรวจสอบได้เกี่ยวกับรูปแบบของคำตอบ เช่น จำกัดความยาว ใช้รายการหรือตาราง ใช้โครงสร้างที่ตายตัว หรือต้องมีประโยคแรกหรือประโยคสุดท้ายที่กำหนดไว้",
            "style": "เขียนพรอมต์ระบบที่กำหนดน้ำเสียงและสไตล์การตอบของผู้ช่วย AI ในหัวข้อนี้ เช่น เป็นทางการหรือเป็นกันเอง ใช้คำง่าย ๆ สำหรับเด็ก ตอบสั้นมาก หรือกำหนดภาษาที่ใช้ตอบ",
            "safety": "เขียนพรอมต์ระบบสำหรับผู้ช่วย AI ในหัวข้อนี้ ที่ระบุว่าผู้ช่วยต้องปฏิเสธหรือหลีกเลี่ยงอะไร และควรตอบอย่างไรในกรณีเหล่านั้น",
            "tutor": "เขียนพรอมต์ระบบสำหรับติวเตอร์หรือโค้ช AI ในหัวข้อนี้ ที่พาผู้เรียนไปทีละขั้น ตั้งคำถาม และไม่บอกคำตอบโดยตรง",
            "organization": "เขียนพรอมต์ระบบสำหรับผู้ช่วย AI ขององค์กรแห่งหนึ่งที่เกี่ยวข้องกับหัวข้อนี้ (เช่น ร้านค้า คลินิก โรงเรียน ชมรม หรือสำนักงาน) โดยใส่ข้อมูลที่เป็นรูปธรรมที่ผู้ช่วยต้องรู้และกฎที่ต้องปฏิบัติตาม",
            "short": "เขียนพรอมต์ระบบที่สั้นมาก หนึ่งถึงสามประโยค สำหรับผู้ช่วย AI ในหัวข้อนี้ โดยมีกฎที่ชัดเจนหนึ่งข้อ",
        },
        "suffix": "เขียนพรอมต์ระบบเป็นภาษาไทย อย่าขึ้นต้นด้วย \"คุณเป็นผู้เชี่ยวชาญ\" ตอบเฉพาะตัวพรอมต์ระบบเท่านั้น ไม่ต้องมีหัวเรื่องหรือคำอธิบายใด ๆ",
        "domains": (
            "การทำอาหารและสูตรอาหาร", "การเดินทาง", "การเงินส่วนบุคคลและการออม", "ฟิตเนสและการออกกำลังกาย", "โภชนาการและการกินเพื่อสุขภาพ",
            "การทำสวน", "การดูแลสัตว์เลี้ยง", "การซ่อมแซมบ้านและงาน DIY", "การเลี้ยงลูก", "ปัญหากฎหมายในชีวิตประจำวัน", "สุขภาพและการแพทย์",
            "ความเครียด การนอนหลับ และสุขภาพจิต", "การเรียนภาษาต่างประเทศ", "คณิตศาสตร์", "ประวัติศาสตร์", "หนังสือและวรรณกรรม", "ดนตรี",
            "ศิลปะและการออกแบบ", "กีฬา", "เกษตรกรรม", "การเช่าและซื้อบ้าน", "การหางานและเส้นทางอาชีพ", "การทำธุรกิจขนาดเล็ก",
            "การช้อปปิ้งออนไลน์", "ธนาคารและประกันภัย", "โรงแรมและการท่องเที่ยว", "ขนส่งสาธารณะ", "สิ่งแวดล้อมและการรีไซเคิล", "วิทยาศาสตร์",
            "บริการภาครัฐและการติดต่อราชการ", "รถยนต์และการขับขี่", "การจัดงานแต่งงานและงานเลี้ยง", "การใช้โทรศัพท์และคอมพิวเตอร์ในชีวิตประจำวัน",
            "การเขียนโปรแกรม",
        ),
    },
    "vi": {
        "topic": "Chủ đề: {domain}",
        "kinds": {
            "persona": "Hãy viết một lời nhắc hệ thống giao cho trợ lý AI một nhân vật hoặc vai trò cụ thể liên quan đến chủ đề này, có tên, tính cách và cách nói chuyện riêng.",
            "task": "Hãy viết một lời nhắc hệ thống cho trợ lý AI giúp mọi người về chủ đề này. Mô tả nhiệm vụ của trợ lý và cách nó nên trả lời.",
            "format": "Hãy viết một lời nhắc hệ thống cho trợ lý AI về chủ đề này, đặt ra những quy tắc rõ ràng và có thể kiểm tra về hình thức câu trả lời, chẳng hạn giới hạn độ dài, dùng danh sách hoặc bảng, một cấu trúc cố định, hoặc câu mở đầu hay câu kết bắt buộc.",
            "style": "Hãy viết một lời nhắc hệ thống quy định giọng điệu và phong cách trả lời của trợ lý AI về chủ đề này, chẳng hạn trang trọng hay thân mật, từ ngữ đơn giản cho trẻ em, câu trả lời rất ngắn, hoặc một ngôn ngữ trả lời cố định.",
            "safety": "Hãy viết một lời nhắc hệ thống cho trợ lý AI về chủ đề này, nêu rõ những gì trợ lý phải từ chối hoặc tránh, và cách trả lời trong những trường hợp đó.",
            "tutor": "Hãy viết một lời nhắc hệ thống cho gia sư hoặc huấn luyện viên AI về chủ đề này, hướng dẫn người học từng bước, đặt câu hỏi và không đưa ngay đáp án.",
            "organization": "Hãy viết một lời nhắc hệ thống cho trợ lý AI của một tổ chức cụ thể liên quan đến chủ đề này (ví dụ cửa hàng, phòng khám, trường học, câu lạc bộ hoặc văn phòng), có những thông tin cụ thể mà trợ lý phải biết và những quy định phải tuân theo.",
            "short": "Hãy viết một lời nhắc hệ thống rất ngắn, từ một đến ba câu, cho trợ lý AI về chủ đề này, có một quy tắc rõ ràng.",
        },
        "suffix": "Viết lời nhắc hệ thống bằng tiếng Việt. Đừng bắt đầu bằng \"Bạn là một chuyên gia\". Chỉ trả lời bằng chính lời nhắc hệ thống, không có tiêu đề hay lời giải thích nào.",
        "domains": (
            "nấu ăn và công thức món ăn", "các chuyến đi", "tài chính cá nhân và tiết kiệm", "thể dục và tập luyện", "dinh dưỡng và ăn uống lành mạnh",
            "làm vườn", "chăm sóc thú cưng", "sửa chữa nhà cửa và tự làm đồ", "nuôi dạy con cái", "các vấn đề pháp lý thường ngày", "sức khỏe và y tế",
            "căng thẳng, giấc ngủ và sức khỏe tinh thần", "học ngoại ngữ", "toán học", "lịch sử", "sách và văn học", "âm nhạc",
            "nghệ thuật và thiết kế", "thể thao", "nông nghiệp", "thuê và mua nhà", "tìm việc và phát triển sự nghiệp", "kinh doanh nhỏ",
            "mua sắm trực tuyến", "ngân hàng và bảo hiểm", "khách sạn và du lịch", "giao thông công cộng", "môi trường và tái chế", "khoa học",
            "dịch vụ hành chính công và giấy tờ", "ô tô và lái xe", "tổ chức đám cưới và tiệc", "sử dụng điện thoại và máy tính hằng ngày", "lập trình",
        ),
    },
    "fil": {
        "topic": "Paksa: {domain}",
        "kinds": {
            "persona": "Sumulat ng system prompt na nagbibigay sa AI assistant ng isang tiyak na karakter o papel na may kaugnayan sa paksang ito, na may pangalan, personalidad at paraan ng pagsasalita.",
            "task": "Sumulat ng system prompt para sa isang AI assistant na tumutulong sa mga tao tungkol sa paksang ito. Ilarawan ang gawain nito at kung paano ito dapat sumagot.",
            "format": "Sumulat ng system prompt para sa isang AI assistant tungkol sa paksang ito na nagtatakda ng malinaw at nasusuring mga patakaran sa anyo ng mga sagot nito, gaya ng limitasyon sa haba, listahan o talahanayan, nakapirming istruktura, o kinakailangang unang o huling pangungusap.",
            "style": "Sumulat ng system prompt na nagtatakda ng tono at istilo ng mga sagot ng isang AI assistant tungkol sa paksang ito, halimbawa pormal o kaswal, simpleng salita para sa mga bata, napakaikling sagot, o isang tiyak na wika ng sagot.",
            "safety": "Sumulat ng system prompt para sa isang AI assistant tungkol sa paksang ito na nagsasabi kung ano ang dapat nitong tanggihan o iwasan, at kung paano ito dapat sumagot sa mga ganoong sitwasyon.",
            "tutor": "Sumulat ng system prompt para sa isang AI tutor o coach tungkol sa paksang ito na gumagabay sa mag-aaral nang hakbang-hakbang, nagtatanong, at hindi agad ibinibigay ang mga sagot.",
            "organization": "Sumulat ng system prompt para sa AI assistant ng isang partikular na organisasyon na may kaugnayan sa paksang ito (halimbawa tindahan, klinika, paaralan, samahan o opisina), na may kongkretong impormasyong dapat nitong malaman at mga patakarang dapat nitong sundin.",
            "short": "Sumulat ng napakaikling system prompt, isa hanggang tatlong pangungusap, para sa isang AI assistant tungkol sa paksang ito, na may isang malinaw na patakaran.",
        },
        "suffix": "Isulat ang system prompt sa wikang Filipino. Huwag magsimula sa \"Ikaw ay isang eksperto\". Ang system prompt lamang ang isagot, walang pamagat o anumang paliwanag.",
        "domains": (
            "pagluluto at mga resipe", "paglalakbay", "personal na pananalapi at pag-iipon", "fitness at ehersisyo", "nutrisyon at malusog na pagkain",
            "paghahalaman", "pag-aalaga ng alagang hayop", "pagkukumpuni sa bahay at DIY", "pagpapalaki ng anak", "mga pang-araw-araw na tanong sa batas",
            "kalusugan at medisina", "stress, tulog at kalusugang pangkaisipan", "pag-aaral ng banyagang wika", "matematika", "kasaysayan",
            "mga aklat at panitikan", "musika", "sining at disenyo", "palakasan", "pagsasaka", "pag-upa at pagbili ng bahay",
            "paghahanap ng trabaho at karera", "pagpapatakbo ng maliit na negosyo", "online na pamimili", "pagbabangko at insurance",
            "mga hotel at turismo", "pampublikong transportasyon", "kapaligiran at pag-recycle", "agham",
            "mga serbisyo ng gobyerno at pag-aasikaso ng dokumento", "mga kotse at pagmamaneho", "pagpaplano ng kasal at handaan",
            "pang-araw-araw na paggamit ng telepono at kompyuter", "programming",
        ),
    },
    "ar": {
        "topic": "الموضوع: {domain}",
        "kinds": {
            "persona": "اكتب موجّه نظام يمنح مساعد الذكاء الاصطناعي شخصية أو دورًا محددًا مرتبطًا بهذا الموضوع، مع اسم وطباع وطريقة في الكلام.",
            "task": "اكتب موجّه نظام لمساعد ذكاء اصطناعي يساعد الناس في هذا الموضوع. صف مهمته وكيف ينبغي أن يجيب.",
            "format": "اكتب موجّه نظام لمساعد ذكاء اصطناعي في هذا الموضوع يضع قواعد واضحة يمكن التحقق منها لشكل إجاباته، مثل حد للطول، أو قائمة أو جدول، أو بنية ثابتة، أو جملة أولى أو أخيرة إلزامية.",
            "style": "اكتب موجّه نظام يحدد نبرة وأسلوب إجابات مساعد ذكاء اصطناعي في هذا الموضوع، مثل الأسلوب الرسمي أو الودي، أو كلمات بسيطة للأطفال، أو ردود قصيرة جدًا، أو لغة محددة للإجابة.",
            "safety": "اكتب موجّه نظام لمساعد ذكاء اصطناعي في هذا الموضوع يبيّن ما يجب أن يرفضه أو يتجنبه، وكيف ينبغي أن يرد في هذه الحالات.",
            "tutor": "اكتب موجّه نظام لمعلم خصوصي أو مدرب يعمل بالذكاء الاصطناعي في هذا الموضوع، يرشد المتعلم خطوة بخطوة، ويطرح عليه الأسئلة، ولا يعطيه الإجابات مباشرة.",
            "organization": "اكتب موجّه نظام للمساعد الذكي لجهة محددة مرتبطة بهذا الموضوع (مثل متجر أو عيادة أو مدرسة أو نادٍ أو مكتب)، يتضمن معلومات ملموسة يجب أن يعرفها وقواعد يجب أن يلتزم بها.",
            "short": "اكتب موجّه نظام قصيرًا جدًا من جملة إلى ثلاث جمل لمساعد ذكاء اصطناعي في هذا الموضوع، مع قاعدة واحدة واضحة.",
        },
        "suffix": "اكتب موجّه النظام باللغة العربية. لا تبدأ بعبارة \"أنت خبير\". أجب بموجّه النظام فقط، دون عنوان أو أي شرح.",
        "domains": (
            "الطبخ والوصفات", "السفر", "الشؤون المالية الشخصية والادخار", "اللياقة البدنية والتمارين", "التغذية والأكل الصحي", "البستنة",
            "رعاية الحيوانات الأليفة", "إصلاحات المنزل والأعمال اليدوية", "تربية الأطفال", "المسائل القانونية اليومية", "الصحة والطب",
            "التوتر والنوم والصحة النفسية", "تعلم لغة أجنبية", "الرياضيات", "التاريخ", "الكتب والأدب", "الموسيقى", "الفن والتصميم", "الرياضة",
            "الزراعة", "استئجار المنازل وشراؤها", "البحث عن عمل والمسار المهني", "إدارة مشروع صغير", "التسوق عبر الإنترنت", "البنوك والتأمين",
            "الفنادق والسياحة", "النقل العام", "البيئة وإعادة التدوير", "العلوم", "الخدمات الحكومية والمعاملات الرسمية", "السيارات والقيادة",
            "تنظيم حفلات الزفاف والمناسبات", "الاستخدام اليومي للهاتف والحاسوب", "البرمجة",
        ),
    },
    "tr": {
        "topic": "Konu: {domain}",
        "kinds": {
            "persona": "Bir yapay zekâ asistanına bu konuyla ilgili belirli bir karakter ya da rol veren, adı, kişiliği ve konuşma tarzı olan bir sistem istemi yaz.",
            "task": "Bu konuda insanlara yardım eden bir yapay zekâ asistanı için bir sistem istemi yaz. Asistanın görevini ve nasıl cevap vermesi gerektiğini anlat.",
            "format": "Bu konudaki bir yapay zekâ asistanı için, cevaplarının biçimine dair açık ve kontrol edilebilir kurallar koyan bir sistem istemi yaz; örneğin bir uzunluk sınırı, bir liste ya da tablo, sabit bir yapı veya zorunlu bir ilk ya da son cümle.",
            "style": "Bu konudaki bir yapay zekâ asistanının cevaplarının tonunu ve üslubunu belirleyen bir sistem istemi yaz; örneğin resmî ya da samimi, çocuklar için basit kelimeler, çok kısa cevaplar veya sabit bir cevap dili.",
            "safety": "Bu konudaki bir yapay zekâ asistanı için, neleri reddetmesi ya da nelerden kaçınması gerektiğini ve bu durumlarda nasıl cevap vermesi gerektiğini söyleyen bir sistem istemi yaz.",
            "tutor": "Bu konudaki bir yapay zekâ öğretmeni ya da koçu için, öğrenciyi adım adım yönlendiren, sorular soran ve cevapları doğrudan vermeyen bir sistem istemi yaz.",
            "organization": "Bu konuyla ilgili belirli bir kurumun (örneğin bir mağaza, klinik, okul, kulüp ya da ofis) yapay zekâ asistanı için, bilmesi gereken somut bilgileri ve uyması gereken kuralları içeren bir sistem istemi yaz.",
            "short": "Bu konudaki bir yapay zekâ asistanı için bir ila üç cümlelik çok kısa bir sistem istemi yaz; içinde açık bir kural olsun.",
        },
        "suffix": "Sistem istemini Türkçe yaz. \"Sen bir uzmansın\" diye başlama. Başlık ya da açıklama eklemeden yalnızca sistem istemini yaz.",
        "domains": (
            "yemek pişirme ve tarifler", "seyahat", "kişisel finans ve birikim", "fitness ve egzersiz", "beslenme ve sağlıklı yemek", "bahçecilik",
            "evcil hayvan bakımı", "ev tamiri ve kendin yap işleri", "çocuk yetiştirme", "günlük hukuki sorular", "sağlık ve tıp",
            "stres, uyku ve ruh sağlığı", "yabancı dil öğrenme", "matematik", "tarih", "kitaplar ve edebiyat", "müzik", "sanat ve tasarım", "spor",
            "tarım", "ev kiralama ve satın alma", "iş arama ve kariyer", "küçük bir işletme yönetme", "internetten alışveriş", "bankacılık ve sigorta",
            "oteller ve turizm", "toplu taşıma", "çevre ve geri dönüşüm", "bilim", "kamu hizmetleri ve resmî işlemler", "arabalar ve araç kullanma",
            "düğün ve parti planlama", "telefon ve bilgisayarın günlük kullanımı", "programlama",
        ),
    },
    "fa": {
        "topic": "موضوع: {domain}",
        "kinds": {
            "persona": "یک پرامپت سیستمی بنویس که به یک دستیار هوش مصنوعی شخصیت یا نقشی مشخص و مرتبط با این موضوع بدهد، همراه با نام، منش و شیوه حرف زدن.",
            "task": "یک پرامپت سیستمی برای دستیار هوش مصنوعی بنویس که در این موضوع به مردم کمک می کند. وظیفه او و اینکه چگونه باید پاسخ دهد را شرح بده.",
            "format": "یک پرامپت سیستمی برای دستیار هوش مصنوعی در این موضوع بنویس که برای شکل پاسخ ها قواعد روشن و قابل بررسی تعیین کند، مثلا محدودیت طول، فهرست یا جدول، ساختاری ثابت، یا جمله اول یا آخر اجباری.",
            "style": "یک پرامپت سیستمی بنویس که لحن و سبک پاسخ های دستیار هوش مصنوعی را در این موضوع تعیین کند، مثلا رسمی یا خودمانی، واژه های ساده برای کودکان، پاسخ های بسیار کوتاه، یا زبانی مشخص برای پاسخ.",
            "safety": "یک پرامپت سیستمی برای دستیار هوش مصنوعی در این موضوع بنویس که بگوید چه چیزهایی را باید رد کند یا از آن ها پرهیز کند و در این موارد چگونه پاسخ دهد.",
            "tutor": "یک پرامپت سیستمی برای معلم خصوصی یا مربی هوش مصنوعی در این موضوع بنویس که یادگیرنده را گام به گام راهنمایی کند، از او سؤال بپرسد و پاسخ ها را مستقیم به او ندهد.",
            "organization": "یک پرامپت سیستمی برای دستیار هوش مصنوعی یک سازمان مشخص مرتبط با این موضوع بنویس (مثلا یک فروشگاه، درمانگاه، مدرسه، باشگاه یا اداره)، با اطلاعات مشخصی که باید بداند و قواعدی که باید رعایت کند.",
            "short": "یک پرامپت سیستمی بسیار کوتاه، از یک تا سه جمله، برای دستیار هوش مصنوعی در این موضوع بنویس که یک قاعده روشن داشته باشد.",
        },
        "suffix": "پرامپت سیستمی را به زبان فارسی بنویس. با «تو یک متخصص هستی» شروع نکن. فقط خود پرامپت سیستمی را بنویس، بدون عنوان و بدون هیچ توضیحی.",
        "domains": (
            "آشپزی و دستور غذا", "سفر", "امور مالی شخصی و پس انداز", "تناسب اندام و تمرین بدنی", "تغذیه و رژیم غذایی سالم", "باغبانی",
            "نگهداری از حیوانات خانگی", "تعمیرات خانه و کارهای دستی", "تربیت فرزند", "مسائل حقوقی روزمره", "سلامت و پزشکی",
            "استرس، خواب و سلامت روان", "یادگیری زبان خارجی", "ریاضیات", "تاریخ", "کتاب و ادبیات", "موسیقی", "هنر و طراحی", "ورزش", "کشاورزی",
            "اجاره و خرید خانه", "کاریابی و مسیر شغلی", "اداره یک کسب و کار کوچک", "خرید اینترنتی", "بانکداری و بیمه", "هتل و گردشگری",
            "حمل و نقل عمومی", "محیط زیست و بازیافت", "علوم", "خدمات دولتی و کارهای اداری", "خودرو و رانندگی", "برنامه ریزی عروسی و مهمانی",
            "استفاده روزمره از تلفن همراه و رایانه", "برنامه نویسی",
        ),
    },
}

# 系统提示第一行中出现这些词（不区分大小写）时，多半是「这是一段系统提示：」一类的说明或标题，不是系统提示本身
SYSTEM_PROMPT_LABELS = (
    "system prompt",
    "system-prompt",
    "systemprompt",
    "系统提示",
    "系統提示",
    "システムプロンプト",
    "시스템 프롬프트",
    "prompt système",
    "prompt systeme",
    "системный промпт",
    "prompt de sistema",
    "prompt del sistema",
    "prompt di sistema",
    "prompt sistem",
    "พรอมต์ระบบ",
    "พรอมท์ระบบ",
    "lời nhắc hệ thống",
    "موجه النظام",
    "موجّه النظام",
    "موجه نظام",
    "موجّه نظام",
    "sistem istemi",
    "پرامپت سیستمی",
    "پرامپت سیستم",
)
MIN_COPY_CHARS = 20
COPY_SHARE = 0.6


def meta_text(lang, kind, domain):
    """一条元指令：主题行、措辞、结尾要求，各占一行。"""
    text = TEXT[lang]
    topic = text["topic"].format(domain = text["domains"][DOMAINS.index(domain)])
    return "\n".join([topic, text["kinds"][kind], text["suffix"]])


def assign(per_lang, seed):
    """按语言均衡地分配元指令：每种语言 per_lang 条，措辞和主题在语言内轮流使用后打乱（两者分别打乱，组合随机），
    再按轮次交错各语言（每一轮每种语言一条，语言顺序每轮打乱），于是提示清单的任何前缀都大致按语言均衡。
    返回 [{"meta_lang", "meta_kind", "meta_domain"}]。"""
    rng = random.Random(seed)
    columns = {}
    for lang in LANGS:
        kinds = [KINDS[number % len(KINDS)] for number in range(per_lang)]
        domains = [DOMAINS[number % len(DOMAINS)] for number in range(per_lang)]
        rng.shuffle(kinds)
        rng.shuffle(domains)
        columns[lang] = list(zip(kinds, domains))
    rows = []
    for number in range(per_lang):
        order = list(LANGS)
        rng.shuffle(order)
        for lang in order:
            kind, domain = columns[lang][number]
            rows.append({"meta_lang": lang, "meta_kind": kind, "meta_domain": domain})
    return rows


def normalize(text):
    text = unicodedata.normalize("NFKC", text or "").casefold()
    return re.sub(r"\s+", " ", text).strip()


def shingles(text, n = 5):
    if len(text) <= n:
        return {text}
    return {text[start: start + n] for start in range(len(text) - n + 1)}


def system_problem(text):
    """系统提示不能用的原因（没有问题时返回 None）：empty_system、think_tag、special_token、code_fence、meta_label。"""
    stripped = (text or "").strip()
    if not stripped:
        return "empty_system"
    if "<think>" in stripped or "</think>" in stripped:
        return "think_tag"
    if "<|im_start|>" in stripped or "<|im_end|>" in stripped or "<|endoftext|>" in stripped:
        return "special_token"
    if stripped.startswith("```"):
        return "code_fence"
    first_line = normalize(stripped.split("\n", 1)[0])
    if any(label in first_line for label in SYSTEM_PROMPT_LABELS):
        return "meta_label"
    return None


def user_problem(system, user):
    """用户消息不能用的原因（没有问题时返回 None）：empty_user（为空或只有空白）、copied_user（抄写系统提示）。"""
    user_norm = normalize(user)
    if not user_norm:
        return "empty_user"
    if len(user_norm) < MIN_COPY_CHARS:
        return None
    system_norm = normalize(system)
    if user_norm in system_norm:
        return "copied_user"
    user_grams = shingles(text = user_norm)
    if len(user_grams & shingles(text = system_norm)) / len(user_grams) >= COPY_SHARE:
        return "copied_user"
    return None
