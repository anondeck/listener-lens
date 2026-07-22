#!/usr/bin/env python3
"""Per-language data behind the full listener-lens matrix.

Three tables, each keyed by locale, each O(N) in the number of languages:

  PHONEMIC_INVENTORY  the contrastive system, used to derive which of a
                      source language's categories collapse onto a listener's.
  INVENTORY_WORDS     a coverage word list per language. Running espeak over
                      it reports the *surface* symbols that locale can emit,
                      which is what the Azure IPA map has to cover — espeak
                      emits allophones the phonemic inventory does not list
                      (Spanish /b d ɡ/ surface as β ð ɣ).
  LISTENER_*          how a listener language reshapes what it hears. Keyed by
                      listener rather than by pair: an English listener files
                      a trill as /ɹ/ whether it arrived from Italian, Spanish
                      or Russian, so the correction is stated once and every
                      source composes against it.

Keying corrections by listener is what keeps a 20-language menu tractable:
380 ordered directions are generated from 20 correction blocks, not authored
380 times. Where a correction genuinely depends on the source language, the
per-source override table carries it — English hears Spanish /x/ as /h/
(jalapeño) but German /x/ as /k/ (Bach -> back), and that distinction is a
real property of the pair, not an inconsistency to be smoothed away.

Build-time only. The committed JSON tables are what the lane reads.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Contrastive inventories.
#
# Standard, uncontested descriptions of each system. Scraping these from
# espeak does not work: espeak emits surface allophones, so a scraped Spanish
# inventory lacks /b d ɡ/ entirely and the deriver concludes Spanish devoices
# English stops. A listener rule table has to compare phoneme systems.
# --------------------------------------------------------------------------
PHONEMIC_INVENTORY: dict[str, dict[str, list[str]]] = {
    "en-US": {
        "vowels": ["i", "ɪ", "eɪ", "ɛ", "æ", "ɑ", "ɔ", "oʊ", "ʊ", "u", "ʌ",
                   "ɜ", "ə", "aɪ", "aʊ", "ɔɪ"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "tʃ", "dʒ", "f", "v",
                       "θ", "ð", "s", "z", "ʃ", "ʒ", "h", "m", "n", "ŋ",
                       "l", "ɹ", "j", "w"],
    },
    # Seven oral vowels plus the five contrastive nasals, and the palatalised
    # /tʃ dʒ/ that ti/di surface as.
    "pt-BR": {
        "vowels": ["a", "e", "ɛ", "i", "o", "ɔ", "u",
                   "ɐ̃", "ẽ", "ĩ", "õ", "ũ"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "tʃ", "dʒ", "f", "v",
                       "s", "z", "ʃ", "ʒ", "m", "n", "ɲ", "l", "ʎ", "ɾ",
                       "ʁ", "j", "w"],
    },
    # European Spanish: keeps /θ/ (distinción) and conservative /ʎ/.
    "es-ES": {
        "vowels": ["a", "e", "i", "o", "u"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "tʃ", "f", "θ", "s",
                       "x", "m", "n", "ɲ", "l", "ʎ", "ɾ", "r", "j", "w"],
    },
    # Mexican Spanish: seseo (no /θ/) and yeísmo (no /ʎ/) — the majority
    # system. Differs from es-ES by exactly those two categories.
    "es-MX": {
        "vowels": ["a", "e", "i", "o", "u"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "tʃ", "f", "s",
                       "x", "m", "n", "ɲ", "l", "ɾ", "r", "j", "w"],
    },
    "fr-FR": {
        "vowels": ["i", "e", "ɛ", "a", "ɔ", "o", "u", "y", "ø", "œ", "ə",
                   "ɛ̃", "ɑ̃", "ɔ̃"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z",
                       "ʃ", "ʒ", "m", "n", "ɲ", "l", "ʁ", "j", "w"],
    },
    "it-IT": {
        "vowels": ["i", "e", "ɛ", "a", "ɔ", "o", "u"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "tʃ", "dʒ", "ts", "dz",
                       "f", "v", "s", "z", "ʃ", "m", "n", "ɲ", "l", "ʎ",
                       "r", "j", "w"],
    },
    "de-DE": {
        "vowels": ["i", "ɪ", "e", "ɛ", "a", "ɔ", "o", "ʊ", "u", "y", "ʏ",
                   "ø", "œ", "ə", "aɪ", "aʊ", "ɔʏ"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z",
                       "ʃ", "ʒ", "ç", "x", "h", "m", "n", "ŋ", "l", "ʁ",
                       "j", "ts", "pf"],
    },
    # Five vowels; owns θ ð x ɣ and lacks ʃ tʃ, which makes it unusually
    # productive as a lens in both directions.
    "el-GR": {
        "vowels": ["a", "e", "i", "o", "u"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð",
                       "s", "z", "x", "ɣ", "ç", "ʝ", "m", "n", "l", "r"],
    },
    # The palatalised series is contrastive, not allophonic, so it is listed
    # as its own set of phonemes.
    "ru-RU": {
        "vowels": ["a", "e", "i", "o", "u", "ɨ"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z",
                       "ʂ", "ʐ", "x", "ts", "tɕ", "ɕ", "m", "n", "l", "r",
                       "j", "pʲ", "bʲ", "tʲ", "dʲ", "fʲ", "vʲ", "sʲ", "zʲ",
                       "mʲ", "nʲ", "lʲ", "rʲ"],
    },
    # Four-way stop contrast (voice x aspiration) and a retroflex series.
    "hi-IN": {
        "vowels": ["ə", "aː", "ɪ", "iː", "ʊ", "uː", "eː", "ɛː", "oː", "ɔː"],
        "consonants": ["p", "pʰ", "b", "bʱ", "t̪", "t̪ʰ", "d̪", "d̪ʱ",
                       "ʈ", "ʈʰ", "ɖ", "ɖʱ", "k", "kʰ", "ɡ", "ɡʱ",
                       "tʃ", "tʃʰ", "dʒ", "dʒʱ", "s", "ʃ", "h",
                       "m", "n", "ɳ", "l", "r", "ɽ", "ʋ", "j"],
    },
    "nl-NL": {
        "vowels": ["ɑ", "aː", "ɛ", "eː", "ɪ", "i", "ɔ", "oː", "ʏ", "yː",
                   "u", "øː", "ə", "ɛi", "œy", "ɑu"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z",
                       "x", "ɣ", "h", "m", "n", "ŋ", "l", "r", "ʋ", "j"],
    },
    # The ɕ/ʑ, ʂ/ʐ, tɕ/dʑ series is the signature lens target.
    "pl-PL": {
        "vowels": ["a", "ɛ", "i", "ɨ", "ɔ", "u", "ɛ̃", "ɔ̃"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z",
                       "ʂ", "ʐ", "ɕ", "ʑ", "x", "ts", "dz", "tʂ", "dʐ",
                       "tɕ", "dʑ", "m", "n", "ɲ", "l", "r", "j", "w"],
    },
    # Vowel harmony with front rounded y/ø and back unrounded ɯ.
    "tr-TR": {
        "vowels": ["a", "e", "ɯ", "i", "o", "ø", "u", "y"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "tʃ", "dʒ", "f", "v",
                       "s", "z", "ʃ", "ʒ", "h", "m", "n", "l", "ɾ", "j"],
    },
    "sv-SE": {
        "vowels": ["ɑː", "a", "eː", "ɛ", "iː", "ɪ", "uː", "ʊ", "yː", "ʏ",
                   "øː", "œ", "ʉː", "ɵ"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "ɧ",
                       "ɕ", "h", "m", "n", "ŋ", "l", "r", "j"],
    },
    "uk-UA": {
        "vowels": ["a", "ɛ", "i", "ɪ", "ɔ", "u"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z",
                       "ʃ", "ʒ", "x", "ɦ", "ts", "tʃ", "m", "n", "ɲ",
                       "l", "r", "j"],
    },
    "id-ID": {
        "vowels": ["a", "e", "i", "o", "u", "ə"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "tʃ", "dʒ", "f", "s",
                       "z", "ʃ", "x", "h", "m", "n", "ɲ", "ŋ", "l", "r",
                       "j", "w"],
    },
    "cs-CZ": {
        "vowels": ["a", "aː", "ɛ", "ɛː", "i", "iː", "o", "oː", "u", "uː"],
        "consonants": ["p", "b", "t", "d", "c", "ɟ", "k", "ɡ", "f", "v",
                       "s", "z", "ʃ", "ʒ", "x", "ɦ", "ts", "tʃ", "r̝",
                       "m", "n", "ɲ", "l", "r", "j"],
    },
    "ro-RO": {
        "vowels": ["a", "e", "i", "o", "u", "ə", "ɨ"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z",
                       "ʃ", "ʒ", "h", "ts", "tʃ", "dʒ", "m", "n", "l",
                       "r", "j", "w"],
    },
    "hu-HU": {
        "vowels": ["ɒ", "aː", "ɛ", "eː", "i", "iː", "o", "oː", "ø", "øː",
                   "u", "uː", "y", "yː"],
        "consonants": ["p", "b", "t", "d", "c", "ɟ", "k", "ɡ", "f", "v",
                       "s", "z", "ʃ", "ʒ", "h", "ts", "tʃ", "dz", "dʒ",
                       "m", "n", "ɲ", "l", "r", "j"],
    },
    "nb-NO": {
        "vowels": ["ɑ", "æ", "e", "ɛ", "i", "ɪ", "o", "ɔ", "u", "ʉ",
                   "y", "ø", "œ"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "ʂ",
                       "ç", "h", "m", "n", "ŋ", "l", "r", "j"],
    },
    # European Portuguese: the reduced unstressed system (ɨ, ɐ) and the
    # uvular rhotic are what separate it from pt-BR, and it does not do
    # BP-style epenthesis — so it needs its own structural block, not a
    # copy of the Brazilian one.
    "pt-PT": {
        "vowels": ["i", "e", "ɛ", "a", "ɐ", "ɔ", "o", "u", "ɨ",
                   "ɐ̃", "ẽ", "ĩ", "õ", "ũ"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z",
                       "ʃ", "ʒ", "m", "n", "ɲ", "l", "ʎ", "ɾ", "ʁ"],
    },
    "ca-ES": {
        "vowels": ["a", "e", "ɛ", "i", "o", "ɔ", "u", "ə"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z",
                       "ʃ", "ʒ", "tʃ", "dʒ", "m", "n", "ɲ", "ŋ", "l",
                       "ʎ", "ɾ", "r", "j", "w"],
    },
    "hr-HR": {
        "vowels": ["a", "e", "i", "o", "u"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z",
                       "ʃ", "ʒ", "x", "ts", "tʃ", "dʒ", "tɕ", "dʑ",
                       "m", "n", "ɲ", "l", "ʎ", "r", "j"],
    },
    "sk-SK": {
        "vowels": ["a", "aː", "ɛ", "ɛː", "i", "iː", "o", "oː", "u", "uː"],
        "consonants": ["p", "b", "t", "d", "c", "ɟ", "k", "ɡ", "f", "v",
                       "s", "z", "ʃ", "ʒ", "x", "ɦ", "ts", "dz", "tʃ",
                       "dʒ", "m", "n", "ɲ", "l", "ʎ", "r", "j"],
    },
    "sl-SI": {
        "vowels": ["a", "e", "ɛ", "i", "o", "ɔ", "u", "ə"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z",
                       "ʃ", "ʒ", "x", "ts", "tʃ", "dʒ", "m", "n", "l",
                       "r", "j", "ʋ"],
    },
    "bg-BG": {
        "vowels": ["a", "ɛ", "i", "ɔ", "u", "ɤ"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z",
                       "ʃ", "ʒ", "x", "ts", "tʃ", "dʒ", "m", "n", "l",
                       "r", "j"],
    },
    "ms-MY": {
        "vowels": ["a", "e", "i", "o", "u", "ə"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "tʃ", "dʒ", "f", "s",
                       "z", "ʃ", "x", "h", "m", "n", "ɲ", "ŋ", "l", "r",
                       "j", "w"],
    },
    "mr-IN": {
        "vowels": ["ə", "aː", "ɪ", "iː", "ʊ", "uː", "eː", "oː", "ɔː"],
        "consonants": ["p", "pʰ", "b", "bʱ", "t̪", "t̪ʰ", "d̪", "d̪ʱ",
                       "ʈ", "ʈʰ", "ɖ", "ɖʱ", "k", "kʰ", "ɡ", "ɡʱ",
                       "ts", "dz", "tʃ", "dʒ", "s", "ʃ", "ʂ", "h",
                       "m", "n", "ɳ", "l", "ɭ", "r", "ʋ", "j"],
    },
    "te-IN": {
        "vowels": ["a", "aː", "i", "iː", "u", "uː", "e", "eː", "o", "oː"],
        "consonants": ["p", "pʰ", "b", "bʱ", "t̪", "t̪ʰ", "d̪", "d̪ʱ",
                       "ʈ", "ʈʰ", "ɖ", "ɖʱ", "k", "kʰ", "ɡ", "ɡʱ",
                       "tʃ", "dʒ", "s", "ʃ", "ʂ", "h", "m", "n", "ɳ",
                       "l", "ɭ", "r", "ʋ", "j"],
    },
    "gu-IN": {
        "vowels": ["ə", "aː", "ɪ", "iː", "ʊ", "uː", "eː", "ɛ", "oː", "ɔ"],
        "consonants": ["p", "pʰ", "b", "bʱ", "t̪", "t̪ʰ", "d̪", "d̪ʱ",
                       "ʈ", "ʈʰ", "ɖ", "ɖʱ", "k", "kʰ", "ɡ", "ɡʱ",
                       "tʃ", "dʒ", "s", "ʃ", "ʂ", "h", "m", "n", "ɳ",
                       "l", "ɭ", "r", "ʋ", "j"],
    },
}

# --------------------------------------------------------------------------
# Coverage word lists.
#
# Chosen to exercise every contrast in the inventory above plus the surface
# allophones espeak reports, so the discovered symbol set is what the Azure
# map actually has to cover. Not used to derive rules.
# --------------------------------------------------------------------------
INVENTORY_WORDS: dict[str, str] = {
    "en-US": (
        "beat bit bait bet bat bot bought boat book boot but bird about "
        "pat bad tack dog cat gap fat vat thin then sat zap ship measure "
        "chat jam mat nap sing lap rat yes wet hat house buy boy now"
    ),
    "pt-BR": (
        "pai mãe gato dedo casa fogo gente filho chave ano caro carro "
        "cinco sapato céu jogo sol luz mão nada peso vinho bom água "
        "terra cadeira ninho olho peixe tia dia irmã põe muito"
    ),
    "es-ES": (
        "padre madre gato dedo casa fuego jamón hijo llave año caro carro "
        "cinco zapato chico mucho sol luz mano nada peso vino bueno agua "
        "tierra silla queso guerra piso puerta"
    ),
    "es-MX": (
        "padre madre gato dedo casa fuego jamón hijo llave año caro carro "
        "cinco zapato chico mucho sol luz mano nada peso vino bueno agua "
        "tierra silla queso guerra piso puerta"
    ),
    "fr-FR": (
        "papa bébé table donner chat gare face vase sac zéro chien jaune "
        "manger nous lit rue oui huile pain bon brun vin peu peur beau "
        "port pur lune fille agneau heure homme quatre film"
    ),
    "it-IT": (
        "padre madre gatto dito casa fuoco gente figlio chiave anno caro "
        "carro cinque zappa cielo gioco sole luce mano nulla peso vino "
        "buono acqua terra sedia gnomo aglio pesce"
    ),
    "de-DE": (
        "Vater Mutter Tag Kind Gast Fisch Wasser Sohn Zeit Katze Bach ich "
        "Buch Loch machen Menschen nein Ring lang rot Haus Bein Leute "
        "schön Bücher Käse Meer bitte Hand Pfand Apfel"
    ),
    "el-GR": (
        "καλημέρα ευχαριστώ σπίτι νερό ψωμί θάλασσα δρόμος γάτα παιδί "
        "μητέρα πατέρας χέρι γιατρός φως ζωή μέρα νύχτα άνθρωπος καρδιά "
        "βιβλίο τραπέζι κόκκινο μεγάλο μικρό"
    ),
    "ru-RU": (
        "привет спасибо хорошо дом мама папа вода хлеб школа книга "
        "человек город время рука сердце белый чёрный большой маленький "
        "щука чай цирк жизнь шапка юг яблоко день тень мать"
    ),
    "hi-IN": (
        "नमस्ते धन्यवाद अच्छा घर माता पिता पानी रोटी किताब स्कूल "
        "आदमी शहर समय हाथ दिल सफेद काला बड़ा छोटा खाना पढ़ना "
        "भाई बहन दूध फूल थाली ठंडा डाल गाना"
    ),
    "nl-NL": (
        "goedemorgen dank je het huis moeder vader water brood school boek "
        "mens stad tijd hand hart wit zwart groot klein nieuw meisje "
        "jongen vuur zeven acht negen leuk graag zien"
    ),
    "pl-PL": (
        "dzień dobry dziękuję dom matka ojciec woda chleb szkoła książka "
        "człowiek miasto czas ręka serce biały czarny duży mały czysty "
        "żaba świat cień szczyt gęś wąż jeść pięć"
    ),
    "tr-TR": (
        "günaydın teşekkürler ev anne baba su ekmek okul kitap insan "
        "şehir zaman el kalp beyaz siyah büyük küçük güzel yol "
        "göz kulak dil yıldız çiçek ağaç yağmur"
    ),
    "sv-SE": (
        "god morgon tack huset mamma pappa vatten bröd skola bok "
        "människa stad tid hand hjärta vit svart stor liten ny "
        "sjuk kjol tjugo hus mjölk fjäll ängel yngre"
    ),
    "uk-UA": (
        "доброго ранку дякую дім мати батько вода хліб школа книга "
        "людина місто час рука серце білий чорний великий малий "
        "щука джміль ґанок їжа юнак яблуко ліс день"
    ),
    "id-ID": (
        "selamat pagi terima kasih rumah ibu ayah air roti sekolah "
        "buku orang kota waktu tangan hati putih hitam besar kecil "
        "baru nyanyi ngengat syukur cinta jalan pohon"
    ),
    "cs-CZ": (
        "dobré ráno děkuji dům matka otec voda chléb škola kniha "
        "člověk město čas ruka srdce bílý černý velký malý "
        "řeka žena ještě ťukat ďábel nůž pět"
    ),
    "ro-RO": (
        "bună dimineața mulțumesc casă mamă tată apă pâine școală carte "
        "om oraș timp mână inimă alb negru mare mic nou "
        "ceas geam jos știu împărat ghiocel chiar"
    ),
    "hu-HU": (
        "jó reggelt köszönöm ház anya apa víz kenyér iskola könyv "
        "ember város idő kéz szív fehér fekete nagy kicsi új "
        "gyerek tyúk cukor dzsem lány zöld tűz"
    ),
    "nb-NO": (
        "god morgen takk huset mor far vann brød skole bok "
        "menneske by tid hånd hjerte hvit svart stor liten ny "
        "kjøre skje gjøre sjø ung øye kylling"
    ),
    "pt-PT": (
        "bom dia obrigado casa mãe pai água pão escola livro "
        "homem cidade tempo mão coração branco preto grande pequeno "
        "chave filho velho carro terra olho peixe rua ponte"
    ),
    "ca-ES": (
        "bon dia gràcies casa mare pare aigua pa escola llibre "
        "home ciutat temps mà cor blanc negre gran petit "
        "clau fill vell cotxe terra ull peix carrer pont"
    ),
    "hr-HR": (
        "dobro jutro hvala kuća majka otac voda kruh škola knjiga "
        "čovjek grad vrijeme ruka srce bijel crn velik malen "
        "ključ sin star đak zemlja oko riba ulica most"
    ),
    "sk-SK": (
        "dobré ráno ďakujem dom matka otec voda chlieb škola kniha "
        "človek mesto čas ruka srdce biely čierny veľký malý "
        "kľúč syn starý džem zem oko ryba ulica most"
    ),
    "sl-SI": (
        "dobro jutro hvala hiša mati oče voda kruh šola knjiga "
        "človek mesto čas roka srce bel črn velik majhen "
        "ključ sin star džez zemlja oko riba ulica most"
    ),
    "bg-BG": (
        "добро утро благодаря къща майка баща вода хляб училище книга "
        "човек град време ръка сърце бял черен голям малък "
        "ключ син стар джоб земя око риба улица мост"
    ),
    "ms-MY": (
        "selamat pagi terima kasih rumah ibu ayah air roti sekolah "
        "buku orang kota waktu tangan hati putih hitam besar kecil "
        "baru nyanyi ngengat syukur cinta jalan pokok"
    ),
    "mr-IN": (
        "सुप्रभात धन्यवाद घर आई वडील पाणी भाकरी शाळा पुस्तक "
        "माणूस शहर वेळ हात हृदय पांढरा काळा मोठा लहान "
        "भाऊ बहीण दूध फूल थाळी थंड डाळ गाणे"
    ),
    "te-IN": (
        "శుభోదయం ధన్యవాదాలు ఇల్లు అమ్మ నాన్న నీరు రొట్టె పాఠశాల పుస్తకం "
        "మనిషి నగరం సమయం చేయి గుండె తెలుపు నలుపు పెద్ద చిన్న "
        "అన్న అక్క పాలు పువ్వు ఆకు చల్లని పాట"
    ),
    "gu-IN": (
        "સુપ્રભાત આભાર ઘર માતા પિતા પાણી રોટલી શાળા પુસ્તક "
        "માણસ શહેર સમય હાથ હૃદય સફેદ કાળો મોટો નાનો "
        "ભાઈ બહેન દૂધ ફૂલ થાળી ઠંડું ડાળ ગીત"
    ),
}

# espeak-ng language id per locale. Kept here so locale ids stay Azure-shaped
# everywhere else in the lane.
ESPEAK_LANGUAGE: dict[str, str] = {
    "en-US": "en-us",
    "pt-BR": "pt-br",
    "es-ES": "es",
    "es-MX": "es-419",
    "fr-FR": "fr-fr",
    "it-IT": "it",
    "de-DE": "de",
    "el-GR": "el",
    "ru-RU": "ru",
    "hi-IN": "hi",
    "nl-NL": "nl",
    "pl-PL": "pl",
    "tr-TR": "tr",
    "sv-SE": "sv",
    "uk-UA": "uk",
    "id-ID": "id",
    "cs-CZ": "cs",
    "ro-RO": "ro",
    "hu-HU": "hu",
    "nb-NO": "nb",
    "pt-PT": "pt",
    "ca-ES": "ca",
    "hr-HR": "hr",
    "sk-SK": "sk",
    "sl-SI": "sl",
    "bg-BG": "bg",
    "ms-MY": "ms",
    "mr-IN": "mr",
    "te-IN": "te",
    "gu-IN": "gu",
}

# Azure voice per locale. One voice per language keeps the per-symbol
# acceptance receipts meaningful: a receipt is only evidence for the voice it
# was taken on.
AZURE_VOICE: dict[str, str] = {
    "en-US": "en-US-AvaNeural",
    "pt-BR": "pt-BR-FranciscaNeural",
    "es-ES": "es-ES-ElviraNeural",
    "es-MX": "es-MX-DaliaNeural",
    "fr-FR": "fr-FR-DeniseNeural",
    "it-IT": "it-IT-ElsaNeural",
    "de-DE": "de-DE-KatjaNeural",
    "el-GR": "el-GR-AthinaNeural",
    "ru-RU": "ru-RU-SvetlanaNeural",
    "hi-IN": "hi-IN-SwaraNeural",
    "nl-NL": "nl-NL-FennaNeural",
    "pl-PL": "pl-PL-AgnieszkaNeural",
    "tr-TR": "tr-TR-EmelNeural",
    "sv-SE": "sv-SE-SofieNeural",
    "uk-UA": "uk-UA-PolinaNeural",
    "id-ID": "id-ID-GadisNeural",
    "cs-CZ": "cs-CZ-VlastaNeural",
    "ro-RO": "ro-RO-AlinaNeural",
    "hu-HU": "hu-HU-NoemiNeural",
    "nb-NO": "nb-NO-PernilleNeural",
    "pt-PT": "pt-PT-RaquelNeural",
    "ca-ES": "ca-ES-JoanaNeural",
    "hr-HR": "hr-HR-GabrijelaNeural",
    "sk-SK": "sk-SK-ViktoriaNeural",
    "sl-SI": "sl-SI-PetraNeural",
    "bg-BG": "bg-BG-KalinaNeural",
    "ms-MY": "ms-MY-YasminNeural",
    "mr-IN": "mr-IN-AarohiNeural",
    "te-IN": "te-IN-ShrutiNeural",
    "gu-IN": "gu-IN-DhwaniNeural",
}

# Locales whose Azure voice was verified to actually *honour* an ipa ph
# attribute: two different ph strings on the same written word must produce
# different audio. Azure fails open here — 19 otherwise-viable locales
# (Swahili, Afrikaans, Estonian, Serbian-Latin, Welsh, ...) return HTTP 200
# and byte-identical audio no matter what phonemes are requested, which would
# ship a lens that silently does nothing. Membership in this set is a
# precondition for a locale entering the menu, re-checked by the probe.
LOCALES = tuple(PHONEMIC_INVENTORY)
IPA_HONOURED = frozenset(LOCALES)

# --------------------------------------------------------------------------
# Listener corrections.
#
# Articulatory feature distance is reliable for vowels and unreliable for
# consonants: the perceptually salient assimilations are exactly the ones a
# distance threshold rejects, because they cross manner (a trill is not
# "near" an approximant in feature space, but every English listener files
# one as /ɹ/). So consonants are resolved by an ordered preference chain
# instead — the listener takes the first category in the chain that its own
# inventory actually contains.
#
# One chain per foreign category, not one per language pair. A Greek listener
# keeps /θ/ because Greek has it; a German listener takes /s/; an Italian
# listener takes /t/ — all three fall out of the same line.
# --------------------------------------------------------------------------
DELETE = "__delete__"

CORRECTION_CHAINS: dict[str, list[str]] = {
    # Rhotics. Every language files a foreign rhotic as its own.
    "ɹ": ["ɹ", "ʁ", "r", "ɾ", "r̝", "ɽ"],
    "r": ["r", "ɾ", "ʁ", "ɹ", "r̝"],
    "ɾ": ["ɾ", "r", "ʁ", "ɹ", "r̝"],
    "ʁ": ["ʁ", "r", "ɾ", "ɹ", "x"],
    "ɽ": ["ɽ", "ɾ", "r", "ɹ"],
    "r̝": ["r̝", "r", "ʒ", "ɾ", "ɹ"],
    # Dental fricatives. The classic learner substitutions.
    "θ": ["θ", "s", "t̪", "t", "f"],
    "ð": ["ð", "z", "d̪", "d", "v"],
    # Nasals and laterals without a palatal series.
    "ŋ": ["ŋ", "n"],
    "ɲ": ["ɲ", "n"],
    "ɳ": ["ɳ", "n"],
    "ʎ": ["ʎ", "l", "j"],
    "ɭ": ["ɭ", "l"],
    # Dorsal and glottal fricatives. Default sends a velar fricative to a
    # velar stop (Bach -> back); the Spanish jota is the documented exception
    # and lives in SOURCE_OVERRIDES.
    "x": ["x", "k", "h", "ʁ"],
    "ɣ": ["ɣ", "ɡ"],
    "β": ["β", "v", "b"],
    "ç": ["ç", "ʃ", "x", "h"],
    "ʝ": ["ʝ", "j", "ʒ"],
    "ɦ": ["ɦ", "h", "x"],
    # A listener with a velar fricative files /h/ there (Spanish hat -> jat);
    # one without simply does not perceive it (French haricot). The uvular
    # rhotic is deliberately NOT a fallback: French has ʁ, and offering it
    # here produced a bogus h->ʁ rule that fought the real h-deletion.
    "h": ["h", "ɦ", "x", DELETE],
    "ɧ": ["ɧ", "ʃ", "x", "h"],
    # Sibilants. The Slavic three-way ɕ/ʂ/ʃ contrast collapses for everyone
    # who does not own it.
    "ʂ": ["ʂ", "ʃ", "s"],
    "ʐ": ["ʐ", "ʒ", "z"],
    "ɕ": ["ɕ", "ʃ", "s"],
    "ʑ": ["ʑ", "ʒ", "z"],
    "ʃ": ["ʃ", "s"],
    "ʒ": ["ʒ", "ʃ", "z", "s"],
    "z": ["z", "s"],
    "v": ["v", "ʋ", "b", "w"],
    "ʋ": ["ʋ", "v", "w"],
    # Affricates.
    "ts": ["ts", "s", "t"],
    "dz": ["dz", "z", "d"],
    "tʃ": ["tʃ", "ʃ", "ts", "s"],
    "dʒ": ["dʒ", "ʒ", "dz", "z"],
    "tɕ": ["tɕ", "tʃ", "ʃ", "s"],
    "dʑ": ["dʑ", "dʒ", "ʒ", "z"],
    "tʂ": ["tʂ", "tʃ", "ʃ", "s"],
    "dʐ": ["dʐ", "dʒ", "ʒ", "z"],
    "pf": ["pf", "f"],
    # Palatal and retroflex stops.
    "c": ["c", "k", "tʃ"],
    "ɟ": ["ɟ", "ɡ", "dʒ"],
    "ʈ": ["ʈ", "t̪", "t"],
    "ɖ": ["ɖ", "d̪", "d"],
    "t̪": ["t̪", "t"],
    "d̪": ["d̪", "d"],
    # Glides.
    "w": ["w", "v", "u"],
    "ɥ": ["ɥ", "y", "w", "j"],
    # Aspirated series: languages without the contrast hear the plain stop.
    "pʰ": ["pʰ", "p", "f"],
    "t̪ʰ": ["t̪ʰ", "t̪", "t"],
    "ʈʰ": ["ʈʰ", "ʈ", "t"],
    "kʰ": ["kʰ", "k"],
    "tʃʰ": ["tʃʰ", "tʃ", "ʃ"],
    "bʱ": ["bʱ", "b"],
    "d̪ʱ": ["d̪ʱ", "d̪", "d"],
    "ɖʱ": ["ɖʱ", "ɖ", "d"],
    "ɡʱ": ["ɡʱ", "ɡ"],
    "dʒʱ": ["dʒʱ", "dʒ", "ʒ"],
    # Front rounded and central vowels — the hardest vowels to borrow.
    "y": ["y", "yː", "ʏ", "i", "iː", "u"],
    "yː": ["yː", "y", "iː", "i", "u"],
    "ʏ": ["ʏ", "y", "ɪ", "i"],
    "ø": ["ø", "øː", "œ", "e", "eː", "ɛ"],
    "øː": ["øː", "ø", "eː", "e", "ɛ"],
    "œ": ["œ", "ø", "ɛ", "e"],
    "ɨ": ["ɨ", "ɪ", "i"],
    "ɯ": ["ɯ", "u", "ɨ", "i"],
    "ʉ": ["ʉ", "ʉː", "u", "y"],
    "ʉː": ["ʉː", "ʉ", "uː", "u", "y"],
    "ɤ": ["ɤ", "o", "ə", "ʌ"],
    "ɵ": ["ɵ", "ø", "o", "ə"],
    "ə": ["ə", "ɐ", "ɨ", "e", "a"],
    "ɐ": ["ɐ", "ə", "a", "ʌ"],
    "ʌ": ["ʌ", "ɐ", "a", "ə"],
    "æ": ["æ", "ɛ", "a", "e"],
    "ɪ": ["ɪ", "i"],
    "ʊ": ["ʊ", "u"],
    "ɜ": ["ɜ", "ə", "ɛ", "e"],
    "ɔ": ["ɔ", "o"],
    "ɛ": ["ɛ", "e"],
    # Nasal vowels denasalise for listeners without a nasal series.
    "ɛ̃": ["ɛ̃", "ɛ", "e"],
    "ɑ̃": ["ɑ̃", "ɑ", "a"],
    "ɔ̃": ["ɔ̃", "ɔ", "o"],
    "ɐ̃": ["ɐ̃", "ɐ", "a", "ʌ"],
    "ẽ": ["ẽ", "e", "ɛ"],
    "ĩ": ["ĩ", "i"],
    "õ": ["õ", "o", "ɔ"],
    "ũ": ["ũ", "u"],
    # Long vowels shorten for listeners without contrastive length.
    "aː": ["aː", "a", "ɑ"],
    "iː": ["iː", "i"],
    "uː": ["uː", "u"],
    "eː": ["eː", "e", "ɛ"],
    "oː": ["oː", "o", "ɔ"],
    "ɛː": ["ɛː", "ɛ", "e"],
    "ɔː": ["ɔː", "ɔ", "o"],
    "ɑː": ["ɑː", "ɑ", "a"],
}

# Palatalised Russian consonants: a listener without the contrast hears the
# plain consonant. Generated rather than listed, since the chain is always
# "strip the ʲ".
for _base in ("p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z", "m", "n",
              "l", "r"):
    CORRECTION_CHAINS.setdefault(f"{_base}ʲ", [f"{_base}ʲ", _base])

# Corrections that genuinely depend on which language the sound arrived from,
# not only on who is listening. Keyed (listener, source).
SOURCE_OVERRIDES: dict[tuple[str, str], dict[str, str]] = {
    # English spells the Spanish jota with an h (jalapeño -> halapenyo) but
    # the German ach-laut with a k (Bach -> back). Same symbol, same
    # listener, different source — a real property of the pair.
    ("en-US", "es-ES"): {"x": "h"},
    ("en-US", "es-MX"): {"x": "h"},
    # The b/v merger is pan-Spanish and not predicted by any feature vector.
    ("es-ES", "en-US"): {"v": "b"},
    ("es-MX", "en-US"): {"v": "b"},
}

# --------------------------------------------------------------------------
# Listener phonotactics and prosody.
#
# Unlike the segment chains above, these are NOT universally instantiable.
# A phonotactic rule exists only where the listener language actually has a
# repair process, and a stress rule only where stress placement is fixed.
# Russian stress is lexically mobile; Greek has no epenthesis; Hungarian has
# no final devoicing. Inventing a rule for those languages so every profile
# looks equally full would manufacture evidence, which the lane's evidence
# policy forbids — an absent entry here means "this language does not do
# this", not "not written yet".
#
# operations:
#   insert_before / insert_after  epenthesis, prothesis, paragoge
#   delete                        a category the listener does not perceive
#   final_devoicing               context-gated substitution at word edge
# --------------------------------------------------------------------------
LISTENER_PHONOTACTICS: dict[str, list[dict[str, object]]] = {
    # Words may not end in a consonant, so one is added.
    "it-IT": [{"op": "insert_after", "target": "e",
               "contexts": ["any_word_final_consonant"],
               "note": "Italian words are vowel-final; a final consonant "
                       "takes a paragogic vowel (cat -> ˈkete)."},
              {"op": "delete", "source": "h", "contexts": ["any"],
               "note": "Italian orthographic h is silent; the category is "
                       "not perceived at all."}],
    # /s/+consonant is an illegal onset, so a vowel is prefixed.
    "es-ES": [{"op": "insert_before", "target": "e",
               "contexts": ["word_initial_cluster"], "onsets": ["s"],
               "note": "Spanish prothesis: school -> eskˈul, study -> estˈadi."}],
    "es-MX": [{"op": "insert_before", "target": "e",
               "contexts": ["word_initial_cluster"], "onsets": ["s"],
               "note": "Spanish prothesis: school -> eskˈul, study -> estˈadi."}],
    # Same repair, and famously the source of "iskuul" for school.
    "hi-IN": [{"op": "insert_before", "target": "ɪ",
               "contexts": ["word_initial_cluster"], "onsets": ["s"],
               "note": "Indic prothesis before an /s/ cluster (school -> ɪskuːl)."}],
    "mr-IN": [{"op": "insert_before", "target": "ɪ",
               "contexts": ["word_initial_cluster"], "onsets": ["s"],
               "note": "Indic prothesis before an /s/ cluster."}],
    "gu-IN": [{"op": "insert_before", "target": "ɪ",
               "contexts": ["word_initial_cluster"], "onsets": ["s"],
               "note": "Indic prothesis before an /s/ cluster."}],
    # Dravidian words are vowel-final.
    "te-IN": [{"op": "insert_after", "target": "u",
               "contexts": ["any_word_final_consonant"],
               "note": "Telugu words end in a vowel; a final consonant takes "
                       "an enunciative /u/."}],
    # Every vowel-initial word begins with a glottal stop.
    "de-DE": [{"op": "insert_before", "target": "ʔ",
               "contexts": ["word_initial_vowel"],
               "note": "German glottal onset: apple -> ˈʔepᵊl, An -> ʔɐn."}],
    # h is not a category; the letter is silent.
    "fr-FR": [{"op": "delete", "source": "h", "contexts": ["any"],
               "note": "French h is silent (haricot -> aʁiko); listeners do "
                       "not perceive the segment."}],
    "ca-ES": [{"op": "delete", "source": "h", "contexts": ["any"],
               "note": "Catalan h is silent."}],
    "pt-BR": [{"op": "delete", "source": "h", "contexts": ["any"],
               "note": "Portuguese h is silent."}],
    "pt-PT": [{"op": "delete", "source": "h", "contexts": ["any"],
               "note": "Portuguese h is silent."}],
    # English may not begin a word with these clusters and drops the first
    # element (psychology -> saɪ, Zeit -> saɪt).
    "en-US": [{"op": "delete", "source": "p", "contexts": ["word_initial_cluster"],
               "followed_by": ["s", "f"],
               "note": "English resolves an illegal /ps/ or /pf/ onset by "
                       "dropping the stop (psychology, Pfand)."},
              {"op": "delete", "source": "t", "contexts": ["word_initial_cluster"],
               "followed_by": ["s"],
               "note": "English resolves an illegal /ts/ onset by dropping "
                       "the stop (Zeit -> saɪt)."}],
    # Word-final obstruents lose voicing. Very audible and well attested
    # across these families.
    **{
        locale: [{"op": "final_devoicing",
                  "note": "Word-final obstruents are devoiced in this "
                          "language, so a foreign final voiced obstruent is "
                          "heard as its voiceless counterpart."}]
        for locale in ("nl-NL", "ru-RU", "uk-UA", "pl-PL", "cs-CZ", "sk-SK",
                       "tr-TR", "bg-BG", "hr-HR", "sl-SI")
    },
}

# Fixed stress placement, where the language has one. Russian and Greek and
# the Indic languages are omitted deliberately: their stress is lexical or
# mobile, so there is no bias to apply.
LISTENER_STRESS: dict[str, str] = {
    "es-ES": "penultimate", "es-MX": "penultimate", "it-IT": "penultimate",
    "pt-BR": "penultimate", "pt-PT": "penultimate", "ca-ES": "penultimate",
    "pl-PL": "penultimate",
    "de-DE": "initial", "nl-NL": "initial", "cs-CZ": "initial",
    "sk-SK": "initial", "hu-HU": "initial", "sv-SE": "initial",
    "nb-NO": "initial",
    "fr-FR": "final", "tr-TR": "final",
}


