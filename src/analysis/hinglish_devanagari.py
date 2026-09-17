"""Romanised Hinglish -> Devanagari transliteration for the MuRIL script test.

Why this is hand-built
----------------------
AI4Bharat IndicXlit, the trained transliteration model, cannot be installed here:
it requires fairseq, which does not build on Python 3.12. A purely rule-based
ITRANS mapping is too lossy for informal romanisation, because informal spelling
does not mark vowel length or retroflexion:

    bukhar -> बुखर   (correct: बुखार)      pet -> पेत  (correct: पेट)
    khansi -> खन्सि  (correct: खांसी)       meri -> मेरि (correct: मेरी)

So transliteration is done in two stages:

1.  A hand-verified lexicon of the most frequent romanised Hindi tokens in the
    MMCQS queries. The 260 entries below cover ~83% of all token occurrences
    that an English vocabulary does not recognise.
2.  Rule-based ITRANS with word-final schwa restoration for the remaining ~17%.

English tokens inside the code-mixed query ("skin", "doctor", "rash") are left
in Latin script: MuRIL reads Latin-script English natively, and transliterating
them would corrupt content the encoder can already use.

The mapping is reported in the paper as hand-verified-plus-rules, not as a
trained transliterator, and `coverage()` reports the share actually covered by
stage 1 so the claim can be stated exactly.
"""

from __future__ import annotations

import re
from functools import lru_cache

VIRAMA = "्"

#: Hand-verified romanised -> Devanagari mappings, ordered by corpus frequency.
LEXICON: dict[str, str] = {
    "hai": "है", "aur": "और", "ke": "के", "mujhe": "मुझे", "mein": "में",
    "nahi": "नहीं", "nahin": "नहीं", "mere": "मेरे", "ek": "एक", "kya": "क्या",
    "hoon": "हूँ", "raha": "रहा", "kuch": "कुछ", "lekin": "लेकिन", "gaya": "गया",
    "meri": "मेरी", "bahut": "बहुत", "kar": "कर", "koi": "कोई", "neeche": "नीचे",
    "gayi": "गई", "dard": "दर्द", "rahi": "रही", "kiya": "किया", "liye": "लिए",
    "saal": "साल", "pehle": "पहले", "hua": "हुआ", "wo": "वो", "din": "दिन",
    "tak": "तक", "gaye": "गए", "kripya": "कृपया", "jab": "जब", "baad": "बाद",
    "sakta": "सकता", "paas": "पास", "uske": "उसके", "dekhiye": "देखिए",
    "kabhi": "कभी", "apne": "अपने", "abhi": "अभी", "toh": "तो", "to": "तो",
    "hota": "होता", "vartaman": "वर्तमान", "hoti": "होती", "karne": "करने",
    "sujan": "सूजन", "uski": "उसकी", "chahiye": "चाहिए", "khujli": "खुजली",
    "hafte": "हफ्ते", "dekhein": "देखें", "mahine": "महीने", "woh": "वह",
    "thoda": "थोड़ा", "mera": "मेरा", "rahe": "रहे", "lagta": "लगता",
    "hui": "हुई", "suj": "सूज", "kam": "कम", "phir": "फिर", "jaise": "जैसे",
    "shuru": "शुरू", "diya": "दिया", "wajah": "वजह", "sirf": "सिर्फ",
    "pata": "पता", "chote": "छोटे", "usko": "उसको", "saath": "साथ",
    "tarah": "तरह", "hone": "होने", "unhone": "उन्होंने", "karna": "करना",
    "laal": "लाल", "kaha": "कहा", "sakte": "सकते", "taraf": "तरफ",
    "lagbhag": "लगभग", "uska": "उसका", "karta": "करता", "aaj": "आज",
    "zyada": "ज़्यादा", "jyada": "ज्यादा", "madad": "मदद", "haath": "हाथ",
    "kisi": "किसी", "upar": "ऊपर", "samay": "समय", "raat": "रात",
    "iske": "इसके", "iska": "इसका", "kyunki": "क्योंकि", "namaste": "नमस्ते",
    "gardan": "गर्दन", "apni": "अपनी", "pichle": "पिछले", "usne": "उसने",
    "sakti": "सकती", "jata": "जाता", "karti": "करती", "chinta": "चिंता",
    "aankhon": "आँखों", "unhe": "उन्हें", "kal": "कल", "dono": "दोनों",
    "andar": "अंदर", "ganth": "गांठ", "gaanth": "गांठ", "diye": "दिए",
    "liya": "लिया", "chehre": "चेहरे", "laga": "लगा", "isliye": "इसलिए",
    "karan": "कारण", "wala": "वाला", "theek": "ठीक", "thik": "ठीक",
    "muh": "मुँह", "munh": "मुँह", "jagah": "जगह", "aas": "आस",
    "aankh": "आँख", "aankhein": "आँखें", "wale": "वाले", "teen": "तीन",
    "mehsoos": "महसूस", "subah": "सुबह", "samasya": "समस्या", "beti": "बेटी",
    "gale": "गले", "haal": "हाल", "hote": "होते", "karte": "करते",
    "hum": "हम", "kyun": "क्यों", "sab": "सब", "har": "हर",
    "hamesha": "हमेशा", "badh": "बढ़", "shayad": "शायद", "kaam": "काम",
    "isse": "इससे", "ghante": "घंटे", "dhanyavaad": "धन्यवाद",
    "dhanyavad": "धन्यवाद", "jaisa": "जैसा", "beech": "बीच", "unke": "उनके",
    "ja": "जा", "thodi": "थोड़ी", "aisa": "ऐसा", "bahar": "बाहर",
    "dekha": "देखा", "chala": "चला", "sharir": "शरीर", "karein": "करें",
    "cheez": "चीज़", "bada": "बड़ा", "baare": "बारे", "jati": "जाती",
    "jaati": "जाती", "bataya": "बताया", "dekhe": "देखे", "gayab": "गायब",
    "wapas": "वापस", "iski": "इसकी", "alag": "अलग", "aapko": "आपको",
    "pareshan": "परेशान", "piche": "पीछे", "peeche": "पीछे", "dikh": "दिख",
    "dusre": "दूसरे", "salah": "सलाह", "humne": "हमने", "lagi": "लगी",
    "kiye": "किए", "bilkul": "बिल्कुल", "bukhar": "बुखार", "daane": "दाने",
    "naak": "नाक", "kaise": "कैसे", "istemal": "इस्तेमाल", "jahan": "जहाँ",
    "twacha": "त्वचा", "hisse": "हिस्से", "aaya": "आया", "wali": "वाली",
    "chota": "छोटा", "unki": "उनकी", "kaafi": "काफी", "jisme": "जिसमें",
    "lena": "लेना", "usse": "उससे", "baat": "बात", "saare": "सारे",
    "isko": "इसको", "rang": "रंग", "dino": "दिनों", "lene": "लेने",
    "peeth": "पीठ", "dheere": "धीरे", "dekh": "देख", "karwaya": "करवाया",
    "koshish": "कोशिश", "gaal": "गाल", "kaan": "कान", "samajh": "समझ",
    "garam": "गरम", "jiske": "जिसके", "jate": "जाते", "jaate": "जाते",
    "chal": "चल", "bura": "बुरा", "wahan": "वहाँ", "ajeeb": "अजीब",
    "khoon": "खून", "wahi": "वही", "dikhta": "दिखता", "bohot": "बहुत",
    "baal": "बाल", "sahi": "सही", "bade": "बड़े", "paon": "पाँव",
    "apna": "अपना", "jisse": "जिससे", "asar": "असर", "tasveer": "तस्वीर",
    "poore": "पूरे", "daag": "दाग", "bete": "बेटे", "khana": "खाना",
    "jaata": "जाता", "ghar": "घर", "kharab": "खराब", "jise": "जिसे",
    "lakshan": "लक्षण", "sujhav": "सुझाव", "badi": "बड़ी", "khane": "खाने",
    "sath": "साथ", "safed": "सफेद", "jaldi": "जल्दी", "roz": "रोज़",
    "puri": "पूरी", "chot": "चोट", "lagti": "लगती", "aata": "आता",
    "saaf": "साफ", "dawai": "दवाई", "saans": "साँस", "dawa": "दवा",
    "daant": "दाँत", "pairon": "पैरों", "dikhai": "दिखाई", "hoga": "होगा",
    "nikal": "निकल", "dar": "डर", "khansi": "खांसी", "sar": "सर",
    "pet": "पेट", "seene": "सीने", "seena": "सीना", "ulti": "उल्टी",
    "kamzori": "कमज़ोरी",
    # Hindi function words that an English wordlist also contains ("ko", "se",
    # "tha", "hain", "ki"). Without explicit entries these stay in Latin script
    # and the transliterated query is left half-romanised.
    "ki": "की", "ko": "को", "se": "से", "tha": "था", "thi": "थी", "the": "थे",
    "hain": "हैं", "ho": "हो", "hi": "ही", "bhi": "भी", "par": "पर",
    "jisko": "जिसको", "jiska": "जिसका", "jiski": "जिसकी", "unpar": "उनपर",
    "bacche": "बच्चे", "bachche": "बच्चे", "bacha": "बच्चा", "beta": "बेटा",
    "aap": "आप", "main": "मैं", "yeh": "यह", "ye": "ये", "is": "इस",
    "us": "उस", "un": "उन", "in": "इन", "ab": "अब",
}


@lru_cache(maxsize=1)
def _sanscript():
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    return sanscript, transliterate


def rule_based(word: str) -> str:
    """ITRANS mapping with the word-final inherent schwa restored."""
    sanscript, transliterate = _sanscript()
    d = transliterate(word, sanscript.ITRANS, sanscript.DEVANAGARI)
    return d[:-1] if d.endswith(VIRAMA) else d


def transliterate_token(token: str, english_vocab: set[str]) -> tuple[str, str]:
    """Return (devanagari_or_original, source) for one whitespace token.

    source is one of: "english" (left as is), "lexicon", "rules".
    """
    core = "".join(ch for ch in token.lower() if ch.isalpha())
    if not core:
        return token, "english"
    if core in english_vocab and core not in LEXICON:
        return token, "english"
    if core in LEXICON:
        return LEXICON[core], "lexicon"
    return rule_based(core), "rules"


#: Latin letter runs; everything between them (spaces, punctuation, digits) is
#: preserved verbatim so "red/raw" does not fuse into one token and "MRSA+" keeps
#: its shape.
_RUN = re.compile(r"[A-Za-z]+")


def _is_proper_noun(word: str, at_sentence_start: bool) -> bool:
    """Capitalised mid-sentence and unknown to the lexicon -> a name or brand.

    Drug and product names ("Advil", "MRSA") carry clinical content that MuRIL
    reads natively in Latin script; transliterating them destroys it.
    """
    return word[:1].isupper() and not at_sentence_start and word.lower() not in LEXICON


def transliterate_text(text: str, english_vocab: set[str]) -> tuple[str, dict[str, int]]:
    s = str(text)
    counts = {"english": 0, "lexicon": 0, "rules": 0}
    out, pos, sentence_start = [], 0, True
    for m in _RUN.finditer(s):
        gap = s[pos:m.start()]
        out.append(gap)
        if gap.strip():
            sentence_start = bool(re.search(r"[.!?]\s*$", gap))
        word = m.group()
        if word.isupper() and len(word) > 1 or _is_proper_noun(word, sentence_start):
            out.append(word)
            counts["english"] += 1
        else:
            dev, src = transliterate_token(word, english_vocab)
            out.append(dev)
            counts[src] += 1
        sentence_start = False
        pos = m.end()
    out.append(s[pos:])
    return "".join(out), counts


def coverage(texts: list[str], english_vocab: set[str]) -> dict[str, float]:
    """Share of tokens handled by each stage, over a corpus."""
    total = {"english": 0, "lexicon": 0, "rules": 0}
    for t in texts:
        _, c = transliterate_text(t, english_vocab)
        for k, v in c.items():
            total[k] += v
    n = sum(total.values()) or 1
    out = {f"{k}_share": v / n for k, v in total.items()}
    non_english = total["lexicon"] + total["rules"]
    out["lexicon_share_of_hindi"] = total["lexicon"] / non_english if non_english else 0.0
    out["n_tokens"] = float(n)
    return out


def contains_devanagari(text: str) -> bool:
    return bool(re.search(r"[ऀ-ॿ]", str(text)))
