import re

_WORDS = (
    "zero one two three four five six seven eight nine ten "
    "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()

_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()

_SCALES = ["", "thousand", "million", "billion"]


def _int_to_words(n: int) -> str:
    """Convert an integer (0 – 999 999 999 999) to English words."""
    if n == 0:
        return "zero"

    def _hundreds(n: int) -> str:
        if n == 0:
            return ""
        parts = []
        if n >= 100:
            parts.append(_WORDS[n // 100] + " hundred")
            n %= 100
        if n >= 20:
            t, o = divmod(n, 10)
            chunk = _TENS[t - 2]
            if o:
                chunk += "-" + _WORDS[o]
            parts.append(chunk)
        elif n > 0:
            parts.append(_WORDS[n])
        return " ".join(parts)

    result = []
    scale_idx = 0
    while n > 0:
        chunk = n % 1000
        if chunk:
            label = _hundreds(chunk)
            if scale := _SCALES[scale_idx]:
                label += " " + scale
            result.append(label)
        n //= 1000
        scale_idx += 1
    return " ".join(reversed(result))


def _norm_numbers(text: str) -> str:
    """
    Convert numeric tokens to their spoken form so the TTS model's G2P
    doesn't have to guess how to pronounce them.

    "70.8%"  -> "seventy point eight percent"
    "3.14"   -> "three point one four"
    "50%"    -> "fifty percent"
    "42"     -> "forty-two"
    """

    def _replace_decimal(m: re.Match) -> str:
        integer_word = _int_to_words(int(m.group(1)))
        decimal_digits = " ".join(_WORDS[int(d)] for d in m.group(2))
        suffix = " percent" if m.group(3) else ""
        return f"{integer_word} point {decimal_digits}{suffix}"

    def _replace_int_percent(m: re.Match) -> str:
        return f"{_int_to_words(int(m.group(1)))} percent"

    def _replace_int(m: re.Match) -> str:
        return _int_to_words(int(m.group(0)))

    text = re.sub(r'(\d+)\.(\d+)(%)?', _replace_decimal, text)
    text = re.sub(r'(?<!\d)(\d+)%', _replace_int_percent, text)
    text = re.sub(r'(?<!\d)(\d+)(?!\.\d)', _replace_int, text)
    return text


def clean_spoken_text(text: str) -> str:
    """Normalize Wikipedia text for TTS: removes citations, phonetic guides, normalizes numbers."""
    if not text:
        return ""

    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[edit\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(/.*?/\)', '', text)
    text = _norm_numbers(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
