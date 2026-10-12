import re


def wc(s):
    return len(s.split())


def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s]


def _split_long(sentence, max_words):
    words = sentence.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def chunk_sentences(text, max_words, overlap_words=0):
    """Split on sentence boundaries into chunks of <= max_words (unless a single sentence is longer,
    in which case it is cut). Consecutive chunks share ~overlap_words words of context."""
    sents = []
    for s in split_sentences(text):
        sents.extend(_split_long(s, max_words) if wc(s) > max_words else [s])
    chunks, start = [], 0
    while start < len(sents):
        end, count = start, 0
        while end < len(sents) and (end == start or count + wc(sents[end]) <= max_words):
            count += wc(sents[end])
            end += 1
        chunks.append(" ".join(sents[start:end]))
        if end >= len(sents):
            break
        back, ov = end, 0
        while back > start + 1 and ov < overlap_words:
            back -= 1
            ov += wc(sents[back])
        start = back                      # always > old start, so the loop terminates
    return chunks


def collapse_repetitions(text, max_repeats=2):
    """Whisper sometimes loops on one sentence. Keep at most max_repeats consecutive copies."""
    out, last, run = [], None, 0
    for s in split_sentences(text):
        key = re.sub(r"\W+", " ", s.lower()).strip()
        run = run + 1 if key == last else 1
        last = key
        if run <= max_repeats:
            out.append(s)
    return " ".join(out)
