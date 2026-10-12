"""Check model claims against the transcript: quoted evidence, owners and deadlines must really appear."""
import re
from difflib import SequenceMatcher

_ORD = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
_IGNORE = frozenset({"and", "the", "of", "a", "an", "by", "on", "at", "in", "to", "&"})


def norm_token(w):
    w = w.lower().replace("’", "'").replace("‘", "'")
    w = re.sub(r"[^\w']", "", w).strip("'")
    w = re.sub(r"'s$", "", w)
    m = _ORD.match(w)
    return m.group(1) if m else w


class Indexed:
    """A text with a normalised token list and a map back to the original words."""

    def __init__(self, text):
        self.words = text.split()
        self.tokens, self.pos = [], []
        for i, w in enumerate(self.words):
            t = norm_token(w)
            if t:
                self.tokens.append(t)
                self.pos.append(i)
        self.vocab = set(self.tokens)


def find_quote(quote, idx, threshold=0.8, min_tokens=3):
    """Locate `quote` in the indexed text. Returns (score, start_tok, end_tok) or None."""
    q = [t for t in (norm_token(w) for w in (quote or "").split()) if t]
    n, T = len(q), idx.tokens
    if n < min_tokens or n > len(T):
        return None
    for i in range(len(T) - n + 1):
        if T[i:i + n] == q:
            return 1.0, i, i + n
    qset, need, best = set(q), 0.6 * len(set(q)), (0.0, 0, 0)
    for i in range(len(T) - n + 1):
        win = T[i:i + n]
        if len(qset.intersection(win)) < need:
            continue
        r = SequenceMatcher(None, q, win, autojunk=False).ratio()
        if r > best[0]:
            best = (r, i, i + n)
    return best if best[0] >= threshold else None


def evidence_text(idx, s, e):
    """The verbatim transcript words for token range [s, e)."""
    return " ".join(idx.words[idx.pos[s]: idx.pos[e - 1] + 1])


def context_window(idx, s, e, pad=60):
    return " ".join(idx.words[max(0, idx.pos[s] - pad): idx.pos[e - 1] + 1 + pad])


def value_supported(value, idx):
    """True if every meaningful word of an owner/deadline value occurs in the transcript."""
    toks = [t for t in (norm_token(w) for w in str(value).split()) if t and t not in _IGNORE]
    return bool(toks) and all(t in idx.vocab for t in toks)
