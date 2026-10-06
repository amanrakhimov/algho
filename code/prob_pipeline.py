"""Как нейросеть получает вероятности следующего слова.
Сквозной учебный пример: «Кошка сидит на ...»  (чистый Python).
Числа придуманы для иллюстрации; в настоящей модели их подбирает обучение."""
import math

# ---------- Шаг 1. Слова -> векторы (таблица embedding) ----------
EMB = {                       # [живое, поза, место]
    "Кошка": [1.0, 0.0, 0.0],
    "сидит": [0.0, 1.0, 0.0],
    "на":    [0.0, 0.5, 1.0],
}
PHRASE = ["Кошка", "сидит", "на"]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def softmax(xs, T=1.0):
    xs = [x / T for x in xs]
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex)
    return [e / s for e in ex]

# ---------- Шаг 2. Attention: последнее слово смотрит на всю фразу ----------
def attention_last(phrase, emb):
    d = len(emb[phrase[0]])
    q = emb[phrase[-1]]                                  # запрос = слово «на»
    scores = [dot(q, emb[w]) / math.sqrt(d) for w in phrase]
    weights = softmax(scores)
    h = [sum(weights[i] * emb[phrase[i]][k] for i in range(len(phrase)))
         for k in range(d)]
    return scores, weights, h

# ---------- Шаг 3. Логиты: h · вектор каждого слова словаря ----------
OUT = {                        # выходная матрица W_out (по строке на слово)
    "диване": [0.0, 3.0, 0.5],
    "полу":   [0.0, 2.0, 0.5],
    "крыше":  [0.0, 1.0, 1.5],
    "луне":   [-1.0, 0.5, 0.0],
    "облаке": [-1.5, -0.5, 0.5],
}

def top_k(probs, k):
    r = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:k]
    s = sum(p for _, p in r)
    return {w: p / s for w, p in r}

def top_p(probs, p):
    r = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    kept, c = [], 0.0
    for w, pr in r:
        kept.append((w, pr)); c += pr
        if c >= p - 1e-9: break
    s = sum(pr for _, pr in kept)
    return {w: pr / s for w, pr in kept}

if __name__ == "__main__":
    scores, weights, h = attention_last(PHRASE, EMB)
    print("баллы (после /sqrt(d)):", [round(x, 3) for x in scores])
    print("доли внимания         :", dict(zip(PHRASE, [round(x, 3) for x in weights])))
    print("вектор контекста h    :", [round(x, 3) for x in h])

    words = list(OUT)
    logits = [dot(h, OUT[w]) for w in words]
    print("\nлогиты:", {w: round(z, 3) for w, z in zip(words, logits)})

    for T in (1.0, 0.5):
        pr = softmax(logits, T)
        print(f"вероятности T={T}:", {w: round(p, 3) for w, p in zip(words, pr)},
              "сумма =", round(sum(pr), 6))

    probs = dict(zip(words, softmax(logits, 1.0)))
    print("\ntop-k (k=2):", {w: round(p, 3) for w, p in top_k(probs, 2).items()})
    print("top-p (p=0.9):", {w: round(p, 3) for w, p in top_p(probs, 0.9).items()})
    assert abs(sum(probs.values()) - 1) < 1e-9
