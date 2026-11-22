import sys, json, re, math, random, collections
sys.path.insert(0, "src")
from rag_guard.core.config import settings
from rag_guard.rag.embeddings import Embedder
from rag_guard.rag.vector_store import VectorStore
from rag_guard.services.ingestion_service import _load_rows, sample_kb_rows

rows = _load_rows()
kept = {r["id"] for r in sample_kb_rows(rows)}
held = [r for r in rows if r["id"] not in kept]
random.seed(0)
by = collections.defaultdict(list)
for r in held: by[r["intent"]].append(r)
qs = [(r["request"], r["intent"]) for g in by.values() for r in random.sample(g, 8)]
print("held-out queries:", len(qs), "intents:", len(by))

col = VectorStore().collection
got = col.get(include=["documents", "metadatas"])
ids, docs, metas = got["ids"], got["documents"], got["metadatas"]
N = len(ids); intent = [m["intent"] for m in metas]
print("chunks:", N)

tok = lambda s: re.findall(r"\w+", s.lower())
dtoks = [tok(d) for d in docs]
avgdl = sum(map(len, dtoks)) / N
df = collections.Counter(w for t in dtoks for w in set(t))
idf = {w: math.log(1 + (N - n + .5) / (n + .5)) for w, n in df.items()}
def bm25(q):
    qt = tok(q); out = []
    for t in dtoks:
        tf = collections.Counter(t); s = 0.0
        for w in qt:
            if w in tf: s += idf[w] * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * len(t) / avgdl))
        out.append(s)
    return out

emb = Embedder()
def dense_scores(q):
    res = col.query(query_embeddings=[emb.embed_query(q)], n_results=N)
    sc = {i: 1 - d for i, d in zip(res["ids"][0], res["distances"][0])}
    return [sc[i] for i in ids]
def rank(scores): return sorted(range(N), key=lambda i: -scores[i])
def rrf(*rankings, k=60):
    s = collections.defaultdict(float)
    for r in rankings:
        for pos, i in enumerate(r): s[i] += 1 / (k + pos + 1)
    return sorted(s, key=lambda i: -s[i])

methods = collections.defaultdict(lambda: {"top1": 0, "r5": 0, "p5": 0.0, "mrr": 0.0})
dense_all = []
for q, gold in qs:
    d = dense_scores(q); b = bm25(q)
    rd, rb = rank(d), rank(b)
    # hybrid by weighted score fusion (min-max) as a second variant
    nz = lambda x: [(v - min(x)) / ((max(x) - min(x)) or 1) for v in x]
    hs = [0.5 * a + 0.5 * c for a, c in zip(nz(d), nz(b))]
    # oracle metadata filter: only chunks of gold intent (upper bound for any filter/router)
    top_dense_intent = intent[rd[0]]
    filt_top1 = [i for i in rd if intent[i] == top_dense_intent]
    for name, r in [("dense", rd), ("bm25", rb), ("hybrid_rrf", rrf(rd, rb)), ("hybrid_score", rank(hs)),
                    ("dense_then_filter_top1_intent", filt_top1)]:
        m = methods[name]; top5 = r[:5]
        m["top1"] += intent[r[0]] == gold
        m["r5"] += any(intent[i] == gold for i in top5)
        m["p5"] += sum(intent[i] == gold for i in top5) / 5
        pos = next((p for p, i in enumerate(r) if intent[i] == gold), None)
        m["mrr"] += 1 / (pos + 1) if pos is not None else 0
    dense_all.append((q, gold, [intent[i] for i in rd[:5]], [round(d[i], 3) for i in rd[:5]]))
n = len(qs)
print(f"{'method':<32}{'top1':>7}{'recall@5':>10}{'prec@5':>9}{'MRR':>7}")
for name, m in methods.items():
    print(f"{name:<32}{m['top1']/n:>7.3f}{m['r5']/n:>10.3f}{m['p5']/n:>9.3f}{m['mrr']/n:>7.3f}")
miss = [(q, g, t) for q, g, t, s in dense_all if g not in t]
print("dense misses (gold intent not in top5):", len(miss))
for q, g, t in miss[:12]: print("  ", g, "|", t[:3], "|", q[:70])
conf = collections.Counter((g, t[0]) for q, g, t, s in dense_all if t[0] != g)
print("top confusions (gold -> got):", conf.most_common(8))
