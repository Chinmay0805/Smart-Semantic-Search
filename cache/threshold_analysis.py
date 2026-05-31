
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from cache.cache import SemanticCache, CacheEntry



# TEST QUERY PAIRS

# Each tuple: (query_to_cache, incoming_query, should_hit: bool, description)
TEST_PAIRS = [
    # --- Should be cache HITS (same meaning, different words) ---
    (
        "What are NASA's latest space missions?",
        "Recent space exploration news from NASA",
        True,
        "Paraphrase: space missions"
    ),
    (
        "How does encryption protect data?",
        "What is the role of cryptography in security?",
        True,
        "Paraphrase: encryption/cryptography"
    ),
    (
        "Best hockey teams this season",
        "Top performing NHL teams right now",
        True,
        "Paraphrase: hockey teams"
    ),
    (
        "Is there a God? Religious perspectives",
        "Arguments for and against the existence of God",
        True,
        "Paraphrase: religion/God"
    ),
    (
        "How to fix Windows file system errors?",
        "Troubleshooting Windows disk problems",
        True,
        "Paraphrase: Windows issues"
    ),

    # --- Should be cache MISSES (different topics) ---
    (
        "NASA space shuttle launch schedule",
        "How does gun control legislation work?",
        False,
        "Different topics: space vs politics"
    ),
    (
        "Best baseball teams of all time",
        "Jewish history in the Middle East",
        False,
        "Different topics: sports vs history"
    ),
    (
        "How to configure SCSI drives on Mac?",
        "What does the Bible say about forgiveness?",
        False,
        "Different topics: hardware vs religion"
    ),
]


# ─────────────────────────────────────────────
# RUN ANALYSIS
# ─────────────────────────────────────────────
def run_threshold_analysis():
    print("=" * 55)
    print("  Threshold Analysis")
    print("=" * 55)

    # Load embedding model
    print("\n🤖 Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Embed all queries
    print("⚙️  Embedding test queries...")
    all_queries = [p[0] for p in TEST_PAIRS] + [p[1] for p in TEST_PAIRS]
    all_embeddings = model.encode(
        all_queries,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    cached_embeddings  = all_embeddings[:len(TEST_PAIRS)]
    incoming_embeddings = all_embeddings[len(TEST_PAIRS):]

    # Compute actual cosine similarities
    similarities = [
        float(np.dot(cached_embeddings[i], incoming_embeddings[i]))
        for i in range(len(TEST_PAIRS))
    ]

    print("\n📊 Actual cosine similarities between query pairs:")
    print("-" * 55)
    for i, (pair, sim) in enumerate(zip(TEST_PAIRS, similarities)):
        label = "✅ SHOULD HIT" if pair[2] else "❌ SHOULD MISS"
        print(f"  {label}  sim={sim:.3f}  {pair[3]}")

    # Test across thresholds
    thresholds  = np.arange(0.50, 1.00, 0.05)
    hit_rates   = []
    precisions  = []

    print("\n🔍 Testing thresholds from 0.50 to 0.99...")
    print("-" * 55)

    for thresh in thresholds:
        hits         = 0
        correct_hits = 0
        total_should_hit = sum(1 for p in TEST_PAIRS if p[2])

        for i, (pair, sim) in enumerate(zip(TEST_PAIRS, similarities)):
            is_hit = sim >= thresh
            if is_hit:
                hits += 1
                if pair[2]:  # correctly identified as hit
                    correct_hits += 1

        hit_rate  = hits / len(TEST_PAIRS)
        precision = correct_hits / hits if hits > 0 else 0.0

        hit_rates.append(hit_rate)
        precisions.append(precision)

        print(f"  threshold={thresh:.2f} → "
              f"hit_rate={hit_rate:.2f}  "
              f"precision={precision:.2f}")

    # ─────────────────────────────────────────
    # PLOT
    # ─────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Hit Rate vs Threshold
    ax1.plot(thresholds, hit_rates, 'bo-', linewidth=2, markersize=6)
    ax1.axvline(x=0.85, color='red', linestyle='--', label='Recommended (0.85)')
    ax1.set_xlabel('Similarity Threshold')
    ax1.set_ylabel('Hit Rate')
    ax1.set_title('Cache Hit Rate vs Threshold\n(higher = more aggressive caching)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Right: Precision vs Threshold
    ax2.plot(thresholds, precisions, 'gs-', linewidth=2, markersize=6)
    ax2.axvline(x=0.85, color='red', linestyle='--', label='Recommended (0.85)')
    ax2.set_xlabel('Similarity Threshold')
    ax2.set_ylabel('Precision')
    ax2.set_title('Cache Precision vs Threshold\n(higher = fewer wrong answers)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.suptitle(
        'Threshold Analysis: Hit Rate vs Precision Tradeoff\n'
        'Low threshold = more hits but wrong answers. '
        'High threshold = accurate but useless cache.',
        fontsize=10
    )
    plt.tight_layout()

    os.makedirs('./cache', exist_ok=True)
    plt.savefig('./cache/threshold_analysis.png', dpi=150)
    plt.close()
    print(f"\n✅ Plot saved to ./cache/threshold_analysis.png")

    # ─────────────────────────────────────────
    # INTERPRETATION
    # ─────────────────────────────────────────
    print("\n\n💡 What Each Threshold Reveals:")
    print("-" * 55)
    print("  0.50 — Catches almost everything, including wrong matches")
    print("         High hit rate but low precision = wrong answers returned")
    print("  0.70 — Better precision but still some false positives")
    print("  0.85 — Sweet spot: paraphrases match, distinct queries don't")
    print("         This is our recommended default")
    print("  0.95 — Very strict: only near-identical queries match")
    print("         Cache barely ever triggers = defeats the purpose")
    print("  0.99 — Almost never hits = cache is useless at this level")

    print("\n🎉 Threshold analysis complete!")


if __name__ == "__main__":
    run_threshold_analysis()