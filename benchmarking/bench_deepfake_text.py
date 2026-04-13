"""
AI-generated text detection via intrinsic dimension.

Hypothesis: LLM-generated text lives on a lower-dimensional manifold than real
human-written text, because the generator samples from a learned distribution
with a latent bottleneck, while real text reflects many independent authors,
styles, and editorial histories.

Pipeline:
  1. Collect real text: Wikipedia paragraphs on diverse topics
  2. Generate fake text: Gemini articles on the same topics, split into paragraphs
  3. Embed all paragraphs via BGE-small-en-v1.5 (384-dim)
  4. Run DiRe intrinsic dimension sweep on both point clouds
  5. Compare detected intrinsic dimensions

Usage:
  # Step 1: Generate text (uses HPC env for Gemini + sentence-transformers)
  python bench_deepfake_text.py --generate --n-topics 20

  # Step 2: Analyze (uses rapids env for DiRe)
  python bench_deepfake_text.py --analyze
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np


# ── Configuration ─────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "results" / "deepfake_text"

TOPICS = [
    "Photosynthesis",
    "Roman Empire",
    "General relativity",
    "Jazz music",
    "Coral reef",
    "French Revolution",
    "DNA replication",
    "Renaissance art",
    "Plate tectonics",
    "Quantum computing",
    "Industrial Revolution",
    "Honey bee",
    "Ancient Egypt",
    "Machine learning",
    "Volcanic eruption",
    "Impressionism",
    "Game theory",
    "Migration (ecology)",
    "Fermentation",
    "Cold War",
    "Black hole",
    "Gothic architecture",
    "Antibiotic resistance",
    "Silk Road",
    "Ocean current",
]


# ── Wikipedia fetching ────────────────────────────────────────────────────

def fetch_wikipedia_text(topic, min_paragraphs=5):
    """Fetch plain-text extract of a Wikipedia article."""
    title = topic.replace(" ", "_")
    url = (
        f"https://en.wikipedia.org/w/api.php?action=query&titles={title}"
        f"&prop=extracts&explaintext=1&exsectionformat=plain&format=json"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DiReResearch/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        text = page.get("extract", "")
        return text
    except Exception as e:
        print(f"    Failed to fetch '{topic}': {e}")
        return ""


def split_paragraphs(text, min_words=20):
    """Split text into paragraphs, filtering short ones."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return [p for p in paragraphs if len(p.split()) >= min_words]


# ── Gemini text generation ────────────────────────────────────────────────

def generate_fake_article(client, topic):
    """Generate a Wikipedia-style article via Gemini."""
    prompt = (
        f"Write a detailed encyclopedic article about '{topic}', "
        f"similar in style and depth to a Wikipedia article. "
        f"Include an introduction, several sections with headers, "
        f"and cover the key aspects of the subject. "
        f"Write at least 8 substantial paragraphs."
    )

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=prompt,
        )
        text = ""
        for part in response.parts:
            if part.text:
                text += part.text
        return text
    except Exception as e:
        print(f"    Gemini error for '{topic}': {e}")
        return ""


def clean_markdown(text):
    """Strip markdown headers/formatting, keep paragraph text."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        # Remove markdown headers
        line = re.sub(r"^#+\s*", "", line)
        # Remove bold/italic markers
        line = re.sub(r"\*+", "", line)
        if line:
            cleaned.append(line)
    return "\n".join(cleaned)


# ── Embedding ─────────────────────────────────────────────────────────────

def embed_paragraphs(paragraphs, model_name="BAAI/bge-small-en-v1.5", batch_size=64):
    """Embed paragraphs using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(paragraphs, batch_size=batch_size, show_progress_bar=False)
    return np.array(embeddings, dtype=np.float32)


# ── Intrinsic dimension analysis ─────────────────────────────────────────

def compute_intrinsic_dimension(X, label, max_dim=20):
    """Run DiRe dimension sweep and detect intrinsic dimension."""
    from dire_rapids import DiRePyTorch
    from sklearn.neighbors import NearestNeighbors

    n_samples, n_features = X.shape
    print(f"\n  Analyzing '{label}': {n_samples} samples in R^{n_features}")

    k_nn = min(15, n_samples - 2)
    results = []

    # Compute kNN in original space once
    nn_orig = NearestNeighbors(n_neighbors=k_nn + 1, metric="euclidean")
    nn_orig.fit(X)
    _, idx_orig = nn_orig.kneighbors(X)
    idx_orig = idx_orig[:, 1:]

    for d in range(1, max_dim + 1):
        print(f"    d={d}...", end=" ", flush=True)
        try:
            reducer = DiRePyTorch(
                n_neighbors=k_nn, n_components=d,
                max_iter_layout=200, random_state=42, verbose=False,
            )
            emb = reducer.fit_transform(X)

            nn_emb = NearestNeighbors(n_neighbors=k_nn + 1, metric="euclidean")
            nn_emb.fit(emb)
            _, idx_emb = nn_emb.kneighbors(emb)
            idx_emb = idx_emb[:, 1:]

            matches = (idx_orig[:, :, None] == idx_emb[:, None, :]).any(axis=2)
            preservation = float(matches.sum(axis=1).mean() / k_nn)

            results.append({"d": d, "neighbor_preservation": preservation})
            print(f"NP={preservation:.4f}")

        except Exception as e:
            print(f"error: {e}")
            results.append({"d": d, "neighbor_preservation": None})

    # Detect elbow
    preservations = [r["neighbor_preservation"] for r in results if r["neighbor_preservation"] is not None]
    dims = [r["d"] for r in results if r["neighbor_preservation"] is not None]

    if len(preservations) >= 3:
        p = np.array(preservations)
        p_norm = (p - p.min()) / (p.max() - p.min() + 1e-10)
        line = np.linspace(p_norm[0], p_norm[-1], len(p_norm))
        distances = np.abs(p_norm - line)
        elbow_idx = int(np.argmax(distances))
        detected_dim = dims[elbow_idx]
    else:
        detected_dim = None

    return {
        "label": label,
        "n_samples": n_samples,
        "n_features": n_features,
        "sweep": results,
        "detected_dim": detected_dim,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI text detection via intrinsic dimension")
    parser.add_argument("--generate", action="store_true", help="Fetch real + generate fake text, embed both")
    parser.add_argument("--analyze", action="store_true", help="Run intrinsic dimension analysis")
    parser.add_argument("--n-topics", type=int, default=20, help="Number of topics to use")
    parser.add_argument("--max-dim", type=int, default=15, help="Max dimension to sweep")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    topics = TOPICS[:args.n_topics]

    if args.generate:
        print(f"=== Generating text data for {len(topics)} topics ===\n")

        # --- Fetch Wikipedia articles ---
        print("--- Fetching Wikipedia articles ---")
        real_paragraphs = []
        for topic in topics:
            print(f"  {topic}...", end=" ", flush=True)
            text = fetch_wikipedia_text(topic)
            paras = split_paragraphs(text)
            print(f"{len(paras)} paragraphs")
            real_paragraphs.extend(paras)

        print(f"\n  Total real paragraphs: {len(real_paragraphs)}")

        # --- Generate Gemini articles ---
        print("\n--- Generating Gemini articles ---")
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        client = genai.Client(api_key=api_key)
        fake_paragraphs = []

        for topic in topics:
            print(f"  {topic}...", end=" ", flush=True)
            text = generate_fake_article(client, topic)
            text = clean_markdown(text)
            paras = split_paragraphs(text)
            print(f"{len(paras)} paragraphs")
            fake_paragraphs.extend(paras)
            time.sleep(1)  # rate limit

        print(f"\n  Total fake paragraphs: {len(fake_paragraphs)}")

        # --- Embed both ---
        print("\n--- Embedding paragraphs (BGE-small-en-v1.5) ---")
        print(f"  Embedding {len(real_paragraphs)} real paragraphs...")
        X_real = embed_paragraphs(real_paragraphs)
        print(f"  Embedding {len(fake_paragraphs)} fake paragraphs...")
        X_fake = embed_paragraphs(fake_paragraphs)

        print(f"  Real embeddings: {X_real.shape}")
        print(f"  Fake embeddings: {X_fake.shape}")

        # Save embeddings and metadata
        np.save(RESULTS_DIR / "real_embeddings.npy", X_real)
        np.save(RESULTS_DIR / "fake_embeddings.npy", X_fake)

        meta = {
            "topics": topics,
            "n_real_paragraphs": len(real_paragraphs),
            "n_fake_paragraphs": len(fake_paragraphs),
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "embedding_dim": int(X_real.shape[1]),
        }
        with open(RESULTS_DIR / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Also save raw text for inspection
        with open(RESULTS_DIR / "real_paragraphs.json", "w") as f:
            json.dump(real_paragraphs, f, indent=2, ensure_ascii=False)
        with open(RESULTS_DIR / "fake_paragraphs.json", "w") as f:
            json.dump(fake_paragraphs, f, indent=2, ensure_ascii=False)

        print(f"\n  Saved to {RESULTS_DIR}")

    if args.analyze:
        print(f"=== Intrinsic dimension analysis (text) ===\n")

        # Load embeddings
        real_file = RESULTS_DIR / "real_embeddings.npy"
        fake_file = RESULTS_DIR / "fake_embeddings.npy"

        if not real_file.exists() or not fake_file.exists():
            print("Error: embeddings not found. Run --generate first.")
            sys.exit(1)

        X_real = np.load(real_file)
        X_fake = np.load(fake_file)
        print(f"  Loaded real: {X_real.shape}, fake: {X_fake.shape}")

        # Equalize sample sizes for fair comparison
        n_min = min(len(X_real), len(X_fake))
        rng = np.random.default_rng(42)
        if len(X_real) > n_min:
            idx = rng.choice(len(X_real), n_min, replace=False)
            X_real = X_real[idx]
        if len(X_fake) > n_min:
            idx = rng.choice(len(X_fake), n_min, replace=False)
            X_fake = X_fake[idx]
        print(f"  Using {n_min} samples each")

        # Run dimension sweep
        print("\n--- Running dimension sweep ---")
        result_real = compute_intrinsic_dimension(X_real, "real (Wikipedia)", max_dim=args.max_dim)
        result_fake = compute_intrinsic_dimension(X_fake, "fake (Gemini)", max_dim=args.max_dim)

        # Summary
        print("\n" + "=" * 70)
        print("RESULTS — Text Intrinsic Dimension")
        print("=" * 70)
        print(f"  Embedding model: BGE-small-en-v1.5 (384-dim)")
        print(f"  Samples per set: {n_min}")
        print(f"  Detected intrinsic dimension (real):  {result_real['detected_dim']}")
        print(f"  Detected intrinsic dimension (fake):  {result_fake['detected_dim']}")
        print()

        if result_real["detected_dim"] and result_fake["detected_dim"]:
            if result_fake["detected_dim"] < result_real["detected_dim"]:
                print("  >>> HYPOTHESIS SUPPORTED: AI text has lower intrinsic dimension")
            elif result_fake["detected_dim"] == result_real["detected_dim"]:
                print("  >>> INCONCLUSIVE: same detected dimension")
            else:
                print("  >>> HYPOTHESIS REJECTED: AI text has higher intrinsic dimension")

        print(f"\n  {'d':>3} | {'Real NP':>8} | {'Fake NP':>8}")
        print("  " + "-" * 28)
        for r, f in zip(result_real["sweep"], result_fake["sweep"]):
            rp = f"{r['neighbor_preservation']:.4f}" if r["neighbor_preservation"] else "-"
            fp = f"{f['neighbor_preservation']:.4f}" if f["neighbor_preservation"] else "-"
            marker_r = " <" if r["d"] == result_real["detected_dim"] else ""
            marker_f = " <" if f["d"] == result_fake["detected_dim"] else ""
            print(f"  {r['d']:>3} | {rp:>8}{marker_r:2} | {fp:>8}{marker_f:2}")

        # Save
        results = {
            "n_samples": n_min,
            "real": result_real,
            "fake": result_fake,
        }
        results_file = RESULTS_DIR / "intrinsic_dim_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {results_file}")

    if not args.generate and not args.analyze:
        parser.print_help()


if __name__ == "__main__":
    main()
