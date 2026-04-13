"""
Deepfake detection via intrinsic dimension.

Hypothesis: AI-generated images live on a lower-dimensional manifold than real
photographs, because the generator's latent space acts as a bottleneck.

Pipeline:
  1. Collect N real images of a category (web download or dataset)
  2. Generate N fake images via Gemini image generation
  3. Resize all to uniform resolution, flatten to vectors
  4. Run DiRe intrinsic dimension sweep on both point clouds
  5. Compare detected intrinsic dimensions

Usage:
  # Step 1: Generate images (uses HPC env for Gemini API)
  python bench_deepfake_dim.py --generate --category landscape --n-images 50

  # Step 2: Analyze (uses rapids env for DiRe)
  python bench_deepfake_dim.py --analyze --category landscape
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


# ── Configuration ─────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "results" / "deepfake_dim"
IMAGE_SIZE = 64  # resize to 64x64
CATEGORIES = {
    "landscape": {
        "real_prompts": None,  # use downloaded images
        "fake_prompt": "a realistic photograph of a {variant}",
        "variants": [
            "mountain landscape at sunset",
            "mountain landscape with lake reflection",
            "mountain landscape in winter snow",
            "mountain landscape with forest",
            "mountain landscape at dawn",
            "rolling green hills landscape",
            "desert landscape with sand dunes",
            "coastal landscape with cliffs",
            "tropical landscape with waterfall",
            "autumn forest landscape with river",
            "misty mountain valley landscape",
            "rocky mountain landscape with wildflowers",
            "volcanic landscape with lava fields",
            "arctic tundra landscape",
            "canyon landscape at golden hour",
            "prairie landscape under storm clouds",
            "lake landscape surrounded by mountains",
            "bamboo forest landscape",
            "savanna landscape with acacia trees",
            "alpine meadow landscape with snow peaks",
        ],
        "search_query": "landscape photograph nature",
    },
    "face": {
        "fake_prompt": "a realistic portrait photograph of a {variant}",
        "variants": [
            "middle-aged man with glasses",
            "young woman with curly hair",
            "elderly man with beard",
            "young man with short hair",
            "woman with straight dark hair",
            "teenager with freckles",
            "man with mustache",
            "woman with blonde hair",
            "man with bald head",
            "woman with red hair",
            "man in a suit",
            "woman wearing a hat",
            "man with grey hair",
            "young woman with braids",
            "elderly woman with white hair",
            "man with sunglasses outdoors",
            "woman with natural makeup",
            "man with a warm smile",
            "woman looking thoughtful",
            "man with a serious expression",
        ],
        "search_query": "portrait photograph face",
    },
}


# ── Image generation (Gemini) ────────────────────────────────────────────

def generate_fake_images(category, n_images, output_dir):
    """Generate fake images using Gemini."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Source ~/devel/heretic/.env first.")

    client = genai.Client(api_key=api_key)
    cat = CATEGORIES[category]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = cat["variants"]
    generated = 0
    attempt = 0

    while generated < n_images:
        variant = variants[generated % len(variants)]
        # Add variation suffix for repeats beyond the variant list
        suffix = f" (variation {generated // len(variants) + 1})" if generated >= len(variants) else ""
        prompt = cat["fake_prompt"].format(variant=variant) + suffix

        attempt += 1
        print(f"  [{generated+1}/{n_images}] Generating: {variant}...", end=" ", flush=True)

        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-image-preview",
                contents=prompt,
            )

            saved = False
            for part in response.parts:
                if part.inline_data:
                    image = part.as_image()
                    fname = output_dir / f"fake_{generated:04d}.png"
                    image.save(str(fname))
                    print(f"OK ({fname.name})")
                    generated += 1
                    saved = True
                    break

            if not saved:
                # Model returned text instead of image
                text = ""
                for part in response.parts:
                    if part.text:
                        text = part.text[:100]
                        break
                print(f"no image (got text: {text}...)")

        except Exception as e:
            print(f"error: {e}")
            if "429" in str(e) or "quota" in str(e).lower():
                print("  Rate limited, waiting 30s...")
                time.sleep(30)
            else:
                time.sleep(2)

        # Rate limiting
        time.sleep(1)

    print(f"\nGenerated {generated} fake images in {output_dir}")


# ── Real image download ──────────────────────────────────────────────────

def download_real_images_cifar(category, n_images, output_dir):
    """Download real images from CIFAR-10 as a baseline source."""
    import torchvision
    import torchvision.transforms as T

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CIFAR-10 classes that map to our categories
    cifar_class_map = {
        "landscape": [5, 8],   # deer (has landscape bg), ship (seascapes)
        "face": [],            # CIFAR-10 has no faces
    }

    # Download CIFAR-10
    dataset = torchvision.datasets.CIFAR10(
        root="/tmp/cifar10", train=True, download=True,
        transform=T.ToPILImage(),
    )

    # For landscape: grab outdoor-ish images; for face: fall back to web
    classes = cifar_class_map.get(category, [])
    if not classes:
        print(f"  No CIFAR-10 mapping for '{category}', using random natural images")
        # Use nature-ish classes: airplane(0), bird(2), deer(4), horse(7), ship(8), truck(9)
        classes = [2, 4, 5, 7, 8]

    saved = 0
    for i, (img, label) in enumerate(dataset):
        if label in classes and saved < n_images:
            fname = output_dir / f"real_{saved:04d}.png"
            img.save(str(fname))
            saved += 1
        if saved >= n_images:
            break

    print(f"  Saved {saved} real images from CIFAR-10 to {output_dir}")
    return saved


def download_real_images_web(category, n_images, output_dir):
    """Download real images by fetching from public domain sources."""
    import urllib.request

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use Lorem Picsum for real photographs (random photos from Unsplash)
    # These are real photographs, not AI-generated
    saved = 0
    for i in range(n_images + 20):  # extra attempts in case of failures
        if saved >= n_images:
            break
        url = f"https://picsum.photos/seed/{category}{i}/512/512"
        fname = output_dir / f"real_{saved:04d}.jpg"
        try:
            print(f"  [{saved+1}/{n_images}] Downloading real image...", end=" ", flush=True)
            urllib.request.urlretrieve(url, str(fname))
            # Verify it's a valid image
            img = Image.open(fname)
            img.verify()
            print(f"OK ({fname.name})")
            saved += 1
        except Exception as e:
            print(f"failed: {e}")
            fname.unlink(missing_ok=True)
        time.sleep(0.5)

    print(f"\n  Saved {saved} real images to {output_dir}")
    return saved


# ── Image loading and preprocessing ──────────────────────────────────────

def load_images_as_vectors(image_dir, size=IMAGE_SIZE, max_images=None):
    """Load images from directory, resize, and flatten to vectors."""
    image_dir = Path(image_dir)
    files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    if max_images:
        files = files[:max_images]

    vectors = []
    for f in files:
        try:
            img = Image.open(f).convert("RGB").resize((size, size), Image.LANCZOS)
            vec = np.array(img, dtype=np.float32).ravel() / 255.0
            vectors.append(vec)
        except Exception as e:
            print(f"  Warning: skipping {f.name}: {e}")

    if not vectors:
        return np.array([])

    X = np.stack(vectors)
    print(f"  Loaded {X.shape[0]} images as {X.shape[1]}-dim vectors from {image_dir}")
    return X


# ── Intrinsic dimension analysis ─────────────────────────────────────────

def compute_intrinsic_dimension(X, label, max_dim=20):
    """Run DiRe dimension sweep and detect intrinsic dimension via neighbor preservation."""
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
    idx_orig = idx_orig[:, 1:]  # remove self

    for d in range(1, max_dim + 1):
        print(f"    d={d}...", end=" ", flush=True)
        try:
            reducer = DiRePyTorch(
                n_neighbors=k_nn, n_components=d,
                max_iter_layout=200, random_state=42, verbose=False,
            )
            emb = reducer.fit_transform(X)

            # Neighbor preservation
            nn_emb = NearestNeighbors(n_neighbors=k_nn + 1, metric="euclidean")
            nn_emb.fit(emb)
            _, idx_emb = nn_emb.kneighbors(emb)
            idx_emb = idx_emb[:, 1:]

            matches = (idx_orig[:, :, None] == idx_emb[:, None, :]).any(axis=2)
            preservation = float(matches.sum(axis=1).mean() / k_nn)

            results.append({"d": d, "neighbor_preservation": preservation})
            print(f"neighbor_preservation={preservation:.4f}")

        except Exception as e:
            print(f"error: {e}")
            results.append({"d": d, "neighbor_preservation": None})

    # Detect elbow via max second derivative of neighbor preservation
    preservations = [r["neighbor_preservation"] for r in results if r["neighbor_preservation"] is not None]
    dims = [r["d"] for r in results if r["neighbor_preservation"] is not None]

    if len(preservations) >= 3:
        p = np.array(preservations)
        # Normalize to [0,1] range
        p_norm = (p - p.min()) / (p.max() - p.min() + 1e-10)
        # Find elbow: max distance from line connecting first and last point
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
    parser = argparse.ArgumentParser(description="Deepfake detection via intrinsic dimension")
    parser.add_argument("--generate", action="store_true", help="Generate fake + download real images")
    parser.add_argument("--analyze", action="store_true", help="Run intrinsic dimension analysis")
    parser.add_argument("--category", default="landscape", choices=list(CATEGORIES.keys()))
    parser.add_argument("--n-images", type=int, default=50)
    parser.add_argument("--max-dim", type=int, default=15, help="Max dimension to sweep")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()

    cat_dir = RESULTS_DIR / args.category
    real_dir = cat_dir / "real"
    fake_dir = cat_dir / "fake"

    if args.generate:
        print(f"=== Generating images for category '{args.category}' ===\n")

        print("--- Downloading real images (Lorem Picsum / Unsplash) ---")
        download_real_images_web(args.category, args.n_images, real_dir)

        print("\n--- Generating fake images (Gemini) ---")
        generate_fake_images(args.category, args.n_images, fake_dir)

    if args.analyze:
        print(f"=== Intrinsic dimension analysis for '{args.category}' ===\n")

        print("--- Loading images ---")
        X_real = load_images_as_vectors(real_dir, size=args.image_size, max_images=args.n_images)
        X_fake = load_images_as_vectors(fake_dir, size=args.image_size, max_images=args.n_images)

        if len(X_real) == 0 or len(X_fake) == 0:
            print("Error: need both real and fake images. Run --generate first.")
            sys.exit(1)

        print("\n--- Running dimension sweep ---")
        result_real = compute_intrinsic_dimension(X_real, "real", max_dim=args.max_dim)
        result_fake = compute_intrinsic_dimension(X_fake, "fake", max_dim=args.max_dim)

        # Summary
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"  Category: {args.category}")
        print(f"  Image size: {args.image_size}x{args.image_size}")
        print(f"  Real images: {result_real['n_samples']}")
        print(f"  Fake images: {result_fake['n_samples']}")
        print(f"  Detected intrinsic dimension (real):  {result_real['detected_dim']}")
        print(f"  Detected intrinsic dimension (fake):  {result_fake['detected_dim']}")
        print()

        if result_real["detected_dim"] and result_fake["detected_dim"]:
            if result_fake["detected_dim"] < result_real["detected_dim"]:
                print("  >>> HYPOTHESIS SUPPORTED: fake images have lower intrinsic dimension")
            elif result_fake["detected_dim"] == result_real["detected_dim"]:
                print("  >>> INCONCLUSIVE: same detected dimension")
            else:
                print("  >>> HYPOTHESIS REJECTED: fake images have higher intrinsic dimension")

        # Print sweep comparison
        print(f"\n  {'d':>3} | {'Real NP':>8} | {'Fake NP':>8}")
        print("  " + "-" * 28)
        for r, f in zip(result_real["sweep"], result_fake["sweep"]):
            rp = f"{r['neighbor_preservation']:.4f}" if r["neighbor_preservation"] else "-"
            fp = f"{f['neighbor_preservation']:.4f}" if f["neighbor_preservation"] else "-"
            marker_r = " <" if r["d"] == result_real["detected_dim"] else ""
            marker_f = " <" if f["d"] == result_fake["detected_dim"] else ""
            print(f"  {r['d']:>3} | {rp:>8}{marker_r:2} | {fp:>8}{marker_f:2}")

        # Save results
        cat_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "category": args.category,
            "image_size": args.image_size,
            "real": result_real,
            "fake": result_fake,
        }
        results_file = cat_dir / "intrinsic_dim_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {results_file}")

    if not args.generate and not args.analyze:
        parser.print_help()


if __name__ == "__main__":
    main()
