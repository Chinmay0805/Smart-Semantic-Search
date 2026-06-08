import os
import re
import json
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


DATA_HOME         = "dataset"                      # where your dataset lives
CHROMA_PATH       = "./embeddings/chroma_db"      # where ChromaDB will be saved
EMBEDDINGS_PATH   = "./embeddings/embeddings.npy" # raw embeddings saved here
TEXTS_PATH        = "./embeddings/texts.json"     # cleaned texts saved here
EMBED_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
MIN_DOC_LENGTH    = 100                           # drop docs shorter than this
BATCH_SIZE        = 64                            # embedding batch size



def load_dataset():
    dataset = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),  # remove noise
        data_home=DATA_HOME
    )

    print(f" Loaded {len(dataset.data)} documents across {len(dataset.target_names)} categories")
    print(f"   Categories: {dataset.target_names[:5]} ... (and more)")
    return dataset



def clean_text(text: str) -> str:
    """
    Cleaning decisions:
    - Remove email addresses: not meaningful for topic search
    - Remove URLs: same reason
    - Remove non-ASCII chars: reduces noise from encoding artifacts
    - Collapse whitespace: normalizes spacing
    """
    text = re.sub(r'\S+@\S+', ' ', text)           # remove emails
    text = re.sub(r'https?://\S+', ' ', text)       # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)        # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()         # collapse whitespace
    return text


def clean_dataset(dataset):
    print("\n Cleaning documents...")

    cleaned_texts  = []
    cleaned_labels = []
    skipped        = 0

    for text, label in zip(dataset.data, dataset.target):
        cleaned = clean_text(text)

        # Drop documents that are too short after cleaning
        # Reason: very short docs don't embed meaningfully
        if len(cleaned) < MIN_DOC_LENGTH:
            skipped += 1
            continue

        cleaned_texts.append(cleaned)
        cleaned_labels.append(int(label))

    print(f" Kept {len(cleaned_texts)} documents")
    print(f" Skipped {skipped} documents (too short after cleaning)")
    return cleaned_texts, cleaned_labels



def embed_documents(texts: list[str]) -> np.ndarray:
    print(f"\n Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"Embedding {len(texts)} documents in batches of {BATCH_SIZE}...")
    print("This will take a few minutes on first run...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalize → cosine sim = dot product (faster)
    )

    print(f" Embeddings shape: {embeddings.shape}")  # should be (N, 384)
    return embeddings



def save_artifacts(texts, labels, embeddings):
    print("\n Saving artifacts...")

    # Save embeddings as numpy array
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Embeddings saved to {EMBEDDINGS_PATH}")

    # Save texts + labels as JSON
    artifacts = [
        {"id": i, "text": t, "label": l, "snippet": t[:200]}
        for i, (t, l) in enumerate(zip(texts, labels))
    ]
    with open(TEXTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(artifacts, f, ensure_ascii=False, indent=2)
    print(f"Texts saved to {TEXTS_PATH}")



def store_in_chromadb(texts, labels, embeddings):
    print("\n  Storing in ChromaDB...")

    # Create persistent ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete collection if it already exists (for clean re-runs)
    try:
        client.delete_collection("newsgroups")
        print(" Deleted existing collection for fresh start")
    except:
        pass

    collection = client.create_collection(
        name="newsgroups",
        metadata={"hnsw:space": "cosine"}  # use cosine similarity for search
    )

    # Insert in batches (ChromaDB has limits per insert)
    CHROMA_BATCH = 500
    total = len(texts)

    for start in range(0, total, CHROMA_BATCH):
        end = min(start + CHROMA_BATCH, total)

        collection.add(
            ids        = [str(i) for i in range(start, end)],
            embeddings = embeddings[start:end].tolist(),
            documents  = texts[start:end],
            metadatas  = [
                {
                    "doc_id":   i,
                    "category": labels[i],
                    "snippet":  texts[i][:200]
                }
                for i in range(start, end)
            ]
        )
        print(f"   Inserted {end}/{total} documents...")

    print(f"ChromaDB collection ready at {CHROMA_PATH}")
    print(f"   Total documents in DB: {collection.count()}")
    return collection


def main():
    print("=" * 55)
    print("Embedding & Vector Database Setup")
    print("=" * 55)

    # Make sure output directories exist
    os.makedirs("./embeddings", exist_ok=True)

    # Run pipeline
    dataset          = load_dataset()
    texts, labels    = clean_dataset(dataset)
    embeddings       = embed_documents(texts)
    save_artifacts(texts, labels, embeddings)
    collection       = store_in_chromadb(texts, labels, embeddings)

   


if __name__ == "__main__":
    main()