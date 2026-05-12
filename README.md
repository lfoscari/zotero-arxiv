# ArXiv-Zotero Semantic Recommender

A Python script that recommends recent arXiv papers based on the semantic similarity to the user's existing Zotero library. It uses Sentence Transformers to compute embeddings and filters papers by a cosine similarity threshold.

## Prerequisites

- Python 3.7+
- A Zotero API key (create one in your Zotero settings)
- A Zotero User ID (your numeric user ID)

## Setup

1.  Install the required packages:

    ```bash
    pip install python-dotenv arxiv pyzotero sentence-transformers huggingface-hub numpy
    ```

2.  Create a `.env` file in the same directory as the script and add your credentials:

    ```dotenv
    ZOTERO_ID=your_numeric_zotero_id
    ZOTERO_API_KEY=your_zotero_api_key
    HUGGING_FACE_API_KEY=hf_your_huggingface_token  # Optional, but recommended to avoid rate limiting
    ```

## Usage

Run the script from the command line:

```bash
python gimme-some-good-papers-please.py
```

The script will print a list of recent arXiv papers from the targeted categories whose abstracts are semantically similar to papers in your Zotero library.

## Configuration

The script has several configuration options at the top of the file:

- **`CATEGORIES`**: The arXiv categories to search. Defaults to a mix of Computer Science, Economics, and Quantitative Finance.
- **`THRESHOLD`**: The cosine similarity threshold (between 0 and 1). Only papers with a score above this will be printed. Defaults to `0.75`.
- **`MODEL`**: The SentenceTransformer model to use for embeddings. Defaults to `all-MiniLM-L6-v2`.
- **Zotero Limit**: `zot.top(limit=50)` controls how many papers from your library are used as seed material.
- **arXiv Limit**: `max_results=1000` controls how many recent arXiv papers are scanned.

## How It Works

1.  **Fetch Seed Papers**: Retrieves the latest papers from your Zotero library and extracts their abstracts.
2.  **Encode**: Generates semantic embeddings for your Zotero paper abstracts using the Sentence Transformer model.
3.  **Fetch Candidates**: Queries the arXiv API for the most recent papers in the specified categories.
4.  **Score**: Embeddings are computed for the arXiv abstracts, and each is compared against all of your Zotero embeddings using cosine similarity.
5.  **Filter and Output**: Papers with a maximum similarity score above the threshold are printed to the console along with their authors, link, categories, and score.
