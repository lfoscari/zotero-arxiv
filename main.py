import os
import numpy as np

from load_dotenv import load_dotenv
load_dotenv()

# Get papers from arXiv
import arxiv

# Get papers from Zotero
from pyzotero import zotero
ZOTERO_ID=os.getenv("ZOTERO_ID")
ZOTERO_API_KEY=os.getenv("ZOTERO_API_KEY")

# Disable Hugging Face logs
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import huggingface_hub as hf

# Load the models for the embdeddings
from sentence_transformers import SentenceTransformer, util


# --- Personalization

# See https://arxiv.org/category_taxonomy
CATEGORIES = ["cs:AI", "cs.GT", "cs.LG", "cs.MA", "econ.EM", "q-fin.EC", "q-fin.TR"]

# For cosine similarity
THRESHOLD = 0.75

# The model to compute the embeddings
MODEL = "all-MiniLM-L6-v2"


# --- Setup clients

if "HUGGING_FACE_API_KEY" in os.environ:
    # Log into Hugging Face to avoid rate limiting
    hf.login(token=os.getenv("HUGGING_FACE_API_KEY"))

client = arxiv.Client()
model = SentenceTransformer(MODEL)
zot = zotero.Zotero(ZOTERO_ID, "user", ZOTERO_API_KEY)


# --- Get papers from Zotero

# Fetch last 50 papers you added
zot_items = zot.top(limit=50)
liked_abstracts = [item["data"].get("abstractNote", "")
    for item in zot_items if item["data"].get("abstractNote")]
liked_embeddings = model.encode(liked_abstracts)


# --- Fetch latest arXiv papers and compute embeddings

query_string = " OR ".join([f"cat:{c}" for c in CATEGORIES])
search = arxiv.Search(
    query=query_string,
    max_results=1000,
    sort_by=arxiv.SortCriterion.SubmittedDate
)
arxiv_papers = list(client.results(search))
arxiv_abstracts = [p.summary for p in arxiv_papers]
arxiv_embeddings = model.encode(arxiv_abstracts)


# --- Score and filter

for i, arxiv_vec in enumerate(arxiv_embeddings):
    # Calculate max similarity against ANY of your Zotero papers
    cosine_scores = util.cos_sim(arxiv_vec, liked_embeddings)
    max_score = np.max(cosine_scores.tolist())

    paper = arxiv_papers[i]
    tags = [t.term for t in paper._raw.tags]

    if max_score > THRESHOLD:
        print(
            f"\n{paper.title}\n" +
            f"{', '.join([a.name for a in paper.authors])}\n" +
            f"{paper.entry_id}\n" +
            f"[tag {" ".join(tags)}] [score {max_score:.2f}]"
        )
