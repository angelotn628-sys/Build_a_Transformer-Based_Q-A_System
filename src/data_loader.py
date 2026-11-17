"""
Data Loading Module for Semantic Retrieval Project using Official BEIR Package
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import gzip
import shutil
import pathlib
import zipfile
import logging

import requests

# BEIR imports
from beir.datasets.data_loader import GenericDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory that holds the BEIR zips (nq.zip, scifact.zip, ...).
# The trailing slash matters: without it, some servers send back a redirect or a 404.
BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"


def _is_valid_beir_folder(data_path: str) -> bool:
    """A usable BEIR folder has corpus.jsonl, queries.jsonl and a qrels/ directory."""
    return (
        os.path.isfile(os.path.join(data_path, "corpus.jsonl"))
        and os.path.isfile(os.path.join(data_path, "queries.jsonl"))
        and os.path.isdir(os.path.join(data_path, "qrels"))
    )


def _download_file(url: str, dest: str) -> None:
    """Stream a file to disk. Raises on 404 and other HTTP errors instead of
    quietly saving the error page as the file."""
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
        print()
    os.replace(tmp, dest)


def _download_from_ukp(dataset_name: str, out_dir: str) -> str:
    """Download <dataset_name>.zip from the UKP server and unzip it into out_dir."""
    url = f"{BEIR_BASE_URL}{dataset_name}.zip"
    zip_file = os.path.join(out_dir, f"{dataset_name}.zip")

    # A zip left over from an earlier failed download (often an HTML 404 page
    # saved as .zip) makes the extract step fail every time, so remove it.
    if os.path.exists(zip_file) and not zipfile.is_zipfile(zip_file):
        logger.warning(f"{zip_file} is not a valid zip (probably a failed download). Deleting it.")
        os.remove(zip_file)

    if not os.path.exists(zip_file):
        logger.info(f"Downloading {url}")
        _download_file(url, zip_file)

    logger.info(f"Extracting {zip_file}")
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(out_dir)
    return os.path.join(out_dir, dataset_name)


def _download_from_hf(dataset_name: str, out_dir: str, split: str) -> str:
    """Fallback: rebuild the BEIR folder layout from the BeIR mirror on Hugging Face."""
    from huggingface_hub import hf_hub_download  # pip install huggingface_hub

    data_path = os.path.join(out_dir, dataset_name)
    os.makedirs(os.path.join(data_path, "qrels"), exist_ok=True)

    for fname in ("corpus.jsonl.gz", "queries.jsonl.gz"):
        src = hf_hub_download(f"BeIR/{dataset_name}", fname, repo_type="dataset")
        with gzip.open(src, "rb") as f_in, open(os.path.join(data_path, fname[:-3]), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    src = hf_hub_download(f"BeIR/{dataset_name}-qrels", f"{split}.tsv", repo_type="dataset")
    shutil.copy(src, os.path.join(data_path, "qrels", f"{split}.tsv"))
    return data_path


def load_beir_dataset(dataset_name, split="test", out_dir: Optional[str] = None):
    """
    Load BeIR dataset using the official BEIR package.

    Args:
        dataset_name (str): Dataset name as it appears on the server (e.g. 'nq', 'scifact')
        split (str): 'train', 'dev', or 'test' (default: 'test')
        out_dir (str, optional): Where to store datasets. Defaults to <cwd>/../dataset

    Returns:
        tuple: (corpus, queries, qrels)
            - corpus: dict of {doc_id: {'title': str, 'text': str}}
            - queries: dict of {query_id: str}
            - qrels: dict of {query_id: {doc_id: relevance_score}}
    """
    if out_dir is None:
        out_dir = os.path.join(pathlib.Path.cwd().parent, "dataset")  # project/solution/dataset
    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(out_dir, dataset_name)

    if _is_valid_beir_folder(data_path):
        logger.info(f"Dataset {dataset_name} already exists at {data_path}, skipping download...")
    else:
        try:
            data_path = _download_from_ukp(dataset_name, out_dir)
        except Exception as e:
            logger.warning(f"UKP download failed ({e}). Falling back to Hugging Face mirror...")
            data_path = _download_from_hf(dataset_name, out_dir, split)

    # Note: for NQ, the BEIR test set is the only split in nq.zip.
    # The training data is a separate download: nq-train.zip.
    qrels_file = os.path.join(data_path, "qrels", f"{split}.tsv")
    if not os.path.isfile(qrels_file):
        available = sorted(f[:-4] for f in os.listdir(os.path.join(data_path, "qrels")))
        raise ValueError(f"Split '{split}' not found for {dataset_name}. Available: {available}")

    logger.info(f"Loading {split} split...")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    return corpus, queries, qrels


def load_beir_nq(split="test"):
    """
    Load BeIR/nq (Natural Questions) dataset using the official BEIR package.

    Args:
        split (str): 'test' for nq.zip. For training data, use load_beir_dataset("nq-train", "train").

    Returns:
        tuple: (corpus, queries, qrels)
    """
    return load_beir_dataset("nq", split)


class DataLoader:
    """
    Handles loading and preprocessing of datasets for semantic retrieval using official BEIR package.
    """

    def __init__(self, dataset_name: str = "BeIR/nq"):
        """
        Initialize DataLoader for BeIR datasets.

        Args:
            dataset_name (str): Dataset name ("BeIR/nq" for Natural Questions)
        """
        self.dataset_name = dataset_name
        self.corpus_dict = None
        self.queries_dict = None
        self.qrels_dict = None

    def load_dataset(self, split: str = "test", query_sample_size: Optional[int] = 500,
                     random_seed: int = 42) -> Dict:
        """
        Load dataset using official BEIR package with intelligent sampling.
        Samples queries randomly and includes all their related corpus documents.

        Args:
            split (str): Dataset split ('train', 'dev', 'test')
            query_sample_size (int, optional): Number of queries to sample (default: 500)
            random_seed (int): Random seed for reproducible sampling

        Returns:
            Dict: Loaded dataset with corpus, queries, qrels
        """
        try:
            if self.dataset_name == "BeIR/nq":
                logger.info("Loading BeIR/nq (Natural Questions) dataset using official BEIR package...")
                corpus, queries, qrels = load_beir_nq(split=split)
            else:
                raise ValueError(f"Dataset {self.dataset_name} not supported. Supported dataset: 'BeIR/nq'")

            # Intelligent sampling: sample queries and include all related documents
            if query_sample_size and len(queries) > query_sample_size:
                logger.info(f"Sampling {query_sample_size} queries and their related documents...")

                rng = np.random.default_rng(random_seed)

                # Sample only queries that have relevance judgments
                all_query_ids = [q for q in queries if q in qrels]
                sampled_query_ids = rng.choice(all_query_ids, size=query_sample_size, replace=False).tolist()

                queries = {q_id: queries[q_id] for q_id in sampled_query_ids}

                needed_doc_ids = set()
                sampled_qrels = {}
                for query_id in sampled_query_ids:
                    sampled_qrels[query_id] = qrels[query_id]
                    needed_doc_ids.update(qrels[query_id].keys())

                original_corpus_size = len(corpus)
                corpus = {doc_id: corpus[doc_id] for doc_id in needed_doc_ids if doc_id in corpus}
                qrels = sampled_qrels

                logger.info("Smart sampling completed:")
                logger.info(f"  - Sampled {len(queries):,} queries from {len(all_query_ids):,}")
                logger.info(f"  - Included {len(corpus):,} related documents from {original_corpus_size:,}")
                logger.info(f"  - Maintained all {len(qrels):,} query-document relationships")

            self.corpus_dict = corpus
            self.queries_dict = queries
            self.qrels_dict = qrels

            logger.info("Dataset loaded successfully:")
            logger.info(f"  - Corpus: {len(corpus):,} documents")
            logger.info(f"  - Queries: {len(queries):,} queries")
            logger.info(f"  - QRELs: {len(qrels):,} queries with judgments")

            return {'corpus': corpus, 'queries': queries, 'qrels': qrels}

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def prepare_retrieval_data(self) -> Tuple[List[str], List[str], Dict]:
        """
        Convert BEIR format to retrieval format with sequential IDs.

        Returns:
            Tuple[List[str], List[str], Dict]: corpus_texts, query_texts, qrels_sequential
        """
        if self.corpus_dict is None or self.queries_dict is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        logger.info("Converting BEIR format to retrieval format...")

        corpus_texts = []
        doc_id_to_idx = {}
        for idx, (doc_id, doc_info) in enumerate(self.corpus_dict.items()):
            # BEIR docs have an optional title; prepend it when present
            title = doc_info.get('title', '')
            text = doc_info.get('text', '')
            corpus_texts.append(f"{title}. {text}" if title else text)
            doc_id_to_idx[doc_id] = idx

        query_texts = []
        query_id_to_idx = {}
        for idx, (query_id, query_text) in enumerate(self.queries_dict.items()):
            query_texts.append(query_text)
            query_id_to_idx[query_id] = idx

        qrels_sequential = {}
        if self.qrels_dict:
            for query_id, relevant_docs in self.qrels_dict.items():
                if query_id in query_id_to_idx:
                    q_idx = query_id_to_idx[query_id]
                    qrels_sequential[q_idx] = {
                        doc_id_to_idx[d]: s for d, s in relevant_docs.items() if d in doc_id_to_idx
                    }

        logger.info("Converted to retrieval format:")
        logger.info(f"  - Corpus texts: {len(corpus_texts)}")
        logger.info(f"  - Query texts: {len(query_texts)}")
        logger.info(f"  - QRELs with sequential IDs: {len(qrels_sequential)}")

        return corpus_texts, query_texts, qrels_sequential