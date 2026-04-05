import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../synthetic_benchmark")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from typing import List
from pybloomfilter import BloomFilter
from writers import write_duplicates_to_csv
from tqdm.autonotebook import tqdm
from uniseg import wordbreak

def process_paragraph(tokens: List[str], bf) -> int:
    """
    tokens: List of n-grams of UniSeg tokens, representing a paragraph
    bf: BloomFilter

    Returns: number of duplicate n-grams in this paragraph
    """
    duplicate_ngrams = 0
    for item in tokens:
        tok = item.strip()
        if tok in bf:
            duplicate_ngrams += 1
        else:
            bf.add(tok)
    
    return duplicate_ngrams

def process_document(paragraphs: List[List[str]], bf) -> int:
    """
    paragraphs: List of lists of n-grams, each element representing a paragraph
    bf: BloomFilter

    Returns: number of duplicate n-grams in this document
    """
    duplicate_ngrams = 0
    for p in paragraphs:
        duplicate_ngrams += process_paragraph(p, bf)

    return duplicate_ngrams

def ngram(tokens: List[str], size: int, stride: int) -> List[str]:
    """
    Constructs a list of ngrams from a list of  tokens
    """

    ngrams = []
    for i in range(0, len(tokens) - size + 1, size*stride):
        ngram = ' '.join(tokens[i:i+size])
        ngrams.append(ngram)
    return ngrams

def tokenize_doc(text: str, ngram_size: int) -> (List[List[str]], int):
    """
    Tokenizes a document using the UniSeg tokenizer, separates
    it into paragraphs, and groups items into n-grams.

    text: document text
    ngram_size: size of n-grams (i.e., n)

    Returns: List of paragraphs each represented as lists of n-grams
    as well as the total number of n-grams in the document
    """

    paragraphs = text.split("\n")
    pars = []
    total_ngrams = 0
    for item in paragraphs:
        p = item.strip()
        if p == "":
            continue

        # uniseg and drop whitespace
        words = list(wordbreak.words(p))
        words = list(filter(lambda w: not all(c.isspace() for c in w), words))

        ngrams = ngram(words, ngram_size, stride=1)
        total_ngrams += len(ngrams)
        pars.append(ngrams)
    
    return pars, total_ngrams

def process_jsonl(ngram_size: int, filepath: str, T: float, bf) -> List[str]:
    """
    filepath: path to jsonl file
    T: threshold in [0,1] for n-gram duplication where we 
        mark a document as a duplicate
    bf: BloomFilter

    Returns: list of duplicate documents by string ID
    """

    dup_docs = []
    with tqdm(total=15000, desc=f"Ingesting...") as pbar,  open(filepath, 'r') as f:
        basename = os.path.basename(filepath)
        # for each doc in each jsonl file, find dups
        for i, line in enumerate(f):
            doc = json.loads(line)
            text = doc.get("text")
            if text is not None:
                paragraphs, total_ngrams = tokenize_doc(text, ngram_size)
                if total_ngrams == 0:
                    continue
                num_dup_ngrams = process_document(paragraphs, bf)
                # compare to threshold
                is_dup = (num_dup_ngrams / total_ngrams) >= T
                if is_dup:
                    doc_id = f"{basename}-{str(i)}"
                    dup_docs.append((doc_id,))
            
            pbar.update()

def dclm_dedup(name: str, dir: str, save_dir: str, result_dir: str, ngram_size: int, n: int, fp: float, T: float):
    """
    name: Name of corpus
    dir: Path to directory of jsonl files containing documents in corpus
    save_dir: Path to directory where we save the bloom filter
    result_dir: Path to directory where we save a CSV of duplicate files
    n: Number of estimated tokens in dataset
    fp: Desired false positive probability in [0,1]
    T: Desired threshold in [0,1]
    """

    files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".jsonl")]
    duplicates = []
    save_file = os.path.join(save_dir, "dclm.bf")
    bf = BloomFilter(n, fp, save_file)
    for f in files:
        dup = process_jsonl(ngram_size, f, T, bf)
        duplicates.extend(dup)
    
    write_file = os.path.join(result_dir, "dclm_duplicates.csv")
    write_duplicates_to_csv(duplicates, write_file, name)
    
    


