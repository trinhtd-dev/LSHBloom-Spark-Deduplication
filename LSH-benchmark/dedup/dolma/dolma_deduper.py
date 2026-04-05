import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../synthetic_benchmark")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from typing import List
from pybloomfilter import BloomFilter
from writers import write_duplicates_to_csv
from tqdm.autonotebook import tqdm
import regex as re

def tokenize(s: str) -> List[str]:
    """
    Tokenizes following DOLMA deduper.rs tokenization scheme
    """
    # Split the string into word boundaries
    words = re.findall(r'\X', s, flags=re.UNICODE)
    
    # Filter out whitespace-only tokens
    return list(filter(lambda w: not all(c.isspace() for c in w), words))

def ngram(tokens: List[str], size: int, stride: int) -> List[str]:
    """
    Constructs a list of ngrams from a list of  tokens
    """

    ngrams = []
    for i in range(0, len(tokens) - size + 1, size*stride):
        ngram = ' '.join(tokens[i:i+size])
        ngrams.append(ngram)
    return ngrams


def ingest_doc_paragraphs(text: str, T: float, bf) -> bool:
    """
    Dedup based on whole-paragraph overlap between documents
    """
    if text is None or text == "":
        return False
    
    text_len = 0
    dup_len = 0
    paragraphs = text.split("\n")
    for p in paragraphs:
        if len(p) == 0:
            continue

        text_len += len(p)
        # check if duplicated in bf, insert if not
        if p.strip() in bf:
            dup_len += len(p)
        else:
            bf.add(p.strip())

    text_overlap = dup_len / (text_len or 1)
    return text_overlap >= T
    
def ingest_doc_ngram(text: str, T: float, bf, size, stride) -> bool:
    """
    Dedup based on n-gram overlap between documents
    """
    if text is None or text == "":
        return False
    
    paragraphs = text.split("\n")
    ngrams = []
    for p in paragraphs:
        ngrams += ngram(tokenize(p), size, stride)
    
    ngram_len = len(ngrams)
    if ngram_len == 0:
        return False
    
    dup_cnt = 0
    for ng in ngrams:
        if ng in bf:
            dup_cnt += 1
        else:
            bf.add(ng)
    
    ngram_overlap = dup_cnt / ngram_len
    return ngram_overlap >= T



def ingest_doc(text, T, bf, ngram_setting):
    if not ngram_setting:
        return ingest_doc_paragraphs(text, T, bf)
    
    return ingest_doc_ngram(text, T, bf, ngram_setting["size"], ngram_setting["stride"])

class DolmaDedup:
    """
    Works with a corpus of size n. Queries each paragraph from a document 
    against the bloom filter, if the proportion of text present duplicated 
    in this document is above the threshold, T in [0,1], then we mark this document 
    as a duplicate. This deduplicates documents based on their proportion of
    overlapping spans of paragraphs.
    """
    def __init__(self, n: int, fp=0.001, T=1.0, write_file="dolma_dedup.csv", save_file="dolma.bf", ngram_size=-1, ngram_stride=-1):
        self.n = n
        self.fp = fp
        self.T = T
        self.write_file = write_file
        self.save_file = save_file
        self.bf = BloomFilter(self.n, self.fp, self.save_file)
        self.n_gram_mode = (ngram_size != -1) or (ngram_stride != -1)
        self.n_gram_size = ngram_size if ngram_size != -1 else 1
        self.n_gram_stride = ngram_stride if ngram_stride != -1 else 1


    def dedup_single(self, text: str):
        ngram_setting = None if not self.n_gram_mode else {"size": self.n_gram_size, "stride": self.n_gram_stride}
        is_dup = ingest_doc(text, self.T, self.bf, ngram_setting)
        return is_dup


    def run(self, corpus_dir, corpus_name):
        ngram_setting = None if not self.n_gram_mode else {"size": self.n_gram_size, "stride": self.n_gram_stride}
        files = [os.path.join(corpus_dir, f) for f in os.listdir(corpus_dir) if f.endswith(".jsonl")]
        with tqdm(total=self.n, desc=f"Ingesting...") as pbar:
            for file in files:
                dup_docs = []
                with open(file, 'r') as f:
                    basename = os.path.basename(file)
                    # for each doc in each jsonl file, find dups
                    for i, line in enumerate(f):
                        doc = json.loads(line)
                        text = doc.get("text")
                        is_dup = ingest_doc(text, self.T, self.bf, ngram_setting)
                        if is_dup:
                            doc_id = f"{basename}-{str(i)}"
                            dup_docs.append((doc_id,))
                        
                        pbar.update()

                # write out to csv
                write_duplicates_to_csv(dup_docs, self.write_file, corpus_name)

