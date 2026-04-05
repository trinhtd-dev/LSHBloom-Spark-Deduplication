import json
import os
from pathlib import Path
import pandas as pd
import random
import numpy as np
from collections import defaultdict
from typing import Optional

def truncate(s: str, percentage: float, rel_location: float) -> str:
    """
    Given a (long) string s, truncates `percentage`% of the string off of it (at the location `rel_location`) 
    and returns the truncated string.

    Supposed to serve as a sensitivity check for the Dedup algo.
    
    Inputs:
    - percentage: either {1, 2.5, 5, 7.5, 10, 15, 20}, percentage of the string to be removed.
    - rel_location: either {0, 0.25, 0.5, 0.75, 1.0}, location relative to the string where truncation should start.
    """
    
    # Validate inputs: at most 20% of content is deleted anywhere from beginning `0` to end `1.0`
    valid_percentages = [1, 2, 5, 7, 10, 15, 20]
    valid_rel_locations = [0, 0.25, 0.5, 0.75, 1.0]

    # check if input valid. 
    if percentage not in valid_percentages:
        raise ValueError(f"Invalid percentage: {percentage}. Must be one of {valid_percentages}.")
    
    if rel_location not in valid_rel_locations:
        raise ValueError(f"Invalid relative location: {rel_location}. Must be one of {valid_rel_locations}.")
    
    # Calculate the number of characters to remove
    total_chars = len(s)
    chars_to_remove = int((percentage / 100) * total_chars)

    # char index of midpoint of truncation interval
    start_pos = int(total_chars * rel_location)
    
    # infers first and last char idnex of truncation interval
    start_truncate = max(0, start_pos - chars_to_remove // 2)
    end_truncate = min(total_chars, start_truncate + chars_to_remove)
    
    # truncation by slicing
    truncated_str = s[:start_truncate] + s[end_truncate:]
    
    return truncated_str

def validate_text(text):
    if text is None or text == "":
        return False
    return True


def parse_jsonl_line(line: str) -> Optional[dict]:
    """Parse one JSONL line safely and return None on malformed input."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _first_non_empty_str(data: dict, keys) -> Optional[str]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip() != "":
            return value
    return None


def _extract_text_from_pages(data: dict) -> Optional[str]:
    pages = data.get("pages")
    if not isinstance(pages, list):
        return None

    page_texts = []
    for page in pages:
        if isinstance(page, dict):
            text = page.get("text")
            if isinstance(text, str) and text.strip() != "":
                page_texts.append(text)

    if not page_texts:
        return None
    return "\n".join(page_texts)


def normalize_record(data: dict) -> Optional[dict]:
    """
    Normalize heterogeneous schemas into records with guaranteed `path` and `text`.

    Supports the original schema (`path`, `text`) and user schema (`doc_id`,
    `source_pdf`, `text`, `pages`, nested `metadata`).
    """
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}

    path = _first_non_empty_str(
        data,
        ["path", "text_path", "html_path", "source_pdf", "pdf_url", "html_url", "doc_id"],
    )
    if path is None:
        path = _first_non_empty_str(metadata, ["pdf_url"])

    text = _first_non_empty_str(data, ["text", "content", "body"])
    if text is None:
        text = _extract_text_from_pages(data)
    if text is None:
        text = _first_non_empty_str(metadata, ["summary", "title"])

    if path is None or not validate_text(text):
        return None

    normalized = dict(data)
    normalized["path"] = path
    normalized["text"] = text
    return normalized

def get_parser_folder(p_src, parser):
    p_content = (p_src / f"joint2_to_{parser}") / 'parsed_pdfs'
    # if not p_content.exists():
    #     p_content = (p_src / f"joint2_to_{parser}") / 'parsed_pdfs'
    assert p_content.exists(), f"Content directory does not exist: {p_content}"
    return p_content

def sample_parser_data(parser_sources, N_per_source, p_src, sample_limit=80):
    """
    Sample N_per_source JSONL entries from each parser and store the results.

    Data comes from  `XXX`
    where I have output directories of each parser. E.g for PyMuPDF
    ```
      XXX/joint_to_pymupdf/parsed_pdfs/*.jsonl
    ```
    """
    
    filepaths_set = set()  # Use a set for faster lookups
    data_list = []  # To store non-duplicate samples
    sampled_paths_per_parser = {}

    # loop parser sources
    for i, parser in enumerate(parser_sources):
        print(f'Parser : {parser}')
        p_content = get_parser_folder(p_src, parser)

        json_list_loc = [p_content / f for f in os.listdir(p_content) if f.endswith('.jsonl')]
        random.shuffle(json_list_loc)

        data_list_loc = []
        parser_paths = []

        # sample files and paths
        # random.seed(i * 467 + 88)
        random.shuffle(json_list_loc)

        # parsed text mostly spread across several JSONLs (per parser)
        for jsonl_file in json_list_loc:
            with open(jsonl_file, 'r') as f:
                for line in f:
                    data = parse_jsonl_line(line)
                    if data is None:
                        continue

                    data = normalize_record(data)
                    if data is None:
                        continue

                    # Use filename-like suffix as duplicate key across parsers.
                    file_name = data['path'].replace('\\', '/').split('/')[-1]

                    # control for duplicates: if another parser got the `path` already, skip the PDF
                    if file_name not in filepaths_set:
                        data['parser'] = parser
                        data_list_loc.append(data)
                        filepaths_set.add(file_name)
                        parser_paths.append(file_name)

                    # break when you have enough new PDFs (unseen to previous parsers)
                    if len(data_list_loc) >= N_per_source:
                        break
            # append them
            if len(data_list_loc) >= N_per_source:
                data_list += data_list_loc
                # store 80 paths (and for this parser that serve as dedicated duplication candidates for later sampling)
                sampled_paths_per_parser[parser] = parser_paths[:sample_limit]  

                # job done, no need to visit further JSONLs of this `parser`, move on to next `parser`
                break
        # - jsonl loop (of parser) END - 
    # - parser loop END - 

    print(f"Collected {len(data_list)} samples.")
    return data_list, sampled_paths_per_parser


def collect_duplicates(sampled_paths_per_parser, parser_sources, p_src, sample_limit=80):
    """
    Collect duplicate entries by looking up files in the other parsers' folders.

    Method leverages `sampled_paths_per_parser` that were sampled previously. We won't need all of them though;
    just serves as a basis for a matching.
    Choices below are hardcoded (I'll leave them as is): Namely, 80, 35, 25. Just chosen to get ~100 a hundred duplicates.

    For each sampled PDF of some `parser` (source parser `src_parser`), we sample another parser `random_other_parsers` and look up
    if the file (with the same `path`) is present in any of its JSONLs. 
    
    """
    data_dupl_list = []

    # For each parser, assign random other parser names to 50 entries
    for src_parser, paths in sampled_paths_per_parser.items():
        random_other_parsers = {other_parser: random.sample(paths, min(len(paths), sample_limit)) for other_parser in parser_sources if other_parser != src_parser}
        
        # For each other parser, search for the paths in the respective jsonl files
        for other_parser, assigned_paths in random_other_parsers.items():
            print(f"Searching for paths in parser: {other_parser} (from {src_parser})")
            p_content = get_parser_folder(p_src, other_parser)
            json_list_loc = [p_content / f for f in os.listdir(p_content) if f.endswith('.jsonl')]
            random.shuffle(json_list_loc)

            found_count = 0
            other_parser_data = []  # Temporarily store potential duplicates
            for jsonl_file in json_list_loc:
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        data = parse_jsonl_line(line)
                        if data is None:
                            continue

                        data = normalize_record(data)
                        if data is None:
                            continue

                        file_name = data['path'].replace('\\', '/').split('/')[-1]

                        if file_name in assigned_paths:
                            data['parser'] = other_parser
                            other_parser_data.append(data)
                            found_count += 1

                            # Remove the found path to avoid processing it again
                            assigned_paths.remove(file_name)

            # Filter to keep only paths that appear exactly once across both sets
            selected_data = [d for d in other_parser_data if sum([d['path'] == x['path'] for x in data_dupl_list]) == 0]
            data_dupl_list += selected_data

   
    # shuffle and apply limit
    data_dupl_list = random.sample(data_dupl_list, min(len(data_dupl_list), sample_limit * len(parser_sources)))

    return data_dupl_list


def make_benchmark_dataframe(data_list, data_dupl_list, data_trunc_list):
    # instantiate pandas
    df_dedup = pd.DataFrame(data_list+data_dupl_list+data_trunc_list)

    # Modification labels
    modification = ([0] * len(data_list)) + ([1] * len(data_dupl_list)) + ([2] * len(data_trunc_list))
    df_dedup['modification'] = modification

    # shuffle
    df_dedup = df_dedup.sample(frac=1).reset_index(drop=True)

    # ID
    df_dedup['id'] = range(len(df_dedup))

    is_duplicate = []

    # text length
    df_dedup['text_len'] = (df_dedup['text'].astype(str).str.len()).astype(int)
    df_dedup['new_text'] = list(df_dedup['text'])

    df_dedup['trunc_percentage'] = 0
    df_dedup['rel_location'] = 0

    # Sample the percentage and relative location
    valid_percentages = [1, 2, 5, 7, 10, 15, 20]
    valid_rel_locations = [0, 0.25, 0.5, 0.75, 1.0]

    # Apply truncation to rows that satisfy the condition
    for index, row in df_dedup[df_dedup['modification'] == 2].iterrows():
        # Check if the 'text' field is not None or NaN
        if pd.notna(row['text']) and isinstance(row['text'], str):
            # Randomly sample percentage and location
            trunc_percentage = random.choice(valid_percentages)
            rel_location = random.choice(valid_rel_locations)

            # Apply truncation
            truncated_text = truncate(row['text'], trunc_percentage, rel_location)

            # Store results back in the DataFrame
            df_dedup.at[index, 'trunc_percentage'] = trunc_percentage
            df_dedup.at[index, 'rel_location'] = rel_location
            df_dedup.at[index, 'new_text'] = truncated_text
            # df_dedup.at[index, 'is_duplicate'] = 1
        else:
            # If 'text' is None or invalid, skip truncation and store None
            df_dedup.at[index, 'trunc_percentage'] = 0
            df_dedup.at[index, 'rel_location'] = 0
            df_dedup.at[index, 'new_text'] = str(row['text'])

    # path
    # df_dedup['path'] = df_dedup['path'].apply(lambda x: '/'.join(x.split('/')[-3:]))


    # loop each row of `df_dedup` if `path` was previously seen, add True else False to the list; in the end assign it to column `is_duplicate`
    seen_paths = set()
    is_duplicate = []
    # iterate rows
    for index, row in df_dedup.iterrows():
        # label on the basis of filenames (stem+ext) since we searched on the basis of filenames
        path = row['path']#.split('/')[-1]

        # NB (Arham): this logic accomodates for a formatting error in our path metadata, where some paths for 'biorxiv' are mistakenly
        # using 'bioarxiv', this was apparently eventually fixed but some early parsed pdfs use the incorrect corpus name.
        # Similarly for 'medrxiv'
        path = path.replace('bioarxiv/', 'biorxiv/')
        path = path.replace('medarxiv/', 'medrxiv/')
        
        # Check if the path has been seen before
        if path in seen_paths:
            is_duplicate.append(1)
        else:
            is_duplicate.append(0)
            seen_paths.add(path)

    # Assign the is_duplicate list as a new column in df_dedup
    df_dedup['is_duplicate'] = is_duplicate

    # # path
    # df_dedup['path'] = df_dedup['path'].apply(lambda x: '/'.join(x.split('/')[-3:]))

    # get meta
    df_meta_path = Path('scaling_data/frames/df_mod_10240.csv')
    df_10240 = pd.read_csv(df_meta_path, sep='|')
    df_10240 = df_10240[df_10240['subclass'].str.len() < 50]

    # merge
    df_merged = pd.merge(left=df_dedup, right=df_10240[['path','publisher', 'class', 'subclass']], on='path', how='left')
    df_merged = df_merged[['text', 'path', 'id', 'metadata', 'parser', 'is_duplicate',
       'modification', 'text_len', 'new_text', 'trunc_percentage',
       'rel_location', 'publisher', 'class', 'subclass']]
    
    # Display the modified DataFrame
    print(df_merged.head())

    return df_merged