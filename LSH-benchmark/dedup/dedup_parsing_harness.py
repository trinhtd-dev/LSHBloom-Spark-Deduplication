import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../synthetic_benchmark")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import csv
import json
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from abc import ABC, abstractmethod
from tqdm.autonotebook import tqdm
from config import DATA_SIZE

def score(name: str, output_csv: str, ground_truth_csv: str, result_csv: str):
       df = pd.read_csv(ground_truth_csv, sep='|')
       df_pred = pd.read_csv(output_csv, sep=',')
       df_pred = df_pred.rename(columns={'is_duplicate': 'predicted_duplicate'})
       df_analytics = pd.merge(left=df, right=df_pred, on='id', how='left')
       # some nan text ends up with nan labels, drop these
       print(f"Dropping {df_analytics.predicted_duplicate.isna().sum()} rows for nan labels")
       df_analytics = df_analytics.dropna(subset=['predicted_duplicate'])
       
       y_true = df_analytics['is_duplicate']
       y_pred = df_analytics['predicted_duplicate']
       precision = precision_score(y_true, y_pred)
       recall = recall_score(y_true, y_pred)
       f1 = f1_score(y_true, y_pred)
       accuracy = (df_analytics['is_duplicate'] == df_analytics['predicted_duplicate']).mean()
       auc_roc_score = roc_auc_score(y_true, y_pred)
       bal_acc_score = balanced_accuracy_score(y_true, y_pred)

       TP = ((y_true == 1) & (y_pred == 1)).sum()
       FP = ((y_true == 0) & (y_pred == 1)).sum()
       FN = ((y_true == 1) & (y_pred == 0)).sum()

       # Accuracy for modification type 1 (different parser)
       df_ana_mod1 = df_analytics[df_analytics['modification'] == 1]
       accuracy_mod1 = (df_ana_mod1['is_duplicate'] == df_ana_mod1['predicted_duplicate']).mean()

       # Accuracy for modification type 2 (truncation of text)
       df_ana_mod2 = df_analytics[df_analytics['modification'] == 2]
       accuracy_mod2 = (df_ana_mod2['is_duplicate'] == df_ana_mod2['predicted_duplicate']).mean()

       header = ("name", "precision", "recall", "f1", "auc_roc", "acc", "bal_acc", "tp", "fp", "fn", "accmod1", "accmod2")
       output = (name, precision, recall, f1, auc_roc_score, accuracy, bal_acc_score, TP, FP, FN, accuracy_mod1, accuracy_mod2)

       with open(result_csv, 'w') as fout:
           writer = csv.writer(fout)
           writer.writerow(header)
           writer.writerow(output)

class DedupHarness(ABC):
    def __init__(self, name):
        self.name = name
        self.output_header = ("is_duplicate", "id")

    @abstractmethod
    def deduplicate(self, text: str, id: int) -> bool:
        ...

    def teardown(self) -> None:
        pass

    def skip_text(self, text: str) -> bool:
        # text == "None" is an edge case where one of our examples is NaN in 
        # the ground truth df, but the string "None" in the jsonl data
        return (not isinstance(text, str)) or text is None or text.strip() == "" or text == "None"

    def write_results(self, output, output_csv):
        with open(output_csv, 'w') as fout:
            writer = csv.writer(fout, delimiter=',')
            writer.writerow(self.output_header)
            writer.writerows(output)

    def run(self, data_file, output_csv):
        output = []
        with tqdm(total=DATA_SIZE, desc="Deduplicating...") as pbar, open(data_file, 'r') as f:
            for line in f:
                obj = json.loads(line)
                # we skip null/empty text for all dedup strategies
                if self.skip_text(obj['text']):
                    continue
                is_dup = self.deduplicate(obj['text'], obj['id'])
                output.append([int(is_dup), obj['id']])
                pbar.update()

        self.write_results(output, output_csv)
        self.teardown()

    def score(self, output_csv: str, ground_truth_csv: str, result_csv: str):
        score(self.name, output_csv, ground_truth_csv, result_csv)


