"""Offline training only; never called by the web server."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from collections import Counter
import joblib
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from app.text import normalize

LABELS = {"billing", "technical_issue", "general_inquiry"}

def train(csv_path, output, demo=False):
    rows = list(csv.DictReader(Path(csv_path).open(encoding="utf-8-sig", newline="")))
    seen = {}
    for row in rows:
        if not {"text", "label"} <= row.keys():
            raise ValueError("CSV requires text,label columns")
        text, label = normalize(row["text"]), row["label"].strip()
        if not text or len(row["text"]) > 10000 or label not in LABELS:
            raise ValueError("Empty/oversized text or unsupported label")
        if text in seen and seen[text] != label:
            raise ValueError("Conflicting labels for duplicate text")
        seen[text] = label
    counts = Counter(seen.values())
    if set(counts) != LABELS or min(counts.values()) < 10:
        raise ValueError("Need at least 10 distinct messages per category")
    texts, labels = list(seen), list(seen.values())
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=.25, stratify=labels, random_state=42)
    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, max_features=30000)),
                      ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    accepted = probabilities.max(axis=1) >= .65
    report = {"demo_data": demo, "split": "stratified random 75/25 after normalized exact deduplication",
              "train_count": len(x_train), "test_count": len(x_test), "accuracy": accuracy_score(y_test,pred),
              "macro_f1": f1_score(y_test,pred,average="macro"), "classification_report": classification_report(y_test,pred,output_dict=True,zero_division=0),
              "labels": list(model.classes_), "confusion_matrix": confusion_matrix(y_test,pred,labels=model.classes_).tolist(),
              "coverage_at_065": float(accepted.mean()), "accepted_accuracy_at_065": float((pred[accepted] == __import__('numpy').array(y_test)[accepted]).mean()) if accepted.any() else None}
    # Export the evaluated model; no silent refit on the held-out test set.
    out = Path(output); out.mkdir(parents=True,exist_ok=True)
    temporary = out / "model.joblib.tmp"
    joblib.dump(model, temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    os.replace(temporary, out / "model.joblib")
    metadata = {"version": digest[:12], "sha256": digest, "sklearn_version": sklearn.__version__, "demo_data": demo, "labels": sorted(LABELS)}
    (out / "metadata.json").write_text(json.dumps(metadata,indent=2))
    (out / "evaluation.json").write_text(json.dumps(report,indent=2))
    return report

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",default="data/demo.csv")
    parser.add_argument("--output",default="models")
    parser.add_argument("--demo",action="store_true",help="Mark artifact as trained on synthetic demo data")
    args=parser.parse_args()
    print(json.dumps(train(args.data,args.output,args.demo),indent=2))
