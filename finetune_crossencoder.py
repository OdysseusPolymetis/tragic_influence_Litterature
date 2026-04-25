"""
finetune_crossencoder.py
─────────────────────────
Fine-tune SPhilBERTa en cross-encoder pour la détection d'intertexte
Sénèque ↔ tragiques grecs.

Usage (dans Colab) :
    !python finetune_crossencoder.py

Requiert :
    - /content/crossencoder_training_clean.json  (généré par extract_training_pairs.py)
    - sentence-transformers >= 2.2
    - transformers >= 4.30
"""

import json
import math
import random
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sklearn.model_selection import train_test_split

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR       = Path("/content")
DATA_PATH      = BASE_DIR / "crossencoder_training_clean.json"
MODEL_NAME     = "bowphs/SPhilBerta"
OUTPUT_DIR     = BASE_DIR / "crossencoder_philbert_seneca"
NUM_EPOCHS     = 40
BATCH_SIZE     = 4
WARMUP_RATIO   = 0.1
MAX_LENGTH     = 256
SEED           = 42
TEST_SIZE      = 0.15  # 15 % pour l'évaluation

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Chargement des données ─────────────────────────────────────────────────────
with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

# Filtrer les paires dont l'extraction a échoué
data = [d for d in data if "[EXTRACTION FAILED" not in d["sentence1"]
        and "[EXTRACTION FAILED" not in d["sentence2"]]

print(f"Paires disponibles : {len(data)}")

from collections import Counter
label_counts = Counter(d["label_int"] for d in data)
label_names  = {0: "negatif", 1: "topos", 2: "reminiscence", 3: "emprunt_direct"}
for lbl in sorted(label_counts):
    print(f"  label {lbl} ({label_names[lbl]}) : {label_counts[lbl]}")


# ── Augmentation des données ───────────────────────────────────────────────────
# Pour les labels rares (label 3), on peut créer des variantes légères.
# Stratégie simple : permuter les phrases dans la paire pour les labels 2 et 3
# (l'intertexte est symétrique).

def augment_pairs(data, labels_to_augment=(2, 3)):
    augmented = list(data)
    for d in data:
        if d["label_int"] in labels_to_augment:
            augmented.append({
                **d,
                "id":        d["id"] + "_aug",
                "sentence1": d["sentence2"],
                "sentence2": d["sentence1"],
            })
    return augmented

data_augmented = list(data)
print(f"\nAprès augmentation : {len(data_augmented)} paires")


# ── Split train/test ───────────────────────────────────────────────────────────
train_data, test_data = train_test_split(
    data_augmented,
    test_size=TEST_SIZE,
    stratify=[d["label_int"] for d in data_augmented],
    random_state=SEED,
)

print(f"Train : {len(train_data)} | Test : {len(test_data)}")

# Conversion au format sentence-transformers InputExample
from sentence_transformers import InputExample

train_examples = [
    InputExample(texts=[d["sentence1"], d["sentence2"]], label=float(d["label"]))
    for d in train_data
]
test_examples = [
    InputExample(texts=[d["sentence1"], d["sentence2"]], label=float(d["label"]))
    for d in test_data
]


# ── Modèle ────────────────────────────────────────────────────────────────────
model = CrossEncoder(
    MODEL_NAME,
    num_labels=1,          # régression sur [0, 1]
    max_length=MAX_LENGTH,
    activation_fn=torch.nn.Sigmoid(),
)

print(f"\nModèle chargé : {MODEL_NAME}")
print(f"Device : {model.device}")


# ── Évaluateur ────────────────────────────────────────────────────────────────
evaluator = CECorrelationEvaluator.from_input_examples(
    test_examples,
    name="seneca_greek_test",
)


# ── Entraînement ──────────────────────────────────────────────────────────────
warmup_steps = math.ceil(len(train_examples) / BATCH_SIZE * WARMUP_RATIO * NUM_EPOCHS)
print(f"\nEntraînement — {NUM_EPOCHS} epochs | batch {BATCH_SIZE} | warmup {warmup_steps} steps")

model.fit(
    train_dataloader=DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE),
    evaluator=evaluator,
    epochs=NUM_EPOCHS,
    warmup_steps=warmup_steps,
    show_progress_bar=True,
)

# ── Sauvegarde au format HuggingFace ─────────────────────────────────────────
# sentence-transformers >=3 : model.save() produit un format SentenceTransformer
# non rechargeable par CrossEncoder(path). On sauvegarde via le modele HF sous-jacent.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model.model.save_pretrained(str(OUTPUT_DIR))
model.tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"\nModele sauvegarde (format HF) : {OUTPUT_DIR}")
print(f"  Fichiers : {[p.name for p in OUTPUT_DIR.iterdir()]}")


# ── Evaluation qualitative ────────────────────────────────────────────────────
# On utilise le modele deja en memoire — aucun rechargement depuis le disque.
print("\n── Scores sur les paires de test ──")

test_pairs  = [(d["sentence1"], d["sentence2"]) for d in test_data]
test_labels = [d["label"]     for d in test_data]
test_names  = [d["label_name"] for d in test_data]
test_notes  = [d.get("note", "") for d in test_data]

scores = model.predict(test_pairs)

results = sorted(
    zip(scores, test_labels, test_names, test_notes,
        [d["sentence1"][:60] for d in test_data],
        [d["sentence2"][:60] for d in test_data]),
    key=lambda x: x[0], reverse=True,
)

print(f"\n{'Score':>6}  {'Vrai':>5}  {'Label':<16}  Latin / Grec")
print("─" * 100)
for score, true_label, label_name, note, lat, grc in results:
    flag = "✓" if abs(score - true_label) < 0.25 else "✗"
    print(f"{score:6.3f}  {true_label:5.2f}  {flag} {label_name:<14}  {lat}… / {grc}…")

# Corrélation finale
from scipy.stats import pearsonr, spearmanr
pearson_r,  _ = pearsonr(scores, test_labels)
spearman_r, _ = spearmanr(scores, test_labels)
print(f"\nCorrélation Pearson  : {pearson_r:.3f}")
print(f"Corrélation Spearman : {spearman_r:.3f}")