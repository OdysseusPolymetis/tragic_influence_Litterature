"""
extract_training_pairs.py  (v2)
────────────────────────────────
Extrait les paires d'entraînement pour le cross-encoder depuis le corpus
Perseus (canonical-latinLit / canonical-greekLit) déjà cloné dans /content.

Résolution par work_id direct — aucun matching de titre.
Structure attendue :
  Latin : canonical-latinLit/data/{author_id}/{work_id}/{author_id}.{work_id}.*-lat*.xml
  Grec  : canonical-greekLit/data/{tlg_author}/{work_id}/{tlg_author}.{work_id}.*-grc*.xml

Usage :
    !python /content/extract_training_pairs.py
"""

import json, re, unicodedata
from pathlib import Path
from copy import deepcopy
from lxml import etree
import pandas as pd

BASE_DIR   = Path("/content")
LATIN_REPO = BASE_DIR / "canonical-latinLit" / "data"
GREEK_REPO = BASE_DIR / "canonical-greekLit"  / "data"
REFS_PATH  = BASE_DIR / "parallel_references.json"
OUT_JSON   = BASE_DIR / "crossencoder_training_data.json"
OUT_CE     = BASE_DIR / "crossencoder_training_clean.json"
OUT_CSV    = BASE_DIR / "crossencoder_training_data.csv"
LOG_PATH   = BASE_DIR / "crossencoder_extraction_log.txt"

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
EXCL   = {"speaker", "stage", "note", "head"}


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[\[\]{}()<>]", " ", text)
    text = re.sub(r"\d+",          " ", text)
    text = re.sub(r'["""\'\'.,;:!?·…—–\\/_|*=+#-]', " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── Utilitaires TEI ───────────────────────────────────────────────────────────

def _remove_tags(root, tag_names):
    for tag in tag_names:
        for el in root.xpath(f".//*[local-name()='{tag}']"):
            parent = el.getparent()
            if parent is None:
                continue
            tail, prev = el.tail, el.getprevious()
            parent.remove(el)
            if tail:
                if prev is not None:
                    prev.tail = (prev.tail or "") + tail
                else:
                    parent.text = (parent.text or "") + tail


def _parse_n(n_raw: str) -> int | None:
    digits = re.sub(r"[^0-9]", "", n_raw)
    return int(digits) if digits else None


def extract_line_range(xml_path: Path, line_start: int, line_end: int) -> str:
    """
    Extrait le texte des vers line_start..line_end depuis un fichier TEI.
    Trois stratégies en cascade : <l n>, <milestone>/<lb n>, fallback positionnel.
    """
    try:
        tree = etree.parse(str(xml_path), etree.XMLParser(recover=True))
    except Exception:
        return ""

    root = tree.getroot()
    body = root.find(".//tei:body", namespaces=TEI_NS) or root
    body = deepcopy(body)
    _remove_tags(body, EXCL)

    # ── Stratégie 1 : <l n="X"> ───────────────────────────────────────────────
    l_els = body.xpath(".//*[local-name()='l' and @n]")
    if l_els:
        parts = []
        for el in l_els:
            n = _parse_n(el.get("n", ""))
            if n is not None and line_start <= n <= line_end:
                t = " ".join(el.itertext()).strip()
                if t:
                    parts.append(t)
        if parts:
            return normalize_text(" ".join(parts))

    # ── Stratégie 2 : <milestone n> ou <lb n> ────────────────────────────────
    ms_els = body.xpath(".//*[(local-name()='milestone' or local-name()='lb') and @n]")
    if ms_els:
        collecting, buf = False, []
        for el in body.iter():
            local = el.tag.split("}")[-1] if "}" in el.tag else el.tag
            if local in ("milestone", "lb"):
                n = _parse_n(el.get("n", ""))
                if n is not None:
                    collecting = line_start <= n <= line_end
            if collecting:
                for t in (el.text, el.tail):
                    if t and t.strip():
                        buf.append(t.strip())
        if buf:
            return normalize_text(" ".join(buf))

    # ── Stratégie 3 : fallback positionnel ────────────────────────────────────
    words = " ".join(
        t.strip() for t in body.itertext() if t and t.strip()
    ).split()
    if not words:
        return ""
    span  = max(1, line_end - line_start + 1)
    start = max(0, int(len(words) * (line_start - 1) / max(line_end, 1)))
    end   = min(len(words), start + span * 8)
    return normalize_text(" ".join(words[start:end]))


# ── Résolution des fichiers XML ───────────────────────────────────────────────

def resolve_xml(repo_dir: Path, author_id: str, work_id: str,
                lang_token: str) -> Path | None:
    """
    Trouve le fichier XML d'une œuvre par résolution directe.
    Cherche *-{lang_token}*.xml dans repo_dir/author_id/work_id/.
    """
    work_dir = repo_dir / author_id / work_id
    if not work_dir.exists():
        return None

    # Fichiers correspondant à la langue demandée
    hits = [
        p for p in sorted(work_dir.glob("*.xml"))
        if f"-{lang_token}" in p.name and p.name != "__cts__.xml"
    ]
    if hits:
        return hits[0]

    # Fallback : tout XML non anglais et non __cts__
    fallback = [
        p for p in sorted(work_dir.glob("*.xml"))
        if p.name != "__cts__.xml"
        and "-eng" not in p.name
        and ".eng" not in p.name
    ]
    return fallback[0] if fallback else None


# ── Extraction principale ─────────────────────────────────────────────────────

def extract_all_pairs(refs: list[dict]) -> tuple[list[dict], list[str]]:
    pairs, warnings = [], []

    for ref in refs:
        pid = ref["id"]
        lat = ref["latin"]
        grc = ref["greek"]

        # Résolution Latin
        lat_xml = resolve_xml(LATIN_REPO, lat["author_id"], lat["work_id"], "lat")
        if lat_xml is None:
            warnings.append(f"[{pid}] Latin introuvable : {lat['author_id']}/{lat['work_id']}")
            continue

        # Résolution Grec
        grc_xml = resolve_xml(GREEK_REPO, grc["tlg_author"], grc["work_id"], "grc")
        if grc_xml is None:
            warnings.append(f"[{pid}] Grec introuvable : {grc['tlg_author']}/{grc['work_id']}")
            continue

        # Extraction du texte
        lat_text = extract_line_range(lat_xml, lat["line_start"], lat["line_end"])
        grc_text = extract_line_range(grc_xml, grc["line_start"], grc["line_end"])

        if not lat_text or len(lat_text.split()) < 3:
            warnings.append(
                f"[{pid}] Texte latin vide ({lat_xml.name} "
                f"v.{lat['line_start']}-{lat['line_end']})"
            )
            lat_text = None

        if not grc_text or len(grc_text.split()) < 3:
            warnings.append(
                f"[{pid}] Texte grec vide ({grc_xml.name} "
                f"v.{grc['line_start']}-{grc['line_end']})"
            )
            grc_text = None

        pairs.append({
            "id":             pid,
            "label":          ref["label"],
            "label_name":     ref["label_name"],
            "sentence1":      lat_text,
            "sentence2":      grc_text,
            "label_norm":     ref["label"] / 3.0,
            "lat_file":       lat_xml.name,
            "grc_file":       grc_xml.name,
            "lat_lines":      f"{lat['line_start']}-{lat['line_end']}",
            "grc_lines":      f"{grc['line_start']}-{grc['line_end']}",
            "note":           ref.get("note", ""),
            "source_scholar": ref.get("source_scholar", ""),
        })

    return pairs, warnings


# ── Format pour sentence-transformers CrossEncoder ────────────────────────────

def to_crossencoder_format(pairs: list[dict]) -> list[dict]:
    return [
        {
            "id":             p["id"],
            "sentence1":      p["sentence1"],
            "sentence2":      p["sentence2"],
            "label":          p["label_norm"],
            "label_int":      p["label"],
            "label_name":     p["label_name"],
            "note":           p["note"],
            "source_scholar": p["source_scholar"],
        }
        for p in pairs
        if p["sentence1"] and p["sentence2"]
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Chargement des références...")
    with open(REFS_PATH, encoding="utf-8") as f:
        refs = json.load(f)
    print(f"{len(refs)} références chargées.")

    print("\nExtraction du texte depuis Perseus...")
    pairs, warnings = extract_all_pairs(refs)

    # ── Statistiques ──────────────────────────────────────────────────────────
    from collections import Counter
    label_names = {3: "emprunt_direct", 2: "reminiscence", 1: "topos_commun", 0: "negatif"}

    total     = len(pairs)
    extracted = sum(1 for p in pairs if p["sentence1"] and p["sentence2"])
    failed    = sum(1 for p in pairs if not p["sentence1"] or not p["sentence2"])

    print(f"\n{'─'*50}")
    print(f"Paires résolues      : {total}")
    print(f"  dont texte extrait : {extracted}")
    print(f"  dont extraction KO : {failed}")
    print(f"  avertissements     : {len(warnings)}")
    print(f"{'─'*50}")

    counts = Counter(p["label"] for p in pairs if p["sentence1"] and p["sentence2"])
    for lbl in sorted(counts, reverse=True):
        print(f"  label {lbl} ({label_names[lbl]}) : {counts[lbl]}")

    if warnings:
        LOG_PATH.write_text("\n".join(warnings), encoding="utf-8")
        print(f"\nLog des avertissements → {LOG_PATH}")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    ce_data = to_crossencoder_format(pairs)
    with open(OUT_CE, "w", encoding="utf-8") as f:
        json.dump(ce_data, f, ensure_ascii=False, indent=2)

    pd.DataFrame(pairs).to_csv(OUT_CSV, index=False)

    print(f"\nJSON complet    → {OUT_JSON}")
    print(f"Format CE clean → {OUT_CE}  ({len(ce_data)} paires utilisables)")
    print(f"CSV             → {OUT_CSV}")

    # ── Aperçu ────────────────────────────────────────────────────────────────
    shown = set()
    print(f"\n{'─'*50}")
    print("Aperçu par label :")
    for p in ce_data:
        lbl = p["label_int"]
        if lbl not in shown:
            print(f"\n  [{lbl} — {label_names[lbl]}]")
            print(f"  LAT : {p['sentence1'][:100]}")
            print(f"  GRC : {p['sentence2'][:100]}")
            shown.add(lbl)
        if len(shown) == 4:
            break