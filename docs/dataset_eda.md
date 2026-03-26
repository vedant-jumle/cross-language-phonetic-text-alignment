# Dataset EDA — 04_dataset.jsonl

Generated from `data/pipeline/04_dataset.jsonl` (119,040 entities).

---

## Splits

| Split | Entities |
|---|---|
| train | 95,173 |
| val | 11,893 |
| test | 11,974 |
| **total** | **119,040** |

Splits are deterministic: MD5 hash of `entity_id` modulo 1000, bucketed at 80/10/10.

---

## Positives per Entity

| Metric | Value |
|---|---|
| min | 0 |
| mean | 39.2 |
| max | 72 |
| std | 8.4 |

Each entity has ~39 positive pairs on average: 4 phonetic Latin variants + 5 names (name_en + 4 variants) × 8 scripts, minus deduplication.

---

## Positive Type Distribution

| Type | Count |
|---|---|
| combined | 3,290,117 |
| script | 918,410 |
| phonetic | 460,930 |
| **total** | **4,669,457** |

- **phonetic**: Latin script variants of name_en (LLM-generated phonetic perturbations)
- **script**: name_en transliterated into a non-Latin script
- **combined**: Latin variants (not name_en) transliterated into non-Latin scripts

---

## Source Distribution

| Source | Count | % |
|---|---|---|
| llm | 4,644,259 | 99.5% |
| wikidata | 25,198 | 0.5% |

The dataset is almost entirely LLM-generated. Wikidata provides ground-truth names for a small subset of entities.

---

## Script Distribution (LLM vs Wikidata)

| Script | Total | LLM | LLM % | Wikidata | Wiki % |
|---|---|---|---|---|---|
| ar | 516,898 | 515,080 | 100% | 1,818 | 0% |
| el | 533,170 | 532,597 | 100% | 573 | 0% |
| he | 519,858 | 518,877 | 100% | 981 | 0% |
| hi | 528,773 | 528,607 | 100% | 166 | 0% |
| ja | 529,857 | 522,324 | 99% | 7,533 | 1% |
| ko | 521,051 | 519,566 | 100% | 1,485 | 0% |
| ru | 541,946 | 532,883 | 98% | 9,063 | 2% |
| zh | 516,974 | 513,395 | 99% | 3,579 | 1% |

Script coverage is nearly uniform. Russian and Japanese have the highest Wikidata coverage (2% and 1% respectively), reflecting their prevalence in Wikidata.

---

## Anchor Name Length (UTF-8 bytes)

| Metric | Value |
|---|---|
| min | 2 |
| mean | 16.2 |
| max | 116 |
| std | 6.0 |

All names are well within the model's `max_len=256` byte limit.
