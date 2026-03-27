# Project Proposal: Cross-Script and Phonetically Robust Name Retrieval via Byte-Level Contrastive Embeddings

**Course:** DSAIT4050/Q3 — Information Retrieval  
**Submission Type:** Brief Outline of Proposed Work (Self-Proposed Topic)

---

## Group Details

**Group Number:** 10  
**Members:**
- Vedant Vivek Jumle: 5296196
- Carolyn Alcaraz: 5680581
- Gönenç Turanlı: 5794765
- Ceylin Ece: 5716950

---

## Motivation

Name retrieval across scripts is not an academic curiosity — it is a critical infrastructure problem in several high-stakes domains. In **sanctions and AML screening**, financial institutions are legally required to match transaction counterparties against sanctions lists (OFAC, UN, EU). A sanctioned individual's name recorded in Cyrillic may appear in a SWIFT transaction as a Latin phonetic variant, or a combination of both — false negatives carry serious legal consequences. In **immigration and border control**, names from Arabic, Russian, or Indic language backgrounds are transliterated inconsistently across passports and documents from different countries, requiring robust cross-script identity matching. The same problem appears in **refugee and humanitarian databases**, **hospital patient record matching** for immigrant populations, and **historical genealogical archives** where names were transcribed by hand across scripts and romanization conventions.

In all of these settings, the failure mode is the same: a name referring to the same person is not retrieved because the query and the indexed form differ in script, phonetics, or both. This work directly addresses that failure mode.

---

## Research Questions

**Primary RQ:**
> *Does byte-level contrastive representation learning enable robust cross-script name retrieval without pretrained multilingual models?*

**Supporting RQs:**
- **RQ2:** Does byte-level UTF-8 tokenization outperform Unicode codepoint tokenization for cross-script name retrieval?
- **RQ3:** Does the model generalize to name identities entirely unseen during training?
- **RQ4:** How do approximate indexing schemes (HNSW, IVF-PQ, IVF-Flat) trade off recall against query latency for embedding-based name retrieval?

---

## Research Gap

Name retrieval systems are commonly evaluated under the assumption that queries share the same script and language as the indexed documents. In practice, however, names frequently appear in perturbed forms — either through phonetic variation within the same script (e.g., *Carolyn → Karolyn*) or through transliteration across scripts (e.g., the same name written in Cyrillic or Arabic). Recent work has documented a "script gap" in neural IR systems — even strong multilingual dense retrievers like BGE-M3 fail significantly on transliterated queries (Chari et al., SIGIR 2025). However, this line of work focuses on general document retrieval and relies on pretrained multilingual models fine-tuned with a translate-train paradigm.

The specific setting of **name entity retrieval** remains underexplored. Names are short, lack semantic context, and are precisely where the script gap is most consequential in practice. Furthermore, no prior work addresses the combination of phonetic perturbation within script and cross-script transliteration jointly, in a retrieval setting with proper IR evaluation, without a pretrained backbone.

We identify the following research gap: **there is no lightweight, script-agnostic retrieval model that directly learns phonetic alignment across writing systems from raw character sequences, without relying on pretrained multilingual language models or explicit transliteration preprocessing.**

---

## Proposed Approach

We propose an end-to-end retrieval pipeline consisting of three components:

### 1. Data Construction via LLM-Based Perturbation

We will construct a dataset of name pairs using **Wikidata** as the source of ground-truth cross-lingual name correspondences. Wikidata provides English names alongside their equivalents in multiple scripts (Arabic, Cyrillic, etc.), giving us known positive pairs.

To augment this with phonetic perturbations within the Latin script, we will use **LLMs (Llama 3.1 / Gemma 3 / Qwen 3 / GLM-4)** with carefully scoped prompts to generate realistic phonetic variants of English names. The perturbation scope is explicitly defined as:

- **In scope:** phonetic spelling variants (e.g., *Catherine → Kathryn → Katerin*), transliteration noise (e.g., *John* → *Жон* (Cyrillic), *جون* (Arabic), *Ιωάννης* (Greek), *जॉन* (Hindi))
- **Out of scope:** shortened forms (*Alexander → Alex*), abbreviations (*A. Smith*), initials (*A.S.*)

This gives us three query types for evaluation: **phonetic-only**, **script-only**, and **combined**.

### 2. Byte-Level Contrastive Encoder (Trained from Scratch)

We train a lightweight transformer encoder from scratch using **UTF-8 byte-level tokenization**. Each character in any script decomposes deterministically into 1–4 bytes from a fixed vocabulary of 256 tokens. This design:

- Eliminates out-of-vocabulary tokens entirely, for any script
- Avoids BPE merge rules, which require sufficient corpus diversity to generalize
- Forces the model to learn phonetic similarity purely from the contrastive training signal

We deliberately do not use a pretrained multilingual model (e.g., XLM-R, mBERT), since their tokenization schemes are incompatible with our byte-level approach. The one honest pretrained alternative, ByT5, will be included as an optional comparison point.

**Architecture:** ~3–5-layer transformer encoder, 256–512 hidden dim, 8 attention heads. Trained with **InfoNCE contrastive loss**, using in-batch negatives and hard negatives mined from phonetically similar but distinct names.

**Hard Negative Mining Strategy:** We use a two-stage approach:

- **Stage 1 — Offline mining (initial training):** Two complementary methods are used. First, edit distance mining: for each name, we retrieve other names within a small Levenshtein distance (1–3 characters), which are surface-similar but distinct identities (e.g., *Caroline* vs. *Carolyn*). Second, phonetic code collision mining: names that share the same Soundex or Double Metaphone code are phonetically confusable by definition, making them strong negatives for learning fine-grained distinctions.
- **Stage 2 — Online embedding-based mining (later training):** Once the model has partially converged, we use its own embeddings to identify names it currently places close together but which are distinct identities — so-called *online hard negative mining*. The hard negative pool is refreshed periodically as the model improves. This stage is also an ablation candidate.

In both stages, we filter out *false negatives* — cases where two names appear similar but are actually cross-lingual equivalents of the same name — using the Wikidata ground truth to ensure mining correctness.

### 3. FAISS-Based Retrieval with Index Ablation

The English name database is indexed using FAISS. We compare the following indexing schemes as part of our evaluation:

| Index | Description |
|---|---|
| `IndexFlatL2` | Exact search — upper bound on recall |
| `IVF-Flat` | Inverted file index, approximate |
| `HNSW` | Graph-based ANN, primary production index |
| `IVF-PQ` | Compressed index, efficiency-recall tradeoff |

---

## Baselines

| Baseline | Description |
|---|---|
| Levenshtein edit distance | Classical character-level fuzzy matching |
| Soundex / Metaphone | English phonetic hashing |
| Transliterate → Edit Distance | Convert to Latin script first (ICU), then Levenshtein |
| Character n-gram BM25 | IR-native sparse retrieval baseline |

---

## Evaluation

**Dataset split:** Strictly at the name identity level. All perturbations of a given name belong exclusively to one split (train / val / test). The test set contains only names entirely unseen during training — in any form.

**Metrics:**
- MRR (primary)
- Recall@k (k = 1, 5, 10)
- NDCG@k

**Breakdown dimensions:**
- By perturbation type: phonetic-only vs. script-only vs. combined
- By script: Cyrillic queries vs. Arabic queries
- By index type: recall vs. query latency tradeoff across FAISS indices

**Ablations:**
- Byte-level (UTF-8) vs. Unicode codepoint tokenization
- InfoNCE vs. triplet loss with hard negative mining
- Architecture size: number of layers and hidden dimensionality
- Stage 1 vs. Stage 1 + Stage 2 hard negative mining

---

## Expected Contributions

1. A controlled, LLM-generated benchmark for cross-script and phonetically perturbed name retrieval
2. A compact byte-level contrastive encoder trained from scratch, shown to generalize to unseen names across scripts
3. A systematic comparison of classical string similarity baselines vs. learned dense retrieval
4. An analysis of approximate indexing schemes (FAISS) for embedding-based name retrieval

---

## Core Claim

A lightweight byte-level contrastive encoder trained from scratch, without any pretrained multilingual backbone, is sufficient for robust cross-script name retrieval — outperforming classical phonetic and string similarity baselines, particularly on script-change and combined perturbation queries.

---

## Compute & Infrastructure

- Training: TU Delft DelftBlue cluster (A100 80GB)
- Local development: 8GB GPU
- Retrieval: FAISS (CPU/GPU)
- LLM perturbation generation: Llama 3.1 / GLM-4 via API or local inference