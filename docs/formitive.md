## Why 

Name retrieval across scripts is a critical infrastructure problem in high-stakes domains including sanctions and AML screening, immigration processing, and humanitarian databases. In these settings, the same person's name may appear in a SWIFT transaction in Cyrillic, in a passport in Latin transliteration, or in an Arabic original, and a failure to match across these forms carries serious legal and humanitarian consequences. Recent work has shown that even strong multilingual dense retrievers such as BGE-M3 degrade significantly when queries are transliterated rather than written in their native script (Chari et al., SIGIR 2025). Critically, this failure is most severe for short name entities, where there is no surrounding semantic context to compensate for script-level mismatch. No existing lightweight model directly addresses cross-script phonetic alignment for name retrieval without relying on a pretrained multilingual backbone.

## What 

This is an empirical retrieval study comparing a byte-level contrastive encoder trained from scratch against classical string similarity and phonetic baselines for cross-script name retrieval. The dataset is itself a contribution: ground-truth cross-script name pairs sourced from Wikidata, augmented with LLM-generated phonetic perturbations within the Latin script. Evaluation distinguishes three query types: phonetic-only, script-only, and combined perturbations.

## How 

The encoder uses UTF-8 byte-level tokenization with a compact transformer architecture trained with InfoNCE contrastive loss and hard negatives mined via edit distance and phonetic code collision. The name database is indexed using FAISS, with an ablation over IndexFlatL2, IVF-Flat, HNSW, and IVF-PQ to characterise the recall-latency tradeoff. Performance is evaluated using MRR and Recall@k, broken down by perturbation type and script.

## Research Question 

Can a lightweight encoder trained from scratch, without any pretrained multilingual backbone, close the script gap in name entity retrieval, and does byte-level UTF-8 tokenization provide a sufficient inductive bias for cross-script phonetic alignment?

### References
- Chari, A., Ounis, I., & MacAvaney, S. (2025). Lost in Transliteration: Bridging the Script Gap in Neural IR. Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25). https://doi.org/10.1145/3726302.3730226
- Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. (2024). BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. arXiv:2402.03216
- Johnson, J., Douze, M., & Jégou, H. (2021). Billion-Scale Similarity Search with GPUs. IEEE Transactions on Big Data, 7(3), 535–547. arXiv:1702.08734
- Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). Dense Passage Retrieval for Open-Domain Question Answering. Proceedings of EMNLP 2020, 6769–6781. https://doi.org/10.18653/v1/2020.emnlp-main.550
- Levenshtein, V. I. (1966). Binary Codes Capable of Correcting Deletions, Insertions, and Reversals. Soviet Physics Doklady, 10(8), 707–710.
- Malkov, Y. A., & Yashunin, D. A. (2020). Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(4), 824–836. https://doi.org/10.1109/TPAMI.2018.2889473
- Navarro, G. (2001). A Guided Tour to Approximate String Matching. ACM Computing Surveys, 33(1), 31–88. https://doi.org/10.1145/375360.375365
- Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4), 333–389.
- Robinson, J., Chuang, C.-Y., Sra, S., & Jegelka, S. (2021). Contrastive Learning with Hard Negative Samples. ICLR 2021. arXiv:2010.04592
- van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems 30 (NeurIPS 2017). arXiv:1706.03762
- Xue, L., Barua, A., Constant, N., Al-Rfou, R., Narang, S., Kale, M., Roberts, A., & Raffel, C. (2022). ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models. Transactions of the Association for Computational Linguistics, 10, 291–306. https://doi.org/10.1162/tacl_a_00461