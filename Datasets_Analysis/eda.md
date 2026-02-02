# Dataset Overview and Coverage Analysis

This report summarizes the key characteristics and index coverage of two financial news–related datasets used in the analysis: **FNSPID** and a **Daily Financial News dataset (2009–2020)**. 

---

## 1. FNSPID Dataset

**Source:**  
Dong, Z., Fan, X., & Peng, Z. (2024). *FNSPID: A Comprehensive Financial News Dataset in Time Series*.

**Time coverage:**  
- 1999 – 2023

**Number of entries:**  
- **>10 million news articles**

**Company universe:**  
- ~4,775 distinct stocks

**S&P 500 coverage:**  
- S&P 500 companies (union, 1999–2023): **1,022**
- Companies in dataset that are/were S&P 500 members: **646**

**Frequency statistics:**  
- Number of companies appearing in **at least 30 news articles**: **>5,000**

**Available metadata:**  
- News title  
- Article URL  
- Publisher  
- Publication date  

**Description:**  
FNSPID is a large-scale, time-series financial news dataset designed to support research at the intersection of finance and NLP. It provides broad market coverage over more than two decades, with substantial representation of major publicly traded firms, including a significant subset of S&P 500 constituents.

---

## 2. Daily Financial News Dataset (2009–2020)

**Time coverage:**  
- 2009 – 2020

**Number of entries:**  
- **>880,000 news articles**

**Company universe:**  
- ~6,000 distinct stocks

**S&P 500 coverage:**  
- S&P 500 companies (union, 2009–2020): **692**
- Companies in dataset that are/were S&P 500 members: **533**

**Frequency statistics:**  
- Number of companies appearing in **at least 30 news articles**: **4,562**

**Available metadata:**  
- News title  
- Article URL  
- Publisher  
- Publication date  

**Description:**  
This dataset focuses on daily financial news during the post–financial-crisis period. Compared to FNSPID, it has a narrower temporal range but a dense concentration of coverage for frequently mentioned firms, including strong representation of S&P 500 companies.

---

## 3. Summary Comparison

| Feature                         | FNSPID                     | Daily Financial News |
|---------------------------------|----------------------------|----------------------|
| Time range                      | 1999–2023                  | 2009–2020            |
| Number of entries               | >10 million                | >880,000             |
| Total companies                 | ~4,775                     | ~6,000               |
| S&P 500 (union, relevant years) | 1,022                      | 692                  |
| S&P 500 companies in dataset    | 646                        | 533                  |
| Companies with ≥30 articles     | >5,000                     | 4,562                |
| Metadata                        | Title, URL, Publisher, Date| Title, URL, Publisher, Date |

---

## Reference

```bibtex
@misc{dong2024fnspid,
  title={FNSPID: A Comprehensive Financial News Dataset in Time Series},
  author={Zihan Dong and Xinyu Fan and Zhiyuan Peng},
  year={2024},
  eprint={2402.06698},
  archivePrefix={arXiv},
  primaryClass={q-fin.ST}
}