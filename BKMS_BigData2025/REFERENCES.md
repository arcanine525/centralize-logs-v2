# Web Attack Detection - References & Dataset

## 1. Methodology Paper

### Log-based Anomaly Detection with Deep Learning: How Far Are We?

| Field | Value |
|-------|-------|
| **Authors** | Van-Hoang Le, Hongyu Zhang |
| **Conference** | ICSE 2022 (44th International Conference on Software Engineering) |
| **Year** | 2022 |
| **DOI** | 10.1145/3510003.3510155 |
| **arXiv** | https://arxiv.org/abs/2202.04301 |
| **PDF Download** | https://arxiv.org/pdf/2202.04301.pdf |

### Citation (BibTeX)
```bibtex
@inproceedings{le2022log,
  title={Log-based anomaly detection with deep learning: How far are we?},
  author={Le, Van-Hoang and Zhang, Hongyu},
  booktitle={Proceedings of the 44th International Conference on Software Engineering},
  pages={1356--1367},
  year={2022}
}
```

### Key Contributions Used in Our System
- MLP (Multi-Layer Perceptron) architecture for log anomaly detection
- Feature engineering methodology from log data
- Sliding window approach for temporal feature aggregation
- Evaluation metrics and benchmarking methodology

---

## 2. Training Dataset

### WebAttack-CVSSMetrics

| Field | Value |
|-------|-------|
| **Name** | WebAttack-CVSSMetrics |
| **Source** | HuggingFace Datasets |
| **Author** | chYassine |
| **License** | Apache 2.0 |
| **URL** | https://huggingface.co/datasets/chYassine/WebAttack-CVSSMetrics |
| **Download** | `pip install datasets` then `load_dataset("chYassine/WebAttack-CVSSMetrics")` |

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total Logs | 18,842 |
| Format | Apache Combined Log Format |
| Normal Traffic | 10,000 (53.1%) |
| Attack Traffic | 8,842 (46.9%) |
| File Size | ~2 MB |

### Attack Types Distribution

| Attack Type | Count | Percentage | Description |
|-------------|-------|------------|-------------|
| LFI | 3,088 | 16.4% | Local File Inclusion |
| SSTI | 2,105 | 11.2% | Server-Side Template Injection |
| SQL Injection | 1,706 | 9.1% | SQL Injection attacks |
| XSS | 648 | 3.4% | Cross-Site Scripting |
| SSRF | 576 | 3.1% | Server-Side Request Forgery |
| File Upload | 437 | 2.3% | Malicious file upload |
| CSRF | 282 | 1.5% | Cross-Site Request Forgery |
| **Normal** | 10,000 | 53.1% | Legitimate traffic |

### Data Format

Each row contains:
- `_raw`: Raw Apache log line
- `Type`: Attack type (null for normal traffic)
- `CVSS`: CVSS metrics string
- `Score`: Risk score (0-10)

### Sample Logs

**Normal Traffic:**
```
91.251.15.250 - - [22/Jan/2019:18:18:42 +0330] "GET /apple-touch-icon-precomposed.png HTTP/1.1" 404 392 "-" "MobileSafari/604.1 CFNetwork/976 Darwin/18.2.0" "-"
Type: null, Score: 0
```

**SQL Injection Attack:**
```
192.168.202.110 - - [16/Mar/2012:19:38:49 +0330] "GET /phpmyadmin/index.php?option=weblinks&Itemid=2&catid=-1 UNION SELECT 0,1,2,368612312108,4,5,6,7,8,9,10,11-- HTTP/1.1" 200 8625 "-" "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0)" "-"
Type: Sql Injection, Score: 5.8
```

**LFI Attack:**
```
10.128.0.205 - - [24/Mar/2018:22:45:34 +0330] "GET /..../..../..../..../..../..../..../..../..../windows/win.ini HTTP/1.1" 403 269 "-" "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0)" "-"
Type: LFI, Score: 5.5
```

### Citation (Dataset)
```
chYassine. (2024). WebAttack-CVSSMetrics [Dataset]. 
Hugging Face. https://huggingface.co/datasets/chYassine/WebAttack-CVSSMetrics
```

---

## 3. Feature Engineering

### 16 Features Extracted per Time Window

| # | Feature Name | Type | Description |
|---|--------------|------|-------------|
| 1 | request_count | int | Total requests in window |
| 2 | unique_ips | int | Distinct IP addresses |
| 3 | requests_per_ip | float | Average requests per IP |
| 4 | unique_methods | int | Distinct HTTP methods |
| 5 | get_ratio | float | GET requests / total |
| 6 | post_ratio | float | POST requests / total |
| 7 | avg_bytes | float | Mean response bytes |
| 8 | total_bytes | int | Sum of response bytes |
| 9 | status_2xx_ratio | float | Success responses / total |
| 10 | status_4xx_ratio | float | Client errors / total |
| 11 | status_5xx_ratio | float | Server errors / total |
| 12 | unique_urls | int | Distinct URLs requested |
| 13 | avg_url_length | float | Mean URL length |
| 14 | request_rate | float | Requests per second |
| 15 | ip_entropy | float | Shannon entropy of IP distribution |
| 16 | url_entropy | float | Shannon entropy of URL distribution |

---

## 4. Model Architecture

```
Input Layer (16 features)
        ↓
Linear(16 → 64) + BatchNorm1d + ReLU + Dropout(0.3)
        ↓
Linear(64 → 32) + BatchNorm1d + ReLU + Dropout(0.3)
        ↓
Linear(32 → 16) + BatchNorm1d + ReLU + Dropout(0.3)
        ↓
Linear(16 → 1) + Sigmoid
        ↓
Output (probability 0-1)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | Binary Cross Entropy |
| Early Stopping Patience | 10 epochs |
| Max Epochs | 100 |

### Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 99.75% |
| Precision | 100.00% |
| Recall | 99.50% |
| F1-Score | 99.75% |

### Confusion Matrix
```
              Predicted
            Normal  Attack
Actual Normal   200      0
       Attack     1    199
```

---

## 5. Download Links

| Resource | Link |
|----------|------|
| **Paper (arXiv)** | https://arxiv.org/abs/2202.04301 |
| **Paper PDF** | https://arxiv.org/pdf/2202.04301.pdf |
| **Dataset** | https://huggingface.co/datasets/chYassine/WebAttack-CVSSMetrics |
| **Paper Code** | https://github.com/LogIntelligence/LogADEmpirical |
| **Paper Dataset** | https://zenodo.org/records/8115559 |

---

## 6. How to Download Dataset

```python
# Install
pip install datasets

# Download
from datasets import load_dataset
ds = load_dataset("chYassine/WebAttack-CVSSMetrics")
df = ds['data'].to_pandas()
df.to_csv("webattack.csv", index=False)
```

---

## 7. Related Papers

| Paper | Year | Focus |
|-------|------|-------|
| Le & Zhang, "Log-based Anomaly Detection with Deep Learning" | ICSE 2022 | Log anomaly detection methodology |
| Du et al., "DeepLog: Anomaly Detection and Diagnosis from System Logs" | CCS 2017 | LSTM for log analysis |
| Meng et al., "LogAnomaly: Unsupervised Detection of Sequential and Quantitative Anomalies" | IJCAI 2019 | Template-based detection |

