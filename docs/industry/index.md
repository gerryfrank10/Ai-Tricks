# AI in Industry

AI is transforming every major industry. This section provides domain-specific guides with real code, practical architectures, and the regulatory context you need to deploy responsibly.

---

## Topics

| Page | Industry | Key AI Applications |
|------|----------|-------------------|
| [Finance](finance.md) | Banking, trading, insurance | Fraud detection, credit scoring, algorithmic trading |
| [Healthcare](healthcare.md) | Hospitals, pharma, diagnostics | Medical imaging, EHR analysis, drug discovery |
| [Law Firms](law-firms.md) | Legal services, e-discovery | Contract review, legal research, RAG over case law |
| [Retail](retail.md) | E-commerce, CPG | Recommendations, demand forecasting, visual search |
| [Automation & Robotics](robotics.md) | Manufacturing, logistics | Robot learning, LLM task planning, computer vision |
| [Business Intelligence](business-intelligence.md) | Enterprise analytics | Text2SQL, anomaly detection, automated reporting |
| [AI Ethics](ethics.md) | All industries | Privacy, fairness, differential privacy, GDPR |
| [Architecture & Construction](architecture.md) | AEC, real estate | Generative design, BIM+AI, energy optimization, site analysis |
| [Agriculture](agriculture.md) | Farming, agri-business | Precision farming, crop monitoring, yield prediction, livestock AI |
| [Manufacturing & Industry 4.0](manufacturing.md) | Factory, industrial, supply chain | Predictive maintenance, visual inspection, digital twins, cobots |

---

## Adoption Maturity by Industry (2025)

```
High Adoption  ████████████  Finance, Tech, Retail
               ████████░░░░  Healthcare, Manufacturing
Medium         ██████░░░░░░  Legal, Education, Agriculture
Low/Growing    ████░░░░░░░░  Construction/AEC (rapidly evolving), Government
```

---

## Cross-Industry Patterns

Most industry AI applications use these core building blocks:

1. **Prediction** — XGBoost/LightGBM on structured data
2. **Classification** — CV or NLP transformers on unstructured data
3. **Recommendations** — Collaborative filtering + two-tower models
4. **Anomaly Detection** — Isolation Forest, LSTM autoencoders
5. **LLM Integration** — Summarization, extraction, Q&A over domain docs

---

## Related Topics

- [Machine Learning](../machine-learning/index.md)
- [MLOps](../mlops/index.md)
- [AI Ethics](ethics.md)
