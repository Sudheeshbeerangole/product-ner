# 🧠 product-ner

**product-ner** is a supervised Named Entity Recognition (NER) model built using Hugging Face Transformers and BERT. It is designed to extract structured product attributes like quantity, weight, brand, and product category from natural language input.

---

## 🚀 Project Features

- Uses `bert-base-uncased` with Hugging Face's `Trainer` API
- Custom token classification for product-related entities
- Easy-to-extend training dataset
- Inference method included directly in the training script

---

## 🏷️ Entity Labels

The model is trained to identify the following entity types in product-related sentences:

| Label               | Description                           |
|--------------------|---------------------------------------|
| `B-quantity`        | Start of quantity mention             |
| `B-weight`, `I-weight` | Start and continuation of weight     |
| `B-brand`, `I-brand`   | Start and continuation of brand      |
| `B-product_category`, `I-product_category` | Product type        |
| `O`                | Outside any entity                    |

---
