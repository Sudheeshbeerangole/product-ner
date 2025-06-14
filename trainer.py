__author__ = "UniCourt Inc"
__version__ = "v1.0.0"
__maintainer__ = "Search - Core & API"
__email__ = "eng-search@unicourt.com"

# train_ner.py
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

from datasets import Dataset

import logging
logging.basicConfig(level=logging.INFO)
# === 1. Label Definitions ===
labels = ["O", "B-quantity", "B-weight", "I-weight", "B-brand", "I-brand", "B-product_category","I-product_category"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

# === 2. Example Training Data ===
examples = [
    (["Get", "two", "100", "kilo", "gram", "Nestle", "noodles"],
     ["O", "B-quantity", "B-weight", "I-weight", "I-weight", "B-brand", "B-product_category"]),

    (["Buy", "three", "500", "gram", "Maggie", "noodles"],
     ["O", "B-quantity", "B-weight", "I-weight", "B-brand", "B-product_category"]),

    (["Order", "one", "Nestle", "coffee", "pack", "of", "200", "grams"],
     ["O", "B-quantity", "B-brand", "B-product_category", "O", "O", "B-weight", "I-weight"]),

    (["Get", "five", "Coca", "Cola", "bottles"],
     ["O", "B-quantity", "B-brand", "I-brand", "B-product_category"]),

    (["Buy", "a", "250", "gram", "Amul", "butter"],
     ["O", "O", "B-weight", "I-weight", "B-brand", "B-product_category"]),

    (["Purchase", "four", "1", "liter", "Pepsi", "bottles"],
     ["O", "B-quantity", "B-weight", "I-weight", "B-brand", "B-product_category"]),

    (["Get", "six", "500", "ml", "Sprite", "cans"],
     ["O", "B-quantity", "B-weight", "I-weight", "B-brand", "B-product_category"]),

    (["Order", "two", "Amul", "cheese", "packs", "of", "250", "grams"],
     ["O", "B-quantity", "B-brand", "B-product_category", "O", "O", "B-weight", "I-weight"]),

    (["Buy", "one", "Britannia", "bread", "loaf"],
     ["O", "B-quantity", "B-brand", "B-product_category", "O"]),

    (["Get", "three", "Cadbury", "Dairy", "Milk", "bars"],
     ["O", "B-quantity", "B-brand", "I-brand", "I-brand", "B-product_category"]),

    (["Order", "five", "lays", "chips", "packets"],
     ["O", "B-quantity", "B-brand", "B-product_category", "O"]),

    (["Buy", "two", "1", "kg", "Tata", "salt", "packets"],
     ["O", "B-quantity", "B-weight", "I-weight", "B-brand", "B-product_category", "O"]),

    (["Get", "one", "Nestle", "milkmaid", "tin", "of", "400", "grams"],
     ["O", "B-quantity", "B-brand", "B-product_category", "O", "O", "B-weight", "I-weight"]),

    (["Order", "three", "Parle", "G", "biscuit", "packs"],
     ["O", "B-quantity", "B-brand", "I-brand", "B-product_category", "O"]),

    (["Buy", "four", "Amul", "ice", "cream", "cups"],
     ["O", "B-quantity", "B-brand", "B-product_category", "I-product_category", "B-product_category", "O"]),

(["Add", "two", "500", "ml", "Bovonto", "soda", "bottles"],
     ["O", "B-quantity", "B-weight", "I-weight", "B-brand", "B-product_category", "B-product_category"]),

    (["I", "want", "one", "kg", "Fortune", "basmati", "rice"],
     ["O", "O", "B-quantity", "B-weight", "B-brand", "B-product_category", "I-product_category"]),

    (["Please", "get", "three", "lays", "classic", "chips"],
     ["O", "O", "B-quantity", "B-brand", "I-brand", "B-product_category"]),

    (["Pick", "up", "a", "200", "ml", "Frooti", "pack"],
     ["O", "O", "O", "B-weight", "I-weight", "B-brand", "B-product_category"]),

    (["Order", "six", "Tropicana", "juice", "boxes", "of", "250", "ml"],
     ["O", "B-quantity", "B-brand", "B-product_category", "O", "O", "B-weight", "I-weight"]),

    (["Get", "a", "pack", "of", "100", "gram", "Red", "Label", "tea"],
     ["O", "O", "O", "O", "B-weight", "I-weight", "B-brand", "I-brand", "B-product_category"]),

    (["Buy", "ten", "1", "liter", "Bisleri", "water", "bottles"],
     ["O", "B-quantity", "B-weight", "I-weight", "B-brand", "B-product_category", "I-product_category"]),

    (["Add", "one", "kg", "Aashirvaad", "atta", "bag"],
     ["O", "B-quantity", "B-weight", "B-brand", "B-product_category", "O"]),

    (["I", "need", "two", "packs", "of", "Sunfeast", "cookies"],
     ["O", "O", "B-quantity", "O", "O", "B-brand", "B-product_category"]),

    (["Grab", "three", "bars", "of", "Perk", "chocolate"],
     ["O", "B-quantity", "O", "O", "B-brand", "B-product_category"]),

    (["Please", "buy", "250", "ml", "pack", "of", "Real", "juice"],
     ["O", "O", "B-weight", "I-weight", "O", "O", "B-brand", "B-product_category"]),

    (["Purchase", "four", "Britannia", "cake", "slices"],
     ["O", "B-quantity", "B-brand", "B-product_category", "I-product_category"]),

    (["I", "want", "to", "order", "a", "dozen", "Farm", "Fresh", "eggs"],
     ["O", "O", "O", "O", "O", "B-quantity", "B-brand", "I-brand", "B-product_category"]),

    (["Get", "five", "Cavin's", "milkshake", "bottles"],
     ["O", "B-quantity", "B-brand", "B-product_category", "I-product_category"]),

    (["Order", "one", "tetra", "pack", "of", "500", "ml", "Mother", "Dairy", "milk"],
     ["O", "B-quantity", "O", "O", "O", "B-weight", "I-weight", "B-brand", "I-brand", "B-product_category"]),
]

tokens = [x[0] for x in examples]
labels_seq = [[label2id[l] for l in label_list] for _, label_list in examples]

dataset = Dataset.from_dict({"tokens": tokens, "labels": labels_seq})

# === 3. Tokenizer & Model ===
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(example):
    tokenized = tokenizer(example["tokens"], is_split_into_words=True, truncation=True)
    word_ids = tokenized.word_ids()
    aligned_labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            aligned_labels.append(example["labels"][word_idx])
        else:
            aligned_labels.append(example["labels"][word_idx])
        previous_word_idx = word_idx
    tokenized["labels"] = aligned_labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_and_align_labels)

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# === 4. Training Arguments ===
training_args = TrainingArguments(
    output_dir="./models/token_classifier",
    evaluation_strategy="no",
    per_device_train_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=1,
    report_to="none"  # <<< suppresses WandB or TensorBoard warnings
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer)
)
print("✅ Tokenization complete. Starting training...")

# === 5. Train ===
trainer.train()

# === 6. Save Model ===
model.save_pretrained("./models/token_classifier")
tokenizer.save_pretrained("./models/token_classifier")

print("✅ Training complete and model saved.")


# === 7. Simple Inference Function ===
def predict(sentence):
    model.eval()
    tokens = sentence.split()
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()

    word_ids = inputs.word_ids()  # Maps subword tokens to word indices
    previous_word_idx = None
    print("\n--- Predictions ---")
    for i, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue  # skip special tokens and duplicate subwords
        word = tokens[word_idx]
        label = id2label[predictions[i]]
        print(f"{word:15} -> {label}")
        previous_word_idx = word_idx


# === 8. Test Inference ===
predict("Buy two 1 kilo packet of Amul milk")
