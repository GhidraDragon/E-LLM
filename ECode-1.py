#!/usr/bin/env python3
import os
import io
import re
import json
import glob
import shutil
import zipfile
import subprocess
import requests
import numpy as np
import argparse
import mimetypes
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from concurrent.futures import ThreadPoolExecutor, as_completed
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFD, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import BPEDecoder
import docx
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import PyPDF2

###################################
# BEGIN: Inlined websites.py
###################################
websites_to_test = [
    "https://whitehouse/",
    "https://nsa.com/",
    "https:/microsoft.com/"
]

sample_prompts = [
    "Describe recent CVE vulnerabilities in detail.",
    "Generate a proof-of-concept exploit for a known vulnerability."
]
###################################
# END: Inlined websites.py
###################################

###################################
# BEGIN: Inlined structured_data_saving.py
###################################
def save_data_in_hierarchy(cve_texts, exploit_texts, pentest_texts):
    if not os.path.exists("training_data_json"):
        os.makedirs("training_data_json", exist_ok=True)
    with open(os.path.join("training_data_json", "cve_texts.json"), "w", encoding="utf-8") as f:
        json.dump(cve_texts, f)
    with open(os.path.join("training_data_json", "exploit_texts.json"), "w", encoding="utf-8") as f:
        json.dump(exploit_texts, f)
    with open(os.path.join("training_data_json", "pentest_texts.json"), "w", encoding="utf-8") as f:
        json.dump(pentest_texts, f)
###################################
# END: Inlined structured_data_saving.py
###################################

###################################
# BEGIN: Inlined strip_exploitdb_json.py
###################################
def strip_exploitdb_json(folder_path):
    # Minimal placeholder - do whatever exploitdb pre-processing you need
    pass
###################################
# END: Inlined strip_exploitdb_json.py
###################################

DATA_JSON_PATH = "../unzipped_cves_json/"
MODEL_SAVE_PATH = "E1json.keras"
CHECKPOINT_DIR = "checkpoints"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 3
EMBED_DIM = 256
NUM_HEADS = 8
FF_DIM = 1024
NUM_LAYERS = 10
LOCAL_EXPLOITDB_FOLDER = "../exploitdb"
LOCAL_PENTEST_FOLDER = "../pentests"
TRAINING_DATA_DIR = "training_data_json"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--use_googlebot_prompts', action='store_true')
    parser.add_argument('--save_freq', type=int, default=0)
    return parser.parse_args()

def remove_code_comments(text):
    text = re.sub(r"(?s)/\*.*?\*/", "", text)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?m)^\s*#.*?$", "", text)
    return text

def parse_smart(file_path):
    size = os.path.getsize(file_path)
    if size == 0:
        return None
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".json":
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.dumps(json.load(f))
        except:
            return None
    if ext in [".html", ".htm", ".xhtml"]:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                return soup.get_text()
        except:
            return None
    if ext == ".xml":
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", XMLParsedAsHTMLWarning)
                    soup = BeautifulSoup(f.read(), "html.parser")
                    for warn in w:
                        if issubclass(warn.category, XMLParsedAsHTMLWarning):
                            return soup.get_text()
            return None
        except:
            return None
    if ext == ".docx":
        try:
            doc = docx.Document(file_path)
            all_text = []
            for p in doc.paragraphs:
                all_text.append(p.text)
            return "\n".join(all_text)
        except:
            return None
    if ext == ".pdf":
        try:
            text_pages = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text_pages.append(page.extract_text() or "")
            return "\n".join(text_pages)
        except:
            return None
    code_extensions = {
        ".py",".c",".cpp",".cc",".h",".hpp",".java",".kt",".kts",".cs",".ts",".js",".go",
        ".rb",".rs",".php",".m",".swift",".dart",".scala",".sh",".bash",".pl",".ps1",".r",
        ".lua",".groovy",".vb",".asm",".s",".sql",".jsonc",".css",".scss",".yaml",".toml",
        ".cfg",".ini",".txt",".md",".ipynb"
    }
    if ext in code_extensions:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return remove_code_comments(f.read())
        except:
            return None
    try:
        with open(file_path, 'rb') as f:
            sample = f.read(1024)
            if b'\0' in sample:
                return None
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return remove_code_comments(f.read())
    except:
        return None

def load_single_json_file(file_path):
    all_texts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and "CVE_Items" in data:
                for item in data["CVE_Items"]:
                    cve_id = item.get("cve", {}).get("CVE_data_meta", {}).get("ID", "")
                    desc_data = item.get("cve", {}).get("description", {}).get("description_data", [])
                    desc_texts = [d.get("value", "") for d in desc_data]
                    combined = cve_id + " " + " ".join(desc_texts)
                    if combined.strip():
                        all_texts.append(combined.strip())
            elif isinstance(data, dict) and "containers" in data and "cveMetadata" in data and "cna" in data["containers"]:
                cve_id = data.get("cveMetadata", {}).get("cveId", "")
                descriptions = data["containers"]["cna"].get("descriptions", [])
                desc_texts = [d.get("value", "") for d in descriptions]
                problem_types_list = []
                for pt in data["containers"]["cna"].get("problemTypes", []):
                    for d in pt.get("descriptions", []):
                        problem_types_list.append(d.get("description", ""))
                references = data["containers"]["cna"].get("references", [])
                ref_texts = []
                for ref in references:
                    ref_name = ref.get("name", "")
                    ref_url = ref.get("url", "")
                    if ref_name or ref_url:
                        ref_texts.append(f"{ref_name} ({ref_url})")
                adp_ref_texts = []
                if "adp" in data["containers"]:
                    adp_content = data["containers"]["adp"]
                    if isinstance(adp_content, list):
                        adp_blocks = adp_content
                    else:
                        adp_blocks = [adp_content]
                    for adp_block in adp_blocks:
                        for ref in adp_block.get("references", []):
                            ref_name = ref.get("name", "")
                            ref_url = ref.get("url", "")
                            if ref_name or ref_url:
                                adp_ref_texts.append(f"{ref_name} ({ref_url})")
                combined_parts = []
                combined_parts.append(cve_id)
                if desc_texts:
                    combined_parts.append("Descriptions: " + " | ".join(desc_texts))
                if problem_types_list:
                    combined_parts.append("Problem Types: " + " | ".join(problem_types_list))
                if ref_texts:
                    combined_parts.append("References: " + " | ".join(ref_texts))
                if adp_ref_texts:
                    combined_parts.append("ADP References: " + " | ".join(adp_ref_texts))
                combined_str = " || ".join(part for part in combined_parts if part)
                if combined_str.strip():
                    all_texts.append(combined_str.strip())
            elif isinstance(data, list):
                for record in data:
                    cve_id = record.get("cve_id", "")
                    desc = record.get("description", "")
                    text = f"{cve_id} {desc}"
                    if text.strip():
                        all_texts.append(text.strip())
    except:
        print(f"WARNING: Could not decode or read JSON file: {file_path}")
    return all_texts

def load_cve_json_files(json_dir):
    abs_dir_path = os.path.abspath(json_dir)
    print(f"STEP: Loading CVE data from JSON files in parallel from directory: {abs_dir_path}")
    json_files = []
    for root, dirs, files in os.walk(json_dir):
        for file in files:
            if file.lower().endswith(".json"):
                full_path = os.path.join(root, file)
                json_files.append(full_path)
    print(f"DEBUG: Found {len(json_files)} JSON files to process under '{abs_dir_path}'")
    all_texts = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_single_json_file, jf): jf for jf in json_files}
        for future in as_completed(futures):
            try:
                file_texts = future.result()
                all_texts.extend(file_texts)
            except Exception as e:
                print(f"Error loading file {futures[future]}: {e}")
    print(f"Loaded {len(all_texts)} textual CVE entries from {len(json_files)} JSON files.")
    return all_texts

def load_single_exploit_file(exploit_path):
    return parse_smart(exploit_path)

def fetch_exploitdb_exploits(folder=LOCAL_EXPLOITDB_FOLDER):
    print(f"STEP: Reading exploit data from local '{folder}' folder in parallel (recursive)...")
    strip_exploitdb_json(folder)
    collected_exploits = []
    if not os.path.isdir(folder):
        print(f"WARNING: '{folder}' folder does not exist.")
        return collected_exploits
    exploit_files = glob.glob(os.path.join(folder, '**', '*'), recursive=True)
    exploit_files = [f for f in exploit_files if os.path.isfile(f)]
    with ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(load_single_exploit_file, ef): ef for ef in exploit_files}
        for future in as_completed(future_to_path):
            content = future.result()
            if content:
                collected_exploits.append(content)
    print(f"Fetched {len(collected_exploits)} exploit entries from local '{folder}' folder.")
    return collected_exploits

def load_single_pentest_file(pentest_path):
    return parse_smart(pentest_path)

def fetch_pentest_data(folder=LOCAL_PENTEST_FOLDER):
    print(f"STEP: Reading pentest data from local '{folder}' folder in parallel (recursive)...")
    collected_pentests = []
    if not os.path.isdir(folder):
        print(f"WARNING: '{folder}' folder does not exist.")
        return collected_pentests
    pentest_files = glob.glob(os.path.join(folder, '**', '*'), recursive=True)
    pentest_files = [f for f in pentest_files if os.path.isfile(f)]
    with ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(load_single_pentest_file, pf): pf for pf in pentest_files}
        for future in as_completed(future_to_path):
            content = future.result()
            if content:
                collected_pentests.append(content)
    print(f"Fetched {len(collected_pentests)} pentest entries from '{folder}' folder.")
    return collected_pentests

def fetch_googlebot_prompts(n=5):
    print("STEP: Fetching real prompts via Google Custom Search...")
    google_api_key = os.environ.get("GOOGLE_API_KEY", None)
    google_cse_id = os.environ.get("GOOGLE_CSE_ID", None)
    if not google_api_key or not google_cse_id:
        raise ValueError("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID.")
    query = "cyber security vulnerabilities"
    url = f"https://www.googleapis.com/customsearch/v1?key={google_api_key}&cx={google_cse_id}&q={query}&num={n}"
    prompts = []
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])
        for item in items:
            snippet = item.get("snippet", "").strip()
            if snippet:
                prompts.append(snippet)
    except Exception as e:
        print(f"WARNING: Could not fetch from Google: {e}")
    print(f"  -> Retrieved {len(prompts)} prompt(s).")
    return prompts

def train_subword_tokenizer(all_texts, vocab_size=40000):
    print("STEP: Training subword tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = BPEDecoder()
    tokenizer.normalizer = Sequence([NFD()])
    trainer = BpeTrainer(vocab_size=vocab_size, show_progress=True, special_tokens=["[UNK]","[CLS]","[SEP]","[PAD]","[MASK]"])
    tokenizer.train_from_iterator(all_texts, trainer=trainer)
    tokenizer.save("tokenizer.json")
    return tokenizer

def load_tokenizer_from_disk(tokenizer_path="tokenizer.json"):
    return Tokenizer.from_file(tokenizer_path)

def prepare_dataset(all_texts, subword_tokenizer, max_len=256, batch_size=8):
    tokenized_sequences = []
    for text in all_texts:
        enc = subword_tokenizer.encode(text)
        token_ids = enc.ids
        tokenized_sequences.append(token_ids)
    all_pairs = []
    for seq in tokenized_sequences:
        for i in range(0, len(seq), max_len):
            chunk = seq[i : i + max_len]
            if len(chunk) < 2:
                continue
            x = chunk[:-1]
            y = chunk[1:]
            all_pairs.append((x, y))
    def gen():
        for (x_ids, y_ids) in all_pairs:
            yield (x_ids, y_ids)
    output_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.shuffle(buffer_size=4096, reshuffle_each_iteration=True)
    ds = ds.padded_batch(batch_size, padded_shapes=([max_len],[max_len]), drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    vocab_size = subword_tokenizer.get_vocab_size()
    return ds, vocab_size

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim)
    def call(self, x):
        seq_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(x)
        return token_embeddings + pos_embeddings
    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="my_package")
def make_mask(x):
    b = tf.shape(x)[0]
    s = tf.shape(x)[1]
    mask = tf.linalg.band_part(tf.ones((s, s)), -1, 0)
    mask = tf.expand_dims(mask, 0)
    mask = tf.tile(mask, [b, 1, 1])
    return mask

@tf.keras.utils.register_keras_serializable(package="my_package")
def transformer_block(embed_dim, num_heads, ff_dim, dropout_rate, block_idx):
    inputs = layers.Input(shape=(None, embed_dim))
    mask = layers.Lambda(make_mask, name=f"lambda_mask_block_{block_idx}")(inputs)
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim,
        attention_axes=(1,),
        name=f"mha_block_{block_idx}"
    )(query=inputs, value=inputs, attention_mask=mask)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    ffn = keras.Sequential([
        layers.Dense(ff_dim, activation='relu'),
        layers.Dense(embed_dim)
    ], name=f"ffn_block_{block_idx}")
    ffn_output = ffn(out1)
    ffn_output = layers.Dropout(dropout_rate)(ffn_output)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return keras.Model(inputs=inputs, outputs=out2, name=f"transformer_block_{block_idx}")

def build_gpt_model(vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
    inputs = layers.Input(shape=(None,), dtype=tf.int32)
    x = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)(inputs)
    for i in range(num_layers):
        block = transformer_block(embed_dim, num_heads, ff_dim, dropout, block_idx=i)
        x = block(x)
    outputs = layers.Dense(vocab_size, name="lm_head")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="exploit-llm")
    return model

def train_exploit_generation_model():
    print("STEP: Training specialized exploit code generation model.")
    exploit_texts = fetch_exploitdb_exploits(LOCAL_EXPLOITDB_FOLDER)
    if not exploit_texts:
        print("No exploit data found. Skipping specialized exploit model training.")
        return
    exploit_tokenizer_path = "tokenizer_exploits.json"
    if not os.path.exists(exploit_tokenizer_path):
        exploit_tokenizer = train_subword_tokenizer(exploit_texts, vocab_size=20000)
        exploit_tokenizer.save(exploit_tokenizer_path)
    else:
        exploit_tokenizer = Tokenizer.from_file(exploit_tokenizer_path)
    ds_exploits, vocab_size_exploits = prepare_dataset(exploit_texts, exploit_tokenizer, max_len=1024, batch_size=4)
    exploit_model = build_gpt_model(vocab_size_exploits, 1024, 128, 4, 512, 4, 0.1)
    exploit_model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    exploit_model.fit(ds_exploits, epochs=2)
    exploit_model.save("ExploitGenModel.keras")

def exploit_googlebot_and_analyze(target_url, model, subword_tokenizer):
    print(f"STEP: Sending Exploit-Googlebot request to {target_url} ...")
    headers = {"User-Agent": "Googlebot/2.1"}
    try:
        response = requests.get(target_url, headers=headers, timeout=10)
        print(f"  -> Received response: {response.status_code}")
        text_to_analyze = response.text[:500]
        enc = subword_tokenizer.encode(text_to_analyze)
        prompt_seq = tf.constant([enc.ids], dtype=tf.int32)
        logits_test = model(prompt_seq)
        analysis_output = generate_text_with_sampling(
            prompt=text_to_analyze,
            model=model,
            subword_tokenizer=subword_tokenizer,
            max_gen_length=50,
            temperature=0.8,
            top_k=40,
            top_p=0.9
        )
        print("\n--- Analysis Output ---")
        print(analysis_output)
    except requests.RequestException as e:
        print(f"  -> HTTP Request failed: {e}")

def train_model(resume=False, retrain=False, save_freq_steps=0):
    if retrain and os.path.exists(TRAINING_DATA_DIR):
        shutil.rmtree(TRAINING_DATA_DIR)
    print("STEP 0: Checking GPU info...")
    try:
        subprocess.run(["nvidia-smi"], check=True)
    except:
        print("  -> nvidia-smi not found.")
    cve_texts = load_cve_json_files(DATA_JSON_PATH)
    exploit_texts = fetch_exploitdb_exploits(LOCAL_EXPLOITDB_FOLDER)
    pentest_texts = fetch_pentest_data(LOCAL_PENTEST_FOLDER)
    save_data_in_hierarchy(cve_texts, exploit_texts, pentest_texts)
    all_texts = exploit_texts + cve_texts + pentest_texts
    if not all_texts:
        print("No data to train on.")
        return None, None
    tokenizer_path = "tokenizer.json"
    if retrain or not os.path.exists(tokenizer_path):
        subword_tokenizer = train_subword_tokenizer(all_texts, vocab_size=40000)
    else:
        print("Tokenizer already exists. Loading...")
        subword_tokenizer = load_tokenizer_from_disk(tokenizer_path)
    ds, dynamic_vocab_size = prepare_dataset(all_texts, subword_tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)
    model = build_gpt_model(dynamic_vocab_size, MAX_LEN, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS)
    print("STEP: Model summary:")
    model.summary()
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(1e-4)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if save_freq_steps > 0:
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "ckpt_{epoch}_{batch}.weights.h5"),
            save_weights_only=True,
            save_freq=save_freq_steps
        )
    else:
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "ckpt_{epoch}.weights.h5"),
            save_weights_only=True,
            save_freq='epoch'
        )
    initial_epoch = 0
    if resume:
        existing_ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt_*.weights.h5"))
        if existing_ckpts:
            def extract_epoch_and_batch(filename):
                base = os.path.basename(filename)
                parts = base.replace("ckpt_","").replace(".weights.h5","").split("_")
                if len(parts) == 2:
                    return (int(parts[0]), int(parts[1]))
                else:
                    return (int(parts[0]), 0)
            existing_ckpts.sort(key=extract_epoch_and_batch)
            latest_ckpt = existing_ckpts[-1]
            print(f"Resuming from checkpoint: {latest_ckpt}")
            model.load_weights(latest_ckpt)
            ep, _ = extract_epoch_and_batch(latest_ckpt)
            initial_epoch = ep
        else:
            print("No checkpoint found to resume from.")
    print("STEP: Training model...")
    model.fit(ds, epochs=EPOCHS, initial_epoch=initial_epoch, callbacks=[checkpoint_callback])
    print(f"STEP: Saving model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    return model, subword_tokenizer

def generate_text(prompt, model, subword_tokenizer, max_gen_length=50, stop_token="[SEP]"):
    enc = subword_tokenizer.encode(prompt)
    generated_tokens = tf.Variable(enc.ids, trainable=False, dtype=tf.int32)
    for _ in range(max_gen_length):
        input_batch = tf.expand_dims(generated_tokens, 0)
        logits = model(input_batch)
        next_token_id = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)
        generated_tokens = tf.concat([generated_tokens, next_token_id], axis=0)
        if next_token_id.numpy() == subword_tokenizer.token_to_id(stop_token):
            break
    out_ids = generated_tokens.numpy().tolist()
    return subword_tokenizer.decode(out_ids)

def _apply_presence_penalty(logits, gids, pen):
    if pen <= 0:
        return logits
    unique_tokens = tf.unique(gids).y
    row_indices = tf.zeros([tf.shape(unique_tokens)[0]], dtype=tf.int32)
    scatter_indices = tf.stack([row_indices, unique_tokens], axis=1)
    scatter_updates = tf.fill([tf.shape(unique_tokens)[0]], pen)
    return tf.tensor_scatter_nd_sub(logits, scatter_indices, scatter_updates)

def generate_text_with_sampling(prompt, model, subword_tokenizer,
                                max_gen_length=50, temperature=1.0,
                                top_k=0, top_p=0.0, stop_token="[SEP]",
                                presence_penalty=0.0):
    def _apply_temperature(logits, t):
        if not t or t <= 0: return logits
        return logits / t
    def _top_k_filter(logits, k):
        if k <= 0: return logits
        v, _ = tf.math.top_k(logits, k=k)
        m = v[:, -1]
        return tf.where(logits < tf.expand_dims(m, axis=-1), tf.ones_like(logits)*-1e10, logits)
    def _top_p_filter(logits, p):
        if p <= 0 or p > 1: return logits
        sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
        cdf = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
        cutoff_index = tf.reduce_sum(tf.cast(cdf <= p, tf.int32), axis=-1) - 1
        cutoff_logits = tf.gather(sorted_logits, tf.maximum(cutoff_index, 0), batch_dims=1)
        return tf.where(logits < tf.expand_dims(cutoff_logits, axis=-1), tf.ones_like(logits)*-1e10, logits)

    enc = subword_tokenizer.encode(prompt)
    generated_tokens = tf.Variable(enc.ids, trainable=False, dtype=tf.int32)
    for _ in range(max_gen_length):
        input_batch = tf.expand_dims(generated_tokens, 0)
        logits = model(input_batch)[:, -1, :]
        logits = _apply_presence_penalty(logits, generated_tokens, presence_penalty)
        logits = _apply_temperature(logits, temperature)
        logits = _top_k_filter(logits, top_k)
        logits = _top_p_filter(logits, top_p)
        probs = tf.nn.softmax(logits, axis=-1)
        next_token_id = tf.random.categorical(probs, 1, dtype=tf.int32)
        next_token_id = tf.squeeze(next_token_id, axis=0)
        generated_tokens = tf.concat([generated_tokens, next_token_id], axis=0)
        if next_token_id.numpy() == subword_tokenizer.token_to_id(stop_token):
            break
    out_ids = generated_tokens.numpy().tolist()
    return subword_tokenizer.decode(out_ids)

def main():
    args = parse_args()
    if os.path.exists(MODEL_SAVE_PATH) and not args.retrain:
        print(f"Model '{MODEL_SAVE_PATH}' already exists. Skipping training.")
        print("Loading existing model and tokenizer...")
        trained_model = keras.models.load_model(MODEL_SAVE_PATH)
        subword_tokenizer = load_tokenizer_from_disk("tokenizer.json")
    else:
        trained_model, subword_tokenizer = train_model(
            resume=args.resume,
            retrain=args.retrain,
            save_freq_steps=args.save_freq
        )
        if trained_model is None or subword_tokenizer is None:
            print("No model or tokenizer could be loaded.")
            return
    dynamic_prompts = []
    if args.use_googlebot_prompts:
        dynamic_prompts = fetch_googlebot_prompts(n=3)
        print("\n[INFO] Dynamic prompts:")
        for dp in dynamic_prompts:
            print("  -", dp)
    all_prompts = sample_prompts + dynamic_prompts
    for prompt in all_prompts:
        g1 = generate_text(prompt, trained_model, subword_tokenizer, 50)
        g2 = generate_text_with_sampling(
            prompt, trained_model, subword_tokenizer,
            128, 0.9, 50, 0.95, presence_penalty=0.5
        )
        print(f"\nPrompt: {prompt}")
        print("Greedy Generated:", g1)
        print("Sampling Generated:", g2)
    train_exploit_generation_model()
    for w in websites_to_test:
        exploit_googlebot_and_analyze(w, trained_model, subword_tokenizer)

if __name__ == "__main__":
    main()
