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

local_dir = '.'
local_pentest_folder = os.path.join(local_dir, "pentests")
local_mitre_attack_folder = os.path.join(local_dir, "MITRE_ATTACK")
local_nvd_folder = os.path.join(local_dir, "NVD")
local_owasp_wstg_folder = os.path.join(local_dir, "OWASP_WSTG")
local_wikitext_folder = os.path.join(local_dir, "wikitext-103-raw")
local_code_dataset_folder = os.path.join(local_dir, "large_code_dataset")
local_cyber_codegen_dataset_folder = os.path.join(local_dir, "cyber_codegen_dataset")

data_json_path = os.path.join(local_dir, "cvelistV5")
model_save_path = os.path.join(local_dir, "E1json.keras")
checkpoint_dir = os.path.join(local_dir, "checkpoints")
training_data_dir = os.path.join(local_dir, "training_data_json")
tokenizer_path = os.path.join(local_dir, "tokenizer.json")

max_len = 512
batch_size = 32
epochs = 3
embed_dim = 320
num_heads = 10
ff_dim = 1280
num_layers = 12

NVD_FEED_URLS = [
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2002.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2003.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2004.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2005.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2006.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2007.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2008.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2009.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2010.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2011.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2012.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2013.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2014.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2015.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2016.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2017.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2018.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2019.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2020.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2021.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2022.json.zip",
    "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2023.json.zip"
]

def clone_repositories():
    if not os.path.exists(os.path.join(local_dir, "cvelistV5")):
        subprocess.run(["git", "clone", "https://github.com/CVEProject/cvelistV5"], check=True)

def download_large_pentest_data(target_folder=local_pentest_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)
    if not os.path.exists(os.path.join(target_folder, "awesome-pentest")):
        subprocess.run(["git", "clone", "https://github.com/enaqx/awesome-pentest.git", "awesome-pentest"], cwd=target_folder, check=True)

def download_owasp_wstg(target_folder=local_owasp_wstg_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)
    if not os.path.exists(os.path.join(target_folder, "wstg")):
        subprocess.run(["git", "clone", "https://github.com/OWASP/wstg.git", "wstg"], cwd=target_folder, check=True)

def download_mitre_attack(target_folder=local_mitre_attack_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)
    url = "https://github.com/mitre/cti/archive/refs/heads/master.zip"
    zip_path = os.path.join(target_folder, "mitre_attack.zip")
    r = requests.get(url, stream=True)
    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target_folder)
    os.remove(zip_path)

def download_nvd_data(target_folder=local_nvd_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)
    for url in NVD_FEED_URLS:
        local_zip = os.path.join(target_folder, os.path.basename(url))
        if not os.path.exists(local_zip):
            r = requests.get(url, stream=True)
            with open(local_zip, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        with zipfile.ZipFile(local_zip, 'r') as zf:
            zf.extractall(target_folder)

def download_wikitext_103(target_folder=local_wikitext_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)
    zip_url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
    zip_path = os.path.join(target_folder, "wikitext-103-raw-v1.zip")
    if not os.path.exists(os.path.join(target_folder, "wiki.valid.raw")):
        r = requests.get(zip_url, stream=True)
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_folder)
        os.remove(zip_path)

def download_additional_code_dataset(target_folder=local_code_dataset_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)
    repo_path = os.path.join(target_folder, "the-stack-samples")
    if not os.path.exists(repo_path):
        subprocess.run(["git", "clone", "https://github.com/bigcode-project/the-stack", "the-stack-samples"], cwd=target_folder, check=True)

def download_cybersecurity_code_generation_dataset(target_folder=local_cyber_codegen_dataset_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)
    repo_path = os.path.join(target_folder, "cyber_code_gen")
    if not os.path.exists(repo_path):
        subprocess.run(["git", "clone", "https://github.com/mr-un1k0d3r/RedTeamPowershellScripts", "cyber_code_gen"], cwd=target_folder, check=True)

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def save_data_in_hierarchy(cve_texts, pentest_texts, mitre_texts, owasp_texts):
    if not os.path.exists(training_data_dir):
        os.makedirs(training_data_dir, exist_ok=True)
    with open(os.path.join(training_data_dir, "cve_texts.json"), "w", encoding="utf-8") as f:
        json.dump(cve_texts, f)
    with open(os.path.join(training_data_dir, "pentest_texts.json"), "w", encoding="utf-8") as f:
        json.dump(pentest_texts, f)
    with open(os.path.join(training_data_dir, "mitre_texts.json"), "w", encoding="utf-8") as f:
        json.dump(mitre_texts, f)
    with open(os.path.join(training_data_dir, "owasp_texts.json"), "w", encoding="utf-8") as f:
        json.dump(owasp_texts, f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save_freq', type=int, default=0)
    args, _ = parser.parse_known_args()
    return args

def remove_code_comments(text):
    text = re.sub(r"(?s)/\*.*?\*/", "", text)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?m)^\s*#.*?$", "", text)
    return text

def parse_smart(file_path):
    size = os.path.getsize(file_path)
    if size == 0:
        return None
    if size > 500 * 1024 * 1024:
        return None
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".json":
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return clean_text(json.dumps(json.load(f)))
        except:
            return None
    if ext in [".html", ".htm", ".xhtml"]:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                return clean_text(soup.get_text())
        except:
            return None
    if ext == ".xml":
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            try:
                soup = BeautifulSoup(content, features="xml")
            except:
                warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
                soup = BeautifulSoup(content, "html.parser")
            return clean_text(soup.get_text())
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
                return clean_text(remove_code_comments(f.read()))
        except:
            return None
    try:
        with open(file_path, 'rb') as f:
            sample = f.read(1024)
            if b'\0' in sample:
                return None
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return clean_text(remove_code_comments(f.read()))
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
                        all_texts.append(clean_text(combined.strip()))
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
                    all_texts.append(clean_text(combined_str.strip()))
            elif isinstance(data, list):
                for record in data:
                    cve_id = record.get("cve_id", "")
                    desc = record.get("description", "")
                    text = f"{cve_id} {desc}"
                    if text.strip():
                        all_texts.append(clean_text(text.strip()))
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

def load_single_pentest_file(pentest_path):
    return parse_smart(pentest_path)

def fetch_pentest_data(folder=local_pentest_folder):
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

def load_single_mitre_file(mitre_path):
    return parse_smart(mitre_path)

def fetch_mitre_attack_data(folder=local_mitre_attack_folder):
    print(f"STEP: Reading MITRE ATT&CK data from local '{folder}' folder in parallel (recursive)...")
    collected_mitre = []
    if not os.path.isdir(folder):
        print(f"WARNING: '{folder}' folder does not exist.")
        return collected_mitre
    mitre_files = glob.glob(os.path.join(folder, '**', '*'), recursive=True)
    mitre_files = [f for f in mitre_files if os.path.isfile(f)]
    with ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(load_single_mitre_file, mf): mf for mf in mitre_files}
        for future in as_completed(future_to_path):
            content = future.result()
            if content:
                collected_mitre.append(content)
    print(f"Fetched {len(collected_mitre)} items from '{folder}' folder.")
    return collected_mitre

def fetch_owasp_wstg_data(folder=local_owasp_wstg_folder):
    print(f"STEP: Reading OWASP WSTG data from local '{folder}' folder in parallel (recursive)...")
    collected_owasp = []
    if not os.path.isdir(folder):
        print(f"WARNING: '{folder}' folder does not exist.")
        return collected_owasp
    owasp_files = glob.glob(os.path.join(folder, '**', '*'), recursive=True)
    owasp_files = [f for f in owasp_files if os.path.isfile(f)]
    with ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(parse_smart, mf): mf for mf in owasp_files}
        for future in as_completed(future_to_path):
            content = future.result()
            if content:
                collected_owasp.append(content)
    print(f"Fetched {len(collected_owasp)} items from '{folder}' folder.")
    return collected_owasp

def load_nvd_json_files(folder=local_nvd_folder):
    if not os.path.isdir(folder):
        print(f"WARNING: '{folder}' folder does not exist.")
        return []
    nvd_json_files = glob.glob(os.path.join(folder, '*.json'))
    nvd_texts = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_single_json_file, jf): jf for jf in nvd_json_files}
        for future in as_completed(futures):
            try:
                file_texts = future.result()
                nvd_texts.extend(file_texts)
            except:
                pass
    print(f"Fetched {len(nvd_texts)} entries from NVD data at '{folder}'.")
    return nvd_texts

def fetch_wikitext_data(folder=local_wikitext_folder):
    files = ["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"]
    collected_wiki = []
    for file_name in files:
        file_path = os.path.join(folder, file_name)
        if not os.path.isfile(file_path):
            continue
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    collected_wiki.append(line)
    print(f"Fetched {len(collected_wiki)} lines from WikiText-103 raw data.")
    return collected_wiki

def fetch_additional_code_data(folder=local_code_dataset_folder):
    if not os.path.isdir(folder):
        return []
    files = glob.glob(os.path.join(folder, '**', '*'), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    data = []
    with ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(parse_smart, fp): fp for fp in files}
        for future in as_completed(future_to_path):
            content = future.result()
            if content:
                data.append(content)
    print(f"Fetched {len(data)} items from additional large code dataset.")
    return data

def fetch_cyber_codegen_data(folder=local_cyber_codegen_dataset_folder):
    if not os.path.isdir(folder):
        return []
    files = glob.glob(os.path.join(folder, '**', '*'), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    data = []
    with ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(parse_smart, fp): fp for fp in files}
        for future in as_completed(future_to_path):
            content = future.result()
            if content:
                data.append(content)
    print(f"Fetched {len(data)} items from cybersecurity code generation dataset.")
    return data

def train_subword_tokenizer(all_texts, vocab_size=40000):
    print("STEP: Training subword tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = BPEDecoder()
    tokenizer.normalizer = Sequence([NFD()])
    trainer = BpeTrainer(vocab_size=vocab_size, show_progress=True, special_tokens=["[UNK]","[CLS]","[SEP]","[PAD]","[MASK]"])
    tokenizer.train_from_iterator(all_texts, trainer=trainer)
    tokenizer.save(tokenizer_path)
    return tokenizer

def load_tokenizer_from_disk(path=tokenizer_path):
    return Tokenizer.from_file(path)

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
    model = keras.Model(inputs=inputs, outputs=outputs, name="exploit-llm-1")
    return model

def train_model(resume=False, retrain=False, save_freq_steps=0):
    if retrain and os.path.exists(training_data_dir):
        shutil.rmtree(training_data_dir)
    print("STEP 0: Checking GPU info...")
    try:
        subprocess.run(["nvidia-smi"], check=True)
    except:
        print("  -> nvidia-smi not found.")

    if not os.path.exists(local_mitre_attack_folder):
        print("Downloading MITRE ATT&CK data...")
        download_mitre_attack(local_mitre_attack_folder)

    if not os.path.exists(local_nvd_folder):
        print("Downloading NVD data...")
        download_nvd_data(local_nvd_folder)

    download_large_pentest_data()
    download_owasp_wstg()
    download_wikitext_103()
    download_additional_code_dataset()
    download_cybersecurity_code_generation_dataset()

    cve_texts = load_cve_json_files(data_json_path)
    pentest_texts = fetch_pentest_data(local_pentest_folder)
    mitre_texts = fetch_mitre_attack_data(local_mitre_attack_folder)
    nvd_texts = load_nvd_json_files(local_nvd_folder)
    owasp_texts = fetch_owasp_wstg_data(local_owasp_wstg_folder)
    wikitext_texts = fetch_wikitext_data(local_wikitext_folder)
    code_data_texts = fetch_additional_code_data(local_code_dataset_folder)
    cyber_codegen_texts = fetch_cyber_codegen_data(local_cyber_codegen_dataset_folder)

    all_cve_texts = cve_texts + nvd_texts
    save_data_in_hierarchy(all_cve_texts, pentest_texts, mitre_texts, owasp_texts)

    all_texts = (
        all_cve_texts
        + pentest_texts
        + mitre_texts
        + owasp_texts
        + wikitext_texts
        + code_data_texts
        + cyber_codegen_texts
    )

    if not all_texts:
        print("No data to train on.")
        return None, None

    if retrain or not os.path.exists(tokenizer_path):
        subword_tokenizer = train_subword_tokenizer(all_texts, vocab_size=40000)
    else:
        print("Tokenizer already exists. Loading...")
        subword_tokenizer = load_tokenizer_from_disk(tokenizer_path)

    ds, dynamic_vocab_size = prepare_dataset(all_texts, subword_tokenizer, max_len=max_len, batch_size=batch_size)
    model = build_gpt_model(dynamic_vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers)
    print("STEP: Model summary:")
    model.summary()

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(1e-4)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    if save_freq_steps > 0:
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "ckpt_{epoch}_{batch}.weights.h5"),
            save_weights_only=True,
            save_freq=save_freq_steps
        )
    else:
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5"),
            save_weights_only=True,
            save_freq='epoch'
        )

    initial_epoch = 0
    if resume:
        existing_ckpts = glob.glob(os.path.join(checkpoint_dir, "ckpt_*.weights.h5"))
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
    model.fit(ds, epochs=epochs, initial_epoch=initial_epoch, callbacks=[checkpoint_callback])
    print(f"STEP: Saving model to {model_save_path}")
    model.save(model_save_path)
    return model, subword_tokenizer

def generate_text(prompt, model, subword_tokenizer, max_gen_length=128, stop_token="[SEP]"):
    enc = subword_tokenizer.encode(prompt)
    generated_tokens = tf.Variable(enc.ids, trainable=False, dtype=tf.int32)
    for _ in range(max_gen_length):
        input_batch = tf.expand_dims(generated_tokens, 0)
        logits = model(input_batch)
        next_token_id = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)
        generated_tokens = tf.concat([generated_tokens, next_token_id], axis=0)
        token_str = subword_tokenizer.id_to_token(int(next_token_id.numpy()))
        print(token_str, end=" ", flush=True)
        if next_token_id.numpy() == subword_tokenizer.token_to_id(stop_token):
            break
    out_ids = generated_tokens.numpy().tolist()
    print()
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
                                max_gen_length=128, temperature=1.0,
                                top_k=0, top_p=0.0, stop_token="[SEP]",
                                presence_penalty=0.0):
    def _apply_temperature(logits, t):
        if not t or t <= 0:
            return logits
        return logits / t
    def _top_k_filter(logits, k):
        if k <= 0:
            return logits
        v, _ = tf.math.top_k(logits, k=k)
        m = v[:, -1]
        return tf.where(logits < tf.expand_dims(m, axis=-1), tf.ones_like(logits)*-1e10, logits)
    def _top_p_filter(logits, p):
        if p <= 0 or p > 1:
            return logits
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
        if tf.random.uniform([]) < 0.8:
            next_token_id = tf.argmax(probs, axis=-1, output_type=tf.int32)
        else:
            next_token_id = tf.random.categorical(probs, 1, dtype=tf.int32)
            next_token_id = tf.squeeze(next_token_id, axis=0)
        generated_tokens = tf.concat([generated_tokens, next_token_id], axis=0)
        token_str = subword_tokenizer.id_to_token(int(next_token_id.numpy()))
        print(token_str, end=" ", flush=True)
        if next_token_id.numpy() == subword_tokenizer.token_to_id(stop_token):
            break
    out_ids = generated_tokens.numpy().tolist()
    print()
    return subword_tokenizer.decode(out_ids)

def main():
    clone_repositories()
    args = parse_args()
    if os.path.exists(model_save_path) and not args.retrain:
        print(f"Model '{model_save_path}' already exists. Skipping training.")
        print("Loading existing model and tokenizer...")
        trained_model = keras.models.load_model(model_save_path)
        subword_tokenizer = load_tokenizer_from_disk(tokenizer_path)
    else:
        trained_model, subword_tokenizer = train_model(
            resume=args.resume,
            retrain=args.retrain,
            save_freq_steps=args.save_freq
        )
        if trained_model is None or subword_tokenizer is None:
            print("No model or tokenizer could be loaded.")
            return

    sample_prompts = [
        "Explain the impact of CVE-2021-34527:",
        "How to mitigate known vulnerabilities in web applications:",
        "Discuss effective techniques for preventing ransomware:",
        "Outline strategies for secure code reviews:",
        "Explore the fundamentals of zero-trust networking:",
        "Explain best practices for email phishing defenses:",
        "Detail methods for incident response planning:",
        "Compare common types of SQL injection attacks:",
        "Discuss encryption standards for data in transit:",
        "Summarize how to conduct a vulnerability assessment:",
        "Analyze the role of AI in cybersecurity:",
        "Explain the concept of network segmentation:",
        "Provide guidelines for secure Docker container management:",
        "Discuss ways to ensure compliance with GDPR:",
        "Detail the difference between black-box and white-box penetration testing:"
    ]
    
    print("\n[Testing Sample Prompts]")
    for prompt in sample_prompts:
        print(f"\nPrompt: {prompt}")
        print("Greedy Generation:")
        _ = generate_text(prompt, trained_model, subword_tokenizer, 128)
        print("Sampling Generation:")
        _ = generate_text_with_sampling(prompt, trained_model, subword_tokenizer, 128, 0.9, 50, 0.95, presence_penalty=0.5)

    print("\n[Now you can enter your own prompts. Press enter on an empty line to exit.]")
    while True:
        user_input = input("Your prompt: ").strip()
        if not user_input:
            break
        print("Greedy Generation:")
        _ = generate_text(user_input, trained_model, subword_tokenizer, 128)
        print("Sampling Generation:")
        _ = generate_text_with_sampling(user_input, trained_model, subword_tokenizer, 128, 0.9, 50, 0.95, presence_penalty=0.5)

if __name__ == "__main__":
    main()