#@title 9. Task T1.7: Pre-Tokenize Text Corpus (FIXED)

print("="*60)
print("TASK T1.7: Pre-Tokenize Text Corpus with DistilBERT")
print("="*60)

labeled_df = pd.read_csv(f"{CONFIG['data']['processed_dir']}/labeled_dataset.csv")

# Check if dataframe is empty
if len(labeled_df) == 0:
    print("❌ ERROR: labeled_dataset.csv is empty!")
    print("   This means the previous pipeline steps filtered out all rows.")
    print("   Possible causes:")
    print("   1. Kaggle download failed - run Cell 3 again")
    print("   2. Thumbnail download had very low success rate")
    print("   3. All channels have < 3 videos")
    print("\n   Check your data/processed/ files for issues.")
    raise ValueError("labeled_dataset.csv is empty!")

print(f"Loaded {len(labeled_df)} samples from labeled_dataset.csv")

# Clean titles - ensure all are valid strings
def clean_title(title):
    if pd.isna(title) or str(title).strip() == '':
        return 'Untitled Video'
    # Remove emojis and special characters
    cleaned = re.sub(r'[\U00010000-\U0010ffff]', '', str(title))
    if cleaned.strip() == '':
        return 'Untitled Video'
    return cleaned

labeled_df['title'] = labeled_df['title'].apply(clean_title)

# Safely print sample title
sample_title = labeled_df['title'].iloc[0] if len(labeled_df) > 0 else 'N/A'
print(f"Cleaned titles. Sample: {sample_title[:50]}...")

print("Loading DistilBERT tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained(
    CONFIG['model']['nlp']['checkpoint'],
    use_fast=True
)

print("Tokenizing titles in batches...")
titles = labeled_df['title'].tolist()
print(f"Total titles to tokenize: {len(titles)}")

# Tokenize in batches to avoid memory issues
batch_size = 5000
all_input_ids = []
all_attention_masks = []

for i in tqdm(range(0, len(titles), batch_size), desc="Tokenizing batches"):
    batch = titles[i:i+batch_size]
    # Tokenize each title individually then stack
    encoded = tokenizer(
        batch,
        max_length=CONFIG['model']['nlp']['max_seq_length'],
        padding='max_length',
        truncation=True,
        return_tensors=None  # Return lists, not tensors
    )
    all_input_ids.extend(encoded['input_ids'])
    all_attention_masks.extend(encoded['attention_mask'])

# Convert to tensors
input_ids = torch.tensor(all_input_ids, dtype=torch.long)
attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)

print(f"Tokenization complete! Shape: {input_ids.shape}")

# Save tensors
torch.save(input_ids, f"{CONFIG['data']['tensor_dir']}/input_ids.pt")
torch.save(attention_masks, f"{CONFIG['data']['tensor_dir']}/attention_masks.pt")

# Compute and save dataset hash
df_hash = hashlib.md5(labeled_df.to_csv(index=False).encode()).hexdigest()

metadata = {
    'dataset_hash': df_hash,
    'num_samples': len(labeled_df),
    'max_seq_length': CONFIG['model']['nlp']['max_seq_length'],
    'tokenizer_checkpoint': CONFIG['model']['nlp']['checkpoint']
}

with open(f"{CONFIG['data']['tensor_dir']}/tokenizer_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Saved: {CONFIG['data']['tensor_dir']}/input_ids.pt - shape {input_ids.shape}")
print(f"✓ Saved: {CONFIG['data']['tensor_dir']}/attention_masks.pt - shape {attention_masks.shape}")
print(f"✓ Saved: {CONFIG['data']['tensor_dir']}/tokenizer_metadata.json")
print(f"✓ Dataset hash: {df_hash}")
