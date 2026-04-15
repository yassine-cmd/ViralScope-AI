# DEBUG CELL - Check what's in your processed data

print("="*60)
print("DEBUG: Checking Processed Data Files")
print("="*60)

import os

files_to_check = [
    f"{CONFIG['data']['processed_dir']}/labeled_dataset.csv",
    f"{CONFIG['data']['processed_dir']}/final_dataset.csv",
    f"{CONFIG['data']['processed_dir']}/clean_dataset.csv",
    f"{CONFIG['data']['processed_dir']}/channel_averages.csv",
    f"{CONFIG['data']['processed_dir']}/thumbnail_manifest.csv",
]

for filepath in files_to_check:
    filename = os.path.basename(filepath)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"\n✓ {filename}: {len(df)} rows")
        if len(df) > 0 and 'is_viral' in df.columns:
            print(f"   Viral: {df['is_viral'].sum()} | Non-viral: {(df['is_viral']==0).sum()}")
    else:
        print(f"\n✗ {filename}: FILE NOT FOUND")

# Check thumbnails
thumb_dir = f"{CONFIG['data']['raw_dir']}/thumbnails"
if os.path.exists(thumb_dir):
    num_thumbs = len([f for f in os.listdir(thumb_dir) if f.endswith('.jpg')])
    print(f"\n✓ Thumbnails: {num_thumbs} files found")
else:
    print(f"\n✗ Thumbnails directory not found")
