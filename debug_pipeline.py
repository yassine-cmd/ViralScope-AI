# DEBUG: Check which step filtered out all data

import os
import pandas as pd

print("="*60)
print("DEBUGGING: Finding where data was lost")
print("="*60)

# Check each processed file
checks = {
    "trending.csv (raw)": f"{CONFIG['data']['raw_dir']}/trending.csv",
    "clean_dataset.csv": f"{CONFIG['data']['processed_dir']}/clean_dataset.csv",
    "thumbnail_manifest.csv": f"{CONFIG['data']['processed_dir']}/thumbnail_manifest.csv",
    "final_dataset.csv": f"{CONFIG['data']['processed_dir']}/final_dataset.csv",
    "channel_averages.csv": f"{CONFIG['data']['processed_dir']}/channel_averages.csv",
    "labeled_dataset.csv": f"{CONFIG['data']['processed_dir']}/labeled_dataset.csv",
}

row_counts = {}
for name, path in checks.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        row_counts[name] = len(df)
        print(f"\n✓ {name}: {len(df)} rows")
        if 'is_viral' in df.columns:
            print(f"   Viral: {df['is_viral'].sum()} | Non-viral: {(df['is_viral']==0).sum()}")
        if 'video_count' in df.columns:
            print(f"   Channels with ≥3 videos: {(df['video_count']>=3).sum()}")
            print(f"   Channels with <3 videos: {(df['video_count']<3).sum()}")
        if 'status' in df.columns:
            success = (df['status'].isin(['success','exists'])).sum()
            print(f"   Successful thumbnails: {success} ({success/len(df)*100:.1f}%)")
    else:
        row_counts[name] = 0
        print(f"\n✗ {name}: FILE NOT FOUND")

# Check thumbnails
thumb_dir = f"{CONFIG['data']['raw_dir']}/thumbnails"
if os.path.exists(thumb_dir):
    thumbs = len([f for f in os.listdir(thumb_dir) if f.endswith('.jpg')])
    print(f"\n✓ Thumbnail files: {thumbs}")
else:
    print(f"\n✗ Thumbnails directory not found")

print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)

if row_counts.get("trending.csv (raw)", 0) == 0:
    print("❌ STEP 1 FAILED: No raw data downloaded from Kaggle")
    print("   → Re-run Cell 3 (T1.1)")
elif row_counts.get("clean_dataset.csv", 0) == 0:
    print("❌ STEP 2 FAILED: Deduplication removed all rows")
    print("   → Check if trending.csv has valid video_ids and views")
elif row_counts.get("thumbnail_manifest.csv", 0) == 0:
    print("❌ STEP 3 FAILED: Thumbnail manifest not created")
    print("   → Re-run Cell 5 (T1.3)")
else:
    manifest = pd.read_csv(f"{CONFIG['data']['processed_dir']}/thumbnail_manifest.csv")
    success_rate = (manifest['status'].isin(['success','exists'])).mean() * 100
    print(f"Thumbnail success rate: {success_rate:.1f}%")
    if success_rate < 50:
        print("❌ STEP 3 WARNING: Very low thumbnail download rate")
        print("   → Check network connection to YouTube")
    
    if row_counts.get("final_dataset.csv", 0) == 0:
        print("❌ STEP 4 FAILED: No videos with valid thumbnails")
        print(f"   → {row_counts.get('clean_dataset.csv', 0)} clean videos, but 0 with thumbnails")
    elif row_counts.get("channel_averages.csv", 0) == 0:
        print("❌ STEP 5 FAILED: No channel statistics computed")
    else:
        channels = pd.read_csv(f"{CONFIG['data']['processed_dir']}/channel_averages.csv")
        reliable = (channels['video_count'] >= 3).sum()
        print(f"Channels with ≥3 videos: {reliable} / {len(channels)}")
        if reliable == 0:
            print("❌ STEP 6 FAILED: All channels have <3 videos")
            print("   → This is unexpected - check your dataset")
