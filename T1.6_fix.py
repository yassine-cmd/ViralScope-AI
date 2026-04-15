#@title 8. Task T1.6: Compute Viral Multiplier and Binary Labels (FIXED)

print("="*60)
print("TASK T1.6: Compute Viral Multiplier and Binary Labels")
print("="*60)

final_df = pd.read_csv(f"{CONFIG['data']['processed_dir']}/final_dataset.csv")
channel_stats = pd.read_csv(f"{CONFIG['data']['processed_dir']}/channel_averages.csv")

print(f"Loaded {len(final_df)} videos and {len(channel_stats)} channels")

# Check channel distribution first
print(f"\nChannel video count distribution:")
print(f"  Channels with 1 video: {(channel_stats['video_count'] == 1).sum()}")
print(f"  Channels with 2 videos: {(channel_stats['video_count'] == 2).sum()}")
print(f"  Channels with 3+ videos: {(channel_stats['video_count'] >= 3).sum()}")

labeled_df = final_df.merge(channel_stats, on='channel_id', how='left')

labeled_df['channel_sum'] = labeled_df.groupby('channel_id')['views'].transform('sum')
labeled_df['channel_count'] = labeled_df.groupby('channel_id')['views'].transform('count')

# Handle channels with only 1 video (LOO would divide by 0)
# For single-video channels, use the video's own views as the "channel average" (meaning multiplier = 1.0)
labeled_df['channel_loo_avg_views'] = (
    (labeled_df['channel_sum'] - labeled_df['views']) / 
    (labeled_df['channel_count'] - 1).clip(lower=1)  # clip to 1 to avoid division by zero
)

# For single-video channels, set multiplier = 1.0 (not viral by definition)
labeled_df['multiplier'] = labeled_df['views'] / (labeled_df['channel_loo_avg_views'] + 1e-5)

# FILTERING: Only filter channels with < 2 videos (not < 3)
reliable_channels = channel_stats[channel_stats['video_count'] >= 2]['channel_id'].tolist()
removed_count = len(labeled_df) - len(labeled_df[labeled_df['channel_id'].isin(reliable_channels)])
labeled_df = labeled_df[labeled_df['channel_id'].isin(reliable_channels)]

print(f"\nAfter filtering channels with <2 videos:")
print(f"  Removed: {removed_count} videos")
print(f"  Remaining: {len(labeled_df)} videos")

if len(labeled_df) == 0:
    print("\n⚠️ WARNING: No videos remaining after filtering!")
    print("   Trying without any channel filtering...")
    labeled_df = final_df.merge(channel_stats, on='channel_id', how='left')
    labeled_df['channel_loo_avg_views'] = labeled_df['loo_avg_views']
    labeled_df['multiplier'] = labeled_df['views'] / (labeled_df['channel_loo_avg_views'] + 1e-5)
    print(f"   Using all {len(labeled_df)} videos without channel filtering")

# Compute binary label
threshold = CONFIG['data']['target_threshold']
labeled_df['is_viral'] = (labeled_df['multiplier'] > threshold).astype(int)

# Dynamic threshold fallback
minority_pct = min(labeled_df['is_viral'].mean(), 1 - labeled_df['is_viral'].mean())
if len(labeled_df) > 0 and minority_pct < 0.10:
    threshold = labeled_df['multiplier'].quantile(0.75)
    print(f"\n⚠️ Class imbalance ({minority_pct:.1%}). Using dynamic threshold: {threshold:.2f}")
    labeled_df['is_viral'] = (labeled_df['multiplier'] > threshold).astype(int)
    minority_pct = min(labeled_df['is_viral'].mean(), 1 - labeled_df['is_viral'].mean())

if len(labeled_df) > 0:
    viral_count = labeled_df['is_viral'].sum()
    non_viral_count = len(labeled_df) - viral_count
    print(f"\nClass distribution:")
    print(f"  - Viral: {viral_count} ({labeled_df['is_viral'].mean():.1%})")
    print(f"  - Non-viral: {non_viral_count} ({1-labeled_df['is_viral'].mean():.1%})")

    output_cols = ['video_id', 'title', 'views', 'channel_loo_avg_views', 'multiplier', 'is_viral', 'channel_id']
    labeled_df = labeled_df[output_cols]

    labeled_df.to_csv(f"{CONFIG['data']['processed_dir']}/labeled_dataset.csv", index=False)
    print(f"\n✓ Saved: {CONFIG['data']['processed_dir']}/labeled_dataset.csv ({len(labeled_df)} rows)")
else:
    print("\n❌ ERROR: Cannot create labeled dataset - no valid data!")
    print("   The dataset may be too small or all thumbnails failed to download.")
