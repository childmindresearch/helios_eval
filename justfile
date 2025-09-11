# Step 1: Assuming you've got access to and downloaded the data,
# and placed them like the stubs in the data folder, this will
# combine them into one database.
organize:
    uv run src/organize_scores.py

# Step 2: Compute the metric for all sets of leaderboard partitions,
# splitting by both sensor collection and public vs private sets.
evaluate:
    uv run src/evaluate_models.py