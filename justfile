# Step 1: Assuming you've got access to and downloaded the data,
# and placed them like the stubs in the data folder, this will
# combine them into one database.
organize:
    uv run src/organize_scores.py

# Step 2: Compute the metric for all sets of leaderboard partitions,
# splitting by both sensor collection and public vs private sets.
evaluate:
    uv run src/evaluate_models.py

# Step 3: Make gif from submission confusion matrices.
make_gif:
    ffmpeg -framerate 10 -i ./data/figs/submission_%d.png ./data/figs/confusion.gif