import pandas as pd
import os

# Create a dataframe for cover images
cover_filenames = [f"{i:06d}.jpg" for i in range(1, 100001) if os.path.exists(f'/home/quinn/workspace/mcse/module8/opdracht/training-images/cover/{i:06d}.jpg')]
cover_labels = [0] * len(cover_filenames)
cover_df = pd.DataFrame({
    'filename': cover_filenames,
    'class': cover_labels
})

# Create a dataframe for stego images
stego_filenames = [f"{i:06d}.jpg" for i in range(1, 100001) if os.path.exists(f'/home/quinn/workspace/mcse/module8/opdracht/training-images/stego/{i:06d}.jpg')]
stego_labels = [1] * len(stego_filenames)
stego_df = pd.DataFrame({
    'filename': stego_filenames,
    'class': stego_labels
})

# Combine the dataframes
df = pd.concat([cover_df, stego_df])

# Save to CSV
df.to_csv('/home/quinn/workspace/mcse/module8/opdracht/training-images/labels.csv', index=False)
