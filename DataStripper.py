import pandas as pd
from tqdm import tqdm


def remove_class_and_save(input_file, output_file, class_to_remove):
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(input_file)

    # Drop the first column
    df = df.drop(df.columns[0], axis=1)
    print("First column dropped.")

    # Display the first few rows to understand the data structure
    print("Original dataset:")
    print(df.head())

    # Check if the class to remove exists in the dataset
    if class_to_remove not in df['label'].unique():
        print(f"Class '{class_to_remove}' not found in the dataset.")
        return

    # Remove the specified class
    print(f"Removing class '{class_to_remove}' from the dataset...")
    df_filtered = df[df['label'] != class_to_remove]

    # **Add this to check the remaining classes**
    print("Remaining classes in the filtered dataset:", df_filtered['label'].unique())

    # Save the new dataset to a file with progress bar
    print(f"Saving new dataset to {output_file}...")
    with tqdm(total=len(df_filtered), desc="Saving", unit="rows") as pbar:
        df_filtered.to_csv(output_file, index=False)
        pbar.update(len(df_filtered))

    print(f"New dataset saved to {output_file}.")
    print("Filtered dataset:")
    print(df_filtered.head())


"""
# Example usage:
input_file = 'iot23_combined_new.csv'
output_file = 'iot23_combined_filtered.csv'
class_to_remove = 'C&C'  # Change this to the class you want to remove

remove_class_and_save(input_file, output_file, class_to_remove)
"""