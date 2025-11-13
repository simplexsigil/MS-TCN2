import pandas as pd
import os # Import the os library for path manipulation


def merge_single_column_csvs(root_dir, file_list, output_dir, split):
    all_dataframes = []

    for file_path in file_list:
        path = os.path.join(root_dir, file_path)

        try:
            df = pd.read_csv(os.path.join(path, f"{split}.csv"), header=None)
            df[0] = file_path + '/' + df[0].astype(str)
            all_dataframes.append(df)
            print(f"Successfully read {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}. Skipping.")

    merged_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\nSuccessfully concatenated data from {len(all_dataframes)} files.")

    output_file = os.path.join(output_dir, f"{split}_cs.csv")
    merged_df.to_csv(output_file, index=False)
    print(f"Merged data saved successfully to {output_file}")


if __name__ == "__main__":
    # List of CSV files to merge
    root_dir = "/pfs/work8/workspace/ffhk/scratch/kf3609-ws/data/omnifall/splits/cs"
    out_dir = "/pfs/work8/workspace/ffhk/scratch/kf3609-ws/data/omnifall/segmentation_annotations/splits"
    dataset_list = [
        "caucafall",
        "cmdfall",
        "edf",
        "gmdcsa24",
        "le2i",
        "mcfd",
        "occu",
        "up_fall",
        "OOPS",  # only for test!
    ]

    # Call the function
    merge_single_column_csvs(root_dir, dataset_list, out_dir, split="test")
