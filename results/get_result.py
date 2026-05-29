import os
import pickle
import pandas as pd

prompt_dataset = pd.read_csv("../dataset_creation/test_dataset_50_50.csv")
# Define the directory containing your pickle files
# Replace '.' with your folder path if needed (e.g., 'path/to/pickles')
directory_path = "pickles"

# List to store data before converting to a DataFrame
data_list = []
merged_df = pd.DataFrame()
# Loop through all files in the directory
for filename in os.listdir(directory_path):
    # Check if the file has a .pickle or .pkl extension
    if filename.endswith(".pickle"):
        file_path = os.path.join(directory_path, filename)

        try:
            # Open and load the pickle file
            with open(file_path, "rb") as f:
                data = pd.DataFrame(pickle.load(f))
            merged_df = pd.concat([merged_df,data], ignore_index=True)
            # Ensure the loaded data is a dictionary (or supports len())
            if isinstance(data, dict):
                file_length = len(data)
            # Append the name and length to our list
            data_list.append({"File Name": filename, "Length": file_length})

        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Convert the list of dictionaries into a pandas DataFrame
df = pd.DataFrame(data_list)

# Display the DataFrame
print(df)