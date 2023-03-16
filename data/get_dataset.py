from clearml import Dataset

ds = Dataset.get(dataset_id="693d0797aab14613b851bdbf37b9efa1")

# The dataset is not download but it has all the meta data
print(ds.list_files()[:5])

# # Get read-only local copy of the data, now we will download
# local_path_read_only = ds.get_local_copy()
# print(local_path_read_only)

# # Get a mutable copy of dataset to the desired folder
local_path_mutable = ds.get_mutable_local_copy(
    target_folder="./clearml-data/tmp/original"
)
