import src.util.utils as Util
import pandas as pd
import random
import warnings
import src.util.file as File

warnings.filterwarnings("ignore")

SAMPLE_SIZE = 100
config = Util.readAllConfig()
root_dir = config["data-directory"]
SOURCES = config["mainstream-sources"]

datasets = []
dataset_sizes = []
sources_size = len(SOURCES)

for source in SOURCES:
    df = pd.read_csv(File.getFilteredDataDir(root_dir) + source + ".csv", index_col=0)
    datasets.append(df)
    dataset_sizes.append(df.shape[0])

sample = None
sample_source_name = []

titles = pd.read_excel(File.getGroundTruthDataDir(root_dir) + 'Relevancy Ground Truth.xlsx')['Title'].tolist()
i = 0

while i < SAMPLE_SIZE:
    source_id = random.randrange(sources_size)
    row_id = random.randrange(dataset_sizes[source_id])

    data = datasets[source_id].iloc[[row_id]]
    if data['Title'].item() not in titles:
        sample_source_name.append(SOURCES[source_id])
        if i == 0:
            sample = data
        else:
            sample = pd.concat([sample, data], ignore_index=False)
        i += 1
    else:
        print("Duplicate: ", data)


sample['Source'] = sample_source_name
sample['Id'] = list(range(0, SAMPLE_SIZE))
sample.apply(File.saveSampleNewspaperAsText, args=root_dir, axis=1)
sample = sample[["Id", "Source", "Title"]]
sample.to_csv(File.getGroundTruthDataDir(root_dir) + "sample.csv")
