import pandas as pd
import src.util.utils as Util
import src.util.relevancy as Relevancy
import src.util.file as File


config = Util.readAllConfig()
root_dir = config["data-directory"]
SOURCES = config["mainstream-sources"]
SEARCH_KEYS = config["mainstream-data-collection"]["keywords"]
RELEVANCY_CONFIG = config["relevancy-ranking"]

for source in SOURCES:
    print("---------Processing - ", source, "------------")
    df = pd.read_csv(File.getFilteredDataDir(root_dir) + source + ".csv", index_col=0)
    print("Number of documents: ", df.shape[0])
    discussion_threshold = RELEVANCY_CONFIG["discussion-threshold"]
    popularity_threshold = RELEVANCY_CONFIG["popularity-threshold"]
    growth_rate = RELEVANCY_CONFIG["growth-rate"]
    keys = RELEVANCY_CONFIG["seed-keys"]

    relevant_doc = Relevancy.getMetaDataBasedRanking(df, discussion_threshold, popularity_threshold,
                                                     growth_rate, keys)
    relevant_doc = relevant_doc[["Title", "Date", "Length", "Body"]]
    relevant_doc = relevant_doc.reset_index(drop=True)
    relevant_doc.to_csv(File.getRelevantDataDir(root_dir) + source + '-relevant.csv')
    Relevancy.generateTextFile(relevant_doc, File.getRelevantDataDir(root_dir), source)
