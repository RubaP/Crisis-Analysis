import src.util.utils as Util
import src.util.file as File
import pandas as pd


def isKeywordThere(data):
    """
    Check whether any of the relevant keywords are there in the body of the newspaper
    @param data: newspaper data
    @return: Boolean indicating whether any of the relevant keywords are there in the body of the newspaper or not
    """
    global RELEVANT_KEYWORDS

    body = data['Body']
    if type(body) != str:
        return False
    content = body.lower()

    for key in RELEVANT_KEYWORDS:
        if key in content:
            return True

    return False


def isRelevantNewsSection(data):
    """
    Check whether the newspaper does not contain any irrelevant news section
    @param data: newspaper data
    @return: Boolean indicating the newspaper does not contain any irrelevant news section or not
    """
    global IRRELEVANT_NEWS_SECTIONS
    section = data['Section']

    if type(section) == str:
        for news_type in IRRELEVANT_NEWS_SECTIONS:
            if news_type in section:
                return False

    return True


def geoGraphicallyRelevant(data):
    """
    Check whether the geography mentioned in the meta-data is relevant
    @param data: newspaper data
    @return: Boolean indicating whether the geography mentioned in the meta-data is relevant or not
    """
    global RELEVANT_GEOGRAPHY

    meta = data["Meta-Geographic"]
    if type(meta) != str:
        return True

    if meta == "National Edition":  # If the geography mentioned is only National Edition
        return True
    else:
        meta = meta.lower()

        for geo in RELEVANT_GEOGRAPHY:
            if geo in meta:
                return True

        countries = meta.split(";")

        for country in countries:
            if country.strip().startswith("wales"):  # Avoid retrieving geography from Australia with the keyword wales
                return True

        if "ukraine" in countries[0]:  # If Ukraine is there as the only geography, then consider it as relevant
            if len(countries) == 2 and "national edition" in countries[1]:
                return True
            elif len(countries) == 1:
                return True

    return False


def filterData(root_dir, sources):
    """
    Filter the newspapers based on multiple relevancy criteria and save filtered dataset
    @param root_dir: data directory
    @param sources: list of sources
    """
    filtered_size = 0

    for source in sources:
        print("Source: ", source)
        df = pd.read_csv(File.getOriginalCSVDataDir(root_dir) + source + ".csv", index_col=0)
        size = df.shape[0]
        print("Original size: ", size)

        df['Filter'] = df.apply(lambda x: (isKeywordThere(x) and geoGraphicallyRelevant(x) and isRelevantNewsSection(x)),
                                axis=1)
        df = df.loc[df.Filter]
        df = df.drop('Filter', axis=1)
        df = df.reset_index(drop=True)
        df.to_csv(File.getFilteredDataDir(root_dir) + source + ".csv")
        size = df.shape[0]
        print("Filtered size: ", size)
        filtered_size += size


config = Util.readAllConfig()
root_dir = config["data-directory"]
SOURCES = config["mainstream-sources"]
filter_config = config["mainstream-filtration"]
RELEVANT_KEYWORDS = filter_config["relevant-keywords"]
RELEVANT_GEOGRAPHY = filter_config["relevant-geography"]
IRRELEVANT_NEWS_SECTIONS = filter_config["irrelevant-section-news"]


filterData(root_dir, SOURCES)
