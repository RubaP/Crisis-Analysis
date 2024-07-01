import pandas as pd
import numpy as np
import collections
import src.util.topic as Topic
import src.util.file as File


def getEntityList(esr_path, source):
    """
    Get list of entities mentioned in the source
    @param esr_path: path which contain entity-level polarity csv file
    @param source: source name
    @return: list of entities mentioned in the source
    """
    entities = pd.read_csv(esr_path + source + "-Entity-Polarity.csv", on_bad_lines='skip')['Entity'].unique()
    entities = [str(entity).lower().strip().replace(" ", "-") for entity in entities]
    return entities


def filtercleanCandidatePhrases(candidate_phrases, entities):
    """
    Filter candidate phrases by removing entities
    @param candidate_phrases: list of candidate phrases
    @param entities: list of entities
    @return: list of filtered candidate phrases
    """
    candidate_phrases = [phrase for phrase in candidate_phrases if phrase not in entities]
    candidate_phrases = [phrase for phrase in candidate_phrases if
                         ("mr-" not in phrase and "mrs-" not in phrase)]  # If the phrase starts with mr or mrs,
    # then it will be an entity
    return candidate_phrases


def getIssues(issue_data, neg_phrases, pos_phrases, Issue_Col_Name, polarization_threshold, cleanPhrases, topic_id,
              popularity, entities):
    """
    Identify issue by filtering candidate phrases and computing the polarization
    @param issue_data: data related to issues to store in a CSV file
    @param neg_phrases: phrases associated with negative sentiment
    @param pos_phrases: phrases associated with positive sentiment
    @param Issue_Col_Name: column name indicating the issue label (differs for Ngram and topic labels)
    @param polarization_threshold: polarization threshold
    @param cleanPhrases: Boolean indicating whether a candidate phrase should be cleaned. This is True for Ngram and
    False for topic labels
    @param topic_id: topic ID
    @param popularity: popularity of the issue
    @param entities: list of entities
    @return: list of issues and updated issue_data
    """
    avg_neg_polarity = neg_phrases.groupby([Issue_Col_Name])['Polarity'].mean()
    avg_pos_polarity = pos_phrases.groupby([Issue_Col_Name])['Polarity'].mean()

    candidate_phrases_neg = avg_neg_polarity.index.values.tolist()
    if cleanPhrases:
        candidate_phrases_neg = filtercleanCandidatePhrases(candidate_phrases_neg, entities)

    polarities_neg = avg_neg_polarity.tolist()

    candidate_phrases_pos = avg_pos_polarity.index.values.tolist()
    if cleanPhrases:
        candidate_phrases_pos = filtercleanCandidatePhrases(candidate_phrases_pos, entities)
    polarities_pos = avg_pos_polarity.tolist()

    issues_list = []

    for i in range(len(candidate_phrases_neg)):
        phrase = candidate_phrases_neg[i]
        if phrase in candidate_phrases_pos:
            j = candidate_phrases_pos.index(phrase)
            diff = polarities_pos[j] - polarities_neg[i]
            if topic_id == None:
                issue_data.append([i, phrase, polarities_pos[j], polarities_neg[i], diff])
            else:
                issue_data.append([topic_id, phrase, popularity[phrase], polarities_pos[j], polarities_neg[i], diff])
        else:
            diff = 0 - polarities_neg[i]
        if diff > polarization_threshold:
            issues_list.append(phrase)

    return issues_list, issue_data


def idenfiyIssuesFromTopicLabels(root_dir, sources, polarization_threshold):
    """
    Identify issues represented by topic labels per source. Identified issues are stored in CSV files
    @param root_dir: data directory
    @param sources: list of source names
    @param polarization_threshold: polarization threshold for issue identification
    """
    for source in sources:
        sa_srl_path = File.getESRDir(root_dir)
        issue_path = File.getIssueDir(root_dir)

        df = pd.read_csv(sa_srl_path + source + "-Topic-Polarity.csv")
        non_zero = df.loc[(df['Polarity'] < 0) | (df['Polarity'] > 0)]
        non_zero.to_csv(sa_srl_path + source + "-Topic-NonZero-Polarity.csv")

        data = []
        print("Total number of entries: ", df.shape[0])

        neg_issues = df.loc[df['Polarity'] < 0]
        print("Total number of negative entries: ", neg_issues.shape[0])
        print("Popular candidate phrases: ", collections.Counter(neg_issues['Topic'].tolist()).most_common())

        pos_issues = df.loc[df['Polarity'] > 0]
        print("Total number of positive entries: ", pos_issues.shape[0])

        issues_list, data = getIssues(data, neg_issues, pos_issues, 'Topic', polarization_threshold, False, None, None, [])
        print("Total number of issues: ", len(issues_list))
        print("Topic Issues: ", issues_list)
        np.savetxt(issue_path + source + "-topic-issues.csv", data, delimiter=", ", fmt='% s', encoding="utf-8",
                   header="Topic-ID,Topic,Avg-Pos-Polarity,Avg-Neg-Polarity,Difference")


def idetifyIssuesFromNgram(root_dir, sources, polarization_threshold):
    """
    Identify issues represented by popular ngrams of topics per source. Identified issues are stored in CSV files
    @param root_dir: data directory
    @param sources: list of source names
    @param polarization_threshold: polarization threshold for issue identification
    """
    sa_srl_path = File.getESRDir(root_dir)
    issue_path = File.getIssueDir(root_dir)
    topic_path = File.getTopicDir(root_dir)

    for source in sources:
        df = pd.read_csv(sa_srl_path + source + "-Issue-Polarity.csv")
        entities = getEntityList(sa_srl_path, source)
        topic_names = Topic.getTopicLabels(topic_path + "Label/", source)

        data = []
        global_issues = []
        POPULARITY_THRESHOLD = 5

        df['Issue'] = df['Issue'].apply(lambda x: x.strip())

        for topic_id in range(0, len(topic_names)):
            issues = df.loc[df['# Topic-ID'] == topic_id]
            topic_issue_strength = collections.Counter(issues['Issue'].tolist())
            print("Total number of entries: ", issues.shape[0])

            neg_issues = issues.loc[issues['Polarity'] < 0]
            print("Total number of negative entries: ", neg_issues.shape[0])
            count = neg_issues.Issue.value_counts().gt(POPULARITY_THRESHOLD - 1)
            neg_issues = neg_issues.loc[neg_issues.Issue.isin(count[count].index)]
            print("Popular negative candidate phrases: ",
                  collections.Counter(neg_issues['Issue'].tolist()).most_common())

            pos_issues = issues.loc[issues['Polarity'] > 0]
            print("Total number of positive entries: ", pos_issues.shape[0])
            count = pos_issues.Issue.value_counts().gt(POPULARITY_THRESHOLD - 1)
            pos_issues = pos_issues.loc[pos_issues.Issue.isin(count[count].index)]
            print("Popular positive candidate phrases: ",
                  collections.Counter(pos_issues['Issue'].tolist()).most_common())

            issues_list, data = getIssues(data, neg_issues, pos_issues, 'Issue', polarization_threshold, True,
                                          topic_id,
                                          topic_issue_strength, entities)
            global_issues.extend(issues_list)
            print("Total number of issues: ", len(issues_list))
            print("Issues: ", issues_list)

        print("Popular global issues: ", collections.Counter(global_issues).most_common())

        np.savetxt(issue_path + source + "-ngram-issues.csv", data, delimiter=", ", fmt='% s', encoding="utf-8",
                   header="Topic-ID,Ngram,Popularity,Avg-Pos-Polarity,Avg-Neg-Polarity,Difference")


def printIssueStats(root_dir, sources):
    """
    Generate stats about issues across sources. This includes global issues, common issues across two sources and
    unique issues per source
    @param root_dir: data directory
    @param sources: list of source names
    """
    issue_path = File.getIssueDir(root_dir)

    issues = {}

    print("--------------Issues Stat---------------")
    for source in sources:
        df = pd.read_csv(issue_path + source + "-ngram-issues.csv", on_bad_lines='skip')
        issue_list = df['Candidate Phrase'].tolist()
        issues[source] = issue_list
        print("Number of Issues in ", source, " : ", len(issue_list))

    global_issues = None
    for i in range(len(sources)):
        s1 = sources[i]

        for j in range(i + 1, len(sources)):
            s2 = sources[j]
            if s1 == s2:
                continue

            print(s1, "vs ", s2, ":")
            common_issues = set(issues[s1]).intersection(issues[s2])
            print("Common Issues Count: ", len(common_issues))
            print("Common Issues: ", common_issues)

            if global_issues is None:
                global_issues = common_issues
            else:
                global_issues = set(global_issues).intersection(common_issues)

    print("Global Issues Count: ", len(global_issues))
    print("Global Issues: ", global_issues)

    for i in range(len(sources)):
        s1 = sources[i]

        for j in range(i + 1, len(sources)):
            s2 = sources[j]
            if s1 == s2:
                continue

            print(s1, "vs ", s2, ":")
            common_issues = set(issues[s1]).intersection(issues[s2])
            print("Unique Issues: ", common_issues - global_issues)


def getIssuesPresentInText(bow, candidate_issues):
    """
    Get list of issues present in bag-of-words representation of a document
    @param bow: bag-of-words representation of a document
    @param candidate_issues: list of candidate issues
    @return: list of issues present in bag-of-words representation of a document
    """
    issues_present = []

    for word in bow:
        if word in candidate_issues:
            issues_present.append(word)

    return issues_present


def getCandidateIssues(top_words):
    """
    Get candidate issues by extracting ngram in popular words of a topic
    @param top_words: list of popular words
    @return: candidate issue phrases
    """
    candidate_issues = []

    for i in range(len(top_words)):
        candidates = []
        top = top_words[i]

        for word in top:
            if "-" in word:
                candidates.append(word)
        candidate_issues.append(candidates)
        print("Candidate Issues for Topic ", i, ": ", candidates)

    return candidate_issues