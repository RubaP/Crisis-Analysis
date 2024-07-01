import randomcolor
from pyvis.network import Network
import pandas as pd
from nltk.corpus import stopwords
import collections
import re
import glob
import src.util.topic as Topic
import src.util.file as File

stop_words = stopwords.words('english')


def getNewColor(existing_colors):
    """
    Get a new color given list of existing colors
    @param existing_colors: list of existing colors
    @return: new color
    """
    color = randomcolor.RandomColor().generate()[0]
    while color in existing_colors:
        color = randomcolor.RandomColor().generate()[0]
    return color


def addTopicNodes(network, topic_names, existing_colors):
    """
    Add topic nodes to the network
    @param network: network
    @param topic_names: list of topic labels
    @param existing_colors: list of existing colors in the network
    @return: new network, new set of existing colors in the network
    """
    for topic_id in range(0, len(topic_names)):
        name = topic_names[topic_id]

        if not Topic.isTopicValid(name): # Filter invalid topics
            continue
        color = getNewColor(existing_colors)
        existing_colors.append(color)
        network.add_node(n_id=topic_id, label=name, color=color, shape="box")

    return network, existing_colors


def addSourceNodes(network, source_names, colors):
    """
    Add source nodes to the network
    @param network: network
    @param source_names: list of source names
    @param colors: predefined list of colors of source
    @return: new network
    """
    for i in range(len(source_names)):
        source = source_names[i]
        network.add_node(n_id=i, label=source, color=colors[source], shape="ellipse")

    return network


def addIssueNodes(issues_list, topic_id, next_node_id, network, node_details, existing_colors, sizes, weights):
    """
    Add issue nodes to the network
    @param issues_list: list of issue names
    @param topic_id: topic id of the issues
    @param next_node_id: next node id in the network
    @param network: network
    @param node_details: existing issue nodes in the network
    @param existing_colors: list of existing colors
    @param sizes: node sizes as a dictionary
    @param weights: edge weights as a dictionary
    @return: updated network, node_details, next_node_id, existing_colors
    """
    for issue in issues_list:
        if issue not in node_details.keys():
            color = getNewColor(existing_colors)
            existing_colors.append(color)
            network.add_node(n_id=next_node_id, label=issue.strip(), color=color, value=int(sizes[issue] * 100))
            node_details[issue] = next_node_id
            current_node_id = next_node_id
            next_node_id += 1
        else:
            current_node_id = node_details[issue]

        network.add_edge(topic_id, current_node_id, value=weights[issue])

    return network, node_details, next_node_id, existing_colors


def addActorNodes(actors_list, topic_id, next_node_id, network, node_details, sizes, weights,
                  polarity_list, colors, edge_labels):
    """
    Add actor nodes to the network
    @param actors_list: list of actor names
    @param topic_id: topic ID of the actors
    @param next_node_id: next node id in the network
    @param network: network
    @param node_details: existing actor nodes in the network
    @param sizes: node sizes as a dictionary
    @param weights: edge weights as a dictionary
    @param polarity_list: list of average polarity of actors with respect to the topic
    @param colors: list of predefined colors for the actor nodes
    @param edge_labels: list of edge labels for the actor nodes
    @return: updated network, node_details, next_node_id
    """
    for i in range(len(actors_list)):
        actor = actors_list[i]
        display_name = actor
        label = getNodeLabel(display_name)
        if label not in node_details.keys():
            network.add_node(n_id=next_node_id, label=display_name, color=colors[i], value=sizes[actor] * 100)
            node_details[label] = next_node_id
            current_node_id = next_node_id
            next_node_id += 1
        else:
            current_node_id = node_details[label]

        edge_color = getEdgeColorByPolarity(polarity_list[i])

        if edge_labels is not None:
            network.add_edge(topic_id, current_node_id, value=weights[actor], color=edge_color, title=edge_labels[i])
        else:
            network.add_edge(topic_id, current_node_id, value=weights[actor], color=edge_color)

    return network, node_details, next_node_id


def getNodeLabel(text):
    """
    Get node label
    @param text: input text
    @return: node label
    """
    return text.lower()


def addArgumentNodes(arguments, topic_id, next_node_id, network, argument_nodes, actor_nodes, sizes,
                     weights, colors, edge_labels):
    """
    Add argument nodes to the network
    @param arguments: list of arguments
    @param topic_id: topic ID of the actors
    @param next_node_id: next node id in the network
    @param network: network
    @param argument_nodes: existing argument nodes in the network
    @param actor_nodes: existing actor nodes in the network
    @param sizes: node sizes as a dictionary
    @param weights: edge weights as a dictionary
    @param colors: list of predefined colors for the actor nodes
    @param edge_labels: list of edge labels for the actor nodes
    @return: updated network, argument_nodes, next_node_id
    """
    for i in range(len(arguments)):
        display_name = arguments[i]
        label = getNodeLabel(display_name)

        if label in argument_nodes.keys():
            current_node_id = argument_nodes[label]
        elif label in actor_nodes.keys():
            current_node_id = actor_nodes[label]
        else:
            network.add_node(n_id=next_node_id, label=display_name, color=colors[i], shape="triangle",
                             value=sizes[display_name] * 100)
            argument_nodes[label] = next_node_id
            current_node_id = next_node_id
            next_node_id += 1

        if edge_labels is not None:
            network.add_edge(topic_id, current_node_id, value=weights[display_name], color="grey", title=edge_labels[i])
        else:
            network.add_edge(topic_id, current_node_id, value=weights[display_name], color="grey")

    return network, argument_nodes, next_node_id


def getEdgeColorByPolarity(polarity):
    """
    Get ege color based on the average polarity of node with respect to the connected node (e.g. topic, issue)
    @param polarity: average polarity with respect to the connected node (e.g. topic, issue)
    @return: edge color
    """
    if polarity < 0:
        return "red"
    elif polarity > 0:
        return "green"
    else:
        return "blue"


def cleanArgument(arg):
    """
    Clean an argument by removing titles and articles
    @param arg: argument
    @return: cleaned argument
    """
    arg = str(arg).strip()
    if arg.lower().startswith('the '):
        arg = arg[4:]
    if arg.lower().startswith('mr '):
        arg = arg[3:]
    if arg.lower().startswith('ms '):
        arg = arg[3:]
    if arg[-2:] == "'s":
        arg = arg[:-2]

    if len(arg) == 0 or arg.lower() in stop_words:
        return None
    else:
        return arg


def getEntityColorList(entity_list, causers, targets):
    """
    Get list of colors for the entities based on their roles
    @param entity_list: list of entities
    @param causers: list of causers in the network
    @param targets: list of targets in the network
    @return: list of colors
    """
    colors = []

    for entity in entity_list:
        if entity in causers and targets:
            colors.append("purple")
        elif entity in causers:
            colors.append("black")
        elif entity in targets:
            colors.append("yellow")
        else:
            colors.append("#ADD8E6")
    return colors


def generateSourceIssueGraph(root_dir, sources, config):
    """
    Generate source-issue graph
    @param root_dir: data directory
    @param sources: list of sources
    @param config: configuration for visualization
    """
    issue_path = File.getIssueDir(root_dir)
    graphs_path = File.getGraphDir(root_dir)
    N = config["num-of-issues"]

    network = Network()
    source_colors = {"Mirror": "#D7EFF6", "Mail": "#F8BED9", "Times": "#C2F7E6", "Guardian": "#F8F3CA"}
    network = addSourceNodes(network, sources, source_colors)

    issue_strength = {}
    source_level_issue_details = {}

    for source in sources:
        issue_df = pd.read_csv(issue_path + source + "-ngram-issues.csv")  # Read all the ngram issues
        issue_popularity = issue_df.groupby("Ngram")['Popularity'].sum()  # Find issue popularity at source level

        issue_list = issue_popularity.index.values.tolist()
        popularity_score = issue_popularity.tolist()

        for i in range(len(issue_list)):
            issue = issue_list[i]

            if issue in issue_strength.keys():
                issue_strength[issue] += popularity_score[i]  # Find issue popularity across all the sources
            else:
                issue_strength[issue] = popularity_score[i]  # Find issue popularity across all the sources

        top_issues = issue_popularity.nlargest(N)  # Choose top N popular issues
        top_issue_list = top_issues.index.values.tolist()
        top_popularity_score = top_issues.tolist()
        source_level_issue_details[source] = {"top-issues": top_issue_list,
                                              "source-strength": dict(zip(top_issue_list, top_popularity_score))}

    next_node_id = len(sources)
    issue_nodes = {}
    existing_colors = []

    for source_id in range(len(sources)):
        source = sources[source_id]
        issues_list = source_level_issue_details[source]["top-issues"]
        topic_issue_strength = source_level_issue_details[source]["source-strength"]

        analysis_network, issue_nodes, next_node_id, existing_colors = addIssueNodes(issues_list, source_id,
                                                                                     next_node_id,
                                                                                     network,
                                                                                     issue_nodes,
                                                                                     existing_colors,
                                                                                     issue_strength,
                                                                                     topic_issue_strength)

    try:
        postfix = ""
        network.write_html(graphs_path + "Sources-Issues" + postfix+ ".html")
    except:
        pass

def generateTopicEntityGraph(root_dir, sources):
    """
    Generate Topic-Entity graph
    @param root_dir: data directory
    @param sources: list of sources
    """
    sa_srl_path = File.getESRDir(root_dir)
    topic_path = File.getTopicDir(root_dir)
    graphs_path = File.getGraphDir(root_dir)

    POPULARITY_THRESHOLD = {"Mirror": 15, "Mail": 30, "Times": 40, "Guardian": 120}

    for source in sources:
        print("source: ", source)
        threshold = POPULARITY_THRESHOLD[source]
        actors = pd.read_csv(sa_srl_path + source + "-Entity-Polarity.csv", on_bad_lines='skip')
        srl = pd.read_csv(sa_srl_path + source + "-Semantic Roles.csv", on_bad_lines='skip')
        actors["Polarity"] = pd.to_numeric(actors["Polarity"], errors="coerce")
        actors = actors.dropna(subset=['Polarity'])

        topic_names = Topic.getTopicLabels(topic_path + "Label/", source)
        K = len(topic_names)

        analysis_network = Network()
        actor_nodes = {}
        argument_nodes = {}
        existing_colors = []

        analysis_network, existing_colors = addTopicNodes(analysis_network, topic_names, existing_colors)

        # Clean entities and arguments
        actors['Entity'] = actors['Entity'].apply(lambda x: cleanArgument(x))
        srl['ARG0'] = srl['ARG0'].apply(lambda x: cleanArgument(x))
        srl['ARG1'] = srl['ARG1'].apply(lambda x: cleanArgument(x))

        actors = actors.dropna(subset=['Entity'])
        srl = srl.dropna(subset=['ARG0', 'ARG1'])

        # Fine popularity of actors and arguments, where arguments can also include actor names
        popularity = collections.Counter(actors['Entity'].tolist())
        popularity.update(srl['ARG0'].tolist())
        popularity.update(srl['ARG1'].tolist())
        popular_actors_list = []
        popular_causer_list = []
        popular_target_list = []

        next_node_id = K

        # Find popular actors based on the popularity threshold of a source
        for topic_id in range(0, len(topic_names)):
            if not Topic.isTopicValid(topic_names[topic_id]): # Filter invalid topics
                continue

            topic_actors = actors.loc[actors['# Topic-ID'] == topic_id]
            count = topic_actors.Entity.value_counts().gt(threshold - 1)
            topic_actors = topic_actors.loc[topic_actors.Entity.isin(count[count].index)]
            popular_actors_list.extend(topic_actors['Entity'].unique())

        popular_actors_list = list(set(popular_actors_list))
        print("Number of Popular Actors: ", len(popular_actors_list))


        # Find popular targets and causers based on the popularity threshold of a source
        for topic_id in range(0, len(topic_names)):
            topic_srl = srl.loc[srl['# Topic-ID'] == topic_id]

            causers = collections.Counter(topic_srl['ARG0'].tolist()).most_common()
            causers = [arg for (arg, count) in causers if count >= threshold or arg in popular_actors_list]
            popular_causer_list.extend(causers)
            targets = collections.Counter(topic_srl['ARG1'].tolist()).most_common()
            targets = [arg for (arg, count) in targets if count >= threshold or arg in popular_actors_list]
            popular_target_list.extend(targets)

        popular_causer_list = list(set(popular_causer_list))
        print("Number of Popular Causers: ", len(popular_causer_list))

        popular_target_list = list(set(popular_target_list))
        print("Number of Popular Targets: ", len(popular_target_list))

        for topic_id in range(0, len(topic_names)):
            if not Topic.isTopicValid(topic_names[topic_id]): # Filter invalid topics
                continue

            print("Topic - ", topic_id)
            topic_actors = actors.loc[actors['# Topic-ID'] == topic_id]
            topic_srl = srl.loc[srl['# Topic-ID'] == topic_id]


            # Find topic level popularity of actos and arguments
            count = topic_actors.Entity.value_counts().gt(threshold - 1)
            topic_actors = topic_actors.loc[topic_actors.Entity.isin(count[count].index)]
            topic_strength = collections.Counter(topic_actors['Entity'].tolist())
            topic_strength.update(topic_srl['ARG0'].tolist())
            topic_strength.update(topic_srl['ARG1'].tolist())


            # Find topic level polarity of actors
            avg_polarity = topic_actors.groupby(['Entity'])['Polarity'].mean()
            entity_list = avg_polarity.index.values.tolist()
            popular_actors_list.extend(entity_list)
            polarity_score = avg_polarity.tolist()
            print("Number of actors: ", len(entity_list))

            # Get color list based on roles of actors
            actor_color_list = getEntityColorList(entity_list, popular_causer_list, popular_target_list)

            # Add actor nodes to the network
            analysis_network, actor_nodes, next_node_id = addActorNodes(entity_list, topic_id,
                                                                        next_node_id,
                                                                        analysis_network,
                                                                        actor_nodes,
                                                                        popularity,
                                                                        topic_strength,
                                                                        polarity_score,
                                                                        actor_color_list,
                                                                        None)

        print("Number of actors added: ", next_node_id)

        for topic_id in range(0, len(topic_names)):
            if not Topic.isTopicValid(topic_names[topic_id]): # Filter invalid topics
                continue

            print("Topic - ", topic_id)
            topic_actors = actors.loc[actors['# Topic-ID'] == topic_id]
            topic_srl = srl.loc[srl['# Topic-ID'] == topic_id]

            # Find topic level popularity of actos and arguments
            count = topic_actors.Entity.value_counts().gt(threshold - 1)
            topic_actors = topic_actors.loc[topic_actors.Entity.isin(count[count].index)]
            topic_strength = collections.Counter(topic_actors['Entity'].tolist())
            topic_strength.update(topic_srl['ARG0'].tolist())
            topic_strength.update(topic_srl['ARG1'].tolist())

            # Find list of causers and targets to be included in the network
            causers = collections.Counter(topic_srl['ARG0'].tolist()).most_common()
            causers = [arg for (arg, count) in causers if count >= threshold]
            print("Number of causers: ", len(causers))
            targets = collections.Counter(topic_srl['ARG1'].tolist()).most_common()
            targets = [arg for (arg, count) in targets if count >= threshold]
            print("Number of targets: ", len(targets))

            # Find list of colors for targets and causers based on their role
            causer_color_list = getEntityColorList(causers, popular_causer_list, popular_target_list)
            target_color_list = getEntityColorList(targets, popular_causer_list, popular_target_list)

            # Add causer nodes to the network
            analysis_network, argument_nodes, next_node_id = addArgumentNodes(causers, topic_id,
                                                                              next_node_id,
                                                                              analysis_network,
                                                                              argument_nodes,
                                                                              actor_nodes,
                                                                              popularity,
                                                                              topic_strength,
                                                                              causer_color_list,
                                                                              None)

            # Add target nodes to the network
            analysis_network, argument_nodes, next_node_id = addArgumentNodes(targets, topic_id,
                                                                              next_node_id,
                                                                              analysis_network,
                                                                              argument_nodes,
                                                                              actor_nodes,
                                                                              popularity,
                                                                              topic_strength,
                                                                              target_color_list,
                                                                              None)
        print("Total number of nodes in the network: ", next_node_id)
        try:
            analysis_network.write_html(graphs_path + "Topic-Entity/" + source + "-Topics-Entities.html")
        except:
            pass

def getCauserSentimentPairs(text):
    """
    Get list of causers and their corresponding sentiment stored as a text in annotation files
    @param text: input text
    @return: list of causers and their corresponding sentiment
    """
    results = []
    pairs = re.findall(r'\((.*?)\)', text)

    for pair in pairs:
        entity = cleanArgument(re.findall(r"\'(.*?)\'", pair)[0])
        sentiment = float(re.findall(r'[-+]?[0-1]*\.?[0-9]+', pair)[0])

        if entity is not None:
            results.append((entity, sentiment))

    return results


def getActorSentimentPairs(text):
    """
    Get list of actors and their corresponding sentiment stored as a text in annotation files
    @param text: input text
    @return: list of actors and their corresponding sentiment
    """
    results = []
    pairs = re.split(",", text)

    for pair in pairs:
        if ":" in pair:
            try:
                entity = cleanArgument(pair.split(":")[0])
                sentiment = float(pair.split(":")[1])
                results.append((entity, sentiment))
            except:
                pass

    return results


def getEntityColorListCauserOnly(actors, causers):
    """
    Get list of colors for an actors when there is only causer nodes in the network (target nodes are not present)
    @param actors: list of actors
    @param causers: list of causers
    @return: list of colors
    """
    colors = []

    for actor in actors:
        if actor in causers:
            colors.append("black")
        else:
            colors.append("#ADD8E6")

    return colors


def getArgumentsText(data, roles):
    """
    Combine all the arguments of a SRL into a single text. This text will be used to search for presence of
    issues and actors in arguments
    @param data: list of SRL generated
    @param roles: list of roles
    @return: string containing all the arguments separated by space
    """
    text = ""
    for role in roles:
        if type(data[role]) is str:
            text += " " + data[role]

    return text.strip().lower()


def getRowsWithIssue(srl_data, issue):
    """
    Find all the rows that contain an issue
    @param srl_data: SRL data
    @param issue: issue name
    @return: list of SRL data which contains an issue in any of the argument
    """
    issue = issue.replace("-", " ")

    srl_data['related'] = srl_data['text'].apply(lambda x: issue in x)
    srl_data = srl_data.loc[srl_data.related]
    srl_data = srl_data.drop('related', axis=1)
    print("Number of sentecnes with SRL: ", srl_data.shape[0])
    return srl_data


def getSRLVerbs(entities, srl_data):
    """
    Get list of verbs of list of entities (verbs that an entity is associated with)
    @param entities: list of entities
    @param srl_data: SRL data
    @return: 2D list of verbs
    """
    labels = []

    for entity in entities:
        label_list = []
        entity = entity.lower()

        for index, row in srl_data.iterrows():
            if entity in row['text']:
                label_list.append(row['Verb'].lower())
        label_list = list(set(label_list))
        labels.append(", ".join(label_list))

    return labels


def generateIssueActorGraph(root_dir, sources, config):
    """
    Generate Issue-Actor graph
    @param root_dir: data directory
    @param sources: list of sources
    @param config: config for visualization
    """
    issue_path = File.getIssueDir(root_dir)
    graphs_path = File.getGraphDir(root_dir)
    annotation_path = File.getAnnotationDir(root_dir)
    sa_srl_path = File.getESRDir(root_dir)
    N_issues = config["num-of-issues"]
    N_entity = config["num-of-actors"]

    source_colors = {"Mirror": "#D7EFF6", "Mail": "#F8BED9", "Times": "#C2F7E6", "Guardian": "#F8F3CA"}

    global_issues = []

    for source in sources:
        issue_df = pd.read_csv(issue_path + source + "-ngram-issues.csv")  # Read all the ngram issues
        issue_popularity = issue_df.groupby("Ngram")['Popularity'].sum() # Find issue popularity at source level

        top_issues = issue_popularity.nlargest(N_issues)  # Chosse top N popular issues
        top_issue_list = top_issues.index.values.tolist()
        global_issues.extend(top_issue_list)

    # Find global issues which are combination of top N issue of each source
    global_issues = list(set(global_issues))
    global_issues = [issue.strip() for issue in global_issues]
    print("Total Number of Global Issues: ", len(global_issues))
    print("Global Issues: ", global_issues)

    srl_data = {}
    SEMANTIC_ROLES = ["ARG0", "ARG1", "ARG2", "ARG3", "ARGM-PRP", "ARGM-TMP", "ARGM-LOC", "ARGM-DIR", "ARGM-MNR"]

    for source in sources:
        srl = pd.read_csv(sa_srl_path + source + "-Semantic Roles.csv", on_bad_lines='skip')
        srl = srl.dropna(subset=['ARG0', 'ARG1'])
        srl['text'] = srl.apply(lambda x: getArgumentsText(x, SEMANTIC_ROLES), axis=1)  # Combine all the arguments
        # into a single text for search of entities and issues
        srl_data[source] = srl[['Verb', 'text']]

    for issue in global_issues:  # For each global issue generate one graph
        print("Processing Issue - ", issue)
        network = Network()
        network = addSourceNodes(network, sources, source_colors)  # Add source nodes to the network first
        next_node_id = len(sources)
        actor_nodes = {}
        causer_nodes = {}

        causer_details = {}
        actors_details = {}
        global_causer_list = []
        global_actor_list = []

        for source in sources:
            path = annotation_path + source + "/"
            causer_details[source] = []
            actors_details[source] = []

            for file_name in glob.glob(path + '*.csv'): # Read all the annotation files of a source
                topic_file = pd.read_csv(file_name)[['Issues', 'Target-Polarity', 'Actor-Polarity']]
                topic_file = topic_file.dropna(subset=['Issues'])

                for index, row in topic_file.iterrows():
                    if issue in row['Issues']:  # if the issue is present in the
                        targets = row['Target-Polarity']
                        actors = row['Actor-Polarity']

                        if type(targets) is str:  # get the list of causers associated with the issue
                            causer_list = getCauserSentimentPairs(targets)
                            causer_details[source].extend(causer_list)
                            [global_causer_list.append(causer) for (causer, polarity) in causer_list]

                        if type(actors) is str:  # get the list of actors associated with the issue
                            actor_list = getActorSentimentPairs(actors)
                            actors_details[source].extend(actor_list)
                            [global_actor_list.append(actor) for (actor, polarity) in actor_list]

        # Merge the causers and actors to find unique list of causers and actors of an issue across all sources
        global_causer_counter = collections.Counter(global_causer_list)
        global_causer_list = list(set(global_causer_list))
        [global_actor_list.append(actor) for actor in global_causer_list if actor in global_actor_list]
        global_actor_counter = collections.Counter(global_actor_list)
        global_actor_list = list(set(global_actor_list))

        print("Total number of causers: ", len(global_causer_list))
        print("Total number of actors: ", len(global_actor_list))

        for i in range(len(sources)):
            source = sources[i]
            new_causers_list = []
            for causer, polarity in causer_details[source]:  # Remove actors from list of causers
                if causer not in global_actor_list:
                    new_causers_list.append((causer, polarity))

            # Find source level strength of actors and causers
            source_actor_strength = collections.Counter([actor for actor, polarity in actors_details[source]])
            source_causers_strength = collections.Counter([actor for actor, polarity in new_causers_list])

            # Choose top N actors and causers per source
            top_actors = [actor for actor, count in source_actor_strength.most_common(N_entity)]
            top_causers = [causer for causer, count in source_causers_strength.most_common(N_entity)]

            actor_df = pd.DataFrame(
                [[actor, polarity] for (actor, polarity) in actors_details[source] if actor in top_actors],
                columns=['actor', 'polarity'])
            causer_df = pd.DataFrame(
                [[causer, polarity] for (causer, polarity) in new_causers_list if causer in top_causers],
                columns=['target', 'polarity'])

            srl = getRowsWithIssue(srl_data[source], issue)  # Find SRL data which contain an issue
            avg_polarity = actor_df.groupby(['actor'])['polarity'].mean()
            source_actor_list = avg_polarity.index.values.tolist()
            actor_polarity_score = avg_polarity.tolist()
            colors = getEntityColorListCauserOnly(source_actor_list, global_causer_list)
            edge_labels = getSRLVerbs(source_actor_list, srl)

            # Add actor nodes
            network, actor_nodes, next_node_id = addActorNodes(source_actor_list, i, next_node_id, network,
                                                               actor_nodes, global_actor_counter,
                                                               source_actor_strength, actor_polarity_score,
                                                               colors, edge_labels)

            source_causer_list = causer_df['target'].unique()
            colors = ["black"] * len(source_causer_list)
            edge_labels = getSRLVerbs(source_causer_list, srl)

            # Add causer nodes
            network, causer_nodes, next_node_id = addArgumentNodes(source_causer_list, i,
                                                                   next_node_id,
                                                                   network,
                                                                   causer_nodes,
                                                                   actor_nodes,
                                                                   global_causer_counter,
                                                                   source_causers_strength,
                                                                   colors, edge_labels)

        try:
            postfix = ""
            network.write_html(graphs_path + "Issue-Actor/" + issue + postfix + ".html")
        except:
            pass

def generateTopicIssueGraph(root_dir, sources):
    """
    Generates Topic-Issue graph
    @param root_dir: data directory
    @param sources: list of sources
    """
    issue_path = File.getIssueDir(root_dir)
    topic_path = File.getTopicDir(root_dir)
    graphs_path = File.getGraphDir(root_dir)

    for source in sources:
        topic_names = Topic.getTopicLabels(topic_path + "Label/", source)  # Read topic labels
        K = len(topic_names)

        analysis_network = Network()
        issue_nodes = {}
        existing_colors = []

        # Add topic nodes to the network
        analysis_network, existing_colors = addTopicNodes(analysis_network, topic_names, existing_colors)
        next_node_id = K

        issue_df = pd.read_csv(issue_path + source + "-ngram-issues.csv")  # Read all the ngram issue labels
        issue_popularity = issue_df.groupby("Ngram")['Popularity'].sum()  # Find popularity across all the topics

        for topic_id in range(0, len(topic_names)):

            if not Topic.isTopicValid(topic_names[topic_id]): # Filter invalid topics
                continue

            issues = issue_df.loc[issue_df['# Topic-ID'] == topic_id]
            issues_list = issues['Ngram'].tolist()
            topic_issue_strength = dict(zip(issues_list, issues['Popularity'].tolist()))


            # Add issue nodes to the network
            analysis_network, issue_nodes, next_node_id, existing_colors = addIssueNodes(issues_list,
                                                                                         topic_id,
                                                                                         next_node_id,
                                                                                         analysis_network,
                                                                                         issue_nodes,
                                                                                         existing_colors,
                                                                                         issue_popularity,
                                                                                         topic_issue_strength)
            print("Total number of issues: ", len(issues_list))
            print("Issues: ", issues_list)

        try:
            analysis_network.write_html(graphs_path + "Topic-Issue/" + source + "-Topics-Issues.html")
        except:
            pass