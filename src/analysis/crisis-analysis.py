import src.util.utils as Util
import src.util.ESR as ESR
import src.util.graph as Graph
import src.util.issues as Issue

config = Util.readAllConfig()
analysis_config = config["analysis"]
root_dir = config["data-directory"]
MAINSTREAM_SOURCES = config["mainstream-sources"]
SOURCES = Util.getSources(MAINSTREAM_SOURCES)

# Extracts actors, sentiments and roles. This line can be commented if the extraction is not required.
ESR.extractActorsSentimentAndRoles(root_dir, MAINSTREAM_SOURCES)

# Generate Ngram issues and topic label issues. Each line can be commented separately to exclude certain type of issue
Issue.idenfiyIssuesFromTopicLabels(root_dir, SOURCES, analysis_config["polarization-threshold"])
Issue.idetifyIssuesFromNgram(root_dir, SOURCES, analysis_config["polarization-threshold"])

# Generate visualization. Each line can be commented separately to exclude certain visualization
Graph.generateTopicIssueGraph(root_dir, SOURCES)
Graph.generateTopicEntityGraph(root_dir, SOURCES)
Graph.generateSourceIssueGraph(root_dir, SOURCES, analysis_config["visualization"])
Graph.generateIssueActorGraph(root_dir, SOURCES, analysis_config["visualization"])
