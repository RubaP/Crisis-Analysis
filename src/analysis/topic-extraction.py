import gc
import numpy as np
import src.util.utils as Util
import src.util.topic as Topic
import src.util.file as File

config = Util.readAllConfig()
analysis_config = config["analysis"]
root_dir = config["data-directory"]
MAINSTREAM_SOURCES = config["mainstream-sources"]
# Get all the sources
SOURCES = Util.getSources(MAINSTREAM_SOURCES)

phrase_path = File.getAutoPhraseDir(root_dir)
topic_path = File.getTopicDir(root_dir)
performance = []

for source in SOURCES:
    print("---------Processing - ", source, "------------")
    documents = File.readTextDocuments(phrase_path + source + "/" + "segmentation.txt")
    bag_of_words = Topic.getBagofWordsOfCorpus(documents)
    documents = None
    gc.collect()

    max_coherence = 0
    optimal_k = -1

    for k in range(10, 21):  # Range of topics searched
        print("---------- k:", k, "-----------------")

        for i in range(3):
            model = Topic.getTrainedModel(k, bag_of_words)

            for t in range(50):  # models are trained for 50 times, each with 10 iteration
                model.train(10)

                result = [source, k, t]

                coherence = Topic.getCoherence(model, 20)
                result.append(coherence)
                print("iteration :", t, " Coherence: ", coherence)

                if coherence > max_coherence:  #If the current model's coherence is the optimal value,
                    # then the current model and coherence score are saved as optimal setting
                    max_coherence = coherence
                    optimal_k = k
                    Topic.saveModel(model, topic_path, source)

                performance.append(result)
            model = None
            gc.collect()

    print("Best model: ")
    print("Max coherence - ", max_coherence)
    print("Optimal number of topics: ", optimal_k)

    performanceArray = np.asarray(performance)
    np.savetxt(topic_path + "Performance/" + "topic-modeling-performance-lda.csv", performanceArray,
               delimiter=",", fmt="%s", header="source,k,iteration,coherence")

Topic.printChatGPTQuery(root_dir, SOURCES)