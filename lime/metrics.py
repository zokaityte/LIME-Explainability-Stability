# Computes the similarity of  items
def _jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

#Computes the similarity of items
def jaccard_similarities(list_of_lists_of_features):
    sim = []
    for l in list_of_lists_of_features:
        i_sim = []
        for j in list_of_lists_of_features:
            i_sim.append(_jaccard_similarity(l, j))
        sim.append(i_sim)
    return sim