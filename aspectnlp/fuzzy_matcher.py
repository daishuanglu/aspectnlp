# standard libraries
import os

# Third-party libraries
from sklearn.metrics.pairwise import cosine_similarity

# project code
from fasttext import load_model
from aspect.w2v import  *

class semanticMatcher():
    def __init__(self):
        import aspect
        rmls_nlp_dir = os.path.dirname(aspect.__file__)
        self.ftmodel = load_model(os.path.join(rmls_nlp_dir,"absa_embedding","custom_emb.vec.bin"))


    def token_similarity(self, tokens,target_list):
        # Input:
        #   sentences1: list of string - a list of sentences
        #   sentences2: list of string - a list of another sentences
        # output:
        #   cosine_scores: list of float numbers - cosine similarity.

        embeddings1 =  np.stack( load_aspect_embedding_from_fasttext(tokens, self.ftmodel))
        embeddings2 = np.stack( load_aspect_embedding_from_fasttext(target_list, self.ftmodel))
        cosine_scores = cosine_similarity(embeddings1, embeddings2)
        return cosine_scores


    def token_matcher(self, tokens, target_list, threshold=0.5):
        # Input:
        #   targets: list of string - list of phrases to be matched.
        #   tokens: list of string - list of tokenized words.
        # Output:
        #   matched: list of list[(string,float)]
        #       - list of matched phrases for each token and its similarity score, empty if no matches.

        sim = self.token_similarity(tokens,target_list)

        matched = []
        for irow in range(sim.shape[0]):
            ind, = np.where(sim[irow, :] > threshold)
            matched.append([{'label': target_list[icol], 'score': sim[irow, icol]} for icol in ind])
        return matched