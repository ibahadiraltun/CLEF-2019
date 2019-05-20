# CLEF2019
CLEF 2019 Submissions
1. Team ID: TOBB ETU
2. Team affiliation: TOBB Economy and Technology University
3. Contact email: mucahidkutlu@gmail.com
4. System specs
- 4.1 Core approach
    We have used a hybrid approach in which we use MART algorithm with 421 linguistic based features. Next, we apply a rule-based approach to decrease the ranks of sentences with phrases like “thank you”. We set the number of trees to 50 and the number of leaves to 2.
- 4.3 Important/interesting/novel novel representations used (features, embeddings...)
    Primary: Our features include:
        1) the existence of named entities identified by Stanford NLP tool [1].
        2) the topic of sentences identified by IBM Watson’s NLP tool [2].
        3) POS tags, bi-grams that appear at least 50 times in only check-worthy sentences or only not-check-worthy sentences in the training set.
        4) whether the sentence is a question sentence or not.
    Contrastive 1: In addition to features used in our primary method, we also used the speaker of the statements as features.
    Contrastive 2: We used the same features as Contrastive 1 but we used logistic regression instead of MART.
- 4.4 Important/interesting/novel tools used
    We used IBM-Watson, Stanford-NLP tool for feature extraction and Ranklib and sci-kit libraries for MART and logistic regression models, respectively.

- 4.5 Significant data pre/post-processing
    We have not applied any preprocessing step.

- 4.6 Other data used (outside of the provided)
    We have not used any external data. We have just used the provided training data.

5 References (if applicable)
    https://nlp.stanford.edu/
    https://www.ibm.com/watson/services/natural-language-understanding/
    https://sourceforge.net/p/lemur/wiki/RankLib/
