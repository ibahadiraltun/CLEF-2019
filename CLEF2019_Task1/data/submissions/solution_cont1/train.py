import os

# os.system("javac Main.java")
# os.system("java Main")
# print("----")
# os.system("python sol_LR_3.py")

for i in range(0, 38):
    if i % 2 == 0: continue
    train_file = 'formatted_vectors_same/vec_{}.txt'.format(i)
    test_file = 'formatted_vectors_same/vec_{}.txt'.format(i - 1)
    rank_file = 'ranks_same_0/rank_{}.txt'.format(i - 1)
    score_file = 'scores_same_0/score_{}.txt'.format(i - 1)
    os.system("java -jar RankLib.jar \
            -train {} -gmax 1 -ranker 0 -tree 50 -leaf 2 -metric2t MAP \
            -save models_same_0/mymodel_{}.txt".format(train_file, i))
    os.system("java -jar RankLib.jar -load models_same_0/mymodel_{}.txt -rank {} -metric2T MAP -score {}".format(i, test_file, score_file))
    # os.system("java -jar RankLib.jar -load models_0/mymodel_{}.txt -indri tmp2.txt".format(i));
    