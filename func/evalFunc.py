import os
import numpy as np
import pickle as pkl
import evaluate
from rouge_score import rouge_scorer
import math
from sklearn.metrics import roc_curve, auc
from sentence_transformers import SentenceTransformer
from metric import *
from plot import *

USE_Roberta = False
USE_EXACT_MATCH = True
##### 导入ROUGE评估函数计算ROUGE-L指标
###### 导入roberta_large模型计算sentence similarity
rougeEvaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
if USE_Roberta:
    SenSimModel = SentenceTransformer('../data/weights/nli-roberta-large')


##### 打印结果信息, resultDict is a list of dict
def printInfo(resultDict):
    print(len(resultDict))
    for item in resultDict:
        for key in item.keys():
            print(key)
        exit()



#### 计算LLM模型输出answer的准确率
def getAcc(resultDict, file_name):
    correctCount = 0
    for item in resultDict:
        ansGT = item["answer"]
        generations = item["most_likely_generation"]
        # print("GT:", ansGT)
        # print("Generation:", generations)
        rougeScore = getRouge(rougeEvaluator, generations, ansGT)
        if "coqa" in file_name or "TruthfulQA" in file_name:
            additional_answers = item["additional_answers"]
            rougeScores = [getRouge(rougeEvaluator, generations, ansGT) for ansGT in additional_answers]
            rougeScore = max(rougeScore, max(rougeScores))
        if rougeScore>0.5:
            correctCount += 1
    print("Acc:", 1.0*correctCount/len(resultDict))



##### 计算皮尔逊相关系数
def getPCC(x, y):
    rho = np.corrcoef(np.array(x), np.array(y))
    return rho[0,1]



##### 计算度量指标的AUROC
def getAUROC(resultDict, file_name):
    Label = []
    Score = []
    Perplexity = []
    Energy = []
    LexicalSimilarity = []
    SentBertScore = []
    Entropy = []
    EigenIndicator = []
    EigenIndicatorOutput = []

    for item in resultDict:
        ansGT = item["answer"]
        generations = item["most_likely_generation"]
        # print("GT:", ansGT)
        # print("Generation:", generations)
        Perplexity.append(-item["perplexity"])
        Energy.append(-item["energy"])
        Entropy.append(-item["entropy"])
        LexicalSimilarity.append(item["lexical_similarity"])
        SentBertScore.append(-item["sent_bertscore"])
        EigenIndicator.append(-item["eigenIndicator"])
        EigenIndicatorOutput.append(-item["eigenIndicatorOutput"])


        if USE_Roberta:
            similarity = getSentenceSimilarity(generations, ansGT, SenSimModel)
            if "coqa" in file_name or "TruthfulQA" in file_name:
                additional_answers = item["additional_answers"]
                similarities = [getSentenceSimilarity(generations, ansGT, SenSimModel) for ansGT in additional_answers]
                similarity = max(similarity, max(similarities))
            if similarity>0.9:
                Label.append(1)
            else:
                Label.append(0)
            Score.append(similarity)
        elif USE_EXACT_MATCH:
            similarity = compute_exact_match(generations, ansGT)
            if "coqa" in file_name or "TruthfulQA" in file_name:
                additional_answers = item["additional_answers"]
                similarities = [compute_exact_match(generations, ansGT) for ansGT in additional_answers]
                similarity = max(similarity, max(similarities))
            if similarity==1:
                Label.append(1)
            else:
                Label.append(0)
            Score.append(similarity)
        else:
            rougeScore = getRouge(rougeEvaluator, generations, ansGT)
            if "coqa" in file_name or "TruthfulQA" in file_name:
                additional_answers = item["additional_answers"]
                rougeScores = [getRouge(rougeEvaluator, generations, ansGT) for ansGT in additional_answers]
                rougeScore = max(rougeScore, max(rougeScores))
            if rougeScore>0.5:
                Label.append(1)
            else:
                Label.append(0)
            Score.append(rougeScore)


######### 计算AUROC ###########
    fpr, tpr, thresholds = roc_curve(Label, Perplexity)
    AUROC = auc(fpr, tpr)
    # thresh_Perplexity = thresholds[np.argmax(tpr - fpr)]
    thresh_Perplexity = get_threshold(thresholds, tpr, fpr)
    print("AUROC-Perplexity:", AUROC)
    # print("thresh_Perplexity:", thresh_Perplexity)
    VisAUROC(tpr, fpr, AUROC, "Perplexity")

    fpr, tpr, thresholds = roc_curve(Label, Energy)
    AUROC = auc(fpr, tpr)
    # thresh_Energy = thresholds[np.argmax(tpr - fpr)]
    thresh_Energy = get_threshold(thresholds, tpr, fpr)
    print("AUROC-Energy:", AUROC)
    # print("thresh_Energy:", thresh_Energy)
    VisAUROC(tpr, fpr, AUROC, "Energy")


    fpr, tpr, thresholds = roc_curve(Label, Entropy)
    AUROC = auc(fpr, tpr)
    # thresh_Entropy = thresholds[np.argmax(tpr - fpr)]
    thresh_Entropy = get_threshold(thresholds, tpr, fpr)
    print("AUROC-Entropy:", AUROC)
    # print("thresh_Entropy:", thresh_Entropy)
    VisAUROC(tpr, fpr, AUROC, "NormalizedEntropy")

    fpr, tpr, thresholds = roc_curve(Label, LexicalSimilarity)
    AUROC = auc(fpr, tpr)
    # thresh_LexicalSim = thresholds[np.argmax(tpr - fpr)]
    thresh_LexicalSim = get_threshold(thresholds, tpr, fpr)
    print("AUROC-LexicalSim:", AUROC)
    # print("thresh_LexicalSim:", thresh_LexicalSim)
    VisAUROC(tpr, fpr, AUROC, "LexicalSim")

    fpr, tpr, thresholds = roc_curve(Label, SentBertScore)
    AUROC = auc(fpr, tpr)
    # thresh_SentBertScore = thresholds[np.argmax(tpr - fpr)]
    thresh_SentBertScore = get_threshold(thresholds, tpr, fpr)
    print("AUROC-SentBertScore:", AUROC)
    # print("thresh_SentBertScore:", thresh_SentBertScore)
    VisAUROC(tpr, fpr, AUROC, "SentBertScore")

    fpr, tpr, thresholds = roc_curve(Label, EigenIndicator)
    AUROC = auc(fpr, tpr)
    # thresh_EigenScore = thresholds[np.argmax(tpr - fpr)]
    thresh_EigenScore = get_threshold(thresholds, tpr, fpr)
    print("AUROC-EigenScore:", AUROC)
    # print("thresh_EigenScore:", thresh_EigenScore)
    VisAUROC(tpr, fpr, AUROC, "EigenScore", file_name.split("_")[1])

    fpr, tpr, thresholds = roc_curve(Label, EigenIndicatorOutput)
    AUROC = auc(fpr, tpr)
    # thresh_EigenScoreOutput = thresholds[np.argmax(tpr - fpr)]
    thresh_EigenScoreOutput = get_threshold(thresholds, tpr, fpr)
    print("AUROC-EigenScore-Output:", AUROC)
    # print("thresh_EigenScoreOutput:", thresh_EigenScoreOutput)
    VisAUROC(tpr, fpr, AUROC, "EigenScoreOutput", file_name.split("_")[1])


######## 计算皮尔逊相关系数 ###############
    rho_Perplexity = getPCC(Score, Perplexity)
    rho_Entropy = getPCC(Score, Entropy)
    rho_Energy = getPCC(Score, Energy)
    rho_LexicalSimilarity = getPCC(Score, LexicalSimilarity)
    rho_EigenIndicator = getPCC(Score, EigenIndicator)
    rho_EigenIndicatorOutput = getPCC(Score, EigenIndicatorOutput)
    print("rho_Perplexity:", rho_Perplexity)
    print("rho_Energy:", rho_Energy)
    print("rho_Entropy:", rho_Entropy)
    print("rho_LexicalSimilarity:", rho_LexicalSimilarity)
    print("rho_EigenScore:", rho_EigenIndicator)
    print("rho_EigenScoreOutput:", rho_EigenIndicatorOutput)



######### 计算幻觉检测准确率(TruthfulQA)
    if "TruthfulQA" in file_name:
        acc = getTruthfulQAAccuracy(Label, Perplexity, thresh_Perplexity)
        print("TruthfulQA Perplexity Accuracy:", acc)
        acc = getTruthfulQAAccuracy(Label, Energy, thresh_Energy)
        print("TruthfulQA Energy Accuracy:", acc)
        acc = getTruthfulQAAccuracy(Label, Entropy, thresh_Entropy)
        print("TruthfulQA Entropy Accuracy:", acc)
        acc = getTruthfulQAAccuracy(Label, LexicalSimilarity, thresh_LexicalSim)
        print("TruthfulQA LexicalSimilarity Accuracy:", acc)
        acc = getTruthfulQAAccuracy(Label, SentBertScore, thresh_SentBertScore)
        print("TruthfulQA SentBertScore Accuracy:", acc)
        acc = getTruthfulQAAccuracy(Label, EigenIndicator, thresh_EigenScore)
        print("TruthfulQA EigenIndicator Accuracy:", acc)
        acc = getTruthfulQAAccuracy(Label, EigenIndicatorOutput, thresh_EigenScoreOutput)
        print("TruthfulQA EigenIndicatorOutput Accuracy:", acc)



# 查找最佳阈值
def get_threshold(thresholds, tpr, fpr):
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    return thresholdOpt



def getTruthfulQAAccuracy(Label, Score, thresh):
    count = 0
    for ind, item in enumerate(Score):
        if item>=thresh and Label[ind]==1:
            count+=1
        if item<thresh and Label[ind]==0:
            count+=1
    return count/len(Score)



def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


if __name__ == "__main__":
    file_name = "../data/output/llama-7b-hf_coqa_1/0.pkl"
    # file_name = "../data/output/llama-7b-hf_triviaqa_3/0.pkl"
    # file_name = "../data/output/llama-7b-hf_nq_open_1/0.pkl"
    # file_name = "../data/output/llama-7b-hf_SQuAD_1/0.pkl"

    # file_name = "../data/output/opt-6.7b_triviaqa_0/0.pkl"
    # file_name = "../data/output/opt-6.7b_nq_open_2/0.pkl"
    # file_name = "../data/output/opt-6.7b_coqa_0/0.pkl"
    # file_name = "../data/output/opt-6.7b_SQuAD_0/0.pkl"

    # file_name = "../data/output/llama-13b-hf_coqa_0/0.pkl"
    # file_name = "../data/output/llama-13b-hf_triviaqa_0/0.pkl"
    # file_name = "../data/output/llama-13b-hf_nq_open_3/0.pkl"
    # file_name = "../data/output/llama-13b-hf_SQuAD_0/0.pkl"

    # file_name = "../data/output/llama-7b-hf_TruthfulQA_7/0.pkl"

    # file_name = "../data/output/llama2-7b-hf_coqa_0/0.pkl"
    # file_name = "../data/output/llama2-7b-hf_nq_open_0/0.pkl"

    # file_name = "../data/output/falcon-7b_coqa_0/0.pkl"
    # file_name = "../data/output/falcon-7b_nq_open_0/0.pkl"


    f = open(file_name, "rb")
    resultDict = pkl.load(f)
    # printInfo(resultDict)
    getAcc(resultDict, file_name)
    getAUROC(resultDict, file_name)

