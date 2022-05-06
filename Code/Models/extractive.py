import os, sys, argparse
import nltk
import pandas as pd
from rouge import Rouge
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

nltk.download("punkt")
rouge = Rouge()

sys.setrecursionlimit(10000)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--text_column", type=str)
    parser.add_argument("--summary_column", type=str)
    parser.add_argument("--sentence_count", type=int, default=3)

    args, _ = parser.parse_known_args()
    
    def summarize(text, sumarizer, SENTENCES_COUNT):
        sentences_ = []
        doc = text
        doc_ = PlaintextParser(doc, Tokenizer("en")).document
        for sentence in sumarizer(doc_, SENTENCES_COUNT):
            sentences_.append(str(sentence))

        summm_ = " ".join(sentences_)
        return summm_

    def RougeScore(ModelScore, ModelSummary):

        standard_summary = df[args.summary_column]
        ModelScore_ = rouge.get_scores(ModelSummary, standard_summary, avg=True)
        ModelDF = pd.DataFrame(ModelScore_).set_index(
            [["recall", "precision", "f-measure"]]
        )
        return ModelDF
    
    
    df = pd.read_csv(args.data_file)

    df["LexRankSummary"] = df[args.text_column].map(lambda x: summarize(x, LexRankSummarizer(), args.sentence_count))
    df["KLSummary"] = df[args.text_column].map(lambda x: summarize(x, KLSummarizer(), args.sentence_count))
    df["TextRankSummary"] = df[args.text_column].map(lambda x: summarize(x, TextRankSummarizer(),args.sentence_count)) 
    df["SumBasicSummary"] = df[args.text_column].map(lambda x: summarize(x, SumBasicSummarizer(), args.sentence_count))
    df["LsaSummary"] = df[args.text_column].map(lambda x: summarize(x, LsaSummarizer(), args.sentence_count))


    LexRouge = RougeScore("LexRouge", df["LexRankSummary"])
    TextRankRouge = RougeScore("TextRankRouge", df["TextRankSummary"])
    SumBasicRouge = RougeScore("SumBasicRouge", df["SumBasicSummary"])
    LsaRouge = RougeScore("LsaRouge", df["LsaSummary"])
    KLRouge = RougeScore("KLRouge", df["KLSummary"])

    os.makedirs(args.output_dir)
    os.chdir(args.output_dir)
    

    TextRankRouge.to_csv("TextRankRouge.csv", header=True, index=True)
    LexRouge.to_csv("LexRouge.csv", header=True, index=True)
    SumBasicRouge.to_csv("SumBasicRouge.csv", header=True, index=True)
    LsaRouge.to_csv("LsaRouge.csv", header=True, index=True)
    KLRouge.to_csv("KLRouge.csv", header=True, index=True)

    df[[args.summary_column,'LexRankSummary','KLSummary','TextRankSummary','SumBasicSummary','LsaSummary']].to_csv( \
                                                                    "Extractiveprediction.csv", index=False, header=True)    
