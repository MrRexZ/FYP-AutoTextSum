import nltk
import os
import csv
from settings import APP_ROOT
import summarization.glove.extractive_summary as ext_sum

data_dir = os.path.join(os.getcwd(), "input")
ref_dir = os.path.join(os.getcwd(), "models")
out_dir = os.path.join(os.getcwd(), "output")
list_of_ref_files = os.listdir(ref_dir)

def create_test_file(model):
    from numpy import max, min, median, average
    unigram_scores  = []
    bigram_scores   = []
    fourgram_scores = []
    with open(os.path.join('output', 'unigramscore.csv'), 'w+', newline='') as uniscore:
        with open(os.path.join('output', 'bigramscore.csv'), 'w+', newline='') as biscore:
            with open(os.path.join('output', 'bleuscore.csv'), 'w+', newline='') as bleufile:
                unigramwriter = csv.writer(uniscore, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                bigramwriter = csv.writer(biscore, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                fourgramwriter = csv.writer(bleufile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                for outer_file in os.listdir(data_dir):
                    for inner_file in os.listdir(os.path.join(data_dir, outer_file)):
                        with open(os.path.join(data_dir, outer_file, inner_file), "r", encoding="UTF-8") as data_file:
                            system_summary = ext_sum.get_ex_sum(model, data_file.read())[0]
                            hypothesis = nltk.word_tokenize(system_summary)
                            references = read_ref(inner_file[:-3])
                            fourgram_score = nltk.translate.bleu_score.sentence_bleu(references, hypothesis)
                            unigram_score = calculate_ngrams(hypothesis,references,1)
                            bigram_score = calculate_ngrams(hypothesis, references, 2)
                            unigram_scores.append(unigram_score)
                            bigram_scores.append(bigram_score)
                            fourgram_scores.append(fourgram_score)
                            unigramwriter.writerow([str(inner_file[:-3]), str(unigram_score)])
                            bigramwriter.writerow(
                                [str(inner_file[:-3]), str(bigram_score)])
                            fourgramwriter.writerow([str(inner_file[:-3]), str(fourgram_score)])

                unigramwriter.writerow(["Max", max(unigram_scores)])
                unigramwriter.writerow(["Min", min(unigram_scores)])
                unigramwriter.writerow(["Median", median(unigram_scores)])
                unigramwriter.writerow(["Average", average(unigram_scores)])

                bigramwriter.writerow(["Max", max(bigram_scores)])
                bigramwriter.writerow(["Min", min(bigram_scores)])
                bigramwriter.writerow(["Median", median(bigram_scores)])
                bigramwriter.writerow(["Average", average(bigram_scores)])

                fourgramwriter.writerow(["Max", max(fourgram_scores)])
                fourgramwriter.writerow(["Min", min(fourgram_scores)])
                fourgramwriter.writerow(["Median", median(fourgram_scores)])
                fourgramwriter.writerow(["Average", average(fourgram_scores)])
    print("Evaluation model successfull!")


def create_refs(filename):
    with open(filename, "r", encoding="UTF-8") as data_file:
        return [nltk.word_tokenize(data_file.read())]


def read_ref(target_filename):
    model_filenames = [filename for filename in list_of_ref_files if target_filename in filename]
    return [references for model_filename in model_filenames for references in
            create_refs(os.path.join(ref_dir, model_filename))]


def calculate_ngrams(hypothesis, references, n):
    return convert_to_float(nltk.translate.bleu_score.modified_precision(references, hypothesis, n))


def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac

if __name__ == '__main__':
    #Change to your own trained GloVe model if you'd like by changing the parameter below
    model = ext_sum.load_model("word2vec")
    create_test_file(model)
