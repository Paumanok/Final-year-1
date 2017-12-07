from processData import processData as p
from learning import learning as l
import os
#import gensim.models.keyedvectors as word2vec
from gensim.models import Word2Vec, Doc2Vec
import gensim
import pandas as pd

REMOVE_STOP_WORDS = True

opd = [ "train_pos.tsv",
        "train_neg.tsv",
        "train_unlab.tsv",
        "test_pos.tsv",
        "test_neg.tsv"]


def process():

    #fn = os.path.join(os.path.dirname(__file__), 'my_file')

    train = "data/aclImdb/train/"
    test = "data/aclImdb/test/"
    #outputdirs


    #process training data 0.pos 1.neg, 2.unsup
    trpfp, trnfp, trufp = p.loadRaw(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", train), True)

    os.rename(trpfp.name,opd[0])
    os.rename(trnfp.name,opd[1])
    os.rename(trufp.name,opd[2])

    #process test data 0.pos, 1.neg
    tp, tn = p.loadRaw(os.path.join(os.path.dirname(__file__),"..",  test))

    os.rename(tp.name,opd[3])
    os.rename(tn.name,opd[4])

    filePointers = [trpfp, trnfp, trufp, tp, tn ]
    return filePointers, opd

def clean(fnList):
    print("gonna clean!")

    reviews = {}


    for fn in fnList:
        review = p.importTSV(fn)
        clean_review, clean_review_dataframe  = p.cleanData(review, REMOVE_STOP_WORDS)
        clean_review_dataframe.to_csv(("clean_" + fn), sep='\t', index=False)
        file_name = os.path.splitext(fn)[0]
        reviews.update({file_name : clean_review})

    print("cleaned!")

    return reviews

def trainW2V():
    training_sets = opd[0:3]
    print("importing clean data")
    ults = p.importTSV("data/" + opd[2])
    lpts = p.importTSV("data/" + opd[0])
    lnts = p.importTSV("data/" + opd[1])
    print("building model")
    model = l.word2vec(ults, lpts, lnts)
    #check if model worked
    model.most_similar("man")

def trainD2V():
    training_sets = opd[0:3]
    ults = p.importTSV("data/" + opd[0])
    lpts = p.importTSV("data/" + opd[1])
    lnts = p.importTSV("data/" + opd[2])

    model = l.doc2vec(ults, lpts, lnts)
    #check if model worked
    model.most_similar("man")


def forest_test():
    lptrs = p.importTSV("data/" + opd[0])
    lntrs = p.importTSV("data/" + opd[1])
    lpts = p.importTSV("data/" + opd[3])
    lnts = p.importTSV("data/" + opd[4])
    model = Word2Vec.load('300features_40minwords_10context')
    #word2vec.KeyedVectors.load_word2vec_format('300features_40minwords_10context', binary=True)
    l.randomForestvec(model,lptrs.append(lntrs),lpts.append(lnts), 300)
    print(lptrs.append(lntrs))

def main():
    #filePointers = process()
    #reviews = clean(opd)
    #trainW2V()
    forest_test()

main()
