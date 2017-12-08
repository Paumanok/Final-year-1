from processData import processData as p
from learning import learning as l
import os
#import gensim.models.keyedvectors as word2vec
from gensim.models import Word2Vec, Doc2Vec
import gensim
import pandas as pd

REMOVE_STOP_WORDS = False

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
        review = p.importTSV("data/" +fn)
        clean_review, clean_review_dataframe  = p.cleanData(review, REMOVE_STOP_WORDS)
        clean_review_dataframe.to_csv(("wsw_clean_" + fn), sep='\t', index=False)
        file_name = os.path.splitext(fn)[0]
        reviews.update({file_name : clean_review})

    print("cleaned!")

    return reviews

def trainW2V(stopwords):
    training_sets = opd[0:3]
    if stopwords:
        ver = "wsw_"
    else:
        ver = ""
    print("importing clean data")
    ults = p.importTSV("data/" + ver + "clean_" + opd[2])
    lpts = p.importTSV("data/" + ver + "clean_" + opd[0])
    lnts = p.importTSV("data/" + ver + "clean_" + opd[1])
    print("building w2v model...")
    model = l.word2vec(ults, lpts, lnts)
    #check if model worked

def trainD2V(stopwords):
    training_sets = opd[0:3]
    if stopwords:
        ver = "wsw_"
    else:
        ver = ""
    ults = p.importTSV("data/" + ver + "clean_" + opd[2])
    lpts = p.importTSV("data/" + ver + "clean_" + opd[0])
    lnts = p.importTSV("data/" + ver + "clean_" + opd[1])
    print("building d2v model...")
    model = l.doc2vec(ults, lpts, lnts)
    #check if model worked


def forest_test(model,stopwords):
    if stopwords:
        ver = "wsw_"
    else:
        ver = ""
    lptrs = p.importTSV("data/" + ver + "clean_" + opd[0])
    lntrs = p.importTSV("data/" + ver + "clean_" + opd[1])
    lpts  = p.importTSV("data/" + ver + "clean_" + opd[3])
    lnts  = p.importTSV("data/" + ver + "clean_" + opd[4])
    model = Doc2Vec.load(model)
   # print(lptrs.append(lntrs))
    #word2vec.KeyedVectors.load_word2vec_format('300features_40minwords_10context', binary=True)
    l.randomForestvec(model,lptrs.append(lntrs),lpts.append(lnts), 300)

def main():
    #filePointers = process()
    #reviews = clean(opd)
    #trainW2V(True)
#    trainD2V(True)
# forest_test(300features_40minwords_10context)
    forest_test("300features40words4workers10context_pvec",True)

main()
