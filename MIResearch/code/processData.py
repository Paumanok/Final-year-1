from os import listdir
from os.path import isfile, join, splitext
import pandas as pd


def loadRaw():
    #todo: add functionality to point to data dir
    #      checkfor/create output file
    #      add tsv header

    #changed to not include parentdir
    positiveFiles = [f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/',f))]

    negativeFiles = [ f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]

    unlabeledFiles = [ f for f in listdir('unlabeledReviews/') if isfile(join('unlabeledReviews/', f))]

    header = "ids \t sentiment \t review"


    for pf in positiveFiles:
        with open(pf, "r", encoding='utf-8') as f:
            review = (splitext(f.name)[1][1:] + '\t' + '1\t' + f.read + '\n') #construct in tsv form
            positiveReviews.write(review) #write each review ddto one larger review file
            #line=f.readline()
            #counter = len(line.split())
            #numWords.append(counter)
    print('Positive files finished')

    for nf in negativeFiles:
        with open(nf, "r", encoding='utf-8') as f:
            review = (splitext(f.name)[1][1:] + '\t' + '1\t' + f.read + '\n') #construct in tsv form
            negativeReviews.write(review) #write each review ddto one larger review file
            #line=f.readline()
            #counter = len(line.split())
            #numWords.append(counter)
    print('Negative files finished')

    for uf in unlabledFiles:
        with open(uf, "r", encoding='utf-8') as f:
            review = (splitext(f.name)[1][1:0] + '\t' + f.read + '\n')
            unlabledReviews.write(review)
    print('Unlabeled files finished')

    #return fps of the reviews in tsv form
    return postiveReviews, negativeReviews, unlabeledReviews


def importTSV(fp):
    reviews = pd.read_csv(os.path.join(fp, 'data', 'labeledTrainData.tsv'), header=0, \
                    delimiter="\t", quoting=3)
    return reviews

def cleanData(reviews, removeStopWords=True):
    #nltk.download()  # Download text data sets, including stop words
    for i in xrange( 0, len(reviews["review"])):
        clean_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], removeStopWords)))

    return clean_reviews
