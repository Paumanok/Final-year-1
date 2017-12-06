from os import listdir
from os.path import isfile, join, splitext
import pandas as pd


class processData():

    @staticmethod
    def loadRaw( dirList, processunsup = False):
        #todo: add functionality to point to data dir
        #      checkfor/create output file
        #      add tsv header

        #changed to not include parentdir
        positiveFiles = [f for f in listdir(join(dirList, 'pos/')) if isfile(join(dirList,'pos/',f))]

        negativeFiles = [ f for f in listdir(join(dirList, 'neg/')) if isfile(join(dirList,'neg/', f))]


        header = "ids \t sentiment \t review"

        positiveReviews = open("positiveReviews.tsv", "w")
        positiveReviews.write(header)
        for pf in positiveFiles:
            with open(join(dirList, 'pos/',pf), "r", encoding='utf-8') as f:
                review = (splitext(pf)[0][1:] + '\t' + '1\t' + f.read() + '\n') #construct in tsv form
                positiveReviews.write(review) #write each review ddto one larger review file
                #line=f.readline()
                #counter = len(line.split())
                #numWords.append(counter)
        print('Positive files finished')


        negativeReviews = open("negativeReviews.tsv", "w")
        negativeReviews.write(header)
        for nf in negativeFiles:
            with open(join(dirList, 'neg/',nf), "r", encoding='utf-8') as f:
                review = (splitext(nf)[0][1:] + '\t' + '1\t' + f.read() + '\n') #construct in tsv form
                negativeReviews.write(review) #write each review ddto one larger review file
                #line=f.readline()
                #counter = len(line.split())
                #numWords.append(counter)
        print('Negative files finished')

        if(processunsup):
            unlabeledFiles = [ f for f in listdir(join(dirList, 'unsup/')) if isfile(join(dirList,'unsup/', f))]
            unlabeledReviews = open("unlabeledReviews.tsv", "w")
            unlabeledReviews.write("id \t review")
            for uf in unlabeledFiles:
                with open(join(dirList,'unsup/',uf), "r", encoding='utf-8') as f:
                    review = (splitext(uf)[0][1:0] + '\t' + f.read() + '\n')
                    unlabeledReviews.write(review)
            print('Unlabeled files finished')

            #return fps of the reviews in tsv form
            return positiveReviews, negativeReviews, unlabeledReviews
        else:
            return positiveReviews, negativeReviews

    @staticmethod
    def importTSV(fp):
        reviews = pd.read_csv(os.path.join(fp, 'data', 'labeledTrainData.tsv'), header=0, \
                        delimiter="\t", quoting=3)
        return reviews

    @staticmethod
    def cleanData(reviews, removeStopWords=True):
        nltk.download()  # Download text data sets, including stop words
        for i in xrange( 0, len(reviews["review"])):
            clean_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], removeStopWords)))

        return clean_reviews
