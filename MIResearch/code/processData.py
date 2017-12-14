from os import listdir
from os.path import isfile, join, splitext
import pandas as pd
#from nltk.corpus import stopword
import nltk.data
from gensim.models import Word2Vec
import gensim
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import hashlib

class processData():

    @staticmethod
    def loadRaw( dirList, processunsup = False):
        #todo: add functionality to point to data dir
        #      checkfor/create output file
        #      add tsv header

        #changed to not include parentdir
        positiveFiles = [f for f in listdir(join(dirList, 'pos/')) if isfile(join(dirList,'pos/',f))]

        negativeFiles = [ f for f in listdir(join(dirList, 'neg/')) if isfile(join(dirList,'neg/', f))]


        header = "id\tsentiment\treview\n"

        positiveReviews = open("positiveReviews.tsv", "w")
        positiveReviews.write(header)
        for pf in positiveFiles:
            with open(join(dirList, 'pos/',pf), "r", encoding='utf-8') as f:
                review = (splitext(pf)[0][1:] + '\t' + '1\t' + f.read().replace('\t', ' ') + '\n') #construct in tsv form
                positiveReviews.write(review) #write each review ddto one larger review file
                #line=f.readline()
                #counter = len(line.split())
                #numWords.append(counter)
        print('Positive files finished')


        negativeReviews = open("negativeReviews.tsv", "w")
        negativeReviews.write(header)
        for nf in negativeFiles:
            with open(join(dirList, 'neg/',nf), "r", encoding='utf-8') as f:
                review = (splitext(nf)[0][1:] + '\t' + '0\t' + f.read().replace('\t', ' ') + '\n') #construct in tsv form
                negativeReviews.write(review) #write each review ddto one larger review file
                #line=f.readline()
                #counter = len(line.split())
                #numWords.append(counter)
        print('Negative files finished')

        if(processunsup):
            unlabeledFiles = [ f for f in listdir(join(dirList, 'unsup/')) if isfile(join(dirList,'unsup/', f))]
            unlabeledReviews = open("unlabeledReviews.tsv", "w")
            unlabeledReviews.write("id\treview\n")
            for uf in unlabeledFiles:
                with open(join(dirList,'unsup/',uf), "r", encoding='utf-8') as f:
                    review = (splitext(uf)[0][1:0] + '\t' + f.read().replace('\t', ' ') + '\n')
                    unlabeledReviews.write(review)
            print('Unlabeled files finished')

            #return fps of the reviews in tsv form
            return positiveReviews, negativeReviews, unlabeledReviews
        else:
            return positiveReviews, negativeReviews

    @staticmethod
    def importTSV(fileName):
        print(fileName)
        reviews = pd.read_csv(fileName, header=0, delimiter="\t", quoting=3, error_bad_lines=False)
        return reviews

    @staticmethod
    def cleanData(reviews, removeStopWords=False):
        #nltk.download()  # Download text data sets, including stop words
        clean_reviews = []
        for i in range( 0, len(reviews["review"])):
            cleaned_review = " ".join(KaggleWord2VecUtility.review_to_wordlist(reviews["review"][i], removeStopWords))
            clean_reviews.append(cleaned_review)
            reviews.loc[i, 'review'] = cleaned_review

        return clean_reviews, reviews

    @staticmethod
    def cleanDataSent(reviews, removeStopWords=False):
        clean_reviews = []
        for i in range( 0, len(reviews["review"])):
            cleaned_review = " ".join(KaggleWord2VecUtility.review_to_sentences(reviews.iloc[i]["review"], removeStopWords))
            clean_reviews.append(cleaned_review)
            reviews.loc[i, 'review'] = cleaned_review

        return clean_reviews, reviews

    @staticmethod
    def GetCleanReviews(reviews, stopwords = False):
        clean_reviews = []
        for review in reviews["review"]:
            clean_reviews.append( " ".join(KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=stopwords )))
        return clean_reviews

    @staticmethod
    def genTaggedDoc(reviews, stopwords=False):
        for review in reviews["review"]:
            if "sentiment" in reviews:
                yield gensim.models.doc2vec.TaggedDocument(KaggleWord2VecUtility.review_to_wordlist(review, stopwords), [1])
            else:
                yield KaggleWord2VecUtility.review_to_wordlist(review, stopwords)

    @staticmethod
    def hashh(text):
        return hashlib.md5(text).hexdigest()

