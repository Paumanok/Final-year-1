import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
from gensim.models import Word2Vec, Doc2Vec
import gensim
import nltk.data
from processData import processData as p
import pandas as pd
from collections import namedtuple


class learning():
    @staticmethod
    def bagOfWords(clean_train_reviews):
        # Initialize the "CountVectorizer" object, which is scikit-learn's
        # bag of words tool.
        vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 1000)

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.
        reviews = []#clean_train_reviews["review"]

        for review in clean_train_reviews["review"]:
            reviews.append(review)
        train_data_features = vectorizer.fit_transform(reviews)
        # Numpy arrays are easy to work with, so convert the result to an
        # array
        np.asarray(train_data_features)
        print(train_data_features.A)
        return train_data_features, vectorizer

    @staticmethod
    def word2vec(unlabeledTrain, labeledTrainPos, labeledTrainNeg):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        sentences = []
        for review in unlabeledTrain["review"]:
            sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

        for review in labeledTrainPos["review"]:
            sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

        for review in labeledTrainNeg["review"]:
            sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

        # Set values for various parameters
        num_features = 300    # Word vector dimensionality
        min_word_count = 40   # Minimum word count
        num_workers = 4       # Number of threads to run in parallel
        context = 20          # Context window size
        downsampling = 1e-3   # Downsample setting for frequent words
        skipgram = 1
        # Initialize and train the model (this will take some time)
        print( "Training Word2Vec model...")
        assert gensim.models.word2vec.FAST_VERSION > -1

        model = Word2Vec(sentences, workers=num_workers, \
                    size=num_features, min_count = min_word_count, \
                    window = context, sample = downsampling, seed=1,sg = skipgram)

        # If you don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        model.init_sims(replace=True)

        # It can be helpful to create a meaningful model name and
        # save the model for later use. You can load it later using Word2Vec.load()

        if skipgram:
            skpg = "_sg"
        else:
            skpg = ""
        model_name = ("models/%dfeatures%dwords%dworkers%dcontext%s") % (num_features, min_word_count, num_workers, context, skpg)
        model.save(model_name)
        return model

    @staticmethod
    def doc2vec(unlabeledTrain, labeledTrainPos, labeledTrainNeg):

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        document = namedtuple('document', 'id words tags')
        docs = []
        for i in range(0, len(unlabeledTrain)):
            ids = unlabeledTrain.iloc[i]["id"]
            words = unlabeledTrain.iloc[i]["review"]
            if i == 1: print(words)
            words = words.split(" ")
            if i==1: print(words)
            tags = [i]
            docs.append(document(ids,words,tags))

        for i in range(0, len(labeledTrainPos)):
            ids = labeledTrainPos.iloc[i]["id"]
            words = labeledTrainPos.iloc[i]["review"]
            tags = [i]
            docs.append(document(ids,words,tags))

        for i in range(0, len(labeledTrainNeg)):
            ids = labeledTrainNeg.iloc[i][ "id"]
            words = labeledTrainNeg.iloc[i]["review"]
            tags = [i]
            docs.append(document(ids,words,tags))

        # Set values for various parameters
        num_features = 300    # Word vector dimensionality
        min_word_count = 40   # Minimum word count
        num_workers = 4       # Number of threads to run in parallel
        context = 20          # Context window size
        downsampling = 1e-3   # Downsample setting for frequent words
        skipgram = 1

        # Initialize and train the model (this will take some time)
        print("Training Doc2Vec model...")
        model = Doc2Vec(docs, workers=num_workers, \
                    size=num_features, min_count = min_word_count, \
                    window = context, sample = downsampling, seed=1, dm = skipgram)
        # If you don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        #model.build_vocab(docs)
        model.init_sims(replace=True)
        if skipgram:
            skpg = "_dm"
        else:
            skpg = "_dbow"
        # It can be helpful to create a meaningful model name and
        # save the model for later use. You can load it later using Word2Vec.load()
        model_name = ("models/%dfeatures%dwords%dworkers%dcontext_pvec%s") % (num_features, min_word_count, num_workers, context,skpg)
        model.save(model_name)
        return model

    @staticmethod
    def makeFeatureVec(words, model, num_features):
        # Function to average all of the word vectors in a given
        # paragraph
        #
        # Pre-initialize an empty numpy array (for speed)
        featureVec = np.zeros((num_features,),dtype="float32")
        #
        nwords = 0.
        #
        # Index2word is a list that contains the names of the words in
        # the model's vocabulary. Convert it to a set, for speed
        index2word_set = set(model.wv.index2word)
        #
        # Loop over each word in the review and, if it is in the model's
        # vocaublary, add its feature vector to the total
        for word in words:
            if word in index2word_set:
                nwords = nwords + 1.
                try:
                    featureVec = np.add(featureVec,model[word])
                    nwords = nwords + 1.
                except:
                    print("word not in model" + word)
        #

        # Divide the result by the number of words to get the average

        featureVec = np.divide(featureVec,nwords)
        return featureVec

    @staticmethod
    def getAvgFeatureVecs(reviews, model, num_features):
        # Given a set of reviews (each one a list of words), calculate
        # the average feature vector for each one and return a 2D numpy array
        #
        # Initialize a counter
        counter = 0.
        #
        # Preallocate a 2D numpy array, for speed
        reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
        #
        # Loop through the reviews
        for review in reviews:
           #
           # Print a status message every 1000th review
           if counter%1000. == 0. and not True:
               print(("Review %d of %d") % (counter, len(reviews)))
           #
           # Call the function (defined above) that makes average feature vectors
           reviewFeatureVecs[int(counter)] = learning.makeFeatureVec(review.split(' '), model, \
               num_features)
           #
           # Increment the counter
           counter = counter + 1.
        return reviewFeatureVecs


    @staticmethod
    def randomForestvec( trained_model, trainlabel,  testlabel, num_features, model_type = "w2v"):

        trainDataVecs = learning.getAvgFeatureVecs(trainlabel["review"], trained_model, num_features)

        testDataVecs = learning.getAvgFeatureVecs(testlabel["review"], trained_model, num_features)


        print("number of nan " + str(len(trainDataVecs[np.isnan(trainDataVecs)])))
        print(trainDataVecs[np.isnan(trainDataVecs)])
        trainDataVecs[np.isnan(trainDataVecs)] = np.median(trainDataVecs[~np.isnan(trainDataVecs)])

        # Initialize a Random Forest classifier with 100 trees
        forest = RandomForestClassifier(n_estimators = 100)

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        #print(len(trainlabel["sentiment"]))
        forest = forest.fit( trainDataVecs, trainlabel["sentiment"] )

        # Use the random forest to make sentiment label predictions
        print( "Predicting test labels...\n")
        result = forest.predict(testDataVecs)

        # Copy the results to a pandas dataframe with an "id" column and
        # a "sentiment" column
        output = pd.DataFrame( data={"id":testlabel["id"], "sentiment":result} )

        correct = 0
        incorrect = 0
        #print(output["sentiment"][1].item())
        testList = testlabel["sentiment"].tolist()
        predList = output["sentiment"].tolist()
        for i in range(0,len(testList)):
            if testList[i] == predList[i]:
                correct += 1
            else:
                incorrect +=1

        print(model_type +" accuracy: " + str(correct/(correct+incorrect)))


        # Use pandas to write the comma-separated output file
        #output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)

    @staticmethod
    def randomForestBow(trained_model,trainlabel, testlabel,vec):

        # Initialize a Random Forest classifier with 100 trees
        forest = RandomForestClassifier(n_estimators = 100)

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        print(trained_model.shape)
        forest = forest.fit( trained_model, trainlabel["sentiment"] )

        # Create an empty list and append the clean reviews one by one
        clean_test_reviews = []

        print("Cleaning and parsing the test set movie reviews...\n")
        #for i in range(0,len(test["review"])):
         #   clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(testlabel["review"][i], True)))

        for review in testlabel["review"]:
            clean_test_reviews.append(review)

        # Get a bag of words for the test set, and convert to a numpy array
        test_data_features = vec.transform(clean_test_reviews)
        np.asarray(test_data_features)

        # Use the random forest to make sentiment label predictions
        print( "Predicting test labels...\n")
        result = forest.predict(test_data_features)

        # Copy the results to a pandas dataframe with an "id" column and
        # a "sentiment" column
        output = pd.DataFrame( data={"id":testlabel["id"], "sentiment":result} )

        correct = 0
        incorrect = 0
        #print(output["sentiment"][1].item())
        testList = testlabel["sentiment"].tolist()
        predList = output["sentiment"].tolist()
        for i in range(0,len(testList)):
            if testList[i] == predList[i]:
                correct += 1
            else:
                incorrect +=1

        print("accuracy: " + str(correct/(correct+incorrect)))
        # Use pandas to write the comma-separated output file
        #output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)

    @staticmethod
    def rnn(trained_model):

        return 0

