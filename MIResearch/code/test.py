from processData import processData as p
import os

def process():

    #fn = os.path.join(os.path.dirname(__file__), 'my_file')

    train = "data/aclImdb/train/"


    test = "data/aclImdb/test/"


    #process training data 0.pos 1.neg, 2.unsup
    trpfp, trnfp, trufp = p.loadRaw(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", train), True)

    os.rename(trpfp.name, "train_pos.tsv")
    os.rename(trnfp.name, "train_neg.tsv")
    os.rename(trufp.name, "train_unlab.tsv")

    #process test data 0.pos, 1.neg
    tp, tn = p.loadRaw(os.path.join(os.path.dirname(__file__),"..",  train))

    os.rename(tp.name, "test_pos.tsv")
    os.rename(tn.name, "test_neg.tsv")



def clean():

    print("no cleaning yet!")
    return 0



def main():
    process()
    clean()

main()
