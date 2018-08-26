import argparse
import glob
import os
import re
import math
from collections import Counter
from itertools import takewhile


def mkdir(output):
    """
    Make directory if does not already exist.
    :param      output:
    :return:    True if no directory exists, and 'output' was made; else, False.
    """
    if not os.path.exists(output):
        os.makedirs(output)
        return True
    return False

def is_file(path):
    """Wrapper for os.path.is_file"""
    return os.path.isfile(str(path))


def is_dir(path):
    """Wrapper for os.path.isdir"""
    return os.path.isdir(str(path))


def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)

def file_base(filename):
    """Return c for filename /a/b/c.ext"""
    (head, tail) = os.path.split(filename)
    (base, ext) = os.path.splitext(tail)
    return base

def clean_text(s):
    """ Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana' """

    s = re.sub("[^a-z A-Z]", "", s)
    s = s.replace(' n ', ' ')
    return s.lower()

def clean_corpus(corpus):
    """ Run clean_text() on each sonnet in the corpus

    :param corpus:  corpus dict with keys set as filenames and contents as a single string of the respective sonnet.
    :type corpus:   dict

    :return     corpus with text cleaned and tokenized. Still a dictionary with keys being file names, but contents
                now the cleaned, tokenized content.
    """
    for key in corpus.keys():
        # clean each exemplar (i.e., sonnet) in corpus
        print("Cleaning", key)
        # call function provided to clean text of all non-alphabetical characters and tokenize by " " via split()
        corpus[key] = clean_text(corpus[key]).split()

    return corpus

def read_sonnets(fin):
    """
    Passes image through network, returning output of specified layer.

    :param fin: fin can be a directory path containing TXT files to process or to a single file,

    :return: (dict) Contents of sonnets with filename (i.e., sonnet ID) as the keys and cleaned text as the values.
    """

    """ reads and cleans list of text files, which are sonnets in this assignment"""

    if is_file(fin):
        f_sonnets = [fin]
    elif is_dir(fin):
        f_sonnets = glob.glob(fin + os.sep + "*.txt")
    else:
        print('Filepath of sonnet not found!')
        return None

    sonnets = {}
    for f in f_sonnets:
        sonnet_id = file_base(f)
        data=[]
        with open(f, 'r') as file:
            data.append(file.readline().replace('\\n', '').replace('\\r', ''))

        sonnets[sonnet_id] = clean_text("".join(data))
    return sonnets

def sort_list(list1):
    """sorts list of tuples in descedning order by the the 2nd elements in each tuple"""
    result_list = sorted(list1, key=lambda x: x[1], reverse=True)
    return result_list

def get_top_k(kv_list, k=20):
    """
    :param kv_list:    list of key-value tuple pairs, where value is some score or count.
    :param k:          number of key-value pairs with top 'k' values (default k=20)
    :return:           k items from kv_list with top scores
    """
    print (kv_list[:k])


def tf(sonnet):
    """Determine the term frequency of text exemplar (i.e, document)

    :return     (tuple) as (word, frequency) of length = number unique words in document.
    """
    doc = read_sonnets(sonnet).items()
    clean_doc = str(doc)
    clean_doc = clean_text(clean_doc)
    clean_doc = clean_doc.split()

    y = Counter(clean_doc)  # counts the frequencty of each word
    x = list(takewhile(lambda x: x[-1] > 0, y.most_common()))  # converts counter to a list of tuples and sorts
    return x



def tf_corpus(corp):
    doc = read_sonnets(corp).items()
    clean_doc = str(doc)
    clean_doc = clean_text(clean_doc)
    clean_doc = clean_doc.split()
    y=Counter(clean_doc) # counts the frequencty of each word
    x=list(takewhile(lambda x: x[-1] > 0, y.most_common())) # converts counter to a list of tuples and sorts
    return x

def idf(corpus):
    """Determine the inverted document frequency of a corpus.

    :return     (tuple) as (word, frequency) of length = number unique words in document.
    """
    list = os.listdir(r"C:\Users\Connor\Desktop\EECE\python\code 2\data\text\shakespeare_sonnets")
    number_files = len(list) #returns the number of sonnets in the directory
    freq_list=[]
    word_list= [item[0]for item in tf_corpus(corpus)] #returns the line of text of each sonnet
    sonnet_list=read_sonnets(corpus).items() #breaks each sonnet into individual words
    for word in word_list:                   # lines 144-150 create a list of all words and their frequencies
        counter=0
        for sonnets in sonnet_list:
            for sonnet_words in sonnets:
                if word in sonnet_words:
                    counter=counter+1
                    freq_list.append((word,counter))
    result1 = sorted(freq_list, key=lambda x: x[1], reverse=True) #sorts in descending order
    seen = set()                                                 #lines 152-157 create a new list with only one copy of each word
    out = []
    for a, b in result1:
        if a not in seen:
            out.append((a, b))
            seen.add(a)
    desired_list=[(a,math.log((number_files/b),10)) for a,b in out] #returns a list of the words and their idf
    desired_list=sort_list(desired_list)
    return (desired_list)



def tf_idf(document,corpus):
    """Determine the inverted document frequency of a corpus.

    :return     (tuple) as (word, frequency) of length = number unique words in document.
    """
    x=tf(document)
    y=idf(corpus)
    z=[]
    for a in x:  #lines 201-204 return a list of each word and its corresponding tf_idf
        for b in y:
            if a[0] == b[0]:
                z.append((a[0], a[1] * b[1]))
    z = sorted(z, key=lambda x: x[1], reverse=True) #sorts in descending order
    return z



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text Analysis through TFIDF computation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i','--input', type=str, default=r"C:\Users\Connor\Desktop\EECE\python\code 2\data\text\shakespeare_sonnets\1.txt", help='Input text file or files.')
    parser.add_argument('-c', '--corpus', type=str, default='results/assignment1/', help='Directory containing document collection (i.e., corpus)')
    parser.add_argument("--tfidf", help="Determine the TF IDF of a document w.r.t. a given corpus", action="store_true")

    args = parser.parse_args()

    # mkdir(args.output)
    sonnets = read_sonnets(args.input)




get_top_k(tf(r"C:\Users\Connor\Desktop\EECE\python\code 2\data\text\shakespeare_sonnets\1.txt"))

get_top_k(tf_corpus(r"C:\Users\Connor\Desktop\EECE\python\code 2\data\text\shakespeare_sonnets"))

get_top_k(idf(r"C:\Users\Connor\Desktop\EECE\python\code 2\data\text\shakespeare_sonnets"))

get_top_k(tf_idf(r"C:\Users\Connor\Desktop\EECE\python\code 2\data\text\shakespeare_sonnets\1.txt",
             r"C:\Users\Connor\Desktop\EECE\python\code 2\data\text\shakespeare_sonnets"))













