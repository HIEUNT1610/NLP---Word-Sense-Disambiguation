#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nltk.corpus import wordnet # This might require "nltk.download('wordnet')" and "nltk.download('omw-1.4')".
import math
import random

from utils import *

class WSDClassifier(object):
    """
    Abstract class for WSD classifiers
    """

    def evaluate(self, instances):
        """
        Evaluates the classifier on a set of instances.
        Returns the accuracy of the classifier, i.e. the percentage of correct predictions.
        
        instances: list[WSDInstance]
        """
        counter = 0 # Integer, count the number of correct predictions.
        
        # Loops over the list of instances and calls the `predict_sense` method on each instance.
        for instance in instances:
            if instance.sense == self.predict_sense(instance):
                counter += 1
            
        return counter / len(instances) # Return accuracy.
        

class RandomSense(WSDClassifier):
    """
    RandomSense baseline
    """
    
    def __init__(self):
        pass # Nothing to do.

    def train(self, instances=[]):
        """
        instances: list[WSDInstance]
        """
        
        pass # Nothing to do.

    def predict_sense(self, instance):
        """
        instance: WSDInstance
        """
        
        senses = list(WN_CORRESPONDANCES[instance.lemma].keys()) # list[string]
        random.shuffle(senses)
        return senses[0]
    
    def __str__(self):
        return "RandomSense"

class MostFrequentSense(WSDClassifier):
    """
    Most Frequent Sense baseline
    """
    
    def __init__(self):
        self.mfs = None # Should be defined as a dictionary from lemmas to most frequent senses (dict[string -> string]) at training.
    
    def train(self, instances):
        """
        instances: list[WSDInstance]
        """
        senses = sense_distribution(instances) # get the distribution of senses
        self.mfs = {}
        # get the most frequent sense for each lemma
        for instance in instances:
            sense_dist_for_lemma = {}
            if instance.lemma not in self.mfs:
                for sense in senses:
                    if instance.lemma in WN_CORRESPONDANCES and sense in WN_CORRESPONDANCES[instance.lemma]:
                        sense_dist_for_lemma[sense] = senses[sense]
                self.mfs[instance.lemma] = max(sense_dist_for_lemma, key=sense_dist_for_lemma.get)
            
    def predict_sense(self, instance):
        """
        instance: WSDInstance
        """
        return self.mfs[instance.lemma]
    
    def __str__(self):
        return "MostFrequentSense"

class SimplifiedLesk(WSDClassifier):
    """
    Simplified Lesk algorithm
    """
    
    def __init__(self, window_size = -1, use_idf = False):
        """
        """
        self.window_size = window_size
        self.use_idf = use_idf
        self.signatures = None # Should be defined as a dictionary from senses to signatures (dict[string -> set[string]]) at training.

    def train(self, instances=[]):
        """
        instances: list[WSDInstance]
        """
        self.signatures = {}
        wn_synset = wordnet.synset # This is a shortcut to the `synset` function of the `wordnet` module.
        # We fill the dictionary `self.signatures` with the definition and examples from WordNet.
        for lemma in WN_CORRESPONDANCES:
            for sense in WN_CORRESPONDANCES[lemma]:
                self.signatures[sense] = set() # Initialize the signature of the sense.
                for synset in WN_CORRESPONDANCES[lemma][sense]:
                    # Update the signature of the sense with the definition and examples of the synset.
                    self.signatures[sense].update(wn_synset(synset).definition().split()) 
                    for example in wn_synset(synset).examples():
                        self.signatures[sense].update(example.split())
        # Update the signature of the sense with the context of the instance.
        for instance in instances:
            if self.window_size == -1:
                self.signatures[instance.sense].update(instance.context)
            else:
                self.signatures[instance.sense].update(instance.left_context[-self.window_size:] + instance.right_context[:self.window_size])
        # For the signature of a sense, use (i) the definition of each of the corresponding WordNet synsets, (ii) all of the corresponding examples in WordNet and (iii) the corresponding training instances.

    def predict_sense(self, instance):
        """
        instance: WSDInstance
        """
        senses = list(WN_CORRESPONDANCES[instance.lemma].keys() ) # list[string]
        score = list(len(senses) * [0]) # Initialize the score of each sense to 0.
        # For each sense, update the score of the sense with the number of words in the context of the instance that are in the signature of the sense.
        for i, sense in enumerate(senses):
            for word in instance.context:
                if word in self.signatures[sense] and word not in STOP_WORDS:
                    if self.use_idf:
                        #TODO: fix the idf score
                        score[i] += math.log(len(self.signatures[sense]) / len(WN_CORRESPONDANCES[instance.lemma][sense]))
                    else:
                        score[i] += 1

        return senses[score.index(max(score))] # Return the sense with the highest score.
    
    def __str__(self):
        return "SimplifiedLesk"

###############################################################



###############################################################

# The body of this conditional is executed only when this file is directly called as a script.
if __name__ == '__main__':
    from twa import WSDCollection
    from optparse import OptionParser

    usage = "Comparison of various WSD algorithms.\n%prog TWA_FILE"
    parser = OptionParser(usage=usage)
    (opts, args) = parser.parse_args()
    if(len(args) > 0):
        sensed_tagged_data_file = args[0]
    else:
        exit(usage + '\nYou need to specify in the command the path to a file of the TWA dataset.\n')

    # Loads the corpus.
    instances = WSDCollection(sensed_tagged_data_file).instances
    
    # Displays the sense distributions of the whole corpus.
    print(f'Sense distributions: {sense_distribution(instances)}')
    
    # Displays the accuracy of the RandomSense baseline.
    clf = RandomSense()
    accuracy = clf.evaluate(instances)
    print("We are doing a RandomSense baseline")
    print(f'{clf} accuracy: {accuracy:.2%}')
    
    # Evaluate the MostFrequentSense baseline on different splits of the dataset.
    clf = MostFrequentSense()
    print("We are doing a MostFrequentSense baseline")
    for k in range(1,9):
        train, test = random_data_split(instances, p = k, n = 10)
        clf.train(train)
        print(f'{clf} accuracy for {k} fold is: {clf.evaluate(test):.2%}')
        
    # Evaluate the SimplifiedLesk algorithm on different splits of the dataset.
    clf = SimplifiedLesk(window_size = 5, use_idf = True)
    print("We are doing a SimplifiedLesk classifier with window size 5 and idf")
    for k in range(1,9):
        train, test = random_data_split(instances, p = k, n = 10)
        clf.train(train)
        print(f'{clf} accuracy for {k} fold is: {clf.evaluate(test):.2%}')    
    
    clf = SimplifiedLesk(window_size = 5, use_idf = False)
    print("We are doing a SimplifiedLesk classifier with window size 5 and no idf")
    for k in range(1,9):
        train, test = random_data_split(instances, p = k, n = 10)
        clf.train(train)
        print(f'{clf} accuracy for {k} fold is: {clf.evaluate(test):.2%}')    
    # Instruction: Split the dataset (with `utils.data_split` or `utils.random_data_split`) into a training and a testing parts, then train and evaluate the different WSD algorithms.