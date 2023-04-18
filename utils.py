#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import re

# correspondances entre synset de WordNet et sensid de TWA
WN_CORRESPONDANCES = {
    'bass': {
        'bass%music': ['bass.n.01', 'bass.n.02', 'bass.n.03', 'bass.n.06', 'bass.n.07'],
        'bass%fish': ['sea_bass.n.01', 'freshwater_bass.n.01', 'bass.n.08']
    },
    'crane': {
        'crane%machine': ['crane.n.04'],
        'crane%bird': ['crane.n.05']
    },
    'motion': {
        'motion%physical': ['gesture.n.02', 'movement.n.03', 'motion.n.03', 'motion.n.04', 'motion.n.06'],
        'motion%legal': ['motion.n.05']
    },
    'palm': {
        'palm%hand': ['palm.n.01'], # +'palm.n.02'?
        'palm%tree': ['palm.n.03']
    },
    'plant': {
        'plant%factory': ['plant.n.01'],
        'plant%living': ['plant.n.02']
    },
    'tank': {
        'tank%vehicle': ['tank.n.01'], # +'tank_car.n.01'?
        'tank%container': ['tank.n.02']
    }
}

# A list of English stop words.
STOP_WORDS = set(['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear', 'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for', 'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers', 'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let', 'like', 'likely', 'may', 'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor', 'not', 'of', 'off', 'often', 'on', 'one', 'only', 'or', 'other', 'our', 'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since', 'so', 'some', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'two', 'us', 'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'yet', 'you', 'your'])
STOP_WORDS.update({"it's", "aren't", "can't"})

# text: string
def normalize_and_split(text):
    chars = ".,':()" # Characters that might be found next to a token but that are not part of it.
    tokens = [token.strip().strip(chars) for token in text.lower().split()] # The text is lowercased and split on spaces to get tokens. Tokens are cleaned based on `chars`.
    return [token for token in tokens if((token not in STOP_WORDS) and re.search('[a-z0-9]', token))] # Stop words and tokens that do not contain any alphanumeric character are filtered out.


def data_split(instances, p=1, n=5):
    """
    Splits `instances` into two parts:
      one that contains p/n of the data,
      another that contains the remaning (n-p)/n of the data.
      
    instances: list[WSDInstance]
    p, n: int
    """
    
    part1 = []; part2 = []
    for i, instance in enumerate(instances):
        i = i % n
        if(i < p): part1.append(instance)
        else: part2.append(instance)
    
    return (part1, part2)

def random_data_split(instances, p=1, n=5):
    """
    Randomly splits `instances` into two parts:
      one that contains p/n of the data,
      another that contains the remaning (n-p)/n of the data.
      
    instances: list[WSDInstance]
    p, n: int
    """
    part1 = []; part2 = []
    instances_copy = instances.copy() # We do not want to modify the original list.
    random.shuffle(instances_copy) # The list is shuffled.
    for i, instance in enumerate(instances_copy):
        i = i % n
        if(i < p): part1.append(instance)
        else: part2.append(instance)
    
    return (part1, part2)    

def sense_distribution(instances):
    """
    Computes the distribution of senses in a list of instances.

    instances: list[WSDInstance]
    """
    
    sense_distrib = {} # dict[string -> int]
    for instance in instances:
        sense = instance.sense
        sense_distrib[sense] = sense_distrib.get(sense, 0) + 1
    
    return sense_distrib

def prettyprint_sense_distribution(instances):
    """
    Prints the distribution of senses in a list of instances.
    
    instances: list[WSDInstance]
    """
    
    sense_distrib = sense_distribution(instances) # dict[string -> int]
    sense_distrib = list(sense_distrib.items()) # list[(string, int)]
    sense_distrib = sorted(sense_distrib, key=(lambda x: x[0])) # Sorts the list in alphabetical order (using the senses' name).
    for sense, count in sense_distrib:
        print(f"{sense}\t{count}") # For (old) versions of Python, use the following instead: print(sense + "\t" + str(count))
    print()
    return sense_distrib
