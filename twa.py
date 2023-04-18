#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import html
import xml.etree.cElementTree as ET
from utils import *

class WSDInstance:
    """
    A WSD instance from the TWA dataset.
    """
    
    # xml_element: xml.etree.ElementTree.Element
    def __init__(self, xml_element):
        self.id = xml_element.get('id') # string
        
        self.sense = xml_element.find('.//answer').get('senseid') # string
        if self.sense == "ank%container": # There is a typo in the corpus.
            self.sense = "tank%container"
        
        context_elt = xml_element.find('.//context')
        # The target token is inside a "head" node.
        head_elt = context_elt.find('.//head')
        self.lemma = head_elt.text # string
        
        # `context_elt.text` contains everything in the context before the "head" node.
        self.left_context = normalize_and_split(context_elt.text) # list[string]
        
        # `head_elt.tail` contains everything in the context after the "head" node.
        self.right_context = normalize_and_split(head_elt.tail) # list[string]
        
        self.context = self.right_context + self.left_context # list[string]

class WSDCollection:
    """
    A collection of WSD instances from the TWA dataset.
    """
    
    def __init__(self, filepath):
        """
        filepath: string (the path to an XML file)
        """
        
        self.instances = []
        self.parse(filepath)

    def parse(self, filepath):
        """
        filepath: string (the path to an XML file)
        """
        
        # The file is read with the XML ETree library.
        # The files in the corpus are not well-formed XML files (but almost): they lack a root node. Therefore, the first thing we do is to add an opening tag <collection> at the beginning of file and a corresponding closing tag </collection> at the end.
        with open(filepath, 'r') as xml_file:
            xml_str = f'<collection>\n{xml_file.read()}\n</collection>' # For (old) versions of Python, use the following instead: '<collection>\n' + str(xml_file.read()) + '\n</collection>'
        
        xml_str = html.unescape(xml_str) # In the corpus, some symbols have been converted the HTML code (ex: "&amp;" is the HTML code for "&").`html.unescape` replaces such codes by the corresponding symbols.
        
        xml_tree = ET.fromstring(xml_str) # Parses the XML.
        inst_elts = xml_tree.findall('.//instance') # Retreives all "instance" node.
        for elt in inst_elts:
            instance = WSDInstance(elt)
            self.instances.append(instance)
        return

##########################

# The body of this conditional is executed only when this file is directly called as a script.
if __name__ == '__main__':
    import sys
    
    if(len(sys.argv) <= 1): exit('You need to specify in the command the path to a file of the TWA dataset.')
    
    f = sys.argv[1]
    sense_tagged_data = WSDCollection(f)
    for instance in sense_tagged_data.instances:
        print(f"{instance.id}: '{instance.lemma}, {instance.context}") # For (old) versions of Python, use the following instead: print(instance.id,, instance.lemma, instance.context)
