from math import exp
from TILDE import TILDE

def sigmoid(x):
    """returns e^x/(1+e^x)
    """

    return (exp(x)/(1+exp(x)))

class GBoost(object):


    def __init__(self,data,pos,neg,bk,target,max_depth=2):
        """classification class to predict target
           conditioned on data and background
           using functional gradient ascent on sigmoid
           probability model assumption of target given data
        """

        self.data = data
        self.examples = {}
        self.pos = pos
        self.neg = neg
        #initial model assumed P(target|data)=0.5 to target=1,0
        for ex in pos+neg:
            self.examples[ex] = 0
        self.bk = bk
        self.target = target
        self.max_depth = max_depth
        self.boosted_trees = []

    def learn(self,k=10):
        """learns set of k boosted trees
        """
        
        gradients = {}
        for i in range(k):

            #create TILDE(R) tree object
            tree_i = TILDE(typ="regression",score="WV",max_depth=self.max_depth)

            #compute gradients as I-P
            for ex in self.examples:
                p = sigmoid(self.examples[ex])
                if ex in self.pos:
                    gradients[ex] = 1-p
                elif ex in self.neg:
                    gradients[ex] = 0-p

            #fit tree on gradients
            tree_i.learn(self.data,self.bk,self.target,examples=gradients)
            
            #recompute example values as previous example value + tree_i value
            for ex in self.examples:
                tree_i_value = tree_i.infer(self.data,ex)
                self.examples[ex] += tree_i_value

            #add tree to boosted_trees
            self.boosted_trees.append(tree_i)

    def infer(self,data,examples,k=10):
        """infer value of examples from data
           and a subset or all of the trees
        """

        example_values = []
        for example in examples:
            example_value = 0
            for i in range(k):
                tree_i = self.boosted_trees[i]
                tree_i_value = tree_i.infer(data,example)
                example_value += tree_i_value
            example_values.append(sigmoid(example_value))

        return example_values
    
