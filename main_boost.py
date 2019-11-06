from FGBoost import GBoost

def classify():
    """shows an example of classification using boosting
       this is data about men,women and dogs
       h(man) means man is happy
       o(man,dog) means man owns dog
       r(man,woman,term) means man is in relationship with woman for long term or short term
    """

    print ("""shows an example of classification
           this is data about men,women and dogs
           h(man) means man is happy
           o(man,dog) means man owns dog
           r(man,woman,term) means man is in relationship with woman for long term or short term
           """)

    print ("\nlearning classification tree for man's happiness\n")

    #inputs to classification: data,examples,target and background
    train_data = ['o(m1,d1)','r(m1,w1,st)','o(m2,d2)','r(m2,w2,st)','o(m3,d3)','r(m3,w3,st)','o(m4,d4)','r(m4,w4,lt)','r(m5,w5,st)','r(m6,w6,lt)','r(m7,w7,lt)']
    train_pos = ['h(m1)','h(m2)','h(m4)','h(m6)']
    train_neg = ['h(m3)','h(m5)','h(m7)']
    target = 'h'
    bk = ['h(+man)','o(+man,-dog)','r(+man,-woman,#term)']


    #get boosting object
    clf = GBoost(train_data,train_pos,train_neg,bk,target)

    
    #learns k boosted trees, k defaults to 10
    clf.learn(k=2)
    
    print ("""learning boosted trees complete.
to see boosted tree ordered clauses, iterate through clf.boosted_trees and print tree.clauses
           """)

    print ()

    print ("\nhere is ordered clauses from first boosted tree:\n")
    print (clf.boosted_trees[0].clauses)

    '''

    #inputs to testing
    test_data = train_data #cheating but you can add your own data
    test_examples = train_pos+train_neg #just to check if working
    infered_value = clf.infer(test_data,test_examples)
    '''

if __name__ == '__main__':

    classify()
