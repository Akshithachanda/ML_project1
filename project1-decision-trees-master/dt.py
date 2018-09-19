"""
In dt.py, you will implement a basic decision tree classifier for
binary classification.  Your implementation should be based on the
minimum classification error heuristic (even though this isn't ideal,
it's easier to code than the information-based metrics).
"""

from numpy import *

from binary import *
import util
import numpy as np
import math

class DT(BinaryClassifier):
    """
    This class defines the decision tree implementation.  It comes
    with a partial implementation for the tree data structure that
    will enable us to print the tree in a canonical form.
    """

    def __init__(self, opts):
        """
        Initialize our internal state.  The options are:
          opts.maxDepth = maximum number of features to split on
                          (i.e., if maxDepth == 1, then we're a stump)
        """

        self.opts = opts

        # initialize the tree data structure.  all tree nodes have a
        # "isLeaf" field that is true for leaves and false otherwise.
        # leaves have an assigned class (+1 or -1).  internal nodes
        # have a feature to split on, a left child (for when the
        # feature value is < 0.5) and a right child (for when the
        # feature value is >= 0.5)
        
        self.isLeaf = True
        self.label  = 1

    def online(self):
        """
        Our decision trees are batch
        """
        return False

    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return self.displayTree(0)

    def displayTree(self, depth):
        # recursively display a tree
        if self.isLeaf:
            return (" " * (depth*2)) + "Leaf " + repr(self.label) + "\n"
        else:
            if self.opts['criterion'] == 'ig':
                return (" " * (depth*2)) + "Branch " + repr(self.feature) + \
                      " [Gain=" + repr(format(self.gain, '.4f')) + "]\n" + \
                      self.left.displayTree(depth+1) + \
                      self.right.displayTree(depth+1)
            else:
                return (" " * (depth*2)) + "Branch " + repr(self.feature) + \
                      "\n" + self.left.displayTree(depth+1) + \
                      self.right.displayTree(depth+1)

    def predict(self, X):
        """
        Traverse the tree to make predictions for a single sample.  
        You should threshold X at 0.5, so <0.5 means left branch and
        >=0.5 means right branch.
        """

        ### TODO: YOUR CODE HERE traverse the treee
        ###util.raiseNotDefined() use syntax like self.left.predict as used in other
        if self.isLeaf:
        	return self.label
        else:
        	if X[self.feature] < 0.5:
        		return self.left.predict(X)
        	
        	else:
        		return self.right.predict(X)
                

    def trainDT(self, X, Y, maxDepth, criterion, used):
        """
        recursively build the decision tree
        """

        # get the size of the data set
        N,D = X.shape                             ### N and D are number of rows and columns of X data matrix

        # check to see if we're either out of depth or no longer
        # have any decisions to make
        if maxDepth <= 0 or len(util.uniq(Y)) <= 1:
            # we'd better end at this point.  need to figure
            # out the label to return
            self.isLeaf =   True                    ###util.raiseNotDefined()    ### TODO: YOUR CODE HERE Boolean true or false

            self.label  =   util.mode(Y)                ###util.raiseNotDefined()  which class to return to for leaf nodes?


        else:
            if criterion == 'ig': # information gain
                # compute the entropy at this node
                ### TODO: YOUR CODE HERE
                def entropy(y):
                    P = np.count_nonzero(y == 1)
                    #print("p")
                    #print(P)
                    N = np.count_nonzero(y == -1)
                    #print("n")
                    #print(N)
                    S = N+P
                    if(P > 0):
                        a = (-(P/S) * math.log((P/S),2))
                    else:
                        a = 0
                    if(N > 0):
                        b = (-(N/S)* math.log((N/S),2))
                    else:
                        b = 0
                             
                    #return ((-(P/S) * math.log((P/S),2)) - ((N/S)* math.log((N/S),2)))
                    return a+b
                
                self.entropy = entropy(Y) # entropy(Y) ###it depends on count at each feature
                print(self.entropy)
            
            # we need to find a feature to split on
            bestFeature = -1     # which feature has lowest error -- split feature 
            
            # use error stats or gain stats (not both) depending on criterion
            
            # initialize error stats
            bestError  = np.finfo('d').max            # finding max value of d -- just initializing
            
            # initialize gain stats
            bestGain = np.finfo('d').min              # Minimum value of d assigned to gain
            
            for d in range(D):                        ### d is FEATURE and iteration variable i  
                # have we used this feature yet
                if d in used:
                    continue

                # suppose we split on this feature; what labels
                # would go left and right?   check the feature value if its less than 0.5 goes left and greater than 0.5 goes rig
                leftY  =   Y[X[:,d] <= 0.5]        ###util.raiseNotDefined()  x[:,d] slicing the matrix to give the dth column  
                left_node = np.count_nonzero(leftY)
                #print("echo")
                #print(left_node)
                rightY =   Y[X[:,d] > 0.5]       ###util.raiseNotDefined()    ### TODO: YOUR CODE HERE
                right_node = np.count_nonzero(rightY)
                #print(right_node)

                # misclassification rate
                if criterion == 'mr':
                    # we'll classify the left points as their most-- so error is difference between all Y on left and Y=0 on left                   # common class and ditto right points.  our error-- for right same except Y=1
                    # is the how many are not their mode. 
                    
                    count_left = util.mode(leftY)    ### finding the most common class on the left (+1 or -1)
                    count_right = util.mode(rightY)  ### finding the most common class on the right (+1 or -1)
                
                    error_lefttree =   len( leftY[leftY!=count_left])
                    error_righttree = len(rightY[ rightY != count_right])
                    error =   error_lefttree + error_righttree #util.raiseNotDefined()#TODO:YOURCODE HERE difference between the
                    
                    # update min, max, bestFeature
                    if error <= bestError:
                        bestFeature = d
                        bestError   = error
                        
                # information gain
                elif criterion == 'ig':
                    # now use information gain
                    Total = np.count_nonzero(Y)
                    #print(Total)
                    N1 = np.count_nonzero(leftY)
                    #print(N1)
                    P1 = np.count_nonzero(rightY)
                    #print(P1)
                    entropy_left = entropy(leftY)
                    #print(entropy_left)
                    entropy_right = entropy(rightY)
                    #print(entropy_right)
                    
                    gain = (entropy(Y)) - ((N1/Total)*entropy(leftY))-((P1/Total)*entropy(rightY))### TODO: YOUR CODE HERE afterwords
                    #print(gain)
                    # update min, max, bestFeature
                    if gain >= bestGain:
                        bestFeature = d
                        bestGain = gain
            
            self.gain = bestGain # information gain corresponding to this split
            if bestFeature < 0:
                # this shouldn't happen, but just in case...
                self.isLeaf = True
                self.label  = util.mode(Y)

            else:
                self.isLeaf  =  False                   ###util.raiseNotDefined()    ### TODO: YOUR CODE HERE

                self.feature =  bestFeature          ###util.raiseNotDefined()    ### TODO: YOUR CODE HERE


                self.left  = DT({'maxDepth': maxDepth-1, 'criterion':criterion}) ## left sub tree
                self.right = DT({'maxDepth': maxDepth-1, 'criterion':criterion}) ## right sub tree
                # recurse on our children by calling
                #   self.left.trainDT(...) 
                # and
                #   self.right.trainDT(...) 
                # with appropriate arguments
                ### TODO: YOUR CODE HERE -- First we need to divide rows and columns to respective trees
                ###util.raiseNotDefined()
                used.append(bestFeature) ## so that the feature does not get used again
                Y_left = Y[X[:,bestFeature] <= 0.5]
                Y_right = Y[X[:,bestFeature] > 0.5]
                X_left = X[X[:,bestFeature] <= 0.5,:]
                X_right = X[X[:,bestFeature] > 0.5,:]
                
                self.left.trainDT(X_left, Y_left, maxDepth-1,criterion, used) 
                self.right.trainDT(X_right, Y_right,maxDepth-1,criterion, used)
                

    def train(self, X, Y):
        """
        Build a decision tree based on the data from X and Y.  X is a
        matrix (N x D) for N many examples on D features.  Y is an
        N-length vector of +1/-1 entries.

        Some hints/suggestions:
          - make sure you don't build the tree deeper than self.opts['maxDepth']
          
          - make sure you don't try to reuse features (this could lead
            to very deep trees that keep splitting on the same feature
            over and over again)
            
          - it is very useful to be able to 'split' matrices and vectors:
            if you want the ids for all the Xs for which the 5th feature is
            on, say X(:,5)>=0.5.  If you want the corresponting classes,
            say Y(X(:,5)>=0.5) and if you want the correspnding rows of X,
            say X(X(:,5)>=0.5,:)
            
          - i suggest having train() just call a second function that
            takes additional arguments telling us how much more depth we
            have left and what features we've used already

          - take a look at the 'mode' and 'uniq' functions in util.py
        """

        # TODO: implement the function below
        if 'criterion' not in self.opts:
          self.opts['criterion'] = 'mr' # misclassification rate
        self.trainDT(X, Y, self.opts['maxDepth'], self.opts['criterion'], [])


    def getRepresentation(self):
        """
        Return our internal representation: for DTs, this is just our
        tree structure -- i.e., ourselves
        """
        
        return self
    
     