import numpy as np
np.random.seed(42)

####################################################################################################
#                                            Part A
####################################################################################################

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.class_value = class_value
        self.dataset = dataset
        self.classDataset = class_data(dataset,class_value)
        self.mean = self.classDataset.mean(axis=0)
        self.std = self.classDataset.std(axis=0)

    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """

        return (len(self.classDataset)/len(self.dataset))
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        prob = 1
        for i in range(len(self.mean)):
           prob *= normal_pdf(x[i], self.mean[i], self.std[i]) 
        return prob
         
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return ((self.get_prior()*self.get_instance_likelihood(x)))


class MultiNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.class_value = class_value
        self.dataset = dataset
        self.classDataset = class_data(dataset,class_value)
        self.mean = self.classDataset.mean(axis=0)
        self.cov = np.cov(self.classDataset,rowvar=False)
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return (len(self.classDataset)/len(self.dataset))
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        return (multi_normal_pdf(x, self.mean, self.cov))
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return ((self.get_prior()*self.get_instance_likelihood(x)))
    
  

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    
    
    pdfCalc = 1/(np.sqrt(2*np.pi*np.square(std)))
    insideExp = -(np.square(x-mean)/(2*np.square(std)))
    expo = np.exp(insideExp)
    pdfCalc *= expo
    return pdfCalc


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variante normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    size = len(x)
    #det - matrix determinent
    det = np.linalg.det(cov)
    norm_const = 1.0/ (((2*np.pi)**float(size)/2) * np.sqrt(det))
    x_mu = np.matrix(x - mean)
    inv = np.linalg.inv(cov)       
    result = np.exp(-0.5 * (x_mu * inv * x_mu.T))
    return (norm_const * result).sum()

# create a data specific to the class
def class_data(dataset,class_value):
    separated = []
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] == class_value):
            vector = np.delete(vector, -1)
            separated.append(vector)
    # returns a numpy array
    return np.array(separated)
####################################################################################################
#                                            Part B
####################################################################################################
EPSILLON = 1e-6 # == 0.000001 It could happen that a certain value will only occur in the test set.
                # In case such a thing occur the probability for that value will EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with la place smoothing.
        
        Input
        - dataset: The dataset from which to compute the probabilites (Numpy Array).
        - class_value : Compute the relevant parameters only for instances from the given class.
        """
        self.class_value = class_value
        self.dataset = dataset
        self.classDataset = class_data(dataset,class_value)
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return (len(self.classDataset)/len(self.dataset))
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        prob = 1
        for i in range(len(x)):
           featureData = self.classDataset[:,i]
           Nij = (featureData == x[i]).sum()
           Ni = float(len(self.classDataset))
           Vj = len(np.unique(featureData))
           prob*= ((Nij+1)/(Ni+Vj))
           #print('Nij :%d , Ni : %d , Vj : %d, prob %d', Nij,Ni,Vj,prob)
        
        return prob 
        
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return ((self.get_prior()*self.get_instance_likelihood(x)))


    
####################################################################################################
#                                            General
####################################################################################################            
class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a postreiori classifier. 
        This class will hold 2 class distribution, one for class 0 and one for class 1, and will predicit and instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
        
        Input
            - An instance to predict.
            
        Output
            - 0 if the posterior probability of class 0 is higher 1 otherwise.
        """
        if(self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x)):
            return 0
        else:
            return 1
    
def compute_accuracy(testset, map_classifier):
    """
    Compute the accuracy of a given a testset and using a map classifier object.
    
    Input
        - testset: The test for which to compute the accuracy (Numpy array).
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / #testset size
    """
    sum = 0.0
    size = len(testset)
    for row in testset:
        X = np.delete(row, -1)
        pred = map_classifier.predict(X)
        if(pred == row[-1]): sum+=1

    return sum/size 
    
            
            
            
            
            
            
            
            
            
    