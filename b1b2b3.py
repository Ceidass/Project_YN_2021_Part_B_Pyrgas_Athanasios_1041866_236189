import numpy as np
import matplotlib.pyplot as plt
import copy

vocablen = 8520 # Number of vocabulary words
path = "DeliciousMIL/Data/"

# Function for export datasets from txt to numpy array
# This function saves dataset to .npy file and also return it in case we want it now
def tf_export(load_path, save_path=None):

    set = np.zeros((0,vocablen))
    docwordNum = np.zeros(shape=(0,))

    with open(load_path, "r") as txt:
        #Iterating through lines
        for line in txt:
            sentfind = False
            wordfind = False
            insentcount = False
            inwordcount = False
            sentNo = ""
            wordNo = ""
            word = ""
            total_words = 0
            spaces = 0
            vect = np.zeros((1,vocablen)) # Helper vector

            # Iterating through line's characters
            for char in range(0,len(line)):
                if not(sentfind): # If we have NOT found the "number of sentences" label 
                    if line[char] == "<":
                        sentfind = True # Indicates that we HAVE found the "number of sentences" label
                        insentcount = True # Indicates that we are inside the "number of sentences" label 
                else:   # If we HAVE found the "number of sentences" label 
                    if insentcount: # If we are inside the "number of sentences" label
                        if line[char] != "<":
                            if line[char] == ">":
                                insentcount = False # Indicates that we EXIT the "number of sentences" label
                                sentNo = int(sentNo)
                            else:
                                sentNo += line[char]
                    else: # If we have exited the "number of sentences" label
                        if not(wordfind): # If we have NOT found a "number of words" label
                            if line[char] == "<":
                                wordfind = True # Indicates that we HAVE found a "number of words" label
                                inwordcount = True # Indicates that we are inside a "number of words" label
                        else: # If we HAVE found a "number of words" label
                            if inwordcount: # If we are inside a "number of words" label
                                if line[char] != "<":
                                    if line[char] == ">":
                                        wordNo = int(wordNo)
                                        total_words += wordNo
                                        #wordNo = "" # try # Re-initializes wordNo to an empty string for the next sentence
                                        inwordcount = False  #Re-initializes inwordcount to False for the next sentence
                                    else:   
                                        wordNo += line[char]
                            else: # If we have exited the "number of words" label
                                if line[char] == " " and not(line[char-1] == ">" and line[char+1] == "<"): # If we hit a SPACE character
                                    if line[char-1] != ">":  
                                        word = int(word)
                                        vect[0,word] += 1
                                    #vect = np.append(vect,) # try
                                    spaces += 1
                                    word = "" # Re-initializes word to an empty string for the next word
                                    if spaces == int(wordNo)+1:
                                        spaces = 0 # Re-initializes spaces to zero for the next sentence
                                        wordfind = False
                                        wordNo = "" # Re-initializes wordNo to an empty string for the next sentence
                                else:
                                    if char == len(line)-1:
                                        word = int(word)
                                        vect[0,word] += 1
                                    else:
                                        word += line[char]

            set = np.append(set, vect, axis=0)
            docwordNum = np.append(docwordNum, total_words)
            #trainset = np.append(trainset, vect/sentNo, axis=0) # try # It will be good if we had word embedings 

    # Create tf for each word by dividing by the total number of words
    for i in range(0,set.shape[0]):
        set[i,:] /= docwordNum[i]

    #If we have set a save_path
    if save_path != None:
        #Save dataset in array format
        data = np.asarray(set)
        np.save(save_path, data)

    return set


# Function for calculating idf
# This function saves dataset to .npy file and also returns it in case we want it now
def idf_export(tf, save_path=None):
    
    tfcopy = np.zeros_like(tf)
    docsNum = tf.shape[0]
    attrNum = tf.shape[1]
    sum = np.zeros((1,attrNum))

    for i in range(0,docsNum):    # Iterating through documents
        for j in range(0, attrNum):   # Iterating through attributes
            if tf[i,j] != 0:        # If the word appears inside this document
                tfcopy[i,j] = 1     # Assign the value one
        sum += tfcopy[i,:]          # Add values of tfcopy line i in sum vector
    
    rep = np.reciprocal(sum)        # Returns 1/sum

    idf = np.log(np.multiply(rep, docsNum)) # Creates idf

    #If we have set a save_path
    if save_path != None:
        #Save dataset in array format
        data = np.asarray(idf)
        np.save(save_path, data)

    return idf


# Function for calculating mean tfidf
# This function saves dataset to .npy file and also returns it in case we want it now
def mean_tfidf_export(tf, idf, save_path=None):

    docsNum = tf.shape[0]
    # Element-wise multiplication of tf and idf tables
    tfidf = np.multiply(tf, idf)

    sum = np.zeros((1,vocablen)) # sum of all tfidf
    for i in range(0,docsNum):
        sum += tfidf[i,:]
    
    # Create the mean tfidf of each word
    sum /= docsNum

    if save_path != None:
        #Save dataset in array format
        data = np.asarray(sum)
        np.save(save_path, data)

    return sum
    

# Function for loading data from .npy format
def load_npy(load_path):
    data = np.load(load_path)
    return data

# Function for creating the initial population 
def initial(indiv_Num = 10, ace_prob=0.5):

    print("Initializing...")
    print("Population Number: ", indiv_Num)

    choose = np.array([0.,1.])
    probs = np.array([1-ace_prob, ace_prob])
    individuals = np.zeros((indiv_Num,vocablen))
    for i in range(indiv_Num):    # Iterating through individuals
        print("Individual ",i) 
        for j in range(vocablen): # Iterating through attributes
            individuals[i,j] = np.random.choice(choose,p=probs)

    return individuals

# Function for evaluating a given population of individuals
def fitness(individuals, old_mean=np.inf):

    print("Evaluating...")

    indvNum = individuals.shape[0]
    attrNum = individuals.shape[1]
    indWordNum = np.zeros((1,indvNum)) # 1-D array for storing number of words inside each individual
    differences = np.zeros((1,indvNum)) # 1-D array for storing the difference between 1000 (the lower limit) and the actual number of words in each individual
    done = False # Indicator for checking if algorithm has converged

    for i in range(indvNum):    # Iterating through individuals
        wordCount = 0
        for j in range(attrNum):    # Iterating through attributes (words)
            if individuals[i,j] == 1:   # If this word exists in the individual
                wordCount += 1          # Increase wordCount
        indWordNum[0,i] = wordCount     # Asign number of individual's words

    differences = indWordNum - 1000     # Number of extra words
    tfidf_sums = np.matmul(mean_tfidf, individuals.T) # 1-D array with sums of tfidfs of each individual

    # Division of each sum with the number of words in this individual
    # This way we calculate the mean of tfidf_mean of all the words in this individual
    fits = np.multiply(tfidf_sums, np.reciprocal(indWordNum)) # Means of the mean tfidfs 

    h = np.ones_like(fits)#try

    # Apply penalties based on differences
    for i in range(indvNum):      # Iterating through differences of the words of each individual
        if differences[0,i] < 0:
            fits[0,i] += differences[0,i] * 3.7231e-07 # 3.7231e-04 / 1000 
        else:
            fits[0,i] -= differences[0,i] * 4.951e-08  # 3.7231e-04 / 7520

    minval = np.amin(fits) # Returns mix of the fitness values

    maxval = np.amax(fits) # Returns max of the fitness values
    mean = np.mean(fits)   # Returns the mean of all vitness values
    bestind = np.argmax(fits) # Returns the index of the best individual
    dist = maxval - minval # Calculating the distance between max and min
    if dist == 0: # Algorithm has converged
        done = True

    # Giving fitness values non-negative values
    # fit = (fit - minfit) / sum of all fits
    fits -= minval
    sum_val = np.dot(fits.reshape((fits.shape[1])),h.reshape((h.shape[1]))) # Calculating sum of all values
    #sum_val = np.mean(fits,dtype=np.float64) * fits.shape[1] # Calculating sum of all values

    
    if not(done):
        fits /= sum_val # Calculating probabilities

    fits = fits.reshape((fits.shape[1],))

    # returns indication if the algorithm has converged
    # returns the probabilities for selection of each individual based on the evaluation
    # Returns maximum real value
    # Returns minimum real value
    # Returns mean real value
    return done, fits, maxval, minval, mean, bestind

# Function for selecting which individuals will pass to the next generation
def selection(old_individuals,probs):

    print("Selecting...")

    new_individuals = np.zeros_like(old_individuals) # Array for storing selected individuals
    indices = np.arange(old_individuals.shape[0])

    for i in range(old_individuals.shape[0]): # Iterate as many times as the old population
        select = np.random.choice(indices,p=probs) # Select an old individual randomly based on given probabilities
        new_individuals[i,:] = old_individuals[select,:] # Store selected individuals in new individuals array

    return new_individuals

def crossover(old_individuals,cut_points=1, prob=0.6):

    print("Crossovering...")

    new_individuals = np.zeros_like(old_individuals)
    indices = np.arange(old_individuals.shape[0])
    gaps = np.arange(1,old_individuals.shape[1])
    parents = np.zeros((2,old_individuals.shape[1])) # Place to store the parents
    offsprings = np.zeros((2,old_individuals.shape[1])) # Place to store the offsprings
    i=0 # Counter for iterating through old_individuals
    j=0 # Counter for iterating through new_individuals
    while i < old_individuals.shape[0]: # Iterating through old_individuals
        if j<new_individuals.shape[0]-2:#try # If there is space for new offsprings
            cross = np.random.choice(np.array([0,1]),p=np.array([1-prob,prob])) # Check if crossover will happen
            if cross==1: # If crossover is selected to occur
                pair = np.random.choice(indices) # Select randomly a pair from old_individuals
                parents[0,:] = old_individuals[i,:] # Store 1st parent
                parents[1,:] = old_individuals[pair,:] # Store 2nd parent
                points = np.random.choice(gaps, size=(cut_points), replace=False) # Select randomly points of the vector to cut
                points = np.append(points,np.array([0,old_individuals.shape[1]]))
                points = np.sort(points, axis=None) # Sort numbers of array in ascending order

                for z in range(1,points.shape[0]):
                    if z % 2 == 1: # If we are in odd number of vector piece
                        # Keep the piecies as they are (do not flip them between parents)
                        offsprings[0,points[z-1]:points[z]] = parents[0,points[z-1]:points[z]] 
                        offsprings[1,points[z-1]:points[z]] = parents[1,points[z-1]:points[z]]
                    elif z % 2 == 0:# If we are in even number of vector piece
                        # FLip the piecies of the parents
                        offsprings[0,points[z-1]:points[z]] = parents[1,points[z-1]:points[z]] 
                        offsprings[1,points[z-1]:points[z]] = parents[0,points[z-1]:points[z]]

                new_individuals[j,:] = offsprings[0,:] # Store 1st offspring in new_individuals
                j += 1 # Increase pointer of new_individuals so it points to free cell
                i += 1 # Increase pointer of old_individuals so it points to free cell
                new_individuals[j,:] = offsprings[1,:] # Store 2nd offspring in new_individuals
                j += 1 # Increase pointer of new_individuals so it points to free cell
                i += 1 # Increase pointer of old_individuals so it points to free cell
            elif cross==0: # If no crossover selected
                new_individuals[j,:] = old_individuals[i,:] # Pass old to new as it is
                j += 1 # Increase pointer of new_individuals so it points to free cell
                i += 1 # Increase pointer of old_individuals so it points to free cell
        else: # If there is NO space for new offsprings
            new_individuals[j,:] = old_individuals[i,:] # Pass old to new as it is
            j += 1 # Increase pointer of new_individuals so it points to free cell
            i += 1 # Increase pointer of old_individuals so it points to free cell

    return new_individuals

def mutation(individuals, prob=None):

    print("Mutating...\n")

    if prob !=0 and prob != None: # If there is probability of mutation
        for i in range(individuals.shape[0]): # Iterate through individuals
            for j in range(individuals.shape[1]): # Iterate through attributes (words)
                mute = np.random.choice(np.array([0,1]), p=np.array([1-prob,prob]))
                if mute == 1: # If mutation is selected to happen
                    # Flip this gene (0 to 1 or 1 to 0)
                    if individuals[i,j]==0:
                        individuals[i,j]=1
                    elif individuals[i,j]==1:
                        individuals[i,j]=0
    
    return individuals

                    
def life(indiv_Num=10, init_prob=0.1174, cprob=0.6, mprob=None, max_gens=100, 
        no_inc_tolerance=10, low_inc_tolerance=20, inc_rate_lim=0.01):
    
    NO_INC = False # Indication that procces ended due to no increase of the best value for most generations than the tolerance limit
    LOW_INC = False # Indication that procces ended due to low increase of best value

    final_solution = None
    solutions = np.zeros((1,vocablen))
    max = np.array([np.NINF])
    bestmax = np.NINF # Best of the max values
    mean = np.array([np.inf])
    min = np.array([np.inf])
    gen = 0 # Generation counter
    nitol = 0 # Counter for how many generations there is not a better solution
    litol = 0 # Counter for how many generations there is not a MUCH better solution

    individuals = initial(indiv_Num, init_prob) #Initialization

    while gen<=max_gens:
        gen += 1 # Increase generation number
        print("Generation ", gen)
        done, probs,tempmax,tempmin,tempmean,bestind = fitness(copy.copy(individuals), mean[gen-1])
        if done: # All individuals are the same
            if mprob == None or mprob == 0: # If there is no chance of mutation
                break
            else:
                probs = np.ones_like(probs) * (1/probs.shape[0]) # Share probs equaly
        print("MAX: ", tempmax) # Print generation's maximum
        print("MEAN: ",tempmean) # Print generation's mean
        print("MIN: ", tempmin) #  Print generation's minimum
        max = np.append(max,tempmax)
        mean = np.append(mean,tempmean)
        min = np.append(min,tempmin)
        #bestind = np.expand_dims(bestind, axis=(0,1))
        solutions = np.append(solutions,individuals[bestind:],axis=0)
        
        if tempmax < bestmax: # If we have a WORSE solution than the last best
            nitol+=1 # Increase the no-increase-counter
            if nitol==no_inc_tolerance:
                NO_INC = True
                break # Exit while loop
        else: # If we have a BETTER solution than the last best
            bestmax = tempmax #Set tempmax as the best max
            nitol = 0 # Set no-increase-counter back to zero
            if tempmax >= (1+inc_rate_lim) * bestmax: # If the new solution is more than 1% better than the previous best one
                litol=0 # Set low-inrease-counter back to zero
            else:
                litol += 1 # Increase the low-increase-counter
                if litol == low_inc_tolerance: # If we reached the low increase tolerance limit
                    LOW_INC = True
                    break # Exit while loop

        individuals = selection(copy.copy(individuals), probs)
        individuals = crossover(copy.copy(individuals), 1, cprob)
        individuals = mutation(copy.copy(individuals), mprob)
    # END OF WHILE

    bestmaxgen = None # Variable for storing the generation number in which we had the best solution

    if done:
        bestmaxgen = np.argmax(max)
        final_solution = solutions[bestmaxgen,:]
        final_max = max[bestmaxgen]
    if NO_INC:
        bestmaxgen = gen-no_inc_tolerance
        final_solution = solutions[bestmaxgen,:]
        final_max = max[bestmaxgen]
    elif LOW_INC:
        bestmaxgen = gen-low_inc_tolerance
        final_solution = solutions[bestmaxgen,:]
        final_max = max[bestmaxgen]
    elif gen == max_gens:
        bestmaxgen = np.argmax(max)
        final_solution = solutions[bestmaxgen,:]
        final_max = max[bestmaxgen]

    if max.shape[0]<100:
        temp = np.zeros((100-max.shape[0]))
        max = np.append(max,temp)

    print("Solution Evaluation: ", final_max)

    return final_solution,final_max, gen, bestmaxgen, max
    


######################################################### MAIN #################################################################

# Save words' tfidfs for each document in npy format
#(we do this only one time and then we load from .npy format if it needed)
# tf = tf_export(path+"train-data.dat", path+"tf.npy")
# idf = idf_export(tf, path+"idf.npy")
# mean_tfidf = mean_tfidf_export(tf, idf, path+"mean_tfidf.npy")

# Load from saved .npy file
mean_tfidf = load_npy(path+"mean_tfidf.npy")

# List for storing results for each question for initialization prob 0.1174
quest = []
for i in range(0,10):
    quest.append({"solutions" : np.zeros((1,vocablen)), # Array of solution vectors
                  "maximum" : np.array([np.NINF]), # Array of maximums for each run
                  "generations" : np.array([0]), # Array of number of generations for each run
                  "topgen" : np.array([0]), # Array of number of generation that best solution appeared
                  "maxes" : np.zeros((1,100)), 
                  "times" : np.zeros((100,))
                })

# quest2 = []
# for i in range(0,10):
#     quest2.append({"solutions" : np.zeros((1,vocablen)),
#                   "maximum" : np.array([np.NINF]),
#                   "generations" : np.array([0]),
#                   "topgen" : np.array([0]),
#                   "maxes" : np.zeros((1,100)),
#                   "times" : np.zeros((1,100))
#                 })


for i in range(10):
    print("Run ",i+1)

    #Number 1 question of the project (Population=20, Crossover Prob=0.6, Mutation Prob=0)
    print("Run ",i+1)
    sol,max,gen,topgen,tops = life(indiv_Num=20, init_prob=0.1174, cprob=0.6, mprob=None) #Initialization prob 0.1174
    quest[0]["solutions"] = np.append(quest[0]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    quest[0]["maximum"] = np.append(quest[0]["maximum"],np.array([max]))
    quest[0]["generations"] = np.append(quest[0]["generations"], np.array([gen]))
    quest[0]["topgen"] = np.append(quest[0]["topgen"],np.expand_dims(topgen,axis=0))
    quest[0]["maxes"] = np.append(quest[0]["maxes"],np.expand_dims(tops,axis=0), axis=0)

    # print("Run ",i+1)
    # sol,max,gen,topgen,tops = life(indiv_Num=20, init_prob=0.5, cprob=0.6, mprob=None) #Initialization prob 0.5
    # quest2[0]["solutions"] = np.append(quest2[0]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    # quest2[0]["maximum"] = np.append(quest2[0]["maximum"],np.array([max]))
    # quest2[0]["generations"] = np.append(quest2[0]["generations"], np.array([gen]))
    # quest2[0]["topgen"] = np.append(quest2[0]["topgen"],np.expand_dims(topgen,axis=0))
    # quest2[0]["maxes"] = np.append(quest2[0]["maxes"],np.expand_dims(tops,axis=0), axis=0)


    #Number 2 question of the project (Population=20, Crossover Prob=0.6, Mutation Prob=0.01)
    print("Run ",i+1)
    sol,max,gen,topgen,tops = life(indiv_Num=20, init_prob=0.1174, cprob=0.6, mprob=0.01) #Initialization prob 0.1174
    quest[1]["solutions"] = np.append(quest[1]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    quest[1]["maximum"] = np.append(quest[1]["maximum"],np.array([max]))
    quest[1]["generations"] = np.append(quest[1]["generations"], np.array([gen]))
    quest[1]["topgen"] = np.append(quest[1]["topgen"],np.expand_dims(topgen,axis=0))
    quest[1]["maxes"] = np.append(quest[1]["maxes"],np.expand_dims(tops,axis=0), axis=0)

    # print("Run ",i+1)
    # sol,max,gen,topgen,tops = life(indiv_Num=20, init_prob=0.5, cprob=0.6, mprob=0.01) #Initialization prob 0.5
    # quest2[1]["solutions"] = np.append(quest2[1]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    # quest2[1]["maximum"] = np.append(quest2[1]["maximum"],np.array([max]))
    # quest2[1]["generations"] = np.append(quest2[1]["generations"], np.array([gen]))
    # quest2[1]["topgen"] = np.append(quest2[1]["topgen"],np.expand_dims(topgen,axis=0))
    # quest2[1]["maxes"] = np.append(quest2[1]["maxes"],np.expand_dims(tops,axis=0), axis=0)


    #Number 3 question of the project (Population=20, Crossover Prob=0.6, Mutation Prob=0.10)
    print("Run ",i+1)
    sol,max,gen,topgen,tops = life(indiv_Num=20, init_prob=0.1174, cprob=0.6, mprob=0.10) #Initialization prob 0.1174
    quest[2]["solutions"] = np.append(quest[2]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    quest[2]["maximum"] = np.append(quest[2]["maximum"],np.array([max]))
    quest[2]["generations"] = np.append(quest[2]["generations"], np.array([gen]))
    quest[2]["topgen"] = np.append(quest[2]["topgen"],np.expand_dims(topgen,axis=0))
    quest[2]["maxes"] = np.append(quest[2]["maxes"],np.expand_dims(tops,axis=0), axis=0)

    #print("Run ",i+1)
    # sol,max,gen,topgen,tops = life(indiv_Num=20, init_prob=0.5, cprob=0.6, mprob=0.10) #Initialization prob 0.5
    # quest2[2]["solutions"] = np.append(quest2[2]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    # quest2[2]["maximum"] = np.append(quest2[2]["maximum"],np.array([max]))
    # quest2[2]["generations"] = np.append(quest2[2]["generations"], np.array([gen]))
    # quest2[2]["topgen"] = np.append(quest2[2]["topgen"],np.expand_dims(topgen,axis=0))
    # quest2[2]["maxes"] = np.append(quest2[2]["maxes"],np.expand_dims(tops,axis=0), axis=0)


    #Number 4 question of the project (Population=20, Crossover Prob=0.9, Mutation Prob=0.01)
    print("Run ",i+1)
    sol,max,gen,topgen,tops = life(indiv_Num=20, init_prob=0.1174, cprob=0.9, mprob=0.01) #Initialization prob 0.1174
    quest[3]["solutions"] = np.append(quest[3]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    quest[3]["maximum"] = np.append(quest[3]["maximum"],np.array([max]))
    quest[3]["generations"] = np.append(quest[3]["generations"], np.array([gen]))
    quest[3]["topgen"] = np.append(quest[3]["topgen"],np.expand_dims(topgen,axis=0))
    quest[3]["maxes"] = np.append(quest[3]["maxes"],np.expand_dims(tops,axis=0), axis=0)

    # print("Run ",i+1)
    # sol,max,gen,topgen,tops = life(indiv_Num=20, init_prob=0.5, cprob=0.9, mprob=0.01) #Initialization prob 0.5
    # quest2[3]["solutions"] = np.append(quest2[3]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    # quest2[3]["maximum"] = np.append(quest2[3]["maximum"],np.array([max]))
    # quest2[3]["generations"] = np.append(quest2[3]["generations"], np.array([gen]))
    # quest2[3]["topgen"] = np.append(quest2[3]["topgen"],np.expand_dims(topgen,axis=0))
    # quest2[3]["maxes"] = np.append(quest2[3]["maxes"],np.expand_dims(tops,axis=0), axis=0)


    #Number 5 question of the project (Population=20, Crossover Prob=0.1, Mutation Prob=0.01)
    print("Run ",i+1)
    sol,max,gen,topgen,tops = life(indiv_Num=20, init_prob=0.1174, cprob=0.1, mprob=0.01) #Initialization prob 0.1174
    quest[4]["solutions"] = np.append(quest[4]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    quest[4]["maximum"] = np.append(quest[4]["maximum"],np.array([max]))
    quest[4]["generations"] = np.append(quest[4]["generations"], np.array([gen]))
    quest[4]["topgen"] = np.append(quest[4]["topgen"],np.expand_dims(topgen,axis=0))
    quest[4]["maxes"] = np.append(quest[4]["maxes"],np.expand_dims(tops,axis=0), axis=0)

    #print("Run ",i+1)
    # sol,max,gen,topgen,tops = life(indiv_Num=20, init_prob=0.5, cprob=0.1, mprob=0.01) #Initialization prob 0.5
    # quest2[4]["solutions"] = np.append(quest2[4]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    # quest2[4]["maximum"] = np.append(quest2[4]["maximum"],np.array([max]))
    # quest2[4]["generations"] = np.append(quest2[4]["generations"], np.array([gen]))
    # quest2[4]["topgen"] = np.append(quest2[4]["topgen"],np.expand_dims(topgen,axis=0))
    # quest2[4]["maxes"] = np.append(quest2[4]["maxes"],np.expand_dims(tops,axis=0), axis=0)


    #Number 6 question of the project (Population=200, Crossover Prob=0.6, Mutation Prob=0)
    print("Run ",i+1)
    sol,max,gen,topgen,tops = life(indiv_Num=200, init_prob=0.1174, cprob=0.6, mprob=None) #Initialization prob 0.1174
    quest[5]["solutions"] = np.append(quest[5]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    quest[5]["maximum"] = np.append(quest[5]["maximum"],np.array([max]))
    quest[5]["generations"] = np.append(quest[5]["generations"], np.array([gen]))
    quest[5]["topgen"] = np.append(quest[5]["topgen"],np.expand_dims(topgen,axis=0))
    quest[5]["maxes"] = np.append(quest[5]["maxes"],np.expand_dims(tops,axis=0), axis=0)

    # print("Run ",i+1)
    # sol,max,gen,topgen,tops = life(indiv_Num=200, init_prob=0.5, cprob=0.6, mprob=None) #Initialization prob 0.5
    # quest2[5]["solutions"] = np.append(quest2[5]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    # quest2[5]["maximum"] = np.append(quest2[5]["maximum"],np.array([max]))
    # quest2[5]["generations"] = np.append(quest2[5]["generations"], np.array([gen]))
    # quest2[5]["topgen"] = np.append(quest2[5]["topgen"],np.expand_dims(topgen,axis=0))
    # quest2[5]["maxes"] = np.append(quest2[5]["maxes"],np.expand_dims(tops,axis=0), axis=0)

    #Number 7 question of the project (Population=200, Crossover Prob=0.6, Mutation Prob=0.01)
    print("Run ",i+1)
    sol,max,gen,topgen,tops = life(indiv_Num=200, init_prob=0.1174, cprob=0.6, mprob=0.01) #Initialization prob 0.1174
    quest[6]["solutions"] = np.append(quest[6]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    quest[6]["maximum"] = np.append(quest[6]["maximum"],np.array([max]))
    quest[6]["generations"] = np.append(quest[6]["generations"], np.array([gen]))
    quest[6]["topgen"] = np.append(quest[6]["topgen"],np.expand_dims(topgen,axis=0))
    quest[6]["maxes"] = np.append(quest[6]["maxes"],np.expand_dims(tops,axis=0), axis=0)

    # print("Run ",i+1)
    # sol,max,gen,topgen,tops = life(indiv_Num=200, init_prob=0.5, cprob=0.6, mprob=0.01) #Initialization prob 0.5
    # quest2[6]["solutions"] = np.append(quest2[6]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    # quest2[6]["maximum"] = np.append(quest2[6]["maximum"],np.array([max]))
    # quest2[6]["generations"] = np.append(quest2[6]["generations"], np.array([gen]))
    # quest2[6]["topgen"] = np.append(quest2[6]["topgen"],np.expand_dims(topgen,axis=0))
    # quest2[6]["maxes"] = np.append(quest2[6]["maxes"],np.expand_dims(tops,axis=0), axis=0)


    #Number 8 question of the project (Population=200, Crossover Prob=0.6, Mutation Prob=0.10)
    print("Run ",i+1)
    sol,max,gen,topgen,tops = life(indiv_Num=200, init_prob=0.1174, cprob=0.6, mprob=0.10) #Initialization prob 0.1174
    quest[7]["solutions"] = np.append(quest[7]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    quest[7]["maximum"] = np.append(quest[7]["maximum"],np.array([max]))
    quest[7]["generations"] = np.append(quest[7]["generations"], np.array([gen]))
    quest[7]["topgen"] = np.append(quest[7]["topgen"],np.expand_dims(topgen,axis=0))
    quest[7]["maxes"] = np.append(quest[7]["maxes"],np.expand_dims(tops,axis=0), axis=0)


    # print("Run ",i+1)
    # sol,max,gen,topgen,tops = life(indiv_Num=200, init_prob=0.5, cprob=0.6, mprob=0.10) #Initialization prob 0.5
    # quest2[7]["solutions"] = np.append(quest2[7]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    # quest2[7]["maximum"] = np.append(quest2[7]["maximum"],np.array([max]))
    # quest2[7]["generations"] = np.append(quest2[7]["generations"], np.array([gen]))
    # quest2[7]["topgen"] = np.append(quest2[7]["topgen"],np.expand_dims(topgen,axis=0))
    # quest2[7]["maxes"] = np.append(quest2[7]["maxes"],np.expand_dims(tops,axis=0), axis=0)


    #Number 9 question of the project (Population=200, Crossover Prob=0.9, Mutation Prob=0.01)
    print("Run ",i+1)
    sol,max,gen,topgen,tops = life(indiv_Num=200, init_prob=0.1174, cprob=0.9, mprob=0.01) #Initialization prob 0.1174
    quest[8]["solutions"] = np.append(quest[8]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    quest[8]["maximum"] = np.append(quest[8]["maximum"],np.array([max]))
    quest[8]["generations"] = np.append(quest[8]["generations"], np.array([gen]))
    quest[8]["topgen"] = np.append(quest[8]["topgen"],np.expand_dims(topgen,axis=0))
    quest[8]["maxes"] = np.append(quest[8]["maxes"],np.expand_dims(tops,axis=0), axis=0)


    # print("Run ",i+1)
    # sol,max,gen,topgen,tops = life(indiv_Num=200, init_prob=0.5, cprob=0.9, mprob=0.01) #Initialization prob 0.5
    # quest2[8]["solutions"] = np.append(quest2[8]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    # quest2[8]["maximum"] = np.append(quest2[8]["maximum"],np.array([max]))
    # quest2[8]["generations"] = np.append(quest2[8]["generations"], np.array([gen]))
    # quest2[8]["topgen"] = np.append(quest2[8]["topgen"],np.expand_dims(topgen,axis=0))
    # quest2[8]["maxes"] = np.append(quest2[8]["maxes"],np.expand_dims(tops,axis=0), axis=0)


    #Number 10 question of the project (Population=200, Crossover Prob=0.1, Mutation Prob=0.01)
    print("Run ",i+1)
    sol,max,gen,topgen,tops = life(indiv_Num=200, init_prob=0.1174, cprob=0.1, mprob=0.01) #Initialization prob 0.1174
    quest[9]["solutions"] = np.append(quest[9]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    quest[9]["maximum"] = np.append(quest[9]["maximum"],np.array([max]))
    quest[9]["generations"] = np.append(quest[9]["generations"], np.array([gen]))
    quest[9]["topgen"] = np.append(quest[9]["topgen"],np.expand_dims(topgen,axis=0))
    quest[9]["maxes"] = np.append(quest[9]["maxes"],np.expand_dims(tops,axis=0), axis=0)


    # print("Run ",i+1)
    # sol,max,gen,topgen,tops = life(indiv_Num=200, init_prob=0.5, cprob=0.1, mprob=0.01) #Initialization prob 0.5
    # quest2[9]["solutions"] = np.append(quest2[9]["solutions"],np.expand_dims(sol,axis=0), axis=0)
    # quest2[9]["maximum"] = np.append(quest2[9]["maximum"],np.array([max]))
    # quest2[9]["generations"] = np.append(quest2[9]["generations"], np.array([gen]))
    # quest2[9]["topgen"] = np.append(quest2[9]["topgen"],np.expand_dims(topgen,axis=0))
    # quest2[9]["maxes"] = np.append(quest2[9]["maxes"],np.expand_dims(tops,axis=0), axis=0)

    # END OF FOR LOOP

for i in range(10): # For each question
    for j in range(1,11): # For each run of the question
        if quest[i]["generations"][j] != 0:
            for z in range(1,quest[i]["generations"][j]+1):
                quest[i]["times"][z] += 1 # Increase generations counter

# for i in range(0,11): # For each question
#     for j in range(1,10): # For each run of the question
#         if quest2[i]["generations"][j] != 0:
#             for z in range(1,quest2[i]["generations"][j]+1):
#                 quest2[i]["times"][z] += 1 # Increase generations counter

# SAVE ARRAYS
for i in range(10):

####################### FIRST #######################

    data = np.asarray(quest[i]["solutions"])
    np.save("results/arrays/first/"+str(i+1)+"/solutions.npy",data)

    data = np.asarray(quest[i]["maximum"])
    np.save("results/arrays/first/"+str(i+1)+"/maximum.npy",data)

    data = np.asarray(quest[i]["generations"])
    np.save("results/arrays/first/"+str(i+1)+"/generations.npy",data)

    data = np.asarray(quest[i]["topgen"])
    np.save("results/arrays/first/"+str(i+1)+"/topgen.npy",data)

    data = np.asarray(quest[i]["maxes"])
    np.save("results/arrays/first/"+str(i+1)+"/maxes.npy",data)

    data = np.asarray(quest[i]["times"])
    np.save("results/arrays/first/"+str(i+1)+"/times.npy",data)

########################### SECOND #######################

    # data = np.asarray(quest2[i]["solutions"])
    # np.save("results/arrays/second/"+str(i+1)+"/solutions.npy",data)

    # data = np.asarray(quest2[i]["maximum"])
    # np.save("results/arrays/second/"+str(i+1)+"/maximum.npy",data)

    # data = np.asarray(quest2[i]["generations"])
    # np.save("results/arrays/second/"+str(i+1)+"/generations.npy",data)

    # data = np.asarray(quest2[i]["topgen"])
    # np.save("results/arrays/second/"+str(i+1)+"/topgen.npy",data)

    # data = np.asarray(quest2[i]["maxes"])
    # np.save("results/arrays/second/"+str(i+1)+"/maxes.npy",data)

    # data = np.asarray(quest2[i]["times"])
    # np.save("results/arrays/second/"+str(i+1)+"/times.npy",data)


# Retrieving data
for i in range(10):
    quest[i]["solutions"] = load_npy("results/arrays/first/"+str(i+1)+"/solutions.npy")
    quest[i]["maximum"] = load_npy("results/arrays/first/"+str(i+1)+"/maximum.npy")
    quest[i]["generations"] = load_npy("results/arrays/first/"+str(i+1)+"/generations.npy")
    quest[i]["topgen"] = load_npy("results/arrays/first/"+str(i+1)+"/topgen.npy")
    quest[i]["maxes"] = load_npy("results/arrays/first/"+str(i+1)+"/maxes.npy")
    quest[i]["times"] = load_npy("results/arrays/first/"+str(i+1)+"/times.npy")

    # quest2[i]["solutions"] = load_npy("results/arrays/second/"+str(i+1)+"/solutions.npy")
    # quest2[i]["maximum"] = load_npy("results/arrays/second/"+str(i+1)+"/maximum.npy")
    # quest2[i]["generations"] = load_npy("results/arrays/second/"+str(i+1)+"/generations.npy")
    # quest2[i]["topgen"] = load_npy("results/arrays/second/"+str(i+1)+"/topgen.npy")
    # quest2[i]["maxes"] = load_npy("results/arrays/second/"+str(i+1)+"/maxes.npy")
    # quest2[i]["times"] = load_npy("results/arrays/second/"+str(i+1)+"/times.npy")


# Finding the mean of best solutions of each question
print("Mean of best solutions")
for i in range(10):
    maxsum = 0
    topgensum = 0
    for j in range(1,quest[i]["maximum"].shape[0]):
        maxsum += quest[i]["maximum"][j]
        topgensum += quest[i]["topgen"][j]

    maxsum /= quest[i]["maximum"].shape[0]
    topgensum /= quest[i]["topgen"].shape[0]
    print("Question "+str(i+1)+" maximum mean : ",maxsum)
    print("Question "+str(i+1)+" topgen mean : ",topgensum)

# PLOTTING

################# FIRST ###########################
plots = []
for i in range(10):
    count = 0
    for j in range(quest[i]["times"].shape[0]):
        if quest[i]["times"][j] != 0:
            count += 1

    plots.append(np.arange(1,count+1))
    
    summaxes = np.zeros((count))
    for j in range(quest[i]["maxes"].shape[0]):
        summaxes += quest[i]["maxes"][j,1:count+1]
    
    yaxis = summaxes / quest[i]["times"][1:count+1]
    plots.append(yaxis)

    plt.style.use('ggplot')
    plt.plot(np.arange(1,count+1), yaxis)
    plt.xlabel("Generations")
    plt.ylabel("Mean Best Value")
    plt.title("Case "+str(i+1))
    plt.savefig("results/plots/first/"+str(i+1)+"/avgbst"+str(i+1)+".png")
    plt.show()

#Compare plot
plt.style.use('ggplot')
plt.plot(plots[0], plots[1], plots[2], plots[3], plots[4], plots[5], plots[6], plots[7], plots[8], plots[9], 
    plots[10], plots[11], plots[12], plots[13], plots[14], plots[15], plots[16], plots[17], plots[18], plots[19])
plt.xlabel("Generations")
plt.ylabel("Mean Best Value")
plt.title("Comparison")
plt.legend(["1", "2", "3","4","5","6","7","8","9","10"], loc ="lower right")
plt.savefig("results/plots/first/comp/avgbstcomp.png")
plt.show()
################# SECOND ###########################

# for i in range(10):
#     for j in range(quest2[i]["times"].shape[0]):
#         if quest2[i]["times"][j] == 0:
#             quest2[i]["times"][j] == 1

#     summaxes = np.zeros((quest2[i]["maxes"].shape[1]))
#     for j in range(quest2[i]["maxes"].shape[0]):
#         summaxes += quest2[i]["maxes"][j,:]
    
#     yaxis = summaxes / quest2[i]["times"]
#     plt.style.use('ggplot')
#     plt.plot(np.arange(100), yaxis)
#     plt.xlabel("Generations")
#     plt.ylabel("Mean Best Value")
#     plt.title("Case "+str(i+1))
#     plt.savefig("results/plots/second/"+str(i+1)+"/avgbst"+str(i+1)+".png")
