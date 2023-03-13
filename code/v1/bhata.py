#bhattacharye distance

#input p = proposedhistograms
# q = target histrogram

def bhatta_dist(p,q):

    h = [ p , q];
    import math
    import numpy as np


    def mean( hist ):
        mean = 0.0;
        for i in hist:
            mean += i;
        mean/= len(hist);
        return mean;

    def bhatta ( hist1,  hist2):
        # calculate mean of hist1
        h1_ = mean(hist1);

        # calculate mean of hist2
        h2_ = mean(hist2);

        # calculate score
        score = 0;
        for i in range(len(hist1)):
            score += math.sqrt( abs(hist1[i] * hist2[i]) );
        

        #print (hist1,hist2)#score;
        score = math.sqrt(abs( 1 - ( 1 / math.sqrt(abs(h1_*h2_*8*8) ) ) * score ));
        return score;

    # generate and output scores
    scores = [];
    for i in range(len(h)):
        score = [];
        for j in range(len(h)):
            score.append( bhatta(h[i],h[j]) );
        scores.append(score);

    #print(1 - scores[0][1])
    sum2 = max(scores[0])
    
    scores[0][i] = 1 - (scores[0][i]/sum2)

    #print(scores[0][1])

    return (scores[0][1])
#cv2.imshow(img2)

