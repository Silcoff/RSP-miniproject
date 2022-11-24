# from math import dist
import cv2 as cv
import numpy as np
import math

# mean values calculated in BGR for each type of space, to be used to mach new spaces
castleMean = [79,  115, 119] #0
# castleMean = [57,  84,  85 ] #better for castles whith the cardboard cutout
meadowMean = [27,  138, 99 ] #1
forestMean = [20,  54,  43 ] #2
waterMean  = [131, 77,  25 ] #3
wasteMean  = [52,  89,  101] #4
fieldMean  = [8 ,  149, 173] #5
# mineMean   = [27,  52,  62 ] #6
mineMean   = [23,  44,  53 ] #6 - slightly reduced by 15% to reduce false positives

meanArrayBGR = [castleMean, meadowMean, forestMean, waterMean, wasteMean, fieldMean, mineMean]

# same values but in RGB (not currently in use)
castleMeanRGB = [119,  115, 79] #0
meadowMeanRGB = [99,  138, 27 ] #1
forestMeanRGB = [43,  54,  20 ] #2
waterMeanRGB  = [25, 77,  131 ] #3
wasteMeanRGB  = [101,  89,  52] #4
fieldMeanRGB  = [173 ,  149, 8] #5
mineMeanRGB   = [62,  52,  27 ] #6

meanArrayRGB = [castleMeanRGB, meadowMeanRGB, forestMeanRGB, waterMeanRGB, wasteMeanRGB, fieldMeanRGB, mineMeanRGB]

# corresponding array of names
nameArray = ["Castle", "Meadow", "Forest", "Water", "Wastes", "Field", "Mine"]

# test array used in debugging crown counting
testCrown = [[0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0],
             [0, 2, 0, 2, 1],
             [0, 1, 0, 1, 0]]

# importing the crown templates used in template matching, one for each orientation
template0 = cv.imread('Crown_Template.png',0)
template90 = cv.imread('Crown_Template90.png',0)
template180 = cv.imread('Crown_Template180.png',0)
template270 = cv.imread('Crown_Template270.png',0)
w, h = template0.shape[::-1]

# correct score for boards 1 - 10
# 1  = 36 +0
# 2  = 43 -1
# 3  = 52 +0
# 4  = 42 -13
# 5  = 36 +3
# 6  = 43 +1
# 7  = 52 -17
# 8  = 42 -13
# 9  = 45 -4
# 10 = 37 +7
#              1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
trueScores = [36, 43, 52, 42, 36, 42, 52, 42, 45, 37, 49, 22, 45, 38, 49, 22, 40, 60, 36, 52]
############################################################################################
# imgnr decides what image is being analyzed
imgnr = int(input('Load image: '))
threshold = 0.6
debug = False
board = cv.imread('dataset/Cropped and perspective corrected boards/' + str(imgnr) + '.jpg')
############################################################################################

# grayscale convertion of the board used in template matching
grayBoard = cv.cvtColor(board, cv.COLOR_BGR2GRAY)

# constant that defines the maximum sidelength of the board
boardSize = 5

# simple is used to hold the mean value colors of each space and gets displayed at the end
simple = np.zeros((boardSize,boardSize,3), dtype='uint8')

# array that holds the guess for each corresponding space
guessArray = np.zeros((boardSize,boardSize), dtype='U16')

# array that holds the index of the guess in each space
guessIndexArray = np.zeros((boardSize,boardSize), dtype='uint8')

# array that holds the color-distance to the mached type for each space
distArray = np.zeros((boardSize,boardSize), dtype='uint8')

# arrays used to hold the groupings of each type
groups = np.zeros((6,boardSize,boardSize))
group = np.zeros((boardSize,boardSize))
scores = np.zeros((6,boardSize*boardSize))

##### IDENTIFY TILES #######################################################################

# for loop that goes through the rows and columns of the board.
for row in range(boardSize):
    for column in range(boardSize):
        # each space is isolated as its own image
        space = board[row*100:(row+1)*100, column*100:(column+1)*100]

        # the space gets split into color channels
        b,g,r = cv.split(space)

        # the average of each color channels gets added to the "simple" array 
        simple[row, column, 0] = b.mean()
        simple[row, column, 1] = g.mean()
        simple[row, column, 2] = r.mean()

        # these variables are used in finding the minimum distance to a space type, and its index
        minDist = 1000
        minIndex = 0
        # for loop going through all the space types and finding what type has the smallest distance
        for i in range(7):
            dist = np.linalg.norm(meanArrayBGR[i] - simple[row,column])
            if dist < minDist:
                # print(dist)
                minDist = dist
                minIndex = i
        
        # the arrays record the minimum distance, its corresponding index and what guess is made
        distArray[row,column] = minDist
        guessIndexArray[row,column] = minIndex
        guessArray[row,column] = nameArray[minIndex]

# variable used in iterating group numbers
groupCount = 1
# go through each tile type
for type in range(6):
    # go through the rows and columns if the board
    for row in range(boardSize):
        for column in range(boardSize):
            # variables for storing content of the spaces that are checked, don't remember why i named them that
            tr = 0
            tc = 0
            # only look at a space if it contains the current type
            if guessArray[row,column] == nameArray[type+1]:
                # store the content (the group) of the space above and the space to the left, unless it...
                # ...is on the edge
                if row != 0:
                    tr = groups[type, row-1, column]
                if column != 0:
                    tc = groups[type, row, column-1]
                
                # if the adjacent space isn't empty, set current space equal to that space (join that group)...
                # ...otherwise, create a new group and move the group counter up by one
                if tr != 0:
                    groups[type, row, column] = tr
                elif tc != 0:
                    groups[type, row, column] = tc
                else:
                    groups[type, row, column] = groupCount
                    groupCount = groupCount + 1
                
                # if the spaces above and to the left are different, then set all spaces of one group equal...
                # ...to the other (merge the groups)
                if (tr and tc != 0) and (tr != tc):
                    for i in range(boardSize):
                        for j in range(boardSize):
                            if groups[type, i, j] == tc:
                                groups[type, i, j] = tr

    # debugging printout of each type and corresponding groups
    if debug:
        print(nameArray[type+1])
        print(groups[type])

# all groups are added togehter in one array
for type in range(6):
    group += groups[type]

##### IDENTIFY CROWNS ######################################################################

# empty array for storing crown locations
crowns = np.zeros((5,5))

# template matching using each of the rotated crown templates
res0 = cv.matchTemplate(grayBoard,template0,cv.TM_CCOEFF_NORMED)
res90 = cv.matchTemplate(grayBoard,template90,cv.TM_CCOEFF_NORMED)
res180 = cv.matchTemplate(grayBoard,template180,cv.TM_CCOEFF_NORMED)
res270 = cv.matchTemplate(grayBoard,template270,cv.TM_CCOEFF_NORMED)

# find where in the template matched image, the match is over the threshhold
# this template match is for the up-right oriented crown template, the process is repeated for the others
loc0 = np.where( res0 >= threshold)

# variable used in order to identify duplicate crowns
crownCounter = 0

# go through all locations found in the template matching
for pt in zip(*loc0[::-1]):
    # boolean value, used in deciding wheter to count a crown, starts as true
    countThis = True
    # go through all the crowns counted so far
    for i in range(crownCounter):
        # calculate the distance to a given crown in x and y
        xDist = abs(loc0[0][i] - pt[1])
        yDist = abs(loc0[1][i] - pt[0])
        # if the crown is too close, don't count it
        if (abs(loc0[0][i] - pt[1]) < 5) and (abs(loc0[1][i] - pt[0]) < 5):
            countThis = False
    # add one to crown counter
    crownCounter += 1
    # if the crown is not found to be a duplicate, add it to the crowns array
    if countThis:
        cv.rectangle(board, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        crowns[math.floor(pt[1]/100),math.floor(pt[0]/100)] += 1

# the same procedure is used for the 90, 180 and 270 degree templates as for the up-right one
crownCounter = 0
loc90 = np.where( res90 >= threshold)
for pt in zip(*loc90[::-1]):
    countThis = True
    for i in range(crownCounter):
        if (abs(loc90[0][i] - pt[1]) < 5) and (abs(loc90[1][i] - pt[0]) < 5):
            countThis = False
    crownCounter += 1
    if countThis:
        cv.rectangle(board, pt, (pt[0] + h, pt[1] + w), (0,0,255), 2)
        crowns[math.floor(pt[1]/100),math.floor(pt[0]/100)] += 1

crownCounter = 0
loc180 = np.where( res180 >= threshold)
for pt in zip(*loc180[::-1]):
    countThis = True
    for i in range(crownCounter):
        if (abs(loc180[0][i] - pt[1]) < 5) and (abs(loc180[1][i] - pt[0]) < 5):
            countThis = False
    crownCounter += 1
    if countThis:
        cv.rectangle(board, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        crowns[math.floor(pt[1]/100),math.floor(pt[0]/100)] += 1

crownCounter = 0
loc270 = np.where( res270 >= threshold)
for pt in zip(*loc270[::-1]):
    countThis = True
    for i in range(crownCounter):
        if (abs(loc270[0][i] - pt[1]) < 5) and (abs(loc270[1][i] - pt[0]) < 5):
            countThis = False
    crownCounter += 1
    if countThis:
        cv.rectangle(board, pt, (pt[0] + h, pt[1] + w), (0,0,255), 2)
        crowns[math.floor(pt[1]/100),math.floor(pt[0]/100)] += 1

# all crown locations are combined into a singe array.
crownLoc = [loc0[:], loc90[:], loc180[:], loc270[:]]

# prints out the final groups
print("groups:")
print(group)

# prints out the final crowns
print("crowns:")
print(crowns[:])

# final score is calculated by going through the crowns array, and multipying by the amount of spaces in...
# ...its corresponding group
score = 0
for row in range(boardSize):
    for column in range(boardSize):
        if crowns[row][column] != 0:
            target = group[row, column]
            targetCount = np.count_nonzero(group == target)
            score = score + (targetCount * crowns[row][column])
            
# final score is printed out
print("The calculated score is: ", score)
if imgnr-1 < len(trueScores):
    print("The correct score is:    ", trueScores[imgnr-1])
    print("The difference is:       ", score - trueScores[imgnr-1])
else:
    print("The correct score is unknown")


# the "simple" image is rezised before being displayed
simple = cv.resize(simple, (500,500), interpolation=cv.INTER_NEAREST)

# text is added to the "simple" image:
# - guess for the type of space
# - measured color value at space
# - guess ideal (mean) value
# - color-distance to the ideal (smaller is better fit)
for row in range(boardSize):
    for column in range(boardSize):
        cv.putText(simple, guessArray[column,row], (row*100+20, column*100+20), cv.FONT_HERSHEY_PLAIN, 1.0, (0,0,0), 2)
        cv.putText(simple, str(simple[row*100,column*100,0]) + ', ' + str(simple[row*100,column*100,1]) + ', ' + str(simple[row*100,column*100,2]), (column*100+5, row*100+40), cv.FONT_HERSHEY_PLAIN, 0.8, (0,0,0), 1)
        cv.putText(simple, str(meanArrayBGR[guessIndexArray[row,column]]), (column*100+5, row*100+60), cv.FONT_HERSHEY_PLAIN, 0.8, (0,0,0), 1)
        cv.putText(simple, str(distArray[column,row]), (row*100+5, column*100+80), cv.FONT_HERSHEY_PLAIN, 0.8, (0,0,0), 1)
        

# display "simple" and starting board
cv.imshow("Simple", simple)
cv.imshow("Board", board)
cv.waitKey(0)
