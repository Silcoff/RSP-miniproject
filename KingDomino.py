# from math import dist
import cv2 as cv
import numpy as np

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

############################################################################################
# imgnr decides what image is being analyzed
imgnr = 1

board = cv.imread('dataset/Cropped and perspective corrected boards/' + str(imgnr) + '.jpg')
############################################################################################

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


for type in range(6):
    groupCount = 1
    for row in range(boardSize):
        for column in range(boardSize):
            tr = 0
            tc = 0
            if guessArray[row,column] == nameArray[type+1]:
                if row != 0:
                    tr = groups[type, row-1, column]
                if column != 0:
                    tc = groups[type, row, column-1]
                
                if tr != 0:
                    groups[type, row, column] = tr
                elif tc != 0:
                    groups[type, row, column] = tc
                else:
                    groups[type, row, column] = groupCount
                    groupCount = groupCount + 1
                
                if (tr and tc != 0) and (tr != tc):
                    for i in range(boardSize):
                        for j in range(boardSize):
                            if groups[type, i, j] == tc:
                                groups[type, i, j] = tr
                    groupCount = groupCount - 1
                    
    print(nameArray[type+1])
    print(groups[type])



# the "simple" image is rezised before being displayed
simple = cv.resize(simple, (500,500), interpolation=cv.INTER_NEAREST)

# text is added to the "simple" image:
#   guess for the type of space
#   measured color value at space
#   guess ideal (mean) value
#   color-distance to the ideal (smaller is better fit)
for row in range(boardSize):
    for column in range(boardSize):
        cv.putText(simple, guessArray[column,row], (row*100+20, column*100+20), cv.FONT_HERSHEY_PLAIN, 1.0, (0,0,0), 2)
        cv.putText(simple, str(simple[row*100,column*100,0]) + ', ' + str(simple[row*100,column*100,1]) + ', ' + str(simple[row*100,column*100,2]), (column*100+5, row*100+40), cv.FONT_HERSHEY_PLAIN, 0.8, (0,0,0), 1)
        cv.putText(simple, str(meanArrayBGR[guessIndexArray[row,column]]), (column*100+5, row*100+60), cv.FONT_HERSHEY_PLAIN, 0.8, (0,0,0), 1)
        cv.putText(simple, str(distArray[column,row]), (row*100+5, column*100+80), cv.FONT_HERSHEY_PLAIN, 0.8, (0,0,0), 1)
        
# debugging print guess array
# print(guessArray)

# display "simple" and starting board
cv.imshow("Simple", simple)
cv.imshow("Board", board)
cv.waitKey(0)