import cv2 as cv
import numpy as np

def whiteYellow(obs):
    white = cv.inRange(obs,(250,250,250),(255,255,255),obs)
    yellow = cv.inRange(obs,(220,220,0),(255,255,190),obs)
    res = white+yellow
    #res[res != 0] = 100
    res = res.T
    res = np.array([res,res,res]).T
    return res

def edgeDetect(obs):
    can = cv.blur(cv.Canny(obs,200,255),(7,7))
    t = 30
    can[can > t] = 255
    can[can <= t] = 0
    can = can.T
    edges = cv.Sobel(obs,cv.CV_8U,1,1,ksize=5) + np.array([can,can,can]).T
    edges = cv.blur(edges,(5,5))
    edges = cv.inRange(edges,(100,100,100),(255,255,255),edges)
    edges = edges.T
    edges = np.array([edges,edges,edges]).T
    #edges[edges != 0] = 110
    return edges

def preProcessRGB(obs):
    edges = whiteYellow(obs) + edgeDetect(obs)

    # Testobs = cv.bitwise_not(Testobs)
    # Testobs[Testobs > (150,150,150)] = 255
    obs = obs.astype(np.int64) - (150, 150, 150)

    obs = obs + edges

    obs[obs < 0] = 0
    obs[obs > 255] = 255
    obs = obs.astype(np.uint8)
    return obs

def detectYellow(obs):
    # flip open cv is bgr real life is rgb
    #yellow = cv.inRange(obs, (0, 210, 210), (160, 255, 255), obs)
    yellow = cv.inRange(obs, (210, 210, 0), (255, 255, 160), obs)
    for x in range(10):
        yellow = cv.blur(yellow, (3, 3))
        yellow[yellow > 10] = 255
    yellow = cv.blur(yellow, (33, 33))
    # #yellow[yellow > 10] = 255
    # res = yellow.T
    # res = np.array([res, res, res]).T
    return yellow

def cheat(obs):

    #obs = cv.rotate(obs,-45)
    yw = whiteYellow(obs)
    yw[yw > 200] = 190

    obs = obs.astype(np.int64)
    sh = obs.shape
    #obs = cv.inRange(obs,(0,0,0),(200,200,200))
    blue = obs[:,:,2]
    blue[blue > 200] = 0
    obs[:,:,0] = blue

    blue[blue > 200] = 55
    obs[:, :, 1] = blue

    b = cv.blur(obs,(11,11))
    u = np.median(b,axis=2)#b[:,:,0]+b[:,:,1]+b[:,:,2]
    u = u.T
    #u = u.astype(np.uint8).T
    obs = abs(obs - np.array([u,u,u]).T)
    obs = cv.inRange(obs,(0,0,0),(20,20,20),obs)


    obs = obs.T
    obs = np.array([obs,obs,obs]).T

    cent = int(obs.shape[1] / 2)
    obs[:, cent- 10:cent + 10, :] = 69

    obs = obs+yw

    obs = obs.astype(np.uint8)
    return obs


if __name__ == '__main__':
    testPath = '/home/sami/Pictures/3.png'
    Testobs = cv.imread(testPath)

    #Testobs = preProcessRGB(Testobs)
    #Testobs = cheat(Testobs)
    Testobs = detectYellow(Testobs)

    #Testobs[Testobs < (Testobs[0,:,:].mean(),Testobs[1,:,:].mean(),Testobs[2,:,:].mean())] = 0
    #Testobs = edgeDetect(Testobs)
    cv.imshow('test',Testobs)
    cv.waitKey(0)