import cv2
import mediapipe as mp
import time

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraws=mp.solutions.drawing_utils
#maxHands= max num of hands,modelComplex=model complexity,
#detectionCon=detection Confidence,TrackingCon=Tracking Confidence
class HandDetector():
    def __init__(self,mode=False,maxHands=2,modelComplex=1,detectionCon=0.5,trackingCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.modelComplex=modelComplex
        self.trackingCon=trackingCon
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,
                                      self.maxHands,
                                      self.modelComplex,
                                      self.detectionCon,
                                      self.trackingCon)
        self.mpDraws=mp.solutions.drawing_utils
    
    def findHands(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                     self.mpDraws.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self,img,handNo=0,Draw=True):
        lmList=[]

        if self.results.multi_hand_landmarks:
                myHand=self.results.multi_hand_landmarks[handNo]
                # id for index,lm for landmark
                for id,lm in enumerate(myHand.landmark):
                    #print(id,lm)
                    height,width,channel=img.shape
                    cx,cy=int(lm.x*width),int(lm.y*height)
                    lmList.append([id,cx,cy])
                    #if id==0:
                    if Draw:
                        cv2.circle(img,(cx,cy),15,cv2.FILLED)
        return lmList
def main():
    cap=cv2.VideoCapture(0)
    success,img=cap.read()
    detector=HandDetector()

    pTime=0
    while True:
        success,img=cap.read()
        img=detector.findHands(img)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime  
        lmList=detector.findPosition(img)
    #put text parameters-img,dimension placement,type of font ,scale, rgb , thickness 
        cv2.putText(img,f"fps:{str(int(fps))}",(10,24),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1) 

        cv2.imshow("image",img)
        cv2.waitKey(1)


if __name__=="__main__":
    main()
