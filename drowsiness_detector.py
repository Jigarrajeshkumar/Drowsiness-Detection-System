#imports
import face_recognition                      # for face landmarks detection
import numpy as np                           #for numpy arry
import pygame                                # for audio(alarm)
import cv2                                   #for video camera
from scipy.spatial import distance as dist   #for finding the eculedian distance betweem eyelids

  
pygame.init()
MIN_EAR=0.20              # minimum eye aspect ratio 
EYE_AR_CONS_FRAMES=10   #if the aspect raio of eye is less than MIN_EAR for consecutive 10 
                        #frames then alarm should we fired
COUNTER=0               #to count the consecutive in which aspect ratio is less then the MIN_AR
ALARM=False

def alarm():
    pygame.mixer.music.load("alarm.mp3")
    pygame.mixer.music.play(1)

def eye_aspect_ratio(eye):
    ''' 
        it accept the eye as the parameter 
        and returns the aspect ration of the eye
    '''
    #compute the eculedian distance between the vertical landmarks cooerdinate
    A=dist.euclidean(eye[1],eye[5])  #eculidean func returns the eculedian distance between two points
    B=dist.euclidean(eye[2],eye[4])
    #compute the eculedian distance between the horizontal coordinates of landmark(eye)
    C=dist.euclidean(eye[0],eye[3])

    #compute the eye aspect ratio
    EAR=(A+B)/(2*C)                 #EAR=eye aspect ratio

    return EAR

def main():

    '''
         main function which detect whether the object is conscious or fall asleep
         accept no argument
    '''

    global COUNTER,ALARM
    video_capture=cv2.VideoCapture(0)     #it create the video capture object for the camera
    while True:
        ret,frame=video_capture.read(0)    # read method to read the frames using video_capture object
                                           #ret = receive the boolean value (True,False) (for knowing frame captured correctly)
                                           #frame= receive the next frame captured by camera

        #to get the face correct landmarks

        face_landmarks_list=face_recognition.face_landmarks(frame) # it will collect all the land marks of the face from the given frame
                                                                   #face_landmarks func returns dict of the face feature(like eyes,nose,ears)
        #traversal in the list to get the eyes
        for landmarks in face_landmarks_list:
            left_eye=landmarks['left_eye']
            right_eye=landmarks['right_eye']

            #eye aspect ratio of lefdt eye and the right eye
            left_EAR=eye_aspect_ratio(left_eye)
            right_EAR=eye_aspect_ratio(right_eye)

            #average of the eye aspect ratios of both the eyes
            avg_EAR=(left_EAR+right_EAR)/2

            #converting left_eye right eye value in the numpy array
            lpts=np.array(left_eye)      #left eye points array
            rpts=np.array(right_eye)     #right eye points array
            
            #constructing the polygon lines around the eyes
            cv2.polylines(frame,[lpts],True,(255,255,0),1)
            cv2.polylines(frame,[rpts],True,(255,255,0),1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter

            if avg_EAR <= MIN_EAR:
                COUNTER=COUNTER+1
                if COUNTER>=EYE_AR_CONS_FRAMES:
                    cv2.putText(frame,"ALERT!!!!!",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    if ALARM is False:
                        ALARM=True
                        alarm()
                cv2.putText(frame,"you are falling asleep",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            else:
                COUNTER=0
                ALARM=False

            
            # draw the computed eye aspect ratio on the frame to help
            # with debugging and setting the correct eye aspect ratio
            # thresholds and frame counters
            cv2.putText(frame, "EAR: {:.2f}".format(avg_EAR), (500, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "press Q/q to exit", (430, 430),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            #show the frame
            cv2.imshow('drawsiness detector',frame)


        #to quit the window/to break the loop
        #break the loop if the 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    #to release the video after the loop is break
    video_capture.release()
    #to destroy all the windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()