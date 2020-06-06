import numpy as np 
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []
dataset_path = './data/'
skip = 0


file_name = input("Enter the Name : ")


while True:
	ret,frame = cap.read()

	g_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if(ret == False):
		continue


	cv2.imshow("Gray_Frame",frame)
	faces = face_cascade.detectMultiScale(g_frame,1.3,5)

	if(len(faces)==0):
		continue

	faces = sorted(faces,key = lambda f:f[2]*f[3],reverse = True)


	for face in faces[:1]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		cv2.imshow("Frame",frame)

		offset = 5

		face_selection = frame[y-offset:y+offset+h,x-offset:x+offset+w]
		face_section = cv2.resize(face_selection,(100,100))

		cv2.imshow('Face_Section',face_section)

		skip +=1
		if(skip % 10 ==0 ):
			face_data.append(face_section)
			print(len(face_data))


	key_pressed = cv2.waitKey(1) & 0xFF
	if(key_pressed == ord('q')):
		break

print(face_data)
face_data = np.array(face_data)
print(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))

np.save(dataset_path+file_name+'.npy',face_data)
print("Data Saved at Location : ",dataset_path+file_name+".npy")
cap.release()
cv2.destroyAllWindows()
