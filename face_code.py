import numpy as np 
import cv2
import os

############# KNN#################

def distance(X1,X2):
	return np.sqrt(((X1-X2)**2).sum())

def KNN(train, test, K=51):
	print(train[0,:-1])
	dist = []
	for i in range(train.shape[0]):
		ix = train[i,:-1]
		iy = train[i,-1]
		d = distance(test,ix)
		dist.append([d,iy])

	dist = sorted(dist,key = lambda x:x[0])[:K]

	labels = np.array(dist)[:,-1]

	out = np.unique(labels,return_counts = True)
	index = np.argmax([out[1]])

	return out[0][index]


############ DataSet #############

dataset_path = './data/'
face_data = []
labels = []
names={}
class_id = 0

for fx in os.listdir(dataset_path):
	if(fx.endswith('.npy')):
		print("Data Loaded Successfully",dataset_path+fx)
		names[class_id] = fx[:-4]
		data_item = np.load(dataset_path + fx)
		face_data.append(data_item)

		target = class_id * np.ones((data_item.shape[0],))
		class_id +=1
		labels.append(target)

face_dataset = np.concatenate(face_data,axis = 0)
face_labels = np.concatenate(labels,axis = 0).reshape((-1,1))

trainset = np.concatenate((face_dataset,face_labels),axis = 1)

################# Testing ##############


cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
	ret,frame = cap.read()
	if(ret == False):
		continue

	g_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(g_frame,1.3,5)
	

	#faces = sorted(faces,key = lambda f:f[2]*f[3])

	for face in faces:
		x,y,w,h = face

		offset =5
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		out = KNN(trainset,face_section.flatten())

		cv2.putText(frame,names[int(out)],(x,y-10),font,1,(0,255,255),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		eye_section_gray = g_frame[y:y+h, x:x+w]
		#eye_section_gray = cv2.cvtColor(eye_section_gray,cv2.COLOR_BGR2GRAY)
		eyes = eye_cascade.detectMultiScale(eye_section_gray,1.3,5)

		for eye in eyes:
			ex,ey,ew,eh = eye

			cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+eh+ey),(0,255,0),2)

	cv2.imshow("frame",frame)

	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break

cap.release()
cv2.destroyAllWindows()
