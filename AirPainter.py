import cv2
import mediapipe as mp
import time
import numpy as np

vid = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 1, 1, 0.6, 0.5)
mpDraw = mp.solutions.drawing_utils


colors = ['red', 'yellow', 'blue']

for i in range(3):
	colors[i] = cv2.resize(cv2.imread('{}.jpg'.format(colors[i])), (100,100))


def findLandmarks(img):
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = hands.process(img_rgb)

	h, w, c = img.shape
	landmarks = []
	if results.multi_hand_landmarks:
		for Lms in results.multi_hand_landmarks:
			for idx, i in enumerate(Lms.landmark):
				cx = int(i.x * w)
				cy = int(i.y * h)
				#print(idx, cx, cy)
				landmarks.append([idx, cx, cy])
	return landmarks


def drawLandmarks(img):
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = hands.process(img_rgb)

	h, w, c = img.shape
	#img = np.zeros((h,w,c))
	if results.multi_hand_landmarks:
		for Lms in results.multi_hand_landmarks:
			for idx, i in enumerate(Lms.landmark):
				cx, cy = int(i.x * w), int(i.y * h)
				#print(idx, cx, cy)
			mpDraw.draw_landmarks(img, Lms, mpHands.HAND_CONNECTIONS)
	return img


IndexPos = [[0,0, (43, 28, 233)]]

cols = {
	'red': tuple(colors[0][0][0][:]),
	'yellow': tuple(colors[1][0][0][:]),
	'blue': tuple(colors[2][0][0][:]),
}

color = (43, 28, 233)
canvas = np.zeros((480, 640, 3))
dimx, dimy = 0, 0
while True:

	grabbed, img = vid.read()
	img = cv2.flip(img, 1)
	lms = findLandmarks(img)
	img = drawLandmarks(img)

	try:
		xy1, xy2 = lms[8][1:], lms[12][1:]
		
	except IndexError:
		pass

	
	
	try: 
		if 0<xy2[1]<100:
			if 0<=xy2[0]<=100:
				color = (43, 28, 233)
			elif 100<=xy2[0]<=200:
				color = (42, 207, 234)
			elif 200<=xy2[0]<=300:
				color = (227, 160, 68)

		if (xy2[1] - xy1[1]) > 40:
			xy1.append(color)
			p_t_1=IndexPos[-1]
			n_pt = [int((xy1[0] + 2*p_t_1[0])/3),int((xy1[1] + 2*p_t_1[1])/3)]
			n_pt.append(color)
			IndexPos.append(n_pt)
			n_pt = [int((2*xy1[0] + p_t_1[0])/3),int((2*xy1[1] + p_t_1[1])/3)]
			n_pt.append(color)
			IndexPos.append(n_pt)
			IndexPos.append(xy1)

	except NameError:
		pass

	if len(IndexPos) > 2:
		for i in IndexPos:
			cv2.circle(img, (i[0], i[1]), 6, i[2], -1)


	img[0:100, 0:100, :] = colors[0]
	img[0:100, 100:200, :] = colors[1]
	img[0:100, 200:300, :] = colors[2]


	scale = 130
	dim = (int((img.shape[1]*scale/100)), int((img.shape[0]*scale/100)))
	dimx, dimy = dim[0], dim[1]
	img = cv2.resize(img, dim)
	cv2.imshow('Video', img)
	if cv2.waitKey(1) & 0xFF==ord('q'):
		vid.release()
		cv2.destroyAllWindows()
		break

img = np.zeros((dimx, dimy, 3), dtype = np.uint8)
if len(IndexPos) > 2:
	for i in IndexPos:
		cv2.circle(img, (i[0], i[1]), 6, i[2], -1)


cv2.imshow('black image', img)
cv2.waitKey(0)

cv2.imwrite("output_drawing.jpg", img)  


# def Identify_Image(img):
# 	pass



