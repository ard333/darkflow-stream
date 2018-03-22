import cv2
import os
from darkflow.net.build import TFNet

options = {"model": "/home/ardiansyah/darkflow/cfg/yolo.cfg", "load": "/home/ardiansyah/darkflow/yolo.weights", "threshold": 0.1}

vcap = cv2.VideoCapture("https://videos3.earthcam.com/fecnetwork/5751.flv/playlist.m3u8")
tfnet = TFNet(options)

while(1):
	ret, frame = vcap.read()
	results = tfnet.return_predict(frame)
	label_count = {}
	for result in results:
		label = result['label']
		if label in label_count:
			label_count[label] = label_count[label]+1
		else:
			label_count[label] = 1
		cv2.rectangle(frame,(result['topleft']['x'], result['topleft']['y']), (result['bottomright']['x'], result['bottomright']['y']),(255, 255, 255), 3)
		cv2.putText(frame, label,(result['topleft']['x'], result['topleft']['y']), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
	os.system('clear')
	print(label_count)
	cv2.imshow('DEMO', frame)
	cv2.waitKey(1)