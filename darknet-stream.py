import cv2
from darkflow.net.build import TFNet

options = {"model": "/home/apps/darkflow/cfg/yolo.cfg", "load": "/home/apps/darkflow/yolo.weights", "threshold": 0.1}

vcap = cv2.VideoCapture("https://videos3.earthcam.com/fecnetwork/5751.flv/playlist.m3u8")
tfnet = TFNet(options)

while(1):
        ret, frame = vcap.read()
        result = tfnet.return_predict(frame)
        for x in result:
                print(x['label'])
        #cv2.imshow('VIDEO', result)
        #cv2.waitKey(1)
