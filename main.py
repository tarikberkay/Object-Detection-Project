

import cv2

####### Resimden #######
def ImgFile():
   img = cv2.imread('person.png')  # Burada 'person.png' adlı resmi okuyoruz.

   classNames = []
   classFile = 'coco.names'   # sınıf adlarının bulunduğu 'coco.names' adlı dosyayı okuyoruz.

   with open(classFile, 'rt') as f:
      classNames = f.read().rstrip('\n').split('\n')

   configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # Modeli yapılandırma dosyası
   weightpath = 'frozen_inference_graph.pb'   # Modelin eğitilmiş ağırlık dosyası

   net = cv2.dnn_DetectionModel(weightpath, configPath)
   net.setInputSize(320 , 230)   # Giriş boyutunu ayarlıyoruz.
   net.setInputScale(1.0 / 127.5)   # Giriş ölçeklemesini ayarlıyoruz.
   net.setInputMean((127.5, 127.5, 127.5))    # Giriş ortalama değerlerini ayarlıyoruz.
   net.setInputSwapRB(True)  # Renk kanallarını RGB'den BGR'ye dönüştürüyoruz

   classIds, confs, bbox = net.detect(img, confThreshold=0.5)    # Nesneleri algılıyoruz ve sınıf kimlikleri, güven değerleri ve sınırlayıcı kutuları alıyoruz.
   print(classIds, bbox)

   for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
      cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)    # Sınırlayıcı kutuları resim üzerine çiziyoruz.
      cv2.putText(img, classNames[classId-1], (box[0] + 10, box[1] + 20), 
                  cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)   # Sınıf adlarını resim üzerine yazıyoruz.


   cv2.imshow('Output', img)  # Sonucu görüntülüyoruz
   cv2.waitKey(0)
######################################

####### Videodan veya Kameradan #######
def Camera():
   cam = cv2.VideoCapture(0)   # Kamerayı başlatıyoruz. (0, varsayılan kamera)

   cam.set(3, 740)   # Görüntü genişliğini ayarlıyoruz.
   cam.set(4, 580)   # Görüntü yüksekliğini ayarlıyoruz.

   classNames = []
   classFile = 'coco.names'

   with open(classFile, 'rt') as f:
      classNames = f.read().rstrip('\n').split('\n')

   configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
   weightpath = 'frozen_inference_graph.pb'

   net = cv2.dnn_DetectionModel(weightpath, configPath)
   net.setInputSize(320 , 230)
   net.setInputScale(1.0 / 127.5)
   net.setInputMean((127.5, 127.5, 127.5))
   net.setInputSwapRB(True)

   while True:
      success, img = cam.read()
      classIds, confs, bbox = net.detect(img, confThreshold=0.5)
      print(classIds, bbox)

      if len(classIds) !=0:
         for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId-1], (box[0] + 10, box[1] + 20), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)


      cv2.imshow('Output', img)
      cv2.waitKey(1)
######################################


## Görüntü veya kamera için ImgFile() fonksiyonunu video ve kamera için Camera() fonksiyonunu çağırıyoruz.
#ImgFile()
Camera()