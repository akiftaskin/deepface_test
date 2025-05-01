from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# Görüntüyü yükle
img_path = "test.jpg"
img = cv2.imread(img_path)

# Görseldeki yüzleri analiz et
detections = DeepFace.analyze(img_path=img_path, actions=["emotion"], enforce_detection=False)

# Eğer birden fazla yüz varsa liste olarak döner
if not isinstance(detections, list):
    detections = [detections]

# Yüzlere kutu çiz ve gülümsüyor mu kontrol et
for face in detections:
    x, y, w, h = face["region"]['x'], face["region"]['y'], face["region"]['w'], face["region"]['h']
    emotion = face["dominant_emotion"]

    if emotion == "happy":
        # Kutu çiz
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Yazı yaz
        cv2.putText(img, "Smile", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Sonucu göster
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Gülümseyen Kişiler")
plt.show()