import cv2
import face_recognition


# Carrega o classificador em cascata Haar para detecção facial
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carrega uma imagem de referencia e obtém a codificação facial
reference_image = face_recognition.load_image_file("referencia.jpg")
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# URL do stream RTSP da câmera Intelbras
rtsp_url = "rtsp://usuario:senha@ip_da_camera:porta/stream"

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(3)

if not cap.isOpened():
    print("Não foi possível abrir a câmera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar imagem")
        break

    # Converte a imagem capturada para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecta rostos na imagem
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)


    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        # Compara a codificação do rosto detectado com a codificação de referência
        matches = face_recognition.compare_faces([reference_encoding], face_encoding)
        name = "Desconhecido"

        if True in matches:
            name = "Erick"

        # Desenha um retângulo ao redor do rosto detectado e coloca um texto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Exibe o frame com os rostos detectados
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
