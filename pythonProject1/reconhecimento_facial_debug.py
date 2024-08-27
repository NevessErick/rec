import cv2
import face_recognition


try:
    # Tente capturar vídeo da webcam (índice 0 ou 1)
    video_capture = cv2.VideoCapture(0)

    # Verificar se a webcam foi aberta corretamente
    if not video_capture.isOpened():
        print("Erro ao acessar a webcam.")
        exit()

    print("Webcam acessada com sucesso! Pressione 'q' para sair.")

    # Carregar a imagem de referência e obter a codificação
    try:
        imagem_referencia = face_recognition.load_image_file("referencia.jpg")
        codificacao_referencia = face_recognition.face_encodings(imagem_referencia)[0]
    except Exception as e:
        print(f"Erro ao carregar a imagem de referência: {e}")
        video_capture.release()
        cv2.destroyAllWindows()
        exit()

    # Listas de rostos e nomes conhecidos
    rostos_conhecidos = [codificacao_referencia]
    nomes_conhecidos = ["Nome da Pessoa"]

    while True:
        try:
            # Capturar frame por frame
            ret, frame = video_capture.read()

            if not ret:
                print("Erro ao capturar o frame.")
                break

            # Converter a imagem de BGR (OpenCV) para RGB (face_recognition)
            rgb_frame = frame[:, :, ::-1]

            # Encontrar todas as localizações de rostos e codificações no frame atual
            try:
                localizacao = face_recognition.face_locations(rgb_frame)
                codificacoes = face_recognition.face_encodings(rgb_frame, localizacao)
            except Exception as e:
                print(f"Erro ao processar a imagem: {e}")
                break

            # Percorrer todos os rostos detectados no frame
            for (top, right, bottom, left), codificacao in zip(localizacao, codificacoes):
                # Comparar a codificação do rosto detectado com as codificações conhecidas
                resultado = face_recognition.compare_faces(rostos_conhecidos, codificacao)
                nome = "desconhecido"

                if True in resultado:
                    primeiro_match = resultado.index(True)
                    nome = nomes_conhecidos[primeiro_match]

                # Desenhar um retângulo ao redor do rosto e adicionar o nome
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, nome, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Exibir o frame com as detecções
            cv2.imshow('Video', frame)

            # Verificar se a tecla 'q' foi pressionada para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Saindo...")
                break
        except Exception as e:
            print(f"Ocorreu um erro no loop principal: {e}")
            break

except Exception as e:
    print(f"Erro fatal: {e}")

finally:
    # Liberar a captura de vídeo e fechar as janelas
    if 'video_capture' in locals():
        video_capture.release()
    cv2.destroyAllWindows()
