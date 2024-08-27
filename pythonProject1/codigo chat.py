import cv2
import face_recognition

# Caminho para a imagem de referência
image_path = "referencia.jpg"

try:
    # Carregar a imagem de referência
    imagem_referencia = face_recognition.load_image_file(image_path)
    print("Imagem de referência carregada com sucesso!")

    # Tentar gerar a codificação facial
    codificacao_referencia = face_recognition.face_encodings(imagem_referencia)

    # Verificar se algum rosto foi encontrado e codificado
    if len(codificacao_referencia) > 0:
        print("Codificação facial gerada com sucesso!")
        # Exibir a imagem de referência
        cv2.imshow("Imagem de Referência", imagem_referencia[:, :, ::-1])  # Converte de RGB para BGR para exibir no OpenCV
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Nenhum rosto encontrado na imagem de referência. Verifique a imagem.")

except Exception as e:
    print(f"Ocorreu um erro ao carregar a imagem ou gerar a codificação: {e}")
