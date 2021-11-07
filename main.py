import streamlit as st
import numpy as np
import cv2

from ocr import getLocalizacaoPlaca, getPlacaFromMascara, getTextFromImagePlaca, grab_contours

st.title('PALPR')
st.subheader('Picinin ALPR (Automatic License/Number Plate Recognition)')

upload_imagem = st.file_uploader('Carrege uma imagem de carro com placa', ['png', 'jpg'])
if upload_imagem is not None:
    file_bytes = np.asarray(bytearray(upload_imagem.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption="Imagem carregada", channels="BGR")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.image(gray, caption="Imagem em cinza")

    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    st.image(blur, caption="Imagem desfocada")

    edged = cv2.Canny(blur, 30, 200)
    st.image(edged, caption="Detecção de Bordas")

    # pega contornos
    conts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    conts = grab_contours(conts)
    conts = sorted(conts, key=cv2.contourArea, reverse=True)[:8]

    localizacao = getLocalizacaoPlaca(conts)

    if localizacao is not None:
        mascara = np.zeros(gray.shape, np.uint8)
        img_placa_mascara = cv2.drawContours(mascara, [localizacao], 0, 255, -1)
        st.image(img_placa_mascara, caption="Mascara da placa detectada")

        img_placa = cv2.bitwise_and(img, img, mask=mascara)
        st.image(img_placa, caption="Placa detectada")

        placa = getPlacaFromMascara(gray, mascara)
        st.image(placa, caption="Placa detectada recorte")

        text = getTextFromImagePlaca(placa)
        st.title(text)
