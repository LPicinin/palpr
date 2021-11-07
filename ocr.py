import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt


def getLocalizacaoPlaca(conts):
    localizacao = None
    for c in conts:
        peri = cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, 0.02 * peri, True)
        if cv2.isContourConvex(aprox):
            if len(aprox) == 4:
                localizacao = aprox
                break
    return localizacao


def getPlacaFromMascara(gray, mascara):
    (y, x) = np.where(mascara == 255)
    (inicioX, inicioY) = (np.min(x), np.min(y))
    (fimX, fimY) = (np.max(x), np.max(y))
    return gray[inicioY:fimY, inicioX:fimX]


def getTextFromImagePlaca(placa):
    config_tesseract = "--tessdata-dir tessdata --psm 6"
    texto = pytesseract.image_to_string(placa, lang="por", config=config_tesseract)
    return "".join(caractere for caractere in texto if caractere.isalnum())

