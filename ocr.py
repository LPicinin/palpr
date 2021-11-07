import cv2
import numpy as np
from easyocr import Reader
reader = Reader(['en', 'pt'], gpu=True)


def grab_contours(cnts):
    if len(cnts) == 2:
        cnts = cnts[0]
    elif len(cnts) == 3:
        cnts = cnts[1]
    else:
        raise Exception("Contours tuple must have length 2 or 3")
    return cnts


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
    return reader.readtext(placa, detail=0)
