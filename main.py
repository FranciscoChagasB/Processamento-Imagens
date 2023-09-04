import cv2
import numpy as np
from skimage import filters

imagem_retina = cv2.imread('C:\\Users\\Francisco\\Desktop\\Novapasta\\0.png', cv2.IMREAD_GRAYSCALE)

def realce_contraste(imagem, alpha, beta):
    # Aplicar a transformação linear
    imagem_realce = cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)
    
    return imagem_realce

# Aplique o realce de contraste à imagem
imagem_realce = realce_contraste(imagem_retina, alpha=2, beta=1)
imagem = cv2.cvtColor(imagem_realce, cv2.COLOR_BGR2GRAY)


cv2.imshow('Imagem com Contraste Realçado', imagem_realce)
# Espere até que uma tecla seja pressionada e depois feche as janelas
cv2.waitKey(0)
cv2.destroyAllWindows()

# Converta a imagem para escala de cinza
imagem_gray = imagem_retina

# Aplique o filtro Frangi
frangi_image = filters.frangi(imagem_gray)

# Aplique a binarização (ajuste o limiar conforme necessário)
threshold_value = 0.01  # Ajuste este valor conforme necessário
binary_image = frangi_image > threshold_value


# Redução de ruído com operação de abertura
kernel = np.ones((2, 2), np.uint8)  # Tamanho do kernel (pode ajustar conforme necessário)
binary_image_clean = cv2.morphologyEx(binary_image.astype(np.uint8), cv2.MORPH_OPEN, kernel)

cv2.imshow('Imagem Binarizada com Redução de Ruído', binary_image_clean.astype(np.uint8) * 255)  # Converta para 8 bits

# Exiba a imagem original, a imagem filtrada e a imagem binarizada
cv2.imshow('Imagem Original', imagem_retina)
cv2.imshow('Imagem Filtrada (Frangi)', frangi_image)
cv2.imshow('Imagem Binarizada', binary_image.astype(np.uint8) * 255)  # Converta para 8 bits

# Aguarde até que uma tecla seja pressionada e, em seguida, feche as janelas
cv2.waitKey(0)
cv2.destroyAllWindows()

def filtro(image): 
    # Separando as cores
    r, g, b = cv2.split(image)
    
    # Equalizando - aumentando a nitidez
    g = cv2.equalizeHist(g)
    b = cv2.equalizeHist(b)

    #juntando os canais
    imagem_equalizada = cv2.merge([r,g,b])
    return r, g, b, imagem_equalizada

r, g, b, imagem_filtrada = filtro(imagem_retina)