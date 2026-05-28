# Laboratório 1

Integrantes:
    Samira Haddad, 11201812350
    Lucas Medeiros da Silva, 11202130277
    Gabriel Intackli Pinto, 11201921426
    


## Exercício 1
Resposta: Na linha 4 temos o comando cv.imread('messi5.jpg',0), ele é utilizado para carregar em memória uma determinada imagem, o primeiro param. dessa função é o nome do arquivo, o segundo é a escala que desejamos ler o arquivo, o cv2.IMREAD_GRAYSCALE ou 0 carrega a imagem em preto e branco (grayscale)


## Exercício 2
Resposta: A percepção de movimento em um vídeo, nada mais é do que uma sequência de imagens mostradas a tela em uma velocidade tão grande que temos a ilusão de movimento conforme a sequência de imagens é exibida.
Conforme vamos aumentando a velocidade de frames (imagens) temos também a impressão de que o movimento está ficando + ou - rápido (podemos notar esse efeito quando alteramos a velocidade de um vídeo no youtube de 1x para 2x, o que muda é simplesmente a frequência de apresentação desses frames e a velocidade do aúdio).
Para deixar o vídeo mais ou menos rápido basta alterar essa "velocidade" de exibição dos frames.
O que possibilita controlar a velocidade de exibição dos frames é justamente a linha 15 do arquivo L__2_video.py (time.sleep(1/25)), nela setamos a quantidade de frames que serão exibidos por segundo (fps), o original está setado em 25 fps (1/25 segundos para cada frame), para aumentar a velocidade para 120 fps podemos mudar essa linha para time.sleep(1/120.0) e para diminuir a velocidade para 10 fps podemos mudar essa linha para time.sleep(1/10.0). 
Reexecutando o arquivo com essas duas alterações podemos perceber que o vídeo big_buck_bunny.mp4 aumenta ou diminui nítidamente sua velocidade



