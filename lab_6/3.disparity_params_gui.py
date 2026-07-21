import numpy as np 
import cv2

# --- CONFIGURAÇÃO DOS CAMINHOS ---
CAMINHO_IMG_ESQ = "data/im0.png" 
CAMINHO_IMG_DIR = "data/im1.png"

# Lendo as imagens estáticas e redimensionando para o padrão
imgL_orig = cv2.imread(CAMINHO_IMG_ESQ)
imgR_orig = cv2.imread(CAMINHO_IMG_DIR)

if imgL_orig is None or imgR_orig is None:
    print("Erro: Imagens não encontradas. Verifique o caminho.")
    exit()

imgL = cv2.resize(imgL_orig, (640, 480))
imgR = cv2.resize(imgR_orig, (640, 480))

imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

def nothing(x):
    pass

cv2.namedWindow('Controles', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Controles', 600, 600) 
cv2.namedWindow('Mapa de Disparidade', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Mapa de Disparidade', 640, 480) 

cv2.createTrackbar('numDisparities','Controles',3,17,nothing)
cv2.createTrackbar('blockSize','Controles',5,50,nothing)
cv2.createTrackbar('preFilterType','Controles',1,1,nothing)
cv2.createTrackbar('preFilterSize','Controles',2,25,nothing)
cv2.createTrackbar('preFilterCap','Controles',31,62,nothing)
cv2.createTrackbar('textureThreshold','Controles',10,100,nothing)
cv2.createTrackbar('uniquenessRatio','Controles',15,100,nothing)
cv2.createTrackbar('speckleRange','Controles',2,100,nothing)
cv2.createTrackbar('speckleWindowSize','Controles',25,100,nothing)
cv2.createTrackbar('disp12MaxDiff','Controles',5,25,nothing)
cv2.createTrackbar('minDisparity','Controles',0,25,nothing)

stereo = cv2.StereoBM_create()

while True:
    # Lendo os valores da janela de 'Controles'
    raw_numDisp = cv2.getTrackbarPos('numDisparities','Controles')
    numDisparities = max(1, raw_numDisp) * 16 
    blockSize = cv2.getTrackbarPos('blockSize','Controles')*2 + 5
    preFilterType = cv2.getTrackbarPos('preFilterType','Controles')
    preFilterSize = cv2.getTrackbarPos('preFilterSize','Controles')*2 + 5
    preFilterCap = max(1, cv2.getTrackbarPos('preFilterCap','Controles'))
    textureThreshold = cv2.getTrackbarPos('textureThreshold','Controles')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','Controles')
    speckleRange = cv2.getTrackbarPos('speckleRange','Controles')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','Controles')*2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','Controles')
    minDisparity = cv2.getTrackbarPos('minDisparity','Controles')
    
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(preFilterType)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)

    # Computando a disparidade (sem o remap, pois a imagem de teste é perfeita)
    disparity = stereo.compute(imgL_gray, imgR_gray)
    disparity_float = disparity.astype(np.float32)
    disparity_norm = (disparity_float/16.0 - minDisparity) / numDisparities
    
    # Prevenção contra artefatos brancos
    disparity_norm = np.clip(disparity_norm, 0, 1)

    disp_vis = (disparity_norm * 255).astype(np.uint8)
    disp_color = cv2.cvtColor(disp_vis, cv2.COLOR_GRAY2BGR)
    
    cv2.putText(disp_color, "Aperte ESC para salvar os parametros", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Mapa de Disparidade", disp_color)

    if cv2.waitKey(50) == 27: 
        break

print("Salvando parâmetros de estimativa de profundidade...")
cv_file = cv2.FileStorage("data/depth_estimation_params_py.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("numDisparities", numDisparities)
cv_file.write("blockSize", blockSize)
cv_file.write("preFilterType", preFilterType)
cv_file.write("preFilterSize", preFilterSize)
cv_file.write("preFilterCap", preFilterCap)
cv_file.write("textureThreshold", textureThreshold)
cv_file.write("uniquenessRatio", uniquenessRatio)
cv_file.write("speckleRange", speckleRange)
cv_file.write("speckleWindowSize", speckleWindowSize)
cv_file.write("disp12MaxDiff", disp12MaxDiff)
cv_file.write("minDisparity", minDisparity)
cv_file.write("M", 39.075) # Valor genérico inicial
cv_file.release()

cv2.destroyAllWindows()