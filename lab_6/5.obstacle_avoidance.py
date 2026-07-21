import numpy as np 
import cv2

# --- CONFIGURAÇÃO DOS CAMINHOS ---
CAMINHO_IMG_ESQ = "data/im0.png" 
CAMINHO_IMG_DIR = "data/im1.png"

max_depth = 400 
min_depth = 50 
depth_thresh = 100.0 # Limite de proximidade do objeto

disparity = None
depth_map = None
output_canvas = None

cv_file = cv2.FileStorage("data/depth_estimation_params_py.xml", cv2.FILE_STORAGE_READ)
numDisparities = int(cv_file.getNode("numDisparities").real())
blockSize = int(cv_file.getNode("blockSize").real())
preFilterType = int(cv_file.getNode("preFilterType").real())
preFilterSize = int(cv_file.getNode("preFilterSize").real())
preFilterCap = int(cv_file.getNode("preFilterCap").real())
textureThreshold = int(cv_file.getNode("textureThreshold").real())
uniquenessRatio = int(cv_file.getNode("uniquenessRatio").real())
speckleRange = int(cv_file.getNode("speckleRange").real())
speckleWindowSize = int(cv_file.getNode("speckleWindowSize").real())
disp12MaxDiff = int(cv_file.getNode("disp12MaxDiff").real())
minDisparity = int(cv_file.getNode("minDisparity").real())
M = cv_file.getNode("M").real()
cv_file.release()

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("Distância estimada: %.2f cm" % depth_map[y,x])  

cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp', 700, 700)
cv2.namedWindow('output_canvas', cv2.WINDOW_NORMAL)
cv2.resizeWindow('output_canvas', 700, 700)
cv2.setMouseCallback('disp', mouse_click)
cv2.setMouseCallback('output_canvas', mouse_click)

stereo = cv2.StereoBM_create()

def obstacle_avoid():
    mask = cv2.inRange(depth_map, 10, depth_thresh)
    if np.sum(mask)/255.0 > 0.01*mask.shape[0]*mask.shape[1]:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        if cv2.contourArea(cnts[0]) > 0.01*mask.shape[0]*mask.shape[1]:
            x, y, w, h = cv2.boundingRect(cnts[0])
            mask2 = np.zeros_like(mask)
            cv2.drawContours(mask2, cnts, 0, (255), -1)
            depth_mean, _ = cv2.meanStdDev(depth_map, mask=mask2)
            
            cv2.putText(output_canvas, "WARNING !", (x+5, y-40), 1, 2, (0,0,255), 2, 2)
            cv2.putText(output_canvas, "Object at", (x+5, y), 1, 2, (100,10,25), 2, 2)
            cv2.putText(output_canvas, "%.2f cm" % depth_mean[0][0], (x+5, y+40), 1, 2, (100,10,25), 2, 2)
    else:
        cv2.putText(output_canvas, "SAFE!", (100, 100), 1, 3, (0,255,0), 2, 3)

imgL_orig = cv2.imread(CAMINHO_IMG_ESQ)
imgR_orig = cv2.imread(CAMINHO_IMG_DIR)

imgL = cv2.resize(imgL_orig, (640, 480))
imgR = cv2.resize(imgR_orig, (640, 480))

output_canvas = imgL.copy()

imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

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

disparity_raw = stereo.compute(imgL_gray, imgR_gray)
disparity_float = disparity_raw.astype(np.float32)
disparity = (disparity_float/16.0 - minDisparity)/numDisparities

# Blindagem contra divisão por zero onde não há disparidade
depth_map = M / (disparity + 1e-6) 

mask_temp = cv2.inRange(depth_map, min_depth, max_depth)
depth_map = cv2.bitwise_and(depth_map, depth_map, mask=mask_temp)

obstacle_avoid()

cv2.imshow("disp", disparity)
cv2.imshow("output_canvas", output_canvas)

print("Processamento concluído. Dê duplo-clique nas janelas para ver a distância!")
print("Aperte qualquer tecla na imagem para sair.")

# Espera infinita até você pressionar alguma tecla para fechar
cv2.waitKey(0) 
cv2.destroyAllWindows()