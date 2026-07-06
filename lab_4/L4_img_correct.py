import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# ==============================================================================
# PARTE 1: CALIBRAÇÃO DINÂMICA DA CÂMERA
# ==============================================================================
print("1. Iniciando a detecção do tabuleiro e calibração...")

# Definições do tabuleiro
CHECKERBOARD = (6, 8) 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = [] 

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

images = glob.glob('samples/*.jpg') 

shape_gray = None

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape_gray = gray.shape[::-1]
    
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
     
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape_gray, None, None)

print("Calibração concluída com sucesso! Parâmetros salvos em memória.\n")


# ==============================================================================
# PARTE 2: CORREÇÃO DE DISTORÇÃO (OBJETIVO D)
# ==============================================================================

def corrigir_imagem_dinamica(caminho_img, titulo_base):
    print(f"2. Corrigindo a imagem: {caminho_img}")
    img = cv2.imread(caminho_img)
    if img is None:
        print(f"Erro: Arquivo '{caminho_img}' não encontrado.")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # (i) Refinar a matriz da câmera usando o mtx e dist calculados dinamicamente
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

    # (ii) Correção undistort()
    dst_undistort = cv2.undistort(img_rgb, mtx, dist, None, newcameramtx)
    
    # (iii) Correção remap()
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst_remap = cv2.remap(img_rgb, mapx, mapy, cv2.INTER_LINEAR)

    # Cortando as bordas pretas extras baseadas na ROI
    x, y, w_roi, h_roi = roi
    dst_undistort = dst_undistort[y:y+h_roi, x:x+w_roi]
    dst_remap = dst_remap[y:y+h_roi, x:x+w_roi]

    # Visualização
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title(f"{titulo_base} - Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(dst_undistort)
    plt.title("Método (ii): undistort()")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(dst_remap)
    plt.title("Método (iii): remap()")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

corrigir_imagem_dinamica('gabriel_intackli0.jpg', 'Webcam')
corrigir_imagem_dinamica('gabriel_intackli1.jpg', 'Webcam')

