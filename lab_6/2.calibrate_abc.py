import numpy as np 
import cv2
from tqdm import tqdm
import os

# ==========================================
# CONFIGURAÇÕES
# ==========================================
CHECKERBOARD = (6, 8)
MAX_IMAGENS = 15

# Set the path to the images captured by the left and right cameras
pathL = "data/stereoL/"
pathR = "data/stereoR/"

print("Extracting image coordinates of respective 3D pattern ....\n")

# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Cria a malha de pontos 3D baseada no tamanho do tabuleiro
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

img_ptsL = []
img_ptsR = []
obj_pts = []

for i in tqdm(range(1, MAX_IMAGENS + 1)):
    imgL_path = f"{pathL}captura_L_{i}.png"
    imgR_path = f"{pathR}captura_R_{i}.png"
    
    # Pula se o arquivo não existir
    if not os.path.exists(imgL_path) or not os.path.exists(imgR_path):
        continue

    imgL = cv2.imread(imgL_path)
    imgR = cv2.imread(imgR_path)
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    outputL = imgL.copy()
    outputR = imgR.copy()

    retR, cornersR = cv2.findChessboardCorners(outputR, CHECKERBOARD, None)
    retL, cornersL = cv2.findChessboardCorners(outputL, CHECKERBOARD, None)

    if retR and retL:
        obj_pts.append(objp)
        cv2.cornerSubPix(imgR_gray, cornersR, (11,11), (-1,-1), criteria)
        cv2.cornerSubPix(imgL_gray, cornersL, (11,11), (-1,-1), criteria)

        img_ptsL.append(cornersL)
        img_ptsR.append(cornersR)


print("\nCalculating left camera parameters ... ")
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts, img_ptsL, imgL_gray.shape[::-1], None, None)
hL, wL = imgL_gray.shape[:2]
new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL,hL), 1, (wL,hL))

print("Calculating right camera parameters ... ")
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts, img_ptsR, imgR_gray.shape[::-1], None, None)
hR, wR = imgR_gray.shape[:2]
new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR,hR), 1, (wR,hR))


print("Stereo calibration .....")
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(
    obj_pts, img_ptsL, img_ptsR, new_mtxL, distL, new_mtxR, distR,
    imgL_gray.shape[::-1], criteria_stereo, flags)

rectify_scale = 1 
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(
    new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::-1], Rot, Trns, rectify_scale, (0,0))

Left_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l, imgL_gray.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r, imgR_gray.shape[::-1], cv2.CV_16SC2)

print("\n--- RESULTADOS PARA O RELATÓRIO ---")
print("Matriz de Rotação (Rot):\n", Rot)
print("\nVetor de Translação (Trns):\n", Trns)
print("\nMatriz Essencial (Emat):\n", Emat)
print("\nMatriz Fundamental (Fmat):\n", Fmat)
print("-----------------------------------\n")

print("Saving parameters ......")
xml_path = f"data/params_py.xml"
cv_file = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
cv_file.release()
print(f"Salvo com sucesso em {xml_path}")