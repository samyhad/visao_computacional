import numpy as np 
import cv2
import matplotlib.pyplot as plt

# --- CONFIGURAÇÃO DOS CAMINHOS ---
CAMINHO_IMG_ESQ = "data/im0.png" 
CAMINHO_IMG_DIR = "data/im1.png"

Value_pairs = []
disparity = None 

# Lendo os parâmetros salvos no código anterior
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
cv_file.release()

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        disp_val = disparity[y,x]
        if disp_val > 0:
            print(f"\n[CLIQUE DETECTADO] Disparidade no pixel: {disp_val}")
            try:
                Z_lido = float(input("Digite a distância real medida (em cm): "))
                Value_pairs.append([Z_lido, disp_val])
                print("-> Registrado! Clique em outro ponto ou aperte ESC na imagem para calcular o M.")
            except ValueError:
                print("Valor inválido. Digite apenas números.")
        else:
            print("\nVocê clicou em uma área sem disparidade válida. Tente novamente.")

cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp', 600, 600)
cv2.namedWindow('left image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('left image', 600, 600)
cv2.setMouseCallback('disp', mouse_click)

stereo = cv2.StereoBM_create()

imgL_orig = cv2.imread(CAMINHO_IMG_ESQ)
imgR_orig = cv2.imread(CAMINHO_IMG_DIR)

imgL = cv2.resize(imgL_orig, (640, 480))
imgR = cv2.resize(imgR_orig, (640, 480))

imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

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

while True:
    cv2.imshow("disp", disparity)
    cv2.imshow("left image", imgL)

    if cv2.waitKey(50) == 27: 
        if len(Value_pairs) > 0:
            break
        else:
            print("Aviso: Você precisa registrar pelo menos 1 ponto antes de sair!")

# Matemática e correção de formato da matriz para o OpenCV C++
value_pairs = np.array(Value_pairs)
z = value_pairs[:,0]
disp = value_pairs[:,1]
disp_inv = 1/disp

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12,6))
ax1.plot(disp, z, 'o-')
ax1.set(xlabel='Normalized disparity value', ylabel='Depth from camera (cm)', title='Depth vs Disparity')
ax1.grid()
ax2.plot(disp_inv, z, 'o-')
ax2.set(xlabel='Inverse disparity value (1/disp) ', ylabel='Depth from camera (cm)', title='Depth vs Inverse Disparity')
ax2.grid()
plt.show()

coeff = np.vstack([disp_inv, np.ones(len(disp_inv))]).T
coeff = np.float32(coeff) 
z_matrix = np.float32(z).reshape(-1, 1) 

ret, sol = cv2.solve(coeff, z_matrix, flags=cv2.DECOMP_QR)
M = sol[0,0]
print(f"Valor final de M = {M}")

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
cv_file.write("M", M)
cv_file.release()
cv2.destroyAllWindows()