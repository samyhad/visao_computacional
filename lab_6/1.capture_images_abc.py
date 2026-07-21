import cv2
import os

# ==========================================
# CONFIGURAÇÕES DA EQUIPE
# ==========================================
MAX_IMAGENS = 15       

output_path = "data/"
os.makedirs(output_path + 'stereoL', exist_ok=True)
os.makedirs(output_path + 'stereoR', exist_ok=True)

# IDs das câmeras
CamL_id = 1
CamR_id = 0

print(f"Tentando conectar nas câmeras {CamL_id} e {CamR_id}...")
CamL = cv2.VideoCapture(CamL_id)
CamR = cv2.VideoCapture(CamR_id)

# Forçando resolução baixa para não travar a porta USB
CamL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
CamL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
CamR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
CamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not CamL.isOpened() or not CamR.isOpened():
    print("ERRO: Uma ou ambas as câmeras não abriram. Mude os IDs (0, 1, 2...).")
    exit(-1)

print("\n=== INSTRUÇÕES ===")
print("1. Posicione o tabuleiro de xadrez para que apareça NAS DUAS câmeras.")
print("2. Pressione a tecla 'S' (na janela do vídeo) para SALVAR o par de fotos.")
print("3. Pressione a tecla 'ESC' para sair.")
print("==================\n")

count = 0

while True:
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()
    
    if not retR or not retL:
        print("Erro ao ler o frame das câmeras. Verifique os cabos USB.")
        break
        
    imgL_vis = frameL.copy()
    imgR_vis = frameR.copy()
    
    cv2.putText(imgL_vis, f"Fotos salvas: {count}/{MAX_IMAGENS}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(imgL_vis, "Aperte 'S' para salvar", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Mostra os vídeos
    cv2.imshow('Camera ESQUERDA (L)', imgL_vis)
    cv2.imshow('Camera DIREITA (R)', imgR_vis)

    # Aguarda a tecla do usuário
    key = cv2.waitKey(1) & 0xFF
    
    # Se apertar 'S' (ou 's'), salva a foto
    if key == ord('s') or key == ord('S'):
        count += 1
        nome_img_R = f"{output_path}stereoR/captura_R_{count}.png"
        nome_img_L = f"{output_path}stereoL/captura_L_{count}.png"
        
        cv2.imwrite(nome_img_R, frameR)
        cv2.imwrite(nome_img_L, frameL)
        
        print(f"[{count}/{MAX_IMAGENS}] Foto salva com sucesso!")
     
        cv2.rectangle(imgL_vis, (0,0), (640,480), (0, 255, 0), 10)
        cv2.rectangle(imgR_vis, (0,0), (640,480), (0, 255, 0), 10)
        cv2.imshow('Camera ESQUERDA (L)', imgL_vis)
        cv2.imshow('Camera DIREITA (R)', imgR_vis)
        cv2.waitKey(200) # Pausa rápida
        
    # Encerra se atingir o limite
    if count >= MAX_IMAGENS:
        print(f"\nPerfeito! {MAX_IMAGENS} pares de fotos capturados.")
        break
        
    # Se apertar ESC, sai
    if key == 27: 
        print("\nCaptura encerrada pelo usuário.")
        break

CamL.release()
CamR.release()
cv2.destroyAllWindows()