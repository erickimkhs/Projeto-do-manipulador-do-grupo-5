#========================================================================
# BIBLIOTECAS (Intale: pip install numpy pybullet roboticstoolbox-python)
#========================================================================

import numpy as np
import roboticstoolbox as rtb
import pybullet as p
import pybullet_data
import time
import os
import sys

#=========================================================
# INCLUINDO OS PARÂMETROS E A CINEMÁTICA DIRETA E INVERSA
#=========================================================

a2 = 0.2  # 200mm
d4 = 0.8  # 800mm
a5 = 0.6  # 600mm

# Função para o cálculo da cinmática direta

def cinematica_direta(d1, t3, t4, t5, a2, d4, a5):
    # Elo 1 + 2 (Prismática + Elo Fixo)
    A02 = np.array([
        [1, 0, 0, a2],  # Matriz de 2 para 0
        [0, 1, 0, 0 ],
        [0, 0, 1, d1],
        [0, 0, 0, 1 ]
    ])

    # Elo 3
    A23 = np.array([
        [np.cos(t3),   0  ,  np.sin(t3),   0  ],  # Matriz de 3 para 2
        [np.sin(t3),   0  , -np.cos(t3),   0  ],
        [    0    ,    1  ,      0     ,   0  ],
        [    0    ,    0  ,      0     ,   1  ]
    ])

    # Elo 4
    A34 = np.array([
        [np.cos(t4),   0  ,  np.sin(t4),    0 ],  # Matriz de 4 para 3
        [np.sin(t4),   0  , -np.cos(t4),   0  ],
        [    0    ,    1  ,      0     ,   d4 ],
        [    0    ,    0  ,      0     ,   1  ]
    ])

    # Elo 5
    A45 = np.array([
        [np.cos(t5), -np.sin(t5),  0  , a5*np.cos(t5)],  # Matriz de 5 para 4
        [np.sin(t5),  np.cos(t5),  0  , a5*np.sin(t5)],
        [    0    ,      0     ,  1  ,       0      ],
        [    0    ,      0     ,  0  ,       1      ]
    ])

    # Matriz da Cinemática Direta
    A05 = A02 @ A23 @ A34 @ A45

    # Posições px,py e pz
    x = float(A05[0, 3])
    y = float(A05[1, 3])
    z = float(A05[2, 3])

    return A05, x, y, z

# Função para o cálculo da cinmática inversa

def cinematica_inversa(px, py, pz, az, a2, d4, a5):
    # Inicializando as variáveis
    d1 = 0.0
    t3 = 0.0
    t4 = 0.0
    t5 = 0.0
    sucesso = False

    try:
        # Tetha 4
        c4 = -az
        
        if abs(c4) > 1.0:
            return float(d1), float(t3), float(t4), float(t5), bool(sucesso)

        s4 = np.sqrt(1 - c4**2) 
        t4 = np.arctan2(s4, c4)

        R2 = (px - a2)**2 + py**2

        # Tetha 5
        coef_A = (a5 * s4)**2
        coef_B = 2 * a5 * d4
        coef_C = (a5 * c4)**2 + d4**2 - R2

        if np.abs(coef_A) < 1e-9:
            sols = [-coef_C / coef_B]
        else:
            sols = np.roots([coef_A, coef_B, coef_C])

        sols_validas = []
        for s in sols:
            if np.isreal(s) and abs(s) <= 1.0001: 
                sols_validas.append(np.real(s))
        
        if not sols_validas:
            return float(d1), float(t3), float(t4), float(t5), bool(sucesso)

        s5 = sols_validas[0]
        pos_s5 = [s for s in sols_validas if s >= 0]
        if pos_s5:
            s5 = pos_s5[0]
        
        c5 = np.sqrt(1 - s5**2)
        t5 = np.arctan2(s5, c5)

        # Tetha 3

        Val_A = a5 * c4 * c5
        Val_B = a5 * s5 + d4

        numerador = Val_B * (px - a2) + Val_A * py
        denominador = Val_A * (px - a2) - Val_B * py
        t3 = np.arctan2(numerador, denominador)

        # d1
        d1 = pz - a5 * s4 * c5

        if 0.0 <= d1 <= 0.8:
            sucesso = True
        else:
            sucesso = False 

    except:
        sucesso = False

    return float(d1), float(t3), float(t4), float(t5), bool(sucesso)

# Função para verificação de segurança

def verificar_seguranca(d1, t3, t4, t5, a2, d4, a5):
    # Verifica limite da junta prismática
    if d1 > 0.8 or d1 < 0.0:
        print(f"\n Erro de segurança - Limite de d1 excedido ({d1:.3f}m)")
        return False

    # Verifica zona proibida de ângulo (-90 < theta < 0)
    # Normaliza ângulos para verificar intervalo
    for nome, ang in [('t3', t3), ('t4', t4), ('t5', t5)]:
        graus = np.rad2deg(ang)
        graus = ((graus + 180) % 360) - 180 # Normaliza -180 a 180
        if -90 < graus < 0:
            print(f"\nErro de segurança - Ângulo proibido em {nome} ({graus:.2f}°)")
            return False

    # Calcula posição para verificar colisão com chão e recuo
    _, x_local, y_local, z_local = cinematica_direta(d1, t3, t4, t5, a2, d4, a5)
    
    # Conversão para coordenadas globais do PyBullet
    # Rotação X(-90): Y_local vira -Z_global (Altura)
    offset_base = 0.05
    z_global = -y_local + offset_base
    x_global = x_local

    if z_global < 0.02:
        print(f"\nErro de segurança - Colisão com o chão (Z={z_global:.3f}m)")
        return False

    if x_global < 0:
        print(f"\nErro de segurança - Movimento para trás da base (X={x_global:.3f}m)")
        return False

    return True

#================================
# CONFIGURAÇÃO DO PYBULLET E URDF
#================================

# CONFIGURAÇÃO DE DIRETÓRIO
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Diretório: {os.getcwd()}")

def generate_urdf():
    meshes_dir = "meshes"
    # VALORES DAS MASSAS (KG)
    m = [3.567, #ELO 1
         0.803, #ELO 2
         0.944, #ELO 3
         2.349, #ELO 4
         0.805] #ELO 5
    
    # VETOR DA ORIGEM ATÉ O CG (m)
    rc = [[0, 0, 0],    #ELO 1
          [-0.1, 0, 0], #ELO 2
          [0, 0, -0.1], #ELO 3
          [0, 0, -0.4], #ELO 4
          [-0.3, 0, 0]] #ELO 5

    # VALORES DA INÉRCIA Ixx,Iyy e Izz (Kg.m²)
    I = [(0.006, 0.029, 0.030),  #ELO 1
         (0.0003, 0.011, 0.011), #ELO 2
         (0.0004, 0.013, 0.013), #ELO 3
         (0.273, 0.273, 0.001),  #ELO 4
         (0.011, 0.0003, 0.011)] #ELO 5

    def fmt(lst): return f"{lst[0]:.5f} {lst[1]:.5f} {lst[2]:.5f}"

    # CRIAÇÃO DO ARQUIVO URDF

    urdf_content = f"""<?xml version="1.0"?>
<robot name="Submarino">
  <link name="base_link">
    <visual><origin xyz="-0.025 0.05 -0.105" rpy="0 0 0"/><geometry><mesh filename="{meshes_dir}/link0.stl"/></geometry><material name="yellow"><color rgba="1 1 0 1"/></material></visual>
    <inertial><mass value="1"/><origin xyz="0 0 0"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  </link>
  <link name="link_1">
    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="{meshes_dir}/link1.stl"/></geometry><material name="blue"><color rgba="0 0 0.8 1"/></material></visual>
    <inertial><mass value="{m[0]}"/><origin xyz="{fmt(rc[0])}"/><inertia ixx="{I[0][0]}" ixy="0" ixz="0" iyy="{I[0][1]}" iyz="0" izz="{I[0][2]}"/></inertial>
  </link>
  <link name="link_2">
    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="{meshes_dir}/link2.stl"/></geometry><material name="gray"><color rgba="0.5 0.5 0.5 1"/></material></visual>
    <inertial><mass value="{m[1]}"/><origin xyz="{fmt(rc[1])}"/><inertia ixx="{I[1][0]}" ixy="0" ixz="0" iyy="{I[1][1]}" iyz="0" izz="{I[1][2]}"/></inertial>
  </link>
  <link name="link_3">
    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="{meshes_dir}/link3.stl"/></geometry><material name="orange"><color rgba="1 0.5 0 1"/></material></visual>
    <inertial><mass value="{m[2]}"/><origin xyz="{fmt(rc[2])}"/><inertia ixx="{I[2][0]}" ixy="0" ixz="0" iyy="{I[2][1]}" iyz="0" izz="{I[2][2]}"/></inertial>
  </link>
  <link name="link_4">
    <visual><origin xyz="0 0 -0.6" rpy="0 0 0"/><geometry><mesh filename="{meshes_dir}/link4.stl"/></geometry><material name="blue"/></visual>
    <inertial><mass value="{m[3]}"/><origin xyz="{fmt(rc[3])}"/><inertia ixx="{I[3][0]}" ixy="0" ixz="0" iyy="{I[3][1]}" iyz="0" izz="{I[3][2]}"/></inertial>
  </link>
  <link name="link_5">
    <visual><origin xyz="0 0 0" rpy="0 0 1.57079"/><geometry><mesh filename="{meshes_dir}/link5.stl"/></geometry><material name="red"><color rgba="0.8 0 0 1"/></material></visual>
    <inertial><mass value="{m[4]}"/><origin xyz="{fmt(rc[4])}"/><inertia ixx="{I[4][0]}" ixy="0" ixz="0" iyy="{I[4][1]}" iyz="0" izz="{I[4][2]}"/></inertial>
  </link>

  <joint name="joint_1" type="prismatic">
    <parent link="base_link"/><child link="link_1"/><origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="0.0" upper="0.8" effort="100" velocity="0.5"/>
  </joint>
  <joint name="joint_2" type="fixed">
    <parent link="link_1"/><child link="link_2"/><origin xyz="0.2 0 0" rpy="0 0 0"/>
  </joint>
  
  <joint name="joint_3" type="revolute">
    <parent link="link_2"/><child link="link_3"/><origin xyz="0 0 0" rpy="0 0 -1.57079"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="200" velocity="1.5"/>
  </joint>
  
  <joint name="joint_4" type="revolute">
    <parent link="link_3"/><child link="link_4"/><origin xyz="0.8 0 0" rpy="0 1.57079 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="200" velocity="1.5"/>
  </joint>
  
  <joint name="joint_5" type="revolute">
    <parent link="link_4"/><child link="link_5"/><origin xyz="0 0 0" rpy="0 1.57079 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="100" velocity="2.0"/>
  </joint>
</robot>
"""
    filename = "submarino_kinematic.urdf"
    with open(filename, "w") as f:
        f.write(urdf_content)
    return filename

# INICIAÇÃO DO PYBULLET
physicsClient = p.connect(p.GUI)

# CONFIGURAÇÃO VISUAL
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Remove abas laterais e menus
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0) # Remove caixa mask
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0) # Remove caixa depth
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0) # Remove caixa RGB

p.resetDebugVisualizerCamera(
    cameraDistance=1.5,      # ZOOM
    cameraYaw=45,            # Rotação horizontal
    cameraPitch=-15,         # Rotação vertical 
    cameraTargetPosition=[0, 0, 0.7] # Ponto para onde a câmera olha
)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)

planeId = p.loadURDF("plane.urdf")
urdf_file = generate_urdf()

# POSIÇÃO INICIAL E ROTAÇÃO DO ROBÔ
startPos = [0, 0, 0.05]
startOrientation = p.getQuaternionFromEuler([-1.5708, 0, 0]) # Rotação Global -90 no X

try:
    robotId = p.loadURDF(urdf_file, startPos, startOrientation,useFixedBase=True)
    print("Robô carregado com sucesso!")
except Exception as e:
    print(f"Erro fatal ao carregar URDF: {e}")
    sys.exit()

# Função auxiliar para mover o robô no PyBullet suavemente
def mover_robo_pybullet(qin_array, qf_array):
    # Gera trajetória usando roboticstoolbox
    traj = rtb.jtraj(qin_array, qf_array, 50).q # 50 passos para ficar suave
    
    for q_step in traj:

        # Mapeia para as juntas do PyBullet
        p.resetJointState(robotId, 0, q_step[0]) # d1
        p.resetJointState(robotId, 2, q_step[2]) # t3
        p.resetJointState(robotId, 3, q_step[3]) # t4
        p.resetJointState(robotId, 4, q_step[4]) # t5
        
        p.stepSimulation()
        time.sleep(0.02)

# Guardando o estado atual, iniciando em 0
qin = np.array([0, 0, 0, 0, 0], dtype=float)

# ========================
#  EXECUÇÃO DE TRAJETÓRIA
# ========================

def coef_quinta_ordem(p0,pf,t_traj):

    a0 = p0
    a1 = 0
    a2 = 0
    a3 = 10*(pf-p0)/(t_traj**3)
    a4 = -15*(pf-p0)/(t_traj**4)
    a5 = 6*(pf-p0)/(t_traj**5)

    a = [a0,a1,a2,a3,a4,a5]

    return a

def calc_traj_pos(a,t):

    pos = a[0]+a[1]*t+a[2]*t**2+a[3]*t**3+a[4]*t**4+a[5]*t**5

    return pos

def calc_traj_vel(a,t):

    vel = a[1]+2*a[2]*t+3*a[3]*t**2+4*a[4]*t**3+5*a[5]*t**4

    return vel

def calc_traj_ace(a,t):
    
    ace = 2*a[2]+6*a[3]*t+12*a[4]*t**2+20*a[5]*t**3

    return ace

# =========================
#  INTERAÇÃO COM O USUÁRIO
# =========================

while True:
    print('\n========================')
    print(' Escolha uma opção: ')
    print(' 1 - Cinemática Direta ')
    print(' 2 - Cinemática Inversa ')
    print(' 3 - Trajetória ')
    print(' 0 - Sair')
    print('==========================')

    try:
        opcao_str = input('Escolha uma opção: ').strip()
        if not opcao_str: continue
        opcao = int(opcao_str)
    except ValueError:
        opcao = -1

    if opcao == 0:
        p.disconnect()
        break
    
    elif opcao == 1:
        # Implementação da cinemática direta
        print(
            '\n>> Digite as juntas [d1(m), t3(graus), t4(graus), t5(graus)] (ex: [0.6, 20, 20, 20]):')
        try:
            entrada_str = input('').replace('[', '').replace(']', '')
            entrada = [float(x) for x in entrada_str.split(',')]
        except:
            entrada = []

        if len(entrada) == 4:
            # Converte graus para radianos para o calculo
            d1 = entrada[0]
            t3 = np.deg2rad(entrada[1])
            t4 = np.deg2rad(entrada[2])
            t5 = np.deg2rad(entrada[3])

            # Verifica segurança antes de calcular
            if verificar_seguranca(d1, t3, t4, t5, a2, d4, a5):
                
                A05, x, y, z = cinematica_direta(d1, t3, t4, t5, a2, d4, a5)
                az = float(A05[3,3])
                y=-y
                # Exibe no terminal
                print('\nRESULTADO (Cinemática Direta):')
                print(f'Coordenadas Finais: X={x:.4f}, Y={y:.4f}, Z={z:.4f}, az={az:.4f}')
                print('Matriz Homogênea Final:')
                print(A05)

                # Define a configuração final
                qf = np.array([d1, 0, t3, t4, t5]) 

                # Move o manipulador na simulação
                print("Movendo manipulador...")
                mover_robo_pybullet(qin, qf)
                
                # Atualiza posição inicial para a próxima jogada
                qin = qf
            else:
                print('Movimento cancelado por segurança.')
        else:
            print('Erro: Digite exatamente 4 valores separados por vírgula.')

    elif opcao == 2:
        
        print(
            '\n Digite a posição e ângulo [X, Y, Z, Ângulo da solda(graus)] ([ex: [1.0417, 0.7634, 0.7928, 1]):')
        try:
            coord_str = input('').replace('[', '').replace(']', '')
            coord = [float(x) for x in coord_str.split(',')]
        except:
            coord = []

        if len(coord) == 4:
            px = coord[0]
            py = coord[1]
            py = -py
            pz = coord[2]
            
            angulo_graus = coord[3]
            az_input = np.cos(np.deg2rad(angulo_graus))

            d1i, t3i, t4i, t5i, sucesso = cinematica_inversa(
                px, py, pz, az_input, a2, d4, a5)

            if sucesso:
                # Verifica segurança na solução encontrada
                if verificar_seguranca(d1i, t3i, t4i, t5i, a2, d4, a5):
                    print('\nRESULTADO (Cinemática Inversa):')
                    print('Juntas Calculadas:')
                    print(f'd1: {d1i:.4f} m')
                    print(f't3: {np.rad2deg(t3i):.4f} graus')
                    print(f't4: {np.rad2deg(t4i):.4f} graus')
                    print(f't5: {np.rad2deg(t5i):.4f} graus')

                    # Define a configuração final
                    qfi = np.array([d1i, 0, t3i, t4i, t5i])
                    
                    # Move o robô na simulação
                    print("Movendo robô...")
                    mover_robo_pybullet(qin, qfi)
                    
                    qin = qfi
                else:
                    print('Solução encontrada, mas insegura (colisão ou limites).')
            else:
                print('\nERRO: Ponto fora do espaço de trabalho ou singularidade')
        else:
            print('Erro: Digite 4 coordenadas.')

    elif opcao == 3:

        print('\nDigite a posição e ângulo (graus) inicial e final da solda:')
        try:
            P0 = np.array([float(x) for x in input('P0 [x,y,z,ângulo]: ').replace('[','').replace(']','').split(',')])
            Pf = np.array([float(x) for x in input('Pf [x,y,z,ângulo]: ').replace('[','').replace(']','').split(',')])
        except:
            print('Erro na entrada.')
            continue

        print('\nDigite o tempo da solda (s):')
        try:
            t_solda = float(input('Tempo da Solda: '))

        except:
            print('Erro na entrada.')
            continue

        # Coeficientes para x, y, z e az
        a_x = coef_quinta_ordem(P0[0],Pf[0],t_solda)
        a_y = coef_quinta_ordem(P0[1],Pf[1],t_solda)
        a_z = coef_quinta_ordem(P0[2],Pf[2],t_solda)
        a_az = coef_quinta_ordem(np.cos(np.deg2rad(P0[3])),np.cos(np.deg2rad(Pf[3])),t_solda)

        # Discretização do tempo
        N = max(int(t_solda/0.05),100)
        t = np.linspace(0,t_solda,N)

        pos = np.zeros((N,4))
        vel = np.zeros((N,4))
        ace = np.zeros((N,4))

        # Calculando posição velocidade e aceleração em cada ponto da trajetória
        for i in range(N):

            pos[i,0] = calc_traj_pos(a_x,t[i])
            pos[i,1] = calc_traj_pos(a_y,t[i])
            pos[i,2] = calc_traj_pos(a_z,t[i])
            pos[i,3] = calc_traj_pos(a_az,t[i])

            vel[i,0] = calc_traj_vel(a_x,t[i])
            vel[i,1] = calc_traj_vel(a_y,t[i])
            vel[i,2] = calc_traj_vel(a_z,t[i])
            vel[i,3] = calc_traj_vel(a_az,t[i])

            ace[i,0] = calc_traj_ace(a_x,t[i])
            ace[i,1] = calc_traj_ace(a_y,t[i])
            ace[i,2] = calc_traj_ace(a_z,t[i])
            ace[i,3] = calc_traj_ace(a_az,t[i])

        print('\nExecutando solda...')

        for i in range(N):

            px = pos[i,0]
            py = -pos[i,1]
            pz = pos[i,2]
            az = pos[i,3]

            print(f"Tentando CI para ponto: X={px:.4f}, Y={py:.4f}, Z={pz:.4f}")

            d1,t3,t4,t5,sucesso = cinematica_inversa(px,py,pz,az,a2,d4,a5)

            if not sucesso:
                print('Falha na cinemática inversa. Solda interrompida.')
                break

            if not verificar_seguranca(d1,t3,t4,t5,a2,d4,a5):
                 print('Movimento inseguro. Solda interrompida.')
                 break

            # Move robô diretamente (passo a passo)
            p.resetJointState(robotId,0,d1)
            p.resetJointState(robotId,2,t3)
            p.resetJointState(robotId,3,t4)
            p.resetJointState(robotId,4,t5)

            p.stepSimulation()
            time.sleep(0.05)
      
    p.stepSimulation()

#================================
# ANÁLISE DINÂMICA DO MANIPULADOR
#================================

