import pygame
import random
import re
import numpy as np
import time

# Inicializa o Pygame
pygame.init()

# Configurações da janela
LARGURA = 800
ALTURA = 600
TAMANHO = (LARGURA, ALTURA)
FPS = 60  # Aumentei o FPS para melhorar a suavidade do movimento

# Cores
BRANCO = (255, 255, 255)
AZUL = (0, 0, 255)
VERDE = (0, 255, 0)
VERMELHO = (255, 0, 0)  # Cor para o ponto atual (bola)

# Carrega as trajetórias do arquivo
def carregar_trajetorias(arquivo="trajetoriasClean.txt"):
    trajetorias = []
    with open(arquivo, "r") as f:
        for linha in f:
            if linha.startswith("Traj:"):
                coords = linha.strip().split(":")[1].strip()
                coordenadas = re.findall(r"\((-?\d+),\s*(-?\d+)\)", coords)
                trajetoria = [(int(x), int(y)) for x, y in coordenadas]
                trajetorias.append(trajetoria)
    
    return trajetorias

# Classe para representar o objeto carro
class Carro:
    def __init__(self, trajetoria):
        self.trajetoria = trajetoria  # Lista de coordenadas (trajetória)
        self.pos = trajetoria[0]  # Começa no primeiro ponto
        self.history = [self.pos]  # Histórico de pontos passados
        self.ponto_atual = 0  # Índice do ponto atual na trajetória

    def mover(self, dt):
        if self.ponto_atual < len(self.trajetoria) - 1:
            # Pega o próximo ponto da trajetória
            prox_ponto = self.trajetoria[self.ponto_atual + 1]
            # Calcula a distância total e a direção do movimento
            dx = prox_ponto[0] - self.pos[0]
            dy = prox_ponto[1] - self.pos[1]
            dist_total = (dx**2 + dy**2)**0.5  # Distância Euclidiana

            # Controla a velocidade (fazendo o movimento mais suave e constante)
            if dist_total > 1:  # Evitar movimentos muito pequenos
                movimento_x = (dx / dist_total) * 100 * dt  # Ajuste da velocidade (controle)
                movimento_y = (dy / dist_total) * 100* dt  # Ajuste da velocidade (controle)
                self.pos = (self.pos[0] + movimento_x, self.pos[1] + movimento_y)
            else:
                self.pos = prox_ponto  # Chegou ao ponto, então atualiza para o próximo
                self.ponto_atual += 1

            self.history.append(self.pos)

# Função do EKF (Filtro de Kalman Estendido)
def ekf(previous_points, prediction_range):
    dt = 1.0
    Q = np.eye(6) * 0.1
    R = np.eye(2) * 0.5
    x = np.array([previous_points[0][0], previous_points[0][1], 0, 0, 0, 0])
    P = np.eye(6)
    F_jacobian = np.array([
        [1, 0, dt, 0, 0.5 * dt**2, 0],
        [0, 1, 0, dt, 0, 0.5 * dt**2],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

    for z in previous_points:
        x_pred = np.array([
            x[0] + x[2] * dt + 0.5 * x[4] * dt**2,
            x[1] + x[3] * dt + 0.5 * x[5] * dt**2,
            x[2] + x[4] * dt,
            x[3] + x[5] * dt,
            x[4],
            x[5]
        ])
        P = F_jacobian @ P @ F_jacobian.T + Q
        y = np.array([z[0], z[1]]) - H @ x_pred
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y
        P = (np.eye(len(K)) - K @ H) @ P

    predictions = []
    for _ in range(prediction_range):
        x_pred = np.array([
            x[0] + x[2] * dt + 0.5 * x[4] * dt**2,
            x[1] + x[3] * dt + 0.5 * x[5] * dt**2,
            x[2] + x[4] * dt,
            x[3] + x[5] * dt,
            x[4],
            x[5]
        ])
        predictions.append(x_pred[:2])
        x = x_pred

    return predictions

# Função principal
def main():
    clock = pygame.time.Clock()

    # Carrega as trajetórias do arquivo e escolhe uma
    trajetorias = carregar_trajetorias()
    traj = random.choice(trajetorias)  # Escolhe uma trajetória aleatória
    print(f"Escolheu a trajetória: {traj}")

    # Cria o carro
    carro = Carro(traj)

    # Configura a tela
    tela = pygame.display.set_mode(TAMANHO)
    pygame.display.set_caption("Simulador de Carro com EKF")

    rodando = True
    while rodando:
        dt = clock.tick(FPS) / 1000  # Tempo delta em segundos

        tela.fill(BRANCO)  # Limpa a tela a cada quadro

        # Atualiza a posição do carro e obtém a trajetória real
        carro.mover(dt)

        # Desenha a trajetória real do carro
        for i in range(len(carro.history) - 1):
            pygame.draw.line(tela, AZUL, carro.history[i], carro.history[i + 1], 2)

        # Desenha o ponto atual do carro como uma bola vermelha (visível)
        pygame.draw.circle(tela, VERMELHO, (int(carro.pos[0]), int(carro.pos[1])), 10)

        # Previsões usando EKF
        predictions = ekf(carro.history, 50)

        # Desenha as previsões do EKF
        for i in range(1, len(predictions)):
            pygame.draw.line(tela, VERDE, predictions[i - 1], predictions[i], 2)

        # Atualiza a tela
        pygame.display.flip()

        # Verifica eventos (como fechar a janela)
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                rodando = False

    pygame.quit()

if __name__ == "__main__":
    main()
