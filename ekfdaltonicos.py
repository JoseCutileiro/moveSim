import pygame
import random
import re
import numpy as np
import time
from predictors import ekf
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

            # Define um limite de erro para considerar que o carro chegou ao próximo ponto
            erro_tolerado = 10.0  # Ajuste o valor conforme necessário para precisão

            if dist_total > erro_tolerado:  # Só movimenta se a distância for maior que o erro tolerado
                movimento_x = (dx / dist_total) * 100 * dt  # Velocidade ajustada pela distância
                movimento_y = (dy / dist_total) * 100 * dt  # Velocidade ajustada pela distância
                self.pos = (self.pos[0] + movimento_x, self.pos[1] + movimento_y)
            else:
                self.pos = prox_ponto  # Atualiza a posição para o próximo ponto
                self.ponto_atual += 1  # Avança para o próximo ponto da trajetória
                self.history.append(self.pos)


# Função principal
def main():
    clock = pygame.time.Clock()

    # Carregar a imagem de fundo (certifique-se de que o caminho está correto)
    background = pygame.image.load('road.png')
    background = pygame.transform.scale(background, TAMANHO)

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
        tela.blit(background, (0, 0))  # Coloca a imagem de fundo
        # Atualiza a posição do carro e obtém a trajetória real
        carro.mover(dt)

        # Desenha a trajetória real do carro
        for i in range(len(carro.history) - 1):
            pygame.draw.line(tela, AZUL, carro.history[i], carro.history[i + 1], 13)

        # Desenha o ponto atual do carro como uma bola vermelha (visível)
        pygame.draw.circle(tela, VERMELHO, (int(carro.pos[0]), int(carro.pos[1])), 10)

        # Previsões usando EKF
        predictions = ekf(carro.history, 50)

        # Desenha as previsões do EKF
        for i in range(1, len(predictions)):
            pygame.draw.line(tela, VERDE, predictions[i - 1], predictions[i], 13)
            
            # Adiciona uma cruz nos pontos da trajetória prevista
            x, y = predictions[i]
            tamanho_cruz = 10  # Define o tamanho da cruz
            pygame.draw.line(tela, (0,0,0), (x - tamanho_cruz, y - tamanho_cruz), (x + tamanho_cruz, y + tamanho_cruz), 5)  # Diagonal \
            pygame.draw.line(tela, (0,0,0), (x - tamanho_cruz, y + tamanho_cruz), (x + tamanho_cruz, y - tamanho_cruz), 5)  # Diagonal /
            

        # Atualiza a tela
        pygame.display.flip()

        # Verifica eventos (como fechar a janela)
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                rodando = False

    pygame.quit()

if __name__ == "__main__":
    main()
