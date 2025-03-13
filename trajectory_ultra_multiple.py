import pygame
import random
import re
import numpy as np
import time
from predictors import ekf, predizer_mais_semelhante

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
                self.history.append(self.pos)
            else:
                self.pos = prox_ponto  # Atualiza a posição para o próximo ponto
                self.ponto_atual += 1  # Avança para o próximo ponto da trajetória
                self.history.append(self.pos)

# Função principal
def main():
    clock = pygame.time.Clock()

    # Carregar a imagem de fundo
    background = pygame.image.load('road.png')
    background = pygame.transform.scale(background, TAMANHO)

    # Carregar as trajetórias do arquivo
    trajetorias = carregar_trajetorias("trajetoriasClean.txt")
    temos_acesso = carregar_trajetorias("trajetorias.txt")
    
    traj = random.choice(trajetorias)  # Escolhe uma trajetória aleatória

    # Criação de múltiplos carros
    num_carros = 1  # Defina quantos carros você quer simular
    carros = [Carro(random.choice(trajetorias)) for _ in range(num_carros)]

    # Configura a tela
    tela = pygame.display.set_mode(TAMANHO)
    pygame.display.set_caption("Simulador de Carros com Novo Preditior")

    rodando = True
    while rodando:
        dt = clock.tick(FPS) / 1000  # Tempo delta em segundos

        tela.fill(BRANCO)  # Limpa a tela
        tela.blit(background, (0, 0))  # Coloca a imagem de fundo

        # Atualiza e desenha todos os carros
        for carro in carros:
            carro.mover(dt)

            # Desenha a trajetória real do carro
            for i in range(len(carro.history) - 1):
                pygame.draw.line(tela, AZUL, carro.history[i], carro.history[i + 1], 2)

            # Desenha o ponto atual do carro como uma bola vermelha
            pygame.draw.circle(tela, VERMELHO, (int(carro.pos[0]), int(carro.pos[1])), 10)

            # Predição da trajetória mais semelhante
            traj_predita = predizer_mais_semelhante(carro.history, temos_acesso)

            # Desenha a trajetória prevista
            for i in range(1, len(traj_predita)):
                pygame.draw.line(tela, VERDE, traj_predita[i - 1], traj_predita[i], 2)

        pygame.display.flip()

        # Verifica eventos
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                rodando = False

    pygame.quit()

if __name__ == "__main__":
    main()
