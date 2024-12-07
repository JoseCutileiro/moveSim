import pygame
import random
import re
import time

# Inicializa o Pygame
pygame.init()

# Configurações da janela
LARGURA = 800
ALTURA = 600
TAMANHO = (LARGURA, ALTURA)
FPS = 60

# Cores
BRANCO = (255, 255, 255)
AZUL = (0, 0, 255)
VERDE = (0, 255, 0)

# Carrega as trajetórias do arquivo
def carregar_trajetorias(arquivo="trajetorias.txt"):
    trajetorias = []
    with open(arquivo, "r") as f:
        for linha in f:
            if linha.startswith("Traj:"):
                # Remove o prefixo "Traj:" e espaços extras
                coords = linha.strip().split(":")[1].strip()
                
                # Usando expressão regular para encontrar pares de coordenadas
                coordenadas = re.findall(r"\((-?\d+),\s*(-?\d+)\)", coords)
                
                # Convertendo as coordenadas encontradas em inteiros
                trajetoria = [(int(x), int(y)) for x, y in coordenadas]
                trajetorias.append(trajetoria)
    
    return trajetorias

def adicionar_noise(trajetoria, intensidade=4):
    """
    Adiciona ruído (noise) à trajetória.
    A intensidade controla o quanto de variação é aplicado.
    """
    trajetoria_com_noise = []
    for (x, y) in trajetoria:

        # Adiciona um deslocamento aleatório dentro de um intervalo (-intensidade, +intensidade)
        noise_x = random.uniform(-intensidade, intensidade)
        noise_y = random.uniform(-intensidade, intensidade)
        novo_ponto = (x + noise_x, y + noise_y)
        trajetoria_com_noise.append(novo_ponto)

    
    return trajetoria_com_noise

# Classe para representar cada objeto
class Objeto:
    def __init__(self, trajetoria, cor, velocidade):
        self.trajetoria = trajetoria  # Lista de coordenadas (trajetória)
        self.cor = cor  # Cor do objeto
        self.velocidade = velocidade  # Velocidade do objeto (quantidade de tempo para mover entre os pontos)
        self.pos = trajetoria[0]  # Começa no primeiro ponto
        self.ponto_atual = 0  # Índice do ponto atual na trajetória
        self.tempo_passado = 0  # Tempo passado desde o último movimento

    def mover(self, dt):
        if self.ponto_atual < len(self.trajetoria) - 1:
            # Pega o próximo ponto da trajetória
            ponto_destino = self.trajetoria[self.ponto_atual + 1]
            
            # Calcula a distância entre a posição atual e o ponto destino
            dx = ponto_destino[0] - self.pos[0]
            dy = ponto_destino[1] - self.pos[1]
            
            # Calcula o movimento com base na velocidade
            dist_total = (dx**2 + dy**2)**0.5
            movimento_x = (dx / dist_total) * self.velocidade * dt
            movimento_y = (dy / dist_total) * self.velocidade * dt

            # Atualiza a posição do objeto
            self.pos = (self.pos[0] + movimento_x, self.pos[1] + movimento_y)

            # Se o objeto chegou ou ultrapassou o próximo ponto, vai para o próximo
            if (abs(self.pos[0] - ponto_destino[0]) < 2 and abs(self.pos[1] - ponto_destino[1]) < 2):
                self.ponto_atual += 1

def provoque_failure(traj,intensity=10):
    ret = []
    for i in range(len(traj)):
        if (random.randint(0,intensity) < 2):
            ret += [traj[i]]
    return ret

def save_traj(trajetoria, arquivo="trajetoriasSim.txt"):
    with open(arquivo, "a") as f:  # Usando "a" para adicionar ao arquivo
        traj_str = ", ".join([f"({int(x)}, {int(y)})" for x, y in trajetoria])
        f.write(f"Traj: {traj_str}\n")

# Função principal
def main():
    clock = pygame.time.Clock()

    # Carrega as trajetórias do arquivo
    trajetorias = carregar_trajetorias()

    # Configura a tela
    tela = pygame.display.set_mode(TAMANHO)
    pygame.display.set_caption("Simulador de Trajetórias")

    # Cria uma lista de objetos com trajetórias e velocidades diferentes
    objetos = []
    for i in range(30):
        j = -1
        for traj in trajetorias:
            noise_traj = adicionar_noise(traj)
            noise_traj = provoque_failure(noise_traj)
            save_traj(noise_traj)
            cor_objeto = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Cor aleatória
            velocidade_objeto = random.uniform(50, 150)  # Velocidade aleatória entre 50 e 150 pixels por segundo
            objeto = Objeto(noise_traj, cor_objeto, velocidade_objeto)
            objetos.append(objeto)

    # Itera enquanto a janela estiver aberta
    rodando = True
    while rodando:
        dt = clock.tick(FPS) / 1000  # Tempo delta em segundos

        tela.fill(BRANCO)  # Limpa a tela a cada quadro

        # Atualiza e desenha todos os objetos
        for objeto in objetos:
            objeto.mover(dt)  # Move o objeto com base no tempo
            pygame.draw.circle(tela, objeto.cor, (int(objeto.pos[0]), int(objeto.pos[1])), 10)

        # Atualiza a tela
        pygame.display.flip()

        # Verifica eventos (como fechar a janela)
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                rodando = False

    pygame.quit()

if __name__ == "__main__":
    main()
