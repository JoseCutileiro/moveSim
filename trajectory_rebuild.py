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

# Função principal
def main():
    clock = pygame.time.Clock()

    # Carrega as trajetórias do arquivo
    trajetorias = carregar_trajetorias()

    # Configura a tela
    tela = pygame.display.set_mode(TAMANHO)
    pygame.display.set_caption("Simulador de Trajetórias")

    # Itera sobre as trajetórias
    for traj in trajetorias:
        ponto_atual = 0  # Inicia no primeiro ponto
        total_pontos = len(traj)

        while ponto_atual < total_pontos:
            tela.fill(BRANCO)  # Limpa a tela a cada quadro

            # Desenha os pontos da trajetória até o ponto atual
            for i in range(ponto_atual + 1):
                pygame.draw.circle(tela, AZUL, traj[i], 5)  # Desenha o ponto com cor azul

            # Atualiza a tela
            pygame.display.flip()

            # Espera um pouco antes de ir para o próximo ponto
            pygame.time.wait(5)  # Espera 500ms (meio segundo)

            ponto_atual += 1  # Vai para o próximo ponto

            # Verifica se o evento de fechamento da janela foi acionado
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    pygame.quit()
                    return

            clock.tick(FPS)  # Controla o FPS

    pygame.quit()

if __name__ == "__main__":
    main()
