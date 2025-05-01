import cv2
import os
import numpy as np
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json

# Função para selecionar pontos manualmente
def select_training_samples(image_path, n_classes=3):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    training_samples = {i: [] for i in range(n_classes)}
    current_class = [0]  # lista para permitir modificação dentro do evento

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Selecione pontos para Classe {current_class[0]} (pressione 'n' para mudar de classe)")

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            training_samples[current_class[0]].append((y, x))
            ax.plot(x, y, 'o', label=f"Classe {current_class[0]}")
            fig.canvas.draw()

    def onkey(event):
        if event.key == 'n':
            current_class[0] = (current_class[0] + 1) % n_classes
            ax.set_title(f"Selecione pontos para Classe {current_class[0]} (pressione 'n' para mudar de classe)")
            fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    kid = fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

    # Salvar os pontos em um arquivo json
    with open('training_samples.json', 'w') as f:
        json.dump(training_samples, f)

    print("Pontos de treinamento salvos em training_samples.json")
    return training_samples

if __name__ == "__main__":
    # Se quiser selecionar manualmente pontos de treinamento:
    samples = select_training_samples('cropped_images/foto_28.jpg', 2)
