import tkinter as tk
from tkinter import *
import numpy as np
import tensorflow as tf
from PIL import Image, ImageGrab, ImageOps
import pandas as pd


# =========================
# Funkcja do centralizacji obrazu 28x28
# =========================
def center_image(img_28x28, threshold=0.1):
    """
    Funkcja przyjmuje obrazek 28x28 (float, 0-1).
    1) Wyszukuje piksele powyżej zadanego progu (threshold).
    2) Oblicza bounding box (min/max wiersz/kolumna).
    3) Wycina obszar z cyfrą.
    4) Wkleja go na środek nowej macierzy 28x28.

    Uwaga: Jeżeli w ogóle nie znajdzie pikseli > threshold, 
           zwraca oryginalny obraz.
    """

    mask = img_28x28 > threshold
    coords = np.argwhere(mask)

    # Jeżeli brak "narysowanych" pikseli, zwracamy oryginał
    if coords.size == 0:
        return img_28x28

    # Wyznaczamy bounding box
    row_min, col_min = coords.min(axis=0)
    row_max, col_max = coords.max(axis=0)

    # Wycinamy bounding box
    cropped = img_28x28[row_min:row_max + 1, col_min:col_max + 1]

    # Tworzymy nowy czysty obraz 28x28 (same zera)
    new_img = np.zeros_like(img_28x28)

    # Rozmiary obszaru wyciętego
    h_cropped, w_cropped = cropped.shape
    h, w = new_img.shape

    # Obliczamy, gdzie wkleić wycinek (tak, by był wycentrowany)
    row_offset = (h - h_cropped) // 2
    col_offset = (w - w_cropped) // 2

    # Wklejamy wyciętą część w środek
    new_img[row_offset:row_offset + h_cropped, col_offset:col_offset + w_cropped] = cropped

    return new_img

# =========================
# Funkcja trenująca model na MNIST
# =========================
def train_model_mnist(epochs=5):
    """
    Trenuje prostą sieć neuronową na zbiorze MNIST i zwraca 
    'probability_model' z warstwą Softmax.
    Możesz zmieniać architekturę, optymalizator itp. w tej funkcji.
    """

    # 1. Wczytaj i przygotuj dane MNIST
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()


    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 2. Zbuduj model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.10),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # 3. Kompilacja
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
                  loss=loss_fn,
                  metrics=['accuracy'])

    # 4. Trening
    print("Trwa trenowanie modelu MNIST...")
    model.fit(x_train, y_train, epochs=epochs, verbose=1)

    # 5. Ewaluacja
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")

    # 6. Doklejamy warstwę Softmax do predykcji
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    print("Model gotowy do użycia.\n")

    return probability_model

# =========================
# Klasa interfejsu graficznego
# =========================
class RozpoznawanieCyfr:
    def __init__(self, master, model):
        """
        Konstruktor przyjmuje:
        - master: okno Tkinter
        - model: wytrenowany model (z warstwą Softmax),
                 np. zwrócony przez train_model_mnist()
        """
        self.master = master
        self.master.title("Rozpoznawanie cyfr (MNIST)")
        self.model = model  # Zapamiętaj przekazany model

        self.width = 200
        self.height = 200

        # Stworzenie płótna do rysowania
        self.canvas = Canvas(self.master, width=self.width, height=self.height, bg='white')
        self.canvas.pack()

        # Sekcja przycisków
        frame = Frame(self.master)
        frame.pack()

        self.predict_button = Button(frame, text="Rozpoznaj", command=self.zrob_predykcje)
        self.predict_button.pack(side=LEFT, padx=5)

        self.clear_button = Button(frame, text="Wyczyść", command=self.wyczysc_canvas)
        self.clear_button.pack(side=LEFT, padx=5)

        # Etykieta z wynikiem
        self.label_predykcja = Label(self.master, text="Narysuj cyfrę i kliknij „Rozpoznaj”.", font=("Helvetica", 16))
        self.label_predykcja.pack(pady=10)

        # Obsługa rysowania myszką
        self.canvas.bind("<B1-Motion>", self.rysuj)
        self.last_x, self.last_y = None, None

    def rysuj(self, event):
        # Rysowanie linią między ostatnią a bieżącą pozycją
        if self.last_x is None:
            self.last_x = event.x
            self.last_y = event.y

        self.canvas.create_line(
            self.last_x, self.last_y, event.x, event.y,
            width=8, fill='black', capstyle=ROUND, smooth=TRUE
        )
        self.last_x, self.last_y = event.x, event.y

    def wyczysc_canvas(self):
        self.canvas.delete("all")
        self.last_x, self.last_y = None, None
        self.label_predykcja.config(text="Narysuj cyfrę i kliknij „Rozpoznaj”.")

    def zrob_predykcje(self):
        # Pobierz współrzędne płótna
        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y = self.master.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        # Zrzut płótna (obszaru ekranu)
        obraz = ImageGrab.grab().crop((x, y, x1, y1))

        # Konwertowanie na skalę szarości
        obraz = obraz.convert('L')

        # Odwrócenie kolorów (lepiej pasuje do MNIST)
        obraz = ImageOps.invert(obraz)

        # Skalowanie do 28x28
        obraz = obraz.resize((28, 28))

        # Konwersja do NumPy + normalizacja
        obraz_arr = np.array(obraz) / 255.0

        # Centralizacja rysunku
        obraz_arr = center_image(obraz_arr, threshold=0.1)

        # Reshape do (1, 28, 28)
        obraz_arr = obraz_arr.reshape((1, 28, 28))

        # Predykcja (model posiada warstwę Softmax)
        predictions = self.model(obraz_arr)
        predictions_array = predictions[0].numpy()

        # Debug: wypisz prawdopodobieństwa w konsoli
        print("Prawdopodobieństwo dla każdej cyfry:")
        for i, prob in enumerate(predictions_array):
            print(f"Cyfra {i}: {prob*100:.2f}%")

        # Wybór klasy z najwyższym prawdopodobieństwem
        predicted_label = np.argmax(predictions_array)

        # Wyświetlenie wyniku w GUI
        self.label_predykcja.config(text=f"Rozpoznana cyfra: {predicted_label}")


def evaluate_model_predictions(model, x_data, y_data):
    """
    Przepuszcza dane przez model i zwraca DataFrame z informacjami:
    - Jaka cyfra była (rzeczywista etykieta).
    - Jaka cyfra została sklasyfikowana.
    - Prawdopodobieństwo przypisania przewidywanej cyfry (%).
    - Informacja, czy predykcja była poprawna.

    Args:
    - model: Model TensorFlow z warstwą Softmax.
    - x_data: Dane wejściowe (np. obrazy testowe MNIST).
    - y_data: Rzeczywiste etykiety cyfry.

    Returns:
    - DataFrame z informacjami o każdej próbce.
    """

    # Przepuszczanie danych przez model
    predictions = model.predict(x_data)
    
    # Wyciągnięcie przewidywanej klasy i jej prawdopodobieństwa
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_probs = np.max(predictions, axis=1) * 100  # Prawdopodobieństwo w %

    # Tworzenie list wynikowych
    results = {
        "Numer próbki": list(range(len(x_data))),
        "Rzeczywista cyfra": y_data,
        "Przewidywana cyfra": predicted_labels,
        "Prawdopodobieństwo przewidywania (%)": predicted_probs,
        "Poprawność": predicted_labels == y_data  # True jeśli poprawna predykcja
    }

    # Tworzenie DataFrame
    df = pd.DataFrame(results)

    return df



# =========================
# Funkcja main
# =========================
def main():
    # 1. Trenuj model lub wczytaj gotowy
    probability_model = train_model_mnist(epochs=20)

    # 2. Uruchom interfejs graficzny z przekazanym modelem
    root = tk.Tk()
    app = RozpoznawanieCyfr(root, probability_model)
    root.mainloop()


    # mnist = tf.keras.datasets.mnist
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # df_results = evaluate_model_predictions(probability_model, x_test, y_test)

    # # Wyświetlenie pierwszych 10 wyników
    # print(df_results.head(10))

    # # Zapisanie wyników do pliku CSV (opcjonalnie)
    # df_results.to_csv("mnist_predictions_results0.csv", index=False)

if __name__ == "__main__":
    main()
