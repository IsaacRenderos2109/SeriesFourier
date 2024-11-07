import numpy as np
import scipy.io.wavfile as wav
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QInputDialog
import sys
import sympy as sp

class AudioProcessor:
    def __init__(self):
        self.sample_rate = None
        self.audio_data = None
        self.fft_data = None

    def load_audio(self, filename):
        self.sample_rate, self.audio_data = wav.read(filename)
        if len(self.audio_data.shape) > 1:
            self.audio_data = self.audio_data[:, 0]  # Use only the first channel if stereo

    def apply_fft(self):
        self.fft_data = fft.fft(self.audio_data)

    def filter_noise(self, threshold):
        mask = np.abs(self.fft_data) > threshold
        self.fft_data = self.fft_data * mask

    def apply_ifft(self):
        return np.real(fft.ifft(self.fft_data))

    def plot_spectrum(self):
        plt.figure(figsize=(10, 4))
        freq = fft.fftfreq(len(self.audio_data), 1/self.sample_rate)
        plt.plot(freq, np.abs(self.fft_data))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Frequency Spectrum')
        plt.show()

    def fourier_series(self, func_str, terms=10):
        x = sp.symbols('x')
        func = sp.sympify(func_str)
        
        series = sp.fourier_series(func, (x, -sp.pi, sp.pi)).truncate(n=terms)
        return series

    def plot_fourier_series(self, func_str, terms=10):
        x_vals = np.linspace(-np.pi, np.pi, 400)
        x = sp.symbols('x')
        series = self.fourier_series(func_str, terms)
        
        # Evaluar la serie en puntos x
        y_vals = [series.subs(x, val).evalf() for val in x_vals]

        plt.figure(figsize=(10, 4))
        plt.plot(x_vals, y_vals, label=f'Serie de Fourier ({terms} términos)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'Serie de Fourier de {func_str}')
        plt.legend()
        plt.grid(True)
        plt.show()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Processor")
        self.setGeometry(100, 100, 300, 200)

        self.processor = AudioProcessor()

        layout = QVBoxLayout()

        self.load_button = QPushButton("Load Audio")
        self.load_button.clicked.connect(self.load_audio)
        layout.addWidget(self.load_button)

        self.fft_button = QPushButton("Apply FFT")
        self.fft_button.clicked.connect(self.apply_fft)
        layout.addWidget(self.fft_button)

        self.filter_button = QPushButton("Filter Noise")
        self.filter_button.clicked.connect(self.filter_noise)
        layout.addWidget(self.filter_button)

        self.plot_button = QPushButton("Plot Spectrum")
        self.plot_button.clicked.connect(self.plot_spectrum)
        layout.addWidget(self.plot_button)

        self.fourier_button = QPushButton("Resolve Fourier Series")
        self.fourier_button.clicked.connect(self.resolve_fourier_series)
        layout.addWidget(self.fourier_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_audio(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "WAV Files (*.wav)")
        if filename:
            self.processor.load_audio(filename)
            print("Audio loaded successfully")

    def apply_fft(self):
        if self.processor.audio_data is not None:
            self.processor.apply_fft()
            print("FFT applied")
        else:
            print("Please load audio first")

    def filter_noise(self):
        if self.processor.fft_data is not None:
            self.processor.filter_noise(1000)  # Arbitrary threshold
            print("Noise filtered")
        else:
            print("Please apply FFT first")

    def plot_spectrum(self):
        if self.processor.fft_data is not None:
            self.processor.plot_spectrum()
        else:
            print("Please apply FFT first")

    def resolve_fourier_series(self):
        func_str, ok = QInputDialog.getText(self, "Fourier Series", "Enter function of x (e.g., sin(x), x**2):")
        if ok and func_str:
            terms, ok = QInputDialog.getInt(self, "Fourier Series", "Enter number of terms:", 10, 1, 100)
            if ok:
                self.processor.plot_fourier_series(func_str, terms)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
