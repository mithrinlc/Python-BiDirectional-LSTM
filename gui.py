import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLabel, QLineEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import model

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

    def plot_movements(self, movements):
        x_data = [m['x'] for m in movements]
        y_data = [m['y'] for m in movements]
        self.axes.clear()
        self.axes.plot(x_data, y_data, 'r-')
        self.axes.set_title('Generated Mouse Movements')
        self.draw()


class ModelGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_model = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("TensorFlow Model GUI with PyQt5")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        self.plot_canvas = PlotCanvas(self, width=5, height=4)
        layout.addWidget(self.plot_canvas)

        self.sequence_length_input = QLineEdit("5")
        self.num_features_input = QLineEdit("8")
        layout.addWidget(QLabel("Sequence Length:"))
        layout.addWidget(self.sequence_length_input)
        layout.addWidget(QLabel("Number of Features:"))
        layout.addWidget(self.num_features_input)

        self.create_model_button = QPushButton('Create Model')
        self.create_model_button.clicked.connect(self.create_model)
        layout.addWidget(self.create_model_button)

        self.load_data_button = QPushButton('Load Training Data')
        self.load_data_button.clicked.connect(self.load_training_data)
        layout.addWidget(self.load_data_button)

        self.train_model_button = QPushButton('Train Model')
        self.train_model_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_model_button)

        self.load_button = QPushButton('Load Model')
        self.load_button.clicked.connect(self.load_model)
        layout.addWidget(self.load_button)

        self.save_button = QPushButton('Save Model')
        self.save_button.clicked.connect(self.save_model)
        layout.addWidget(self.save_button)

        self.model_info = QLabel("Model Info: Not Loaded")
        layout.addWidget(self.model_info)

        self.start_x_input = QLineEdit()
        self.start_y_input = QLineEdit()
        self.end_x_input = QLineEdit()
        self.end_y_input = QLineEdit()
        layout.addWidget(QLabel("Start X:"))
        layout.addWidget(self.start_x_input)
        layout.addWidget(QLabel("Start Y:"))
        layout.addWidget(self.start_y_input)
        layout.addWidget(QLabel("End X:"))
        layout.addWidget(self.end_x_input)
        layout.addWidget(QLabel("End Y:"))
        layout.addWidget(self.end_y_input)

        self.predict_button = QPushButton('Predict Movement')
        self.predict_button.clicked.connect(self.predict_movement)
        layout.addWidget(self.predict_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def create_model(self):
        sequence_length = int(self.sequence_length_input.text())
        num_features = int(self.num_features_input.text())
        self.current_model = model.create_model(sequence_length, num_features)
        self.model_info.setText("Model created.")

    def load_training_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Training Data", "", "JSON files (*.json)")
        if file_path:
            sequence_length = int(self.sequence_length_input.text())
            num_features = int(self.num_features_input.text())
            self.x_train, self.y_train = model.load_and_preprocess_data(file_path, sequence_length, num_features)
            self.model_info.setText(f"Training data loaded: {file_path}")

    def train_model(self):
        if self.current_model and self.x_train is not None and self.y_train is not None:
            history = model.train_model(self.current_model, self.x_train, self.y_train)
            self.model_info.setText("Model trained.")
            self.plot_training_history(history)


    def plot_training_history(self, history):
        self.plot_canvas.axes.clear()
        self.plot_canvas.axes.plot(history['loss'], label='Training Loss')
        self.plot_canvas.axes.set_title('Training History')
        self.plot_canvas.axes.set_xlabel('Epoch')
        self.plot_canvas.axes.set_ylabel('Loss')
        self.plot_canvas.axes.legend()
        self.plot_canvas.draw()
        
        
    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Model", "", "HDF5 files (*.h5)")
        if file_path:
            try:
                self.current_model = model.load_model(file_path)
                self.model_info.setText(f"Model loaded: {file_path}")
            except Exception as e:
                self.model_info.setText(f"Failed to load model: {e}")

    def save_model(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "HDF5 files (*.h5)")
        if file_path:
            try:
                model.save_model(self.current_model, file_path)
                self.model_info.setText(f"Model saved: {file_path}")
            except Exception as e:
                self.model_info.setText(f"Failed to save model: {e}")

    def predict_movement(self):
        if self.current_model:
            start_x = float(self.start_x_input.text())
            start_y = float(self.start_y_input.text())
            end_x = float(self.end_x_input.text())
            end_y = float(self.end_y_input.text())
            sequence_length = int(self.sequence_length_input.text())
            num_features = int(self.num_features_input.text())  # Retrieve the number of features
            movements = model.generate_path_to_target(self.current_model, (start_x, start_y), (end_x, end_y), sequence_length, num_features)
            if movements:
                self.plot_canvas.plot_movements(movements)
            else:
                self.model_info.setText("No movements generated. Check input values.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ModelGUI()
    ex.show()
    sys.exit(app.exec_())
