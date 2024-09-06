from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QLineEdit, QComboBox
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMessageBox, QDateEdit, QTextEdit, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QDate
from PySide6.QtGui import QImage, QPixmap
import cv2
import sys
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import socket
import threading
import time
import re  # Importar módulo de expresiones regulares
import psycopg2  # Importar la librería psycopg2 para conectar con PostgreSQL
import time
from datetime import datetime  # Importar datetime para la fecha y hora actuales
from decimal import Decimal
import pandas as pd  # Importar pandas para exportar a CSV
import shutil

# Ruta a las carpetas con imágenes de entrenamiento
base_dir = r'C:\Users\fredd\OneDrive\Documentos\Maestria\BaseImagenes'
os.makedirs(base_dir, exist_ok=True)

# Configuración de la cámara ESP32
camera_url = 'http://192.168.1.108/stream'


class CaptureImages(QWidget):
    def __init__(self, category_name):
        super().__init__()
        self.category_name = category_name
        self.initUI()
        self.cap = None
        self.counter = 0
        self.setup_folder()

    def initUI(self):
        self.setWindowTitle(f'Captura de Imágenes - {self.category_name}')
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        self.label = QLabel(f'Captura 45 imágenes de {self.category_name}')
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.btn_open_camera = QPushButton('Abrir Cámara')
        self.btn_open_camera.clicked.connect(self.open_camera)
        layout.addWidget(self.btn_open_camera)

        self.btn_capture = QPushButton(f'Capturar {self.category_name}')
        self.btn_capture.clicked.connect(self.capture_image)
        layout.addWidget(self.btn_capture)

        self.btn_finish = QPushButton('Finalizar')
        self.btn_finish.clicked.connect(self.finish_capture)
        layout.addWidget(self.btn_finish)

        self.setLayout(layout)

    def setup_folder(self):
        # Crear carpeta para la categoría si no existe
        self.folder_path = os.path.join(base_dir, self.category_name)
        os.makedirs(self.folder_path, exist_ok=True)

    def open_camera(self):
        self.cap = cv2.VideoCapture(camera_url)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "No se pudo abrir la cámara.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            cv2.imshow('Captura de Cámara', frame)

            # Cierra la cámara con la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def capture_image(self):
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Primero abre la cámara.")
            return

        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.critical(self, "Error", "No se pudo capturar la imagen.")
            return

        img_name = os.path.join(self.folder_path, f"{self.category_name}_{self.counter}.png")
        self.counter += 1

        cv2.imwrite(img_name, frame)
        QMessageBox.information(self, "Captura", f"Imagen guardada en {img_name}.")

    def finish_capture(self):
        if self.counter < 45:
            QMessageBox.warning(self, "Advertencia", "Debes capturar 45 imágenes.")
            return

        self.cap.release()
        self.train_model()

    def train_model(self):
        self.training_window = TrainingWindow()
        self.training_window.show()
        self.close()


class TrainingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Entrenamiento del Modelo')
        self.setGeometry(100, 100, 300, 100)

        layout = QVBoxLayout()

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

        # Iniciar el entrenamiento en un hilo separado
        self.thread = TrainingThread()
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.training_finished)
        self.thread.start()

    def training_finished(self):
        QMessageBox.information(self, "Entrenamiento Completado", "Entrenamiento IA terminado.")
        self.close()
        self.main_window = MainWindow()
        self.main_window.show()

class ReportWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent  # Guardar referencia a la ventana principal
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Generar Reporte')
        self.setGeometry(100, 100, 800, 600)  # Ajustar tamaño de la ventana

        layout = QVBoxLayout()

        # Selector de fecha
        date_layout = QHBoxLayout()
        self.date_label = QLabel('Seleccione la fecha:')
        self.date_edit = QDateEdit(self)
        self.date_edit.setDisplayFormat('yyyy-MM-dd')
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)

        date_layout.addWidget(self.date_label)
        date_layout.addWidget(self.date_edit)

        layout.addLayout(date_layout)

        # Botón para consultar
        self.btn_accept = QPushButton('Aceptar')
        self.btn_accept.clicked.connect(self.fetch_report_data)
        layout.addWidget(self.btn_accept)

        # Tabla para mostrar los datos del reporte
        self.report_table = QTableWidget(self)
        layout.addWidget(self.report_table)

        # Botón para exportar los datos
        self.btn_export = QPushButton('Exportar')
        self.btn_export.clicked.connect(self.export_report)
        layout.addWidget(self.btn_export)

        # Botón para volver a la ventana principal
        self.btn_back = QPushButton('Volver')
        self.btn_back.clicked.connect(self.close_and_return)
        layout.addWidget(self.btn_back)

        self.setLayout(layout)

    def fetch_report_data(self):
        selected_date = self.date_edit.date().toString('yyyy-MM-dd')  # Obtener la fecha seleccionada
        try:
            # Conectar a la base de datos PostgreSQL
            connection = psycopg2.connect(
                host="localhost",
                database="Balanza",
                user="postgres",
                password="postgres"
            )
            cursor = connection.cursor()

            # Consulta de datos de la tabla tot por la fecha seleccionada
            query = "SELECT * FROM tot WHERE date = %s"
            cursor.execute(query, (selected_date,))
            rows = cursor.fetchall()

            # Configurar la tabla para mostrar los datos
            if rows:
                column_names = [desc[0] for desc in cursor.description]  # Obtener nombres de columnas
                self.report_table.setColumnCount(len(column_names))
                self.report_table.setRowCount(len(rows))
                self.report_table.setHorizontalHeaderLabels(column_names)

                # Insertar datos en la tabla
                for row_idx, row_data in enumerate(rows):
                    for col_idx, col_data in enumerate(row_data):
                        # Formatear 'totprice' a 4 decimales, 'weight' a 3 decimales y 'time' a 'HH:MM:SS'
                        if column_names[col_idx] == 'totprice':
                            col_data = f"{col_data:.4f}"
                        elif column_names[col_idx] == 'weight':
                            col_data = f"{col_data:.3f}"
                        elif column_names[col_idx] == 'time':
                            col_data = col_data.strftime('%H:%M:%S')

                        self.report_table.setItem(row_idx, col_idx, QTableWidgetItem(str(col_data)))

            else:
                self.report_table.setRowCount(0)
                QMessageBox.information(self, "Sin datos", "No hay datos para la fecha seleccionada.")

            cursor.close()
            connection.close()

        except (Exception, psycopg2.DatabaseError) as error:
            QMessageBox.critical(self, "Error", f"Error al consultar datos: {error}")

    def export_report(self):
        selected_date = self.date_edit.date().toString('yyyy-MM-dd')  # Obtener la fecha seleccionada
        try:
            # Conectar a la base de datos PostgreSQL
            connection = psycopg2.connect(
                host="localhost",
                database="Balanza",
                user="postgres",
                password="postgres"
            )
            cursor = connection.cursor()

            # Consulta de datos de la tabla tot por la fecha seleccionada
            query = "SELECT * FROM tot WHERE date = %s"
            cursor.execute(query, (selected_date,))
            data = pd.read_sql(query, connection, params=(selected_date,))

            # Redondear 'totprice' a 4 decimales, 'weight' a 3 decimales y formatear 'time' a 'HH:MM:SS'
            data['totprice'] = data['totprice'].apply(lambda x: round(x, 4))
            data['weight'] = data['weight'].apply(lambda x: round(x, 3))
            data['time'] = data['time'].apply(lambda x: x.strftime('%H:%M:%S'))

            # Crear la carpeta "C:\Report" si no existe
            export_dir = r'C:\Report'
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)

            # Exportar los datos a un archivo CSV en el formato 'report_(fecha).csv'
            export_path = os.path.join(export_dir, f'report_{selected_date}.csv')
            data.to_csv(export_path, index=False)

            QMessageBox.information(self, "Éxito", f"Reporte exportado a {export_path}")

            cursor.close()
            connection.close()

        except (Exception, psycopg2.DatabaseError) as error:
            QMessageBox.critical(self, "Error", f"Error al exportar datos: {error}")

    def close_and_return(self):
        self.close()
        self.parent.show()  # Volver a la ventana principal

class TrainingThread(QThread):
    progress = Signal(int)

    def run(self):
        # Configuración del entrenamiento del modelo
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            validation_split=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=[0.8, 1.2],
            horizontal_flip=True,
            vertical_flip=True
        )

        train_generator = train_datagen.flow_from_directory(
            base_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            base_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )

        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        epochs = 30
        for epoch in range(epochs):
            model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=1,
                verbose=1
            )
            self.progress.emit(int((epoch + 1) / epochs * 100))  # Actualizar barra de progreso

        model.save('modelo_multiclase.keras')

class AddProductWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window  # Guardar la referencia a la ventana principal
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Agregar Nuevo Producto')
        self.setGeometry(100, 100, 400, 300)  # Ajustar tamaño de ventana

        layout = QVBoxLayout()

        # Campo para el nombre del producto
        name_layout = QHBoxLayout()
        self.label_name = QLabel('Nombre:')
        self.product_name_input = QLineEdit(self)
        name_layout.addWidget(self.label_name)
        name_layout.addWidget(self.product_name_input)
        layout.addLayout(name_layout)

        # Campo para el precio unitario
        price_layout = QHBoxLayout()
        self.label_price = QLabel('Precio unitario:')
        self.product_price_input = QLineEdit(self)
        price_layout.addWidget(self.label_price)
        price_layout.addWidget(self.product_price_input)
        layout.addLayout(price_layout)

        # Campo para el PLU ID
        plu_layout = QHBoxLayout()
        self.label_plu = QLabel('PLU ID:')
        self.product_plu_input = QLineEdit(self)
        plu_layout.addWidget(self.label_plu)
        plu_layout.addWidget(self.product_plu_input)
        layout.addLayout(plu_layout)

        # Campo para la caducidad
        expiry_layout = QHBoxLayout()
        self.label_expiry = QLabel('Caducidad:')
        self.product_expiry_input = QLineEdit(self)
        expiry_layout.addWidget(self.label_expiry)
        expiry_layout.addWidget(self.product_expiry_input)
        layout.addLayout(expiry_layout)

        # Botón para aceptar y agregar el nuevo producto
        self.btn_accept = QPushButton('Aceptar')
        self.btn_accept.clicked.connect(self.accept)
        layout.addWidget(self.btn_accept)

        # Botón para volver a la ventana principal
        self.btn_back = QPushButton('Volver')
        self.btn_back.clicked.connect(self.go_back)
        layout.addWidget(self.btn_back)

        self.setLayout(layout)

    def accept(self):
        product_name = self.product_name_input.text().strip()
        product_price = self.product_price_input.text().strip()
        product_plu = self.product_plu_input.text().strip()
        product_expiry = self.product_expiry_input.text().strip()

        # Validar los campos ingresados
        if not product_name:
            QMessageBox.warning(self, "Advertencia", "Por favor, introduce un nombre para el producto.")
            return
        if not self.is_float(product_price):
            QMessageBox.warning(self, "Advertencia", "Por favor, introduce un precio válido.")
            return
        if not self.is_int(product_plu):
            QMessageBox.warning(self, "Advertencia", "Por favor, introduce un PLU ID válido.")
            return
        if not self.is_int(product_expiry):
            QMessageBox.warning(self, "Advertencia", "Por favor, introduce un valor de caducidad válido.")
            return

        # Guardar los datos en la base de datos PostgreSQL
        self.insert_into_database(int(product_plu), product_name, float(product_price), int(product_expiry))

        # Aquí puedes agregar el código para guardar los datos ingresados
        print(
            f"Producto agregado: {product_name}, Precio: {product_price}, PLU ID: {product_plu}, Caducidad: {product_expiry}")

        # Abrir ventana de captura de imágenes
        self.capture_window = CaptureImages(product_name)
        self.capture_window.show()
        self.close()

    def insert_into_database(self, plu_id, name, unitprice, caducidad):
        try:
            # Conectar a la base de datos PostgreSQL
            connection = psycopg2.connect(
                host="localhost",
                database="Balanza",
                user="postgres",  # Reemplazar con el nombre de usuario correcto
                password="postgres"  # Reemplazar con la contraseña correcta
            )
            cursor = connection.cursor()

            # Insertar los datos en la tabla plu
            insert_query = """
            INSERT INTO plu (id, name, unitprice, itemcode, caducidad)
            VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (plu_id, name, unitprice, plu_id, caducidad))

            # Confirmar la transacción
            connection.commit()
            cursor.close()
            connection.close()

            QMessageBox.information(self, "Éxito", "Se ha creado un nuevo producto.")

        except (Exception, psycopg2.DatabaseError) as error:
            QMessageBox.critical(self, "Error", f"Error al insertar datos en la base de datos: {error}")

    def go_back(self):
        self.close()  # Cerrar la ventana actual
        self.main_window.show()  # Mostrar la ventana principal

    @staticmethod
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_int(value):
        try:
            int(value)
            return True
        except ValueError:
            return False

class ModifyProductWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent  # Guardar referencia a la ventana principal
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Modificar Producto')
        self.setGeometry(100, 100, 400, 300)  # Ajustar tamaño de la ventana

        layout = QVBoxLayout()

        # Lista desplegable para seleccionar el producto
        self.product_combobox = QComboBox(self)
        self.product_combobox.currentIndexChanged.connect(self.load_product_data)
        layout.addWidget(self.product_combobox)

        # Labels y cuadros de texto para editar información del producto
        self.unitprice_label = QLabel('Precio unitario:')
        layout.addWidget(self.unitprice_label)
        self.unitprice_edit = QLineEdit(self)
        layout.addWidget(self.unitprice_edit)

        self.caducidad_label = QLabel('Caducidad:')
        layout.addWidget(self.caducidad_label)
        self.caducidad_edit = QLineEdit(self)
        layout.addWidget(self.caducidad_edit)

        # Cargar productos en la lista desplegable
        self.load_products()

        # Botones de Guardar, Borrar y Volver
        button_layout = QHBoxLayout()

        self.btn_save = QPushButton('Guardar')
        self.btn_save.clicked.connect(self.save_product_changes)
        button_layout.addWidget(self.btn_save)

        self.btn_delete = QPushButton('Borrar')
        self.btn_delete.clicked.connect(self.delete_product)
        button_layout.addWidget(self.btn_delete)

        self.btn_back = QPushButton('Volver')
        self.btn_back.clicked.connect(self.close_and_return)  # Conectar el botón Volver a la función close_and_return
        button_layout.addWidget(self.btn_back)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def load_products(self):
        try:
            # Conectar a la base de datos PostgreSQL
            connection = psycopg2.connect(
                host="localhost",
                database="Balanza",
                user="postgres",
                password="postgres"
            )
            cursor = connection.cursor()

            # Obtener todos los productos de la tabla plu
            cursor.execute("SELECT name FROM plu")
            products = cursor.fetchall()

            # Limpiar el combobox antes de agregar nuevos elementos
            self.product_combobox.clear()

            # Llenar la lista desplegable con los nombres de los productos
            for product in products:
                self.product_combobox.addItem(product[0])

            cursor.close()
            connection.close()

        except (Exception, psycopg2.DatabaseError) as error:
            QMessageBox.critical(self, "Error", f"Error al cargar productos: {error}")

    def load_product_data(self):
        # Cargar datos del producto seleccionado
        selected_product = self.product_combobox.currentText()  # Obtener el producto seleccionado
        if not selected_product:
            return

        try:
            # Conectar a la base de datos PostgreSQL
            connection = psycopg2.connect(
                host="localhost",
                database="Balanza",
                user="postgres",
                password="postgres"
            )
            cursor = connection.cursor()

            # Consultar los detalles del producto seleccionado
            cursor.execute("SELECT unitprice, caducidad FROM plu WHERE name = %s", (selected_product,))
            result = cursor.fetchone()

            if result:
                # Mostrar los datos en los cuadros de texto correspondientes
                self.unitprice_edit.setText(str(result[0]))  # Mostrar unitprice
                self.caducidad_edit.setText(str(result[1]))  # Mostrar caducidad

            cursor.close()
            connection.close()

        except (Exception, psycopg2.DatabaseError) as error:
            QMessageBox.critical(self, "Error", f"Error al cargar datos del producto: {error}")

    def save_product_changes(self):
        # Guardar cambios en la base de datos
        selected_product = self.product_combobox.currentText()
        new_unitprice = self.unitprice_edit.text()
        new_caducidad = self.caducidad_edit.text()

        if not selected_product:
            QMessageBox.warning(self, "Advertencia", "Seleccione un producto.")
            return

        try:
            # Confirmación para guardar los cambios
            reply = QMessageBox.question(self, "Confirmar", f"¿Está seguro de que desea guardar los cambios para {selected_product}?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

            # Conectar a la base de datos PostgreSQL
            connection = psycopg2.connect(
                host="localhost",
                database="Balanza",
                user="postgres",
                password="postgres"
            )
            cursor = connection.cursor()

            # Actualizar los datos del producto en la base de datos
            cursor.execute("UPDATE plu SET unitprice = %s, caducidad = %s WHERE name = %s",
                           (new_unitprice, new_caducidad, selected_product))
            connection.commit()

            QMessageBox.information(self, "Éxito", f"Los cambios para {selected_product} se han guardado correctamente.")

            cursor.close()
            connection.close()

        except (Exception, psycopg2.DatabaseError) as error:
            QMessageBox.critical(self, "Error", f"Error al guardar cambios: {error}")

    def delete_product(self):
        # Eliminar el producto de la base de datos y su carpeta de imágenes
        selected_product = self.product_combobox.currentText()
        if not selected_product:
            QMessageBox.warning(self, "Advertencia", "Seleccione un producto.")
            return

        try:
            # Confirmación para borrar el producto
            reply = QMessageBox.question(self, "Confirmar", f"¿Está seguro de que desea borrar el producto {selected_product} y todos sus registros asociados?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

            # Conectar a la base de datos PostgreSQL
            connection = psycopg2.connect(
                host="localhost",
                database="Balanza",
                user="postgres",
                password="postgres"
            )
            cursor = connection.cursor()

            # Obtener el `id` del producto para eliminar sus registros en las tablas `tot` y `trx`
            cursor.execute("SELECT id FROM plu WHERE name = %s", (selected_product,))
            product_id = cursor.fetchone()[0]

            # Eliminar registros asociados en las tablas `tot` y `trx`
            cursor.execute("DELETE FROM tot WHERE pluid = %s", (product_id,))
            cursor.execute("DELETE FROM trx WHERE pluid = %s", (product_id,))
            connection.commit()

            # Eliminar el producto de la tabla `plu`
            cursor.execute("DELETE FROM plu WHERE name = %s", (selected_product,))
            connection.commit()

            # Eliminar la carpeta de imágenes correspondiente
            image_folder = os.path.join(r'C:\Users\fredd\OneDrive\Documentos\Maestria\BaseImagenes', selected_product)
            if os.path.exists(image_folder):
                shutil.rmtree(image_folder)  # Eliminar la carpeta y todo su contenido

            QMessageBox.information(self, "Éxito", f"El producto {selected_product} y sus registros de transacciones se han borrado correctamente.")

            cursor.close()
            connection.close()

            # Actualizar la lista de productos
            self.product_combobox.clear()
            self.load_products()

        except (Exception, psycopg2.DatabaseError) as error:
            QMessageBox.critical(self, "Error", f"Error al borrar producto: {error}")

    def close_and_return(self):
        # Volver a la ventana principal sin guardar cambios
        self.close()
        self.parent.show()

class RealTimePredictionWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent  # Guardar referencia a la ventana principal
        self.cap = None  # Inicializar la cámara a None
        self.timer = QTimer(self)  # Inicializar el QTimer antes de cualquier uso
        self.timer.timeout.connect(self.update_frame)  # Conectar el QTimer al método update_frame

        # Variables para la captura de datos de peso
        self.peso_registrado = None
        self.captura_activa = False  # Para controlar la captura de peso
        self.thread_captura = None  # Hilo para la captura de peso

        # Cargar el modelo entrenado
        self.model = load_model('modelo_multiclase.keras')

        # Obtener las clases de la carpeta BaseImagenes
        self.classes = sorted(os.listdir(base_dir))

        # Inicializar la interfaz gráfica
        self.initUI()

        # Iniciar captura de peso automáticamente al abrir la ventana
        self.iniciar_captura_peso()

    def initUI(self):
        self.setWindowTitle('Registrar')
        self.setGeometry(100, 100, 800, 600)

        # Crear layout principal
        main_layout = QVBoxLayout()

        # Layout para la cámara y el resultado
        layout = QHBoxLayout()

        # Label para mostrar el stream de la cámara
        self.camera_label = QLabel(self)
        self.camera_label.setFixedSize(640, 480)
        layout.addWidget(self.camera_label)

        # Sub-layout para el resultado y otros elementos
        result_layout = QVBoxLayout()

        # Label para mostrar el resultado de la predicción
        self.result_label = QLabel('Resultado: No hay producto')
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 28px; font-weight: bold;")  # Centrar y estilizar el texto
        result_layout.addWidget(self.result_label)

        # Cuadro de texto para mostrar el valor de 'peso_registrado'
        self.data_label = QLabel('Valor de Peso: 0.0')
        self.data_label.setAlignment(Qt.AlignCenter)
        self.data_label.setStyleSheet("font-size: 24px;")
        result_layout.addWidget(self.data_label)

        layout.addLayout(result_layout)

        # Añadir el layout principal
        main_layout.addLayout(layout)

        # Layout para los botones Aceptar y Volver
        button_layout = QHBoxLayout()

        # Botón "Volver"
        self.btn_back = QPushButton('Volver')
        self.btn_back.clicked.connect(self.close_and_return)
        button_layout.addWidget(self.btn_back, alignment=Qt.AlignRight)  # Alinear a la derecha

        # Botón "Aceptar"
        self.btn_accept = QPushButton('Aceptar')
        self.btn_accept.clicked.connect(self.insert_trx_and_tot_data)
        button_layout.addWidget(self.btn_accept, alignment=Qt.AlignRight)

        # Añadir layout de botones
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # Abrir la cámara y comenzar a capturar
        self.open_camera()

    def open_camera(self):
        self.cap = cv2.VideoCapture(camera_url)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "No se pudo abrir la cámara.")
            return

        self.timer.start(30)  # Actualizar cada 30 ms

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():  # Verificar si la cámara está abierta
            self.timer.stop()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            return

        # Convertir la imagen a formato RGB para PySide6
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_frame.shape
        step = channel * width
        qImg = QImage(rgb_frame.data, width, height, step, QImage.Format_RGB888)

        # Mostrar la imagen en el QLabel
        self.camera_label.setPixmap(QPixmap.fromImage(qImg))

        # Realizar la predicción y actualizar el label
        self.predicted_product = self.predict_product(frame)  # Guardar el producto predicho
        self.result_label.setText(f"Producto Detectado: {self.predicted_product}")

    def predict_product(self, frame):
        img = cv2.resize(frame, (224, 224))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        predictions = self.model.predict(img)
        predicted_index = np.argmax(predictions[0])  # Obtener el índice de la clase con la mayor probabilidad

        # Obtener el nombre de la clase
        predicted_class = self.classes[predicted_index]

        return predicted_class

    def iniciar_captura_peso(self):
        if not self.captura_activa:
            # Iniciar la captura en un hilo separado
            self.thread_captura = threading.Thread(target=self.capturar_datos, args=("192.168.1.125", 502))
            self.thread_captura.start()
            self.captura_activa = True

    def capturar_datos(self, ip, puerto):
        try:
            # Crear el socket de conexión TCP/IP
            con = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            con.settimeout(5)  # Establecer un tiempo de espera de 5 segundos para la conexión

            # Conectar al convertidor USR-W610 (balanza)
            con.connect((ip, puerto))
            print(f"Conectado a la balanza en {ip}:{puerto}")

            peso_anterior = None  # Variable para almacenar el peso anterior

            # Bucle para recibir datos continuamente
            while self.captura_activa:
                try:
                    # Crear un buffer de 1024 bytes para recibir datos
                    buf = bytearray(1024)

                    # Recibir datos de la balanza
                    rec = con.recv_into(buf)  # Leer datos en el buffer

                    # Decodificar los datos recibidos
                    msg = buf[:rec].decode('utf-8', errors='ignore')  # Convertir a cadena de texto
                    if len(msg) == 17:
                        # Extraer los datos que contienen el peso (de índice 2 a 16)
                        data = msg[2:16]  # Extraer la subcadena que debería contener el número
                        # Utilizar expresión regular para encontrar números en la cadena
                        match = re.search(r'\d+\.\d+', data)  # Buscar un número decimal

                        if match:
                            data_float = float(match.group())  # Convertir la parte numérica a float

                            # Verificar si el peso ha cambiado antes de actualizar
                            if peso_anterior is None or data_float != peso_anterior:
                                peso_anterior = data_float  # Actualizar el peso anterior
                                self.peso_registrado = data_float  # Guardar el valor de peso registrado
                                self.update_peso_display()  # Actualizar la interfaz gráfica
                        else:
                            print(f"No se encontró un número válido en los datos recibidos: '{data}'")

                    # Pausa breve para evitar sobrecargar la CPU
                    time.sleep(0.15)

                except socket.timeout:
                    print("Tiempo de espera agotado al recibir datos.")
                    break

        except Exception as ex:
            print(f"Error al conectar con la balanza: {ex}")

        finally:
            con.close()
            self.captura_activa = False
            print("Conexión cerrada.")

    def update_peso_display(self):
        if self.peso_registrado is not None:
            self.data_label.setText(f'Valor de Peso: {self.peso_registrado:.2f}')

    def insert_trx_and_tot_data(self):
        if self.predicted_product is None or self.peso_registrado is None:
            QMessageBox.warning(self, "Advertencia", "No se ha detectado ningún producto o peso.")
            return

        try:
            # Conectar a la base de datos PostgreSQL
            connection = psycopg2.connect(
                host="localhost",
                database="Balanza",
                user="postgres",
                password="postgres"
            )
            cursor = connection.cursor()

            # Consultar itemcode y unitprice de la tabla plu para el producto identificado
            select_query = "SELECT itemcode, unitprice FROM plu WHERE name = %s"
            cursor.execute(select_query, (self.predicted_product,))
            result = cursor.fetchone()

            if result is None:
                QMessageBox.warning(self, "Advertencia", "Producto no encontrado en la base de datos.")
                cursor.close()
                connection.close()
                return

            itemcode, unitprice = result

            # Convertir peso registrado a Decimal antes de multiplicar
            totalprice = unitprice * Decimal(self.peso_registrado)

            # Insertar los datos en la tabla trx
            insert_query = """
            INSERT INTO trx (pluid, name, weight, totprice, date, time)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            current_date = datetime.now().date()
            current_time = datetime.now().time()
            cursor.execute(insert_query, (itemcode, self.predicted_product, Decimal(self.peso_registrado), totalprice, current_date, current_time))

            # Insertar los mismos datos en la tabla tot
            insert_query_tot = """
            INSERT INTO tot (pluid, name, weight, totprice, date, time)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query_tot, (itemcode, self.predicted_product, Decimal(self.peso_registrado), totalprice, current_date, current_time))

            # Confirmar la transacción
            connection.commit()
            QMessageBox.information(self, "Éxito", "Producto añadido a la venta.")

            cursor.close()
            connection.close()

        except (Exception, psycopg2.DatabaseError) as error:
            QMessageBox.critical(self, "Error", f"Error al insertar datos en las tablas trx y tot: {error}")

    def close_and_return(self):
        # Detener la cámara y el temporizador, cerrar la ventana y volver a la principal
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.captura_activa = False  # Detener la captura de peso
        self.close()
        self.parent.show()  # Mostrar la ventana principal

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.captura_activa = False  # Detener la captura de peso
        cv2.destroyAllWindows()
        event.accept()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.reset_trx_table()

    def initUI(self):
        self.setWindowTitle('Ventana Principal')
        self.setGeometry(100, 100, 300, 300)  # Ajustar tamaño de la ventana

        layout = QVBoxLayout()

        # Título
        title_label = QLabel('Visión Artificial y IoT en balanza comercial', self)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")  # Ajustar el estilo del texto
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Imagen
        image_label = QLabel(self)
        pixmap = QPixmap(r'C:\Users\fredd\OneDrive\Documentos\Maestria\Imagenes\Logo_VIU.png')  # Ruta de la imagen
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)  # Centrar la imagen
        layout.addWidget(image_label)

        self.btn_empty_plate = QPushButton('Plato Vacío')
        self.btn_empty_plate.clicked.connect(self.capture_empty_plate)
        layout.addWidget(self.btn_empty_plate)

        self.btn_add = QPushButton('Agregar Producto')
        self.btn_add.clicked.connect(self.add_product)
        layout.addWidget(self.btn_add)

        self.btn_modify = QPushButton('Modificar o Borrar producto')
        self.btn_modify.clicked.connect(self.modify_product)
        layout.addWidget(self.btn_modify)

        self.btn_register = QPushButton('Registrar Venta ')
        self.btn_register.clicked.connect(self.capture_weight)
        layout.addWidget(self.btn_register)

        # Botón "Exportar Venta"
        self.btn_export = QPushButton('Exportar Venta')
        self.btn_export.clicked.connect(self.export_sale)
        layout.addWidget(self.btn_export)  # Añadir el botón "Exportar Venta" entre "Registrar" y "Salir"

        self.btn_report = QPushButton('Reporte Total de Ventas')
        self.btn_report.clicked.connect(self.open_report_window)
        layout.addWidget(self.btn_report)

        self.btn_exit = QPushButton('Salir')  # Botón de Salir
        self.btn_exit.clicked.connect(self.close)  # Cerrar la aplicación
        layout.addWidget(self.btn_exit)

        self.setLayout(layout)

    def reset_trx_table(self):
        try:
            # Conectar a la base de datos PostgreSQL
            connection = psycopg2.connect(
                host="localhost",
                database="Balanza",
                user="postgres",
                password="postgres"
            )
            cursor = connection.cursor()

            # Eliminar todos los registros de la tabla trx
            cursor.execute("DELETE FROM trx")
            connection.commit()

            cursor.close()
            connection.close()
            #print("Tabla 'trx' reiniciada exitosamente.")

        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error al reiniciar la tabla 'trx': {error}")

    def capture_empty_plate(self):
        self.capture_window = CaptureImages("Plato_Vacio")
        self.capture_window.show()
        self.close()

    def add_product(self):
        self.add_product_window = AddProductWindow(self)  # Pasar self como referencia
        self.add_product_window.show()
        self.close()

    def modify_product(self):
        self.modify_product_window = ModifyProductWindow(self)  # Crear ventana de modificación de producto
        self.modify_product_window.show()
        self.close()

    def train_model(self):
        self.training_window = TrainingWindow()
        self.training_window.show()
        self.close()

    def capture_weight(self):
        self.real_time_window = RealTimePredictionWindow(self)  # Pasar self como referencia de la ventana principal
        self.real_time_window.show()
        self.close()

    def export_sale(self):
        try:
            # Conectar a la base de datos PostgreSQL
            connection = psycopg2.connect(
                host="localhost",
                database="Balanza",
                user="postgres",
                password="postgres"
            )
            cursor = connection.cursor()

            # Crear la carpeta "C:\Ventas" si no existe
            export_dir = r'C:\Ventas'
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)

            # Nombre del archivo siempre será "transaccion.csv"
            export_path = os.path.join(export_dir, 'transaccion.csv')

            # Exportar datos de la tabla trx a un DataFrame de pandas excluyendo la columna 'id'
            select_query = "SELECT pluid, name, weight, totprice, date, time FROM trx"
            data = pd.read_sql(select_query, connection)

            # Redondear 'totprice' a 4 decimales y formatear 'time' a 'hora:minutos:segundos'
            data['totprice'] = data['totprice'].apply(lambda x: round(x, 4))
            data['time'] = data['time'].apply(lambda x: x.strftime('%H:%M:%S'))  # Convertir tipo time a string 'HH:MM:SS'

            # Exportar el DataFrame a un archivo CSV sobrescribiendo el archivo existente
            data.to_csv(export_path, index=False)

            # Eliminar todos los datos de la tabla trx
            delete_query = "DELETE FROM trx"
            cursor.execute(delete_query)

            # Reiniciar el contador de la columna id de la tabla trx
            reset_sequence_query = "ALTER SEQUENCE trx_id_seq RESTART WITH 1"  # Ajusta esto según el nombre real de la secuencia
            cursor.execute(reset_sequence_query)

            # Confirmar los cambios
            connection.commit()

            QMessageBox.information(self, "Éxito", f"Datos exportados a {export_path}")

            cursor.close()
            connection.close()

        except (Exception, psycopg2.DatabaseError) as error:
            QMessageBox.critical(self, "Error", f"Error al exportar los datos: {error}")

    def open_report_window(self):
        self.report_window = ReportWindow(self)  # Crear la ventana de reporte
        self.report_window.show()
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())