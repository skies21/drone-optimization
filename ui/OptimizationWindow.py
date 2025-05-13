import math

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QHBoxLayout, QRadioButton, QGroupBox, QMessageBox, QLineEdit, QTextEdit, QCheckBox, QFormLayout
)
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class OptimizationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Оптимизация маршрута БПЛА")

        # Основной вертикальный layout
        main_layout = QVBoxLayout()

        # Ввод параметров дрона
        self.weight_input = QLineEdit()
        self.range_input = QLineEdit()
        self.battery_input = QLineEdit()
        self.objective_combo = QComboBox()
        self.objective_combo.addItems(["Оптимизация по времени", "Оптимизация по энергии"])
        self.return_checkbox = QCheckBox("Возврат к начальному пункту")

        main_layout.addWidget(QLabel("Грузоподъёмность дрона (кг):"))
        main_layout.addWidget(self.weight_input)
        main_layout.addWidget(QLabel("Макс. дальность полёта (км):"))
        main_layout.addWidget(self.range_input)
        main_layout.addWidget(QLabel("Заряд батареи (%):"))
        main_layout.addWidget(self.battery_input)
        main_layout.addWidget(QLabel("Цель оптимизации:"))
        main_layout.addWidget(self.objective_combo)
        main_layout.addWidget(self.return_checkbox)

        # Ввод координат
        main_layout.addWidget(QLabel("Точки маршрута (формат: x,y,z):"))
        self.points_input = QTextEdit()
        self.points_input.setFont(QFont("Courier", 10))
        main_layout.addWidget(self.points_input)

        # Кнопка запуска
        self.run_button = QPushButton("Запустить оптимизацию")
        self.run_button.clicked.connect(self.run_optimization)
        main_layout.addWidget(self.run_button)

        # Поля ввода ветра
        form_layout = QFormLayout()  # нужно создать layout перед использованием
        self.wind_x_input = QLineEdit()
        self.wind_x_input.setPlaceholderText("по X, м/с")
        self.wind_y_input = QLineEdit()
        self.wind_y_input.setPlaceholderText("по Y, м/с")

        form_layout.addRow(QLabel("Скорость ветра X:"), self.wind_x_input)
        form_layout.addRow(QLabel("Скорость ветра Y:"), self.wind_y_input)

        main_layout.addLayout(form_layout)  # добавляем в основной layout

        # График и таблица в горизонтальном layout
        graph_table_layout = QHBoxLayout()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        graph_table_layout.addWidget(self.canvas)

        self.cost_table = QTableWidget()
        self.cost_table.setMinimumWidth(350)  # можно настроить по вкусу
        graph_table_layout.addWidget(self.cost_table)

        # Добавляем горизонтальный layout в основной
        main_layout.addLayout(graph_table_layout)

        self.setLayout(main_layout)

        # Переключатель режима отображения
        self.display_mode_group = QGroupBox("Режим отображения маршрута")
        display_layout = QHBoxLayout()
        self.simple_mode = QRadioButton("Только маршрут")
        self.detailed_mode = QRadioButton("Маршрут + детали")
        self.simple_mode.setChecked(True)  # по умолчанию
        display_layout.addWidget(self.simple_mode)
        display_layout.addWidget(self.detailed_mode)
        self.display_mode_group.setLayout(display_layout)
        main_layout.addWidget(self.display_mode_group)

    def run_optimization(self):
        try:
            weight = float(self.weight_input.text())
            max_distance = float(self.range_input.text())
            battery = float(self.battery_input.text())
            wind_x = float(self.wind_x_input.text() or 0)
            wind_y = float(self.wind_y_input.text() or 0)
            wind_vector = (wind_x, wind_y)

            objective = self.objective_combo.currentText()
            return_to_start = self.return_checkbox.isChecked()

            if not (0 <= battery <= 100):
                raise ValueError("Заряд должен быть от 0 до 100")

            raw_points = self.points_input.toPlainText().strip().split('\n')
            points = []
            for line in raw_points:
                x_str, y_str, z_str = line.strip().split(',')
                points.append((float(x_str), float(y_str), float(z_str)))

            if len(points) < 2:
                raise ValueError("Введите как минимум 2 точки.")

            optimized_route, total_cost, steps, total_distance = self.optimize(
                points, weight, battery, max_distance, objective, return_to_start, wind_vector
            )

            self.visualize_route(optimized_route, objective, total_cost)
            self.display_cost_table(steps)

        except Exception as e:
            QMessageBox.warning(self, "Ошибка", str(e))

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def energy_cost(self, p1, p2, weight, wind_vector=(0, 0)):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        base_distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        if base_distance == 0:
            return 0

        elevation_penalty = 5 + (dz ** 2) / 10 if dz > 0 else 1
        weight_penalty = 1 + (weight / 10)

        # Ветер влияет только на X и Y
        direction = (dx / base_distance, dy / base_distance)
        wind_x, wind_y = wind_vector
        wind_proj = (wind_x * direction[0] + wind_y * direction[1])
        wind_effect = 1 - wind_proj / 10
        wind_effect = max(0.5, min(wind_effect, 1.5))

        return base_distance * elevation_penalty * weight_penalty * wind_effect

    def time_cost(self, p1, p2, wind_vector=(0, 0)):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        base_distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        if base_distance == 0:
            return 0

        direction = (dx / base_distance, dy / base_distance)
        wind_x, wind_y = wind_vector
        wind_proj = (wind_x * direction[0] + wind_y * direction[1])
        wind_effect = 1 - wind_proj / 10
        wind_effect = max(0.5, min(wind_effect, 1.5))

        base_speed = 10
        speed = base_speed * wind_effect
        return base_distance / speed

    def optimize(self, points, weight, battery, max_range, objective, return_to_start, wind_vector):
        visited = [points[0]]
        unvisited = points[1:]
        current = points[0]

        steps = []
        total_cost = 0.0
        total_distance = 0.0

        while unvisited:
            if objective == "Оптимизация по времени":
                next_point = min(unvisited, key=lambda p: self.time_cost(current, p, wind_vector))
                cost = self.time_cost(current, next_point, wind_vector)
            else:
                next_point = min(unvisited, key=lambda p: self.energy_cost(current, p, weight, wind_vector))
                cost = self.energy_cost(current, next_point, weight, wind_vector)

            d = self.distance(current, next_point)
            steps.append((current, next_point, d, cost))

            total_cost += cost
            total_distance += d
            current = next_point
            visited.append(current)
            unvisited.remove(current)

        if return_to_start:
            d = self.distance(current, points[0])
            cost = d if objective == "Оптимизация по времени" else self.energy_cost(current, points[0], weight,
                                                                                    wind_vector)
            steps.append((current, points[0], d, cost))
            total_cost += cost
            total_distance += d
            visited.append(points[0])

        if total_distance > max_range * (battery / 100):
            raise ValueError(
                f"Недостаточная макс. дальность полёта. Требуется {total_distance:.2f} км, доступно {max_range * (battery / 100):.2f} км"
            )

        return visited, total_cost, steps, total_distance

    def visualize_route(self, points, objective, cost):
        self.figure.clear()

        ax_main = self.figure.add_subplot(121, projection='3d')
        x, y, z = zip(*points)
        ax_main.plot(x, y, z, marker='o', linestyle='-', color='blue')

        show_details = self.detailed_mode.isChecked()

        for i, (px, py, pz) in enumerate(points):
            ax_main.text(px, py, pz, str(i), fontsize=9, ha='right')
            if show_details and i > 0:
                prev = points[i - 1]
                ax_main.quiver(prev[0], prev[1], prev[2],
                               px - prev[0], py - prev[1], pz - prev[2],
                               arrow_length_ratio=0.1, color='green')

        ax_main.set_title(f"{objective}\nОбщая стоимость: {cost:.2f}")
        ax_main.set_xlabel("X (км)")
        ax_main.set_ylabel("Y (км)")
        ax_main.set_zlabel("Z (км)")
        ax_main.grid(True)

        # Отображение ветра на компасе
        try:
            wind_x = float(self.wind_x_input.text() or 0)
            wind_y = float(self.wind_y_input.text() or 0)
        except ValueError:
            wind_x = wind_y = 0

        ax_wind = self.figure.add_subplot(122, polar=True)
        ax_wind.set_title("Ветер", fontsize=10)

        if wind_x != 0 or wind_y != 0:
            direction = np.arctan2(wind_y, wind_x)
            magnitude = np.hypot(wind_x, wind_y)

            ax_wind.arrow(direction, 0, 0, magnitude,
                          width=0.1, head_width=0.2, head_length=0.3,
                          fc='red', ec='red')
            ax_wind.set_rlim(0, max(1, magnitude))
        else:
            ax_wind.text(0.5, 0.5, "штиль", transform=ax_wind.transAxes,
                         ha='center', va='center', fontsize=10, color='gray')

        ax_wind.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax_wind.set_xticklabels(['E', 'N', 'W', 'S'])
        ax_wind.set_yticklabels([])

        self.canvas.draw()

    def display_cost_table(self, steps):
        self.cost_table.clear()
        self.cost_table.setColumnCount(4)
        self.cost_table.setRowCount(len(steps))

        unit = "км" if self.objective_combo.currentText() == "Оптимизация по времени" else "кг·км"
        headers = ["Из", "В", "Дистанция (км)", f"Затраты ({unit})"]
        self.cost_table.setHorizontalHeaderLabels(headers)

        for row, (start, end, dist, cost) in enumerate(steps):
            self.cost_table.setItem(row, 0, QTableWidgetItem(f"{start}"))
            self.cost_table.setItem(row, 1, QTableWidgetItem(f"{end}"))
            self.cost_table.setItem(row, 2, QTableWidgetItem(f"{dist:.2f}"))
            self.cost_table.setItem(row, 3, QTableWidgetItem(f"{cost:.2f}"))

        self.cost_table.resizeColumnsToContents()
