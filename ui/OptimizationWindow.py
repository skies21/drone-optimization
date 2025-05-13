import math
import random

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QHBoxLayout, QRadioButton, QGroupBox, QMessageBox, QLineEdit, QTextEdit, QCheckBox, QFormLayout
)
import numpy as np
import matplotlib
from matplotlib.figure import Figure
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class OptimizationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Оптимизация маршрута БПЛА")

        main_layout = QVBoxLayout()
        input_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        # Ввод параметров дрона
        self.weight_input = QLineEdit()
        self.range_input = QLineEdit()
        self.battery_input = QLineEdit()
        self.speed_input = QLineEdit()
        self.objective_combo = QComboBox()
        self.objective_combo.addItems(["Оптимизация по времени", "Оптимизация по энергии"])
        self.return_checkbox = QCheckBox("Возврат к начальному пункту")

        left_layout.addWidget(QLabel("Груз дрона (кг):"))
        left_layout.addWidget(self.weight_input)
        left_layout.addWidget(QLabel("Дальность полёта (км):"))
        left_layout.addWidget(self.range_input)
        left_layout.addWidget(QLabel("Заряд батареи (%):"))
        left_layout.addWidget(self.battery_input)
        left_layout.addWidget(QLabel("Скорость дрона (км/ч):"))
        left_layout.addWidget(self.speed_input)
        left_layout.addWidget(QLabel("Цель оптимизации:"))
        left_layout.addWidget(self.objective_combo)
        left_layout.addWidget(self.return_checkbox)

        # Заголовок
        left_layout.addWidget(QLabel("Начальная точка (формат: x,y,z):"))
        self.start_point_input = QLineEdit()
        self.start_point_input.setFont(QFont("Courier", 10))
        left_layout.addWidget(self.start_point_input)

        # Остальные точки
        left_layout.addWidget(QLabel("Остальные точки маршрута (по одной в строке, формат: x,y,z):"))
        self.points_input = QTextEdit()
        self.points_input.setFont(QFont("Courier", 10))
        left_layout.addWidget(self.points_input)

        # Ввода скорости ветра
        form_layout = QFormLayout()
        self.wind_x_input = QLineEdit()
        self.wind_x_input.setPlaceholderText("по X, м/с")
        self.wind_y_input = QLineEdit()
        self.wind_y_input.setPlaceholderText("по Y, м/с")

        form_layout.addRow(QLabel("Скорость ветра X:"), self.wind_x_input)
        form_layout.addRow(QLabel("Скорость ветра Y:"), self.wind_y_input)

        left_layout.addLayout(form_layout)

        # Переключатель режима отображения
        self.display_mode_group = QGroupBox("Режим отображения маршрута")
        display_layout = QHBoxLayout()
        self.simple_mode = QRadioButton("Только маршрут")
        self.detailed_mode = QRadioButton("Маршрут + детали")
        self.simple_mode.setChecked(True)
        display_layout.addWidget(self.simple_mode)
        display_layout.addWidget(self.detailed_mode)
        self.display_mode_group.setLayout(display_layout)
        left_layout.addWidget(self.display_mode_group)

        input_layout.addLayout(left_layout)

        right_layout = QVBoxLayout()

        # Кнопка запуска
        self.run_button = QPushButton("Запустить оптимизацию")
        self.run_button.clicked.connect(self.run_optimization)
        right_layout.addWidget(self.run_button)

        graph_table_layout = QHBoxLayout()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        graph_table_layout.addWidget(self.canvas)

        self.cost_table = QTableWidget()
        self.cost_table.setMinimumWidth(400)
        graph_table_layout.addWidget(self.cost_table)

        right_layout.addLayout(graph_table_layout)
        input_layout.addLayout(right_layout)
        main_layout.addLayout(input_layout)

        self.setLayout(main_layout)

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

            speed = float(self.speed_input.text() or 10)

            if not (0 <= battery <= 100):
                raise ValueError("Заряд должен быть от 0 до 100")

            points = []
            start_text = self.start_point_input.text().strip()
            if start_text:
                try:
                    x, y, z = map(float, start_text.split(','))
                    points.append((x, y, z))
                except ValueError:
                    QMessageBox.warning(self, "Ошибка", "Неверный формат начальной точки. Используйте формат: x,y,z")
                    return

            for line in self.points_input.toPlainText().strip().splitlines():
                if line.strip():
                    try:
                        x, y, z = map(float, line.strip().split(','))
                        points.append((x, y, z))
                    except ValueError:
                        QMessageBox.warning(self, "Ошибка", f"Неверный формат точки: {line}")
                        return

            if len(points) < 2:
                raise ValueError("Введите как минимум 2 точки.")

            optimized_route, total_cost, steps, total_distance = self.genetic_algorithm(
                points, weight, battery, max_distance, objective, return_to_start, wind_vector, speed
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

        # Увеличиваем энергозатраты в зависимости от веса
        weight_penalty = 1 + (weight ** 1.5) / 100  # Экспоненциально

        # Ветер влияет только на X и Y
        direction = (dx / base_distance, dy / base_distance)
        wind_x, wind_y = wind_vector
        wind_proj = (wind_x * direction[0] + wind_y * direction[1])
        wind_effect = 1 - wind_proj / 10
        wind_effect = max(0.5, min(wind_effect, 1.5))

        return base_distance * elevation_penalty * weight_penalty * wind_effect

    def time_cost(self, p1, p2, wind_vector=(0, 0), speed=10., weight=10., k=100):
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

        # Уменьшаем скорость с учётом веса и влияния ветра
        speed = speed / (1 + weight / k) * wind_effect
        return base_distance / speed

    def genetic_algorithm(self, points, weight, battery, max_range, objective, return_to_start, wind_vector, speed):
        population_size = 100
        generations = 500
        mutation_rate = 0.1
        elite_size = 10

        start_point = points[0]
        middle_points = points[1:]

        # Генерация начальной популяции
        population = [self.create_random_route(points, start_point) for _ in range(population_size)]

        for generation in range(generations):
            population = self.evolve_population(
                population, [start_point] + middle_points, weight, battery, max_range,
                objective, return_to_start, wind_vector, mutation_rate, elite_size, speed
            )

        best_route = self.get_best_route(
            population, [start_point] + middle_points, weight, battery, max_range,
            objective, return_to_start, wind_vector, speed
        )

        return best_route['route'], best_route['cost'], best_route['steps'], best_route['total_distance']

    def create_random_route(self, points, start_point):
        middle_points = points[1:]  # исключаем начальную
        random.shuffle(middle_points)
        return [start_point] + middle_points

    def evolve_population(self, population, points, weight, battery, max_range, objective, return_to_start, wind_vector,
                          mutation_rate, elite_size, speed):
        # Этап отбора
        population = sorted(population,
                            key=lambda route: self.calculate_route_cost(route, points, weight, battery, max_range,
                                                                        objective, return_to_start, wind_vector, speed))
        elite = population[:elite_size]

        # Кроссовер (смешивание лучших маршрутов)
        children = []
        for i in range(len(population) - elite_size):
            parent1, parent2 = random.sample(elite, 2)
            child = self.crossover(parent1, parent2)
            children.append(child)

        # Мутация
        for child in children:
            if random.random() < mutation_rate:
                self.mutate(child)

        # Собираем новую популяцию
        population = elite + children
        return population

    def crossover(self, parent1, parent2):
        start_point = parent1[0]
        middle1 = parent1[1:]
        middle2 = parent2[1:]

        crossover_point = random.randint(1, len(middle1))
        child_middle = middle1[:crossover_point] + [p for p in middle2 if p not in middle1[:crossover_point]]
        return [start_point] + child_middle

    def mutate(self, route):
        # Затрагиваем все точки кроме начальной
        idx1, idx2 = random.sample(range(1, len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]

    def calculate_route_cost(self, route, points, weight, battery, max_range, objective, return_to_start, wind_vector,
                             speed):
        total_cost = 0.0
        total_distance = 0.0
        steps = []

        for i in range(len(route) - 1):
            p1, p2 = route[i], route[i + 1]
            if objective == "Оптимизация по времени":
                cost = self.time_cost(p1, p2, wind_vector, speed, weight)
            else:
                cost = self.energy_cost(p1, p2, weight, wind_vector)
            d = self.distance(p1, p2)
            steps.append((p1, p2, d, cost))
            total_cost += cost
            total_distance += d

        if return_to_start:
            p1, p2 = route[-1], route[0]
            d = self.distance(p1, p2)
            cost = self.time_cost(p1, p2, wind_vector, speed, weight) if objective == "Оптимизация по времени" \
                else self.energy_cost(p1, p2, weight, wind_vector)

            steps.append((p1, p2, d, cost))
            total_cost += cost
            total_distance += d

        return total_cost, total_distance, steps

    def get_best_route(self, population, points, weight, battery, max_range, objective, return_to_start, wind_vector,
                       speed):
        best_route = min(population,
                         key=lambda route: self.calculate_route_cost(route, points, weight, battery, max_range,
                                                                     objective, return_to_start, wind_vector, speed))
        total_cost, total_distance, steps = self.calculate_route_cost(best_route, points, weight, battery, max_range,
                                                                      objective, return_to_start, wind_vector, speed)
        return {'route': best_route, 'cost': total_cost, 'steps': steps, 'total_distance': total_distance}

    def visualize_route(self, points, objective, cost):
        self.figure.clear()

        if self.return_checkbox.isChecked():
            points.append(points[0])

        ax_main = self.figure.add_subplot(111, projection='3d')
        x, y, z = zip(*points)
        ax_main.plot(x, y, z, marker='o', linestyle='-', color='blue')

        show_details = self.detailed_mode.isChecked()

        for i, (px, py, pz) in enumerate(points):
            if i == 0:
                ax_main.text(px, py, pz, str(i), fontsize=12, ha='right')
            elif i != len(points) - 1:
                ax_main.text(px, py, pz, str(i), fontsize=12, ha='right')

            if show_details and i > 0:
                prev = points[i - 1]
                ax_main.quiver(prev[0], prev[1], prev[2],
                               px - prev[0], py - prev[1], pz - prev[2],
                               arrow_length_ratio=0.1, color='green')

        try:
            wind_x = float(self.wind_x_input.text() or 0)
            wind_y = float(self.wind_y_input.text() or 0)
        except ValueError:
            wind_x = wind_y = 0

        if wind_x != 0 or wind_y != 0:
            angle = np.arctan2(wind_y, wind_x)
            angle_deg = np.degrees(angle) % 360

            directions = [
                ("С", 0), ("СВ", 45), ("В", 90), ("ЮВ", 135),
                ("Ю", 180), ("ЮЗ", 225), ("З", 270), ("СЗ", 315)
            ]
            # Ближайшее направление
            wind_direction = min(directions, key=lambda x: abs(x[1] - angle_deg))[0]
            wind_str = f" | Ветер: {wind_direction} ({angle_deg:.0f}°)"
        else:
            wind_str = " | Ветер: штиль"

        ax_main.set_title(f"{objective}\nОбщая стоимость: {cost:.2f}{wind_str}", fontsize=14)

        ax_main.set_xlabel("X (км)", fontsize=12)
        ax_main.set_ylabel("Y (км)", fontsize=12)
        ax_main.set_zlabel("Z (км)", fontsize=12)
        ax_main.grid(True)

        self.canvas.draw()

    def display_cost_table(self, steps):
        self.cost_table.clear()
        self.cost_table.setColumnCount(5)
        self.cost_table.setRowCount(len(steps))

        unit = "км" if self.objective_combo.currentText() == "Оптимизация по затратам" else "часы"
        headers = ["Из", "В", "Дистанция (км)", f"Затраты по времени ({unit})", "Затраты по энергии (кВтч)"]
        self.cost_table.setHorizontalHeaderLabels(headers)

        speed = float(self.speed_input.text() or 10)
        weight = float(self.weight_input.text() or 10)
        wind_x = float(self.wind_x_input.text() or 0)
        wind_y = float(self.wind_y_input.text() or 0)
        wind_vector = (wind_x, wind_y)

        for row, (start, end, dist, cost) in enumerate(steps):
            if self.objective_combo.currentText() == "Оптимизация по времени":
                time_cost = self.time_cost(start, end, wind_vector, speed, weight)
                energy_cost = self.energy_cost(start, end, weight, wind_vector)
            else:
                time_cost = self.time_cost(start, end, wind_vector, speed, weight)
                energy_cost = self.energy_cost(start, end, weight, wind_vector)

            self.cost_table.setItem(row, 0, QTableWidgetItem(f"{start}"))
            self.cost_table.setItem(row, 1, QTableWidgetItem(f"{end}"))
            self.cost_table.setItem(row, 2, QTableWidgetItem(f"{dist:.2f}"))

            if unit == "часы":
                hours = int(time_cost)
                minutes = (time_cost - hours) * 60
                self.cost_table.setItem(row, 3, QTableWidgetItem(f"{hours:02}:{int(minutes):02}"))
            else:
                self.cost_table.setItem(row, 3, QTableWidgetItem(f"{time_cost:.2f}"))

            self.cost_table.setItem(row, 4, QTableWidgetItem(f"{energy_cost:.2f} кВтч"))

        self.cost_table.resizeColumnsToContents()
