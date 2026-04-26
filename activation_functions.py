#Модуль функций активации (ФА) выходного нейрона RBF-сети.

from typing import Callable

# Тип «функция активации»: float -> float
ActivationCallable = Callable[[float], float]

class ActivationFunctions: #Контейнер с реализациями ФА и их производных

    # Идентификаторы ФА
    THRESHOLD: int = 1
    RATIONAL_SIGMOID: int = 2

    # Список доступных в варианте 10 ФА (для GUI)
    AVAILABLE_CODES: tuple = (THRESHOLD, RATIONAL_SIGMOID)

    @staticmethod
    def get_name(code: int) -> str: #Удобочитаемое имя ФА для UI
        if code == ActivationFunctions.THRESHOLD:
            return "1: пороговая"
        if code == ActivationFunctions.RATIONAL_SIGMOID:
            return "2: рациональная сигмоида"
        return f"<неизвестная ФА {code}>"

    # функции активации

    @staticmethod
    def threshold(net_value: float) -> float: #Пороговая
        return 1.0 if net_value >= 0.0 else 0.0

    @staticmethod
    def rational_sigmoid(net_value: float) -> float: #Рациональная сигмоида
        ratio: float = net_value / (1.0 + abs(net_value))
        return 0.5 * (ratio + 1.0)

    # производные

    @staticmethod
    def rational_sigmoid_derivative(net_value: float) -> float: #Производная рациональной сигмоиды
        denominator: float = (1.0 + abs(net_value)) ** 2
        return 0.5 / denominator

    # бинаризация выхода

    @staticmethod
    def binarize(activation_value: float) -> int: #Преобразование непрерывного выхода ФА в двоичный
        return 1 if activation_value >= 0.5 else 0

    # вспомогательные методы

    @staticmethod
    def call(code: int, net_value: float) -> float: #Вызвать ФА по её коду
        if code == ActivationFunctions.THRESHOLD:
            return ActivationFunctions.threshold(net_value)
        if code == ActivationFunctions.RATIONAL_SIGMOID:
            return ActivationFunctions.rational_sigmoid(net_value)
        raise ValueError(f"Неизвестный код функции активации: {code}")

    @staticmethod
    def derivative(code: int, net_value: float) -> float: #Производная ФА для дельта-правила
        if code == ActivationFunctions.THRESHOLD:
            return 1.0
        if code == ActivationFunctions.RATIONAL_SIGMOID:
            return ActivationFunctions.rational_sigmoid_derivative(net_value)
        raise ValueError(f"Неизвестный код функции активации: {code}")
