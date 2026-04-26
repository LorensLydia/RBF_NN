#Модуль RBF-сети с гауссовыми радиальными базисными функциями в скрытом слое и одним выходным нейроном с настраиваемой ФА


import math
from typing import List, Tuple

from activation_functions import ActivationFunctions


class RbfNetwork: #RBF-сеть для моделирования булевой функции 4 переменных

    # Константы
    MIN_LEARNING_RATE: float = 1e-6
    MAX_LEARNING_RATE: float = 1.0

    def __init__(self,
                 centers: List[Tuple[int, int, int, int]],
                 learning_rate: float = 0.3,
                 activation_code: int = ActivationFunctions.THRESHOLD):
        #инициализируем приватные поля
        self._centers: List[Tuple[int, int, int, int]] = list(centers)
        self._learning_rate: float = float(learning_rate)
        self._activation_code: int = int(activation_code)
        self._weights: List[float] = []
        self._bias: float = 0.0
        self._last_total_error: int = 0

        self._initialize_weights()

    # свойства

    @property
    def neuron_count(self) -> int: #Число RBF-нейронов в скрытом слое (J)
        return len(self._centers)

    @property
    def weights(self) -> List[float]: #Веса (копия) v1... выходного нейрона
        return list(self._weights)

    @property
    def bias(self) -> float: #Вес смещения v0 выходного нейрона
        return self._bias

    @property
    def learning_rate(self) -> float: #Норма обучения
        return self._learning_rate

    @property
    def activation_code(self) -> int: #Код ФА выходного нейрона
        return self._activation_code

    # инициализация

    def _initialize_weights(self) -> None: #Начальные веса нулевые
        self._weights = [0.0 for _ in range(self.neuron_count)]
        self._bias = 0.0

    # прямое распространение (рабочий режим)

    def _compute_rbf_outputs(self,
                             input_vector: Tuple[int, int, int, int]
                             ) -> List[float]: #Гауссовы РБФ
        rbf_outputs: List[float] = []
        for center in self._centers:
            squared_distance: float = 0.0
            for index in range(len(input_vector)):
                difference: float = input_vector[index] - center[index]
                squared_distance += difference * difference
            rbf_outputs.append(math.exp(-squared_distance))
        return rbf_outputs

    def _compute_net(self, rbf_outputs: List[float]) -> float: #Сетевой (комбинированный) вход выходного нейрона
        net_value: float = self._bias
        for index in range(self.neuron_count):
            net_value += self._weights[index] * rbf_outputs[index]
        return net_value

    def predict_single(self,
                       input_vector: Tuple[int, int, int, int]) -> int: #Двоичный прогноз сети для одного входного вектора
        rbf_outputs: List[float] = self._compute_rbf_outputs(input_vector)
        net_value: float = self._compute_net(rbf_outputs)
        activation: float = ActivationFunctions.call(
            self._activation_code, net_value
        )
        return ActivationFunctions.binarize(activation)

    def predict_all(self,
                    input_vectors: List[Tuple[int, int, int, int]]
                    ) -> List[int]: #Двоичный прогноз для списка входных векторов
        return [self.predict_single(x) for x in input_vectors]

    # обучение по Видроу — Хоффу

    def train(self,
              input_vectors: List[Tuple[int, int, int, int]],
              target_outputs: List[int],
              max_epochs: int) -> List[int]:
        if max_epochs <= 0:
            raise ValueError("Число эпох должно быть положительным")
        if len(input_vectors) != len(target_outputs):
            raise ValueError("Размеры обучающей выборки не согласованы")

        error_history: List[int] = []
        sample_count: int = len(input_vectors)

        for _ in range(max_epochs):
            for sample_index in range(sample_count):
                self._train_one_sample(
                    input_vectors[sample_index],
                    target_outputs[sample_index]
                )

            current_error: int = self._evaluate_total_error(
                input_vectors, target_outputs
            )
            error_history.append(current_error)
            self._last_total_error = current_error

            if current_error == 0:
                break

        return error_history

    def _train_one_sample(self,
                          input_vector: Tuple[int, int, int, int],
                          target_output: int) -> None:
        rbf_outputs: List[float] = self._compute_rbf_outputs(input_vector)
        net_value: float = self._compute_net(rbf_outputs)
        activation: float = ActivationFunctions.call(
            self._activation_code, net_value
        )
        actual_output: int = ActivationFunctions.binarize(activation)

        local_error: int = target_output - actual_output
        if local_error == 0:
            return  # делать нечего: пример уже распознан

        derivative: float = ActivationFunctions.derivative(
            self._activation_code, net_value
        )
        delta_factor: float = self._learning_rate * local_error * derivative

        for index in range(self.neuron_count):
            self._weights[index] += delta_factor * rbf_outputs[index]
        self._bias += delta_factor

    def _evaluate_total_error(
            self,
            input_vectors: List[Tuple[int, int, int, int]],
            target_outputs: List[int]) -> int: #Суммарная ошибка E(k)
        mismatch_count: int = 0
        for index in range(len(input_vectors)):
            predicted: int = self.predict_single(input_vectors[index])
            if predicted != target_outputs[index]:
                mismatch_count += 1
        return mismatch_count
