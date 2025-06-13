import numpy as np
from math import sqrt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer, plot_bloch_multivector, plot_histogram
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

import qiskit
print(f"Qiskit version: {qiskit.__version__}")

class Complex:
    def __init__(self, real=0, img=0):
        self.real = float(real)
        self.img = float(img)

    def module(self):
        return sqrt(self.real**2 + self.img**2)

    def __add__(self, other):
        return Complex(self.real + other.real, self.img + other.img)

    def __iadd__(self, other):
        self.real += other.real
        self.img += other.img
        return self

    def __neg__(self):
        return Complex(-self.real, -self.img)

    def __sub__(self, other):
        return self + (-other)

    def __isub__(self, other):
        self.real -= other.real
        self.img -= other.img
        return self

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Complex(self.real * other, self.img * other)
        return Complex(
            self.real * other.real - self.img * other.img,
            self.real * other.img + self.img * other.real
        )

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            self.real *= other
            self.img *= other
        else:
            real = self.real * other.real - self.img * other.img
            img = self.real * other.img + self.img * other.real
            self.real = real
            self.img = img
        return self

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Complex(self.real * other, self.img * other)
        return self.__mul__(other)

    def conjugate(self):
        return Complex(self.real, -self.img)

    def __eq__(self, other):
        return self.real == other.real and self.img == other.img

class Qubit:
    def __init__(self, *args):
        if len(args) == 2 and all(isinstance(arg, Complex) for arg in args):
            c1, c2 = args
            self.size = 2
            norm_sq = c1.module()**2 + c2.module()**2
            self.normal = 1 / sqrt(norm_sq) if norm_sq > 0 else 1
            self.items = np.array([c1, c2], dtype=object)
            if self.normal < 1:
                self.normalize()
        elif len(args) == 4 and all(isinstance(arg, Complex) for arg in args):
            c1, c2, c3, c4 = args
            self.size = 4
            norm_sq = sum(c.module()**2 for c in args)
            self.normal = 1 / sqrt(norm_sq) if norm_sq > 0 else 1
            self.items = np.array([c1, c2, c3, c4], dtype=object)
            if self.normal < 1:
                self.normalize()
        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            data = args[0]
            size = len(data)
            if not (size >= 2 and (np.log2(size).is_integer()) and data is not None):
                raise ValueError("Invalid qubit size or data")
            self.size = size
            self.items = np.array([Complex(c.real, c.img) if isinstance(c, Complex) else Complex(c) for c in data], dtype=object)
            norm_sq = sum(c.module()**2 for c in self.items)
            self.normal = 1 / sqrt(norm_sq) if norm_sq > 0 else 1
            if self.normal < 1:
                self.normalize()
        else:
            raise ValueError("Invalid Qubit initialization")

    def normalize(self):
        for i in range(self.size):
            self.items[i] *= self.normal
        self.normal = 1

    def getSize(self):
        return self.size

    def getNormal(self):
        return self.normal

    def __getitem__(self, index):
        if index < 0 or index >= self.size:
            return None
        return self.items[index]

    def __mul__(self, other):
        tmp = np.array([self.items[i] * other[j] for i in range(self.size) for j in range(other.getSize())], dtype=object)
        return Qubit(tmp)

    def __add__(self, other):
        if self.size != other.getSize():
            raise ValueError("Qubit sizes must match for addition")
        tmp = np.array([self.items[i] + other[i] for i in range(self.size)], dtype=object)
        return Qubit(tmp)

class Gate:
    def __init__(self, size, data):
        if not (np.log2(size).is_integer() and data is not None):
            raise ValueError("Invalid gate size or data")
        self.size = size
        self.items = np.array(data, dtype=object).reshape(size, size)

    def __mul__(self, qubit):
        if self.size != qubit.getSize():
            raise ValueError("Gate and qubit dimensions do not match")
        tmp = np.array([Complex(0, 0) for _ in range(self.size)], dtype=object)
        for i in range(self.size):
            for j in range(self.size):
                tmp[i] += self.items[i, j] * qubit[j]
        return Qubit(tmp)

    def getSize(self):
        return self.size

    def __call__(self, i, j):
        if i < 0 or i >= self.size or j < 0 or j >= self.size:
            raise ValueError("Index out of range")
        return self.items[i, j]

def print_complex(c, end='\n'):
    if c.real == 0 and c.img == 0:
        print('0', end=end)
        return
    if c.real != 0:
        print(f"{c.real}", end='')
    if c.real != 0 and c.img != 0:
        print("+", end='')
    if c.img != 0:
        if c.img == 1:
            print("i", end='')
        elif c.img == -1:
            print("-i", end='')
        else:
            print(f"{c.img}i", end='')
    print(end=end)

def print_qubit(q, end='\n'):
    if abs(q.getNormal() - 1) > 1e-10:
        print(f"{q.getNormal()} * ", end='')
    print('(', end='')
    for i in range(q.getSize()):
        print_complex(q[i], ' ')
    print(')', end=end)

def print_gate(g, end='\n'):
    print('(', end='')
    for i in range(g.getSize()):
        for j in range(g.getSize()):
            print_complex(g(i, j), ' ')
        if i != g.getSize() - 1:
            print("\n ", end='')
    print(')', end=end)

class Quantum:
    i = Complex(0, 1)
    b0 = Qubit(Complex(1), Complex(0))
    b1 = Qubit(Complex(0), Complex(1))
    PaulX = Gate(2, [
        Complex(0), Complex(1),
        Complex(1), Complex(0)
    ])
    PaulY = Gate(2, [
        Complex(0), -i,
        i, Complex(0)
    ])
    PaulZ = Gate(2, [
        Complex(1), Complex(0),
        Complex(0), Complex(-1)
    ])
    Hadamard = Gate(2, [
        Complex(1/sqrt(2)), Complex(1/sqrt(2)),
        Complex(1/sqrt(2)), Complex(-1/sqrt(2))
    ])
    CNOT = Gate(4, [
        Complex(1), Complex(0), Complex(0), Complex(0),
        Complex(0), Complex(1), Complex(0), Complex(0),
        Complex(0), Complex(0), Complex(0), Complex(1),
        Complex(0), Complex(0), Complex(1), Complex(0)
    ])

    @staticmethod
    def getOracle_f1():
        ans = np.array([[Complex(0, 0) for _ in range(8)] for _ in range(8)], dtype=object)
        for i in range(0, 8, 2):
            for j in range(8):
                if j - i == 1:
                    ans[i, j] = Complex(1)
                    ans[j, i] = Complex(1)
        return ans.flatten()

    @staticmethod
    def getOracle_fxor():
        ans = np.array([[Complex(0, 0) for _ in range(8)] for _ in range(8)], dtype=object)
        ans[0, 0] = Complex(1)
        ans[1, 1] = Complex(1)
        ans[2, 3] = Complex(1)
        ans[3, 2] = Complex(1)
        ans[4, 5] = Complex(1)
        ans[5, 4] = Complex(1)
        ans[6, 6] = Complex(1)
        ans[7, 7] = Complex(1)
        return ans.flatten()

    Oracle_f1 = Gate(8, getOracle_f1.__func__())
    Oracle_fxor = Gate(8, getOracle_fxor.__func__())

    @staticmethod
    def Grover(f):
        psy = Qubit(Complex(1), Complex(1), Complex(1), Complex(1))
        tU = np.zeros((4, 4), dtype=object)
        tU[0, 0] = Complex(-1 if f(0, 0) else 1)
        tU[1, 1] = Complex(-1 if f(0, 1) else 1)
        tU[2, 2] = Complex(-1 if f(1, 0) else 1)
        tU[3, 3] = Complex(-1 if f(1, 1) else 1)
        U = Gate(4, tU.flatten())
        tD = np.full((4, 4), Complex(0.5), dtype=object)
        np.fill_diagonal(tD, Complex(-0.5))
        D = Gate(4, tD.flatten())

        # Визуализация
        qreg = QuantumRegister(2, 'q')
        
        # Шаг 1: Создаём суперпозицию (отдельная схема)
        display(Markdown("### Шаг 1: Создаём суперпозицию"))
        circuit1 = QuantumCircuit(qreg)
        circuit1.h(qreg)
        circuit1.save_statevector()
        
        simulator = AerSimulator(method='statevector')
        job = simulator.run(circuit1)
        result = job.result()
        statevector = result.get_statevector(0)
        display(plot_bloch_multivector(statevector))
        display(circuit1.draw(output='mpl'))

        # Шаг 2: Применяем оракул (отдельная схема)
        display(Markdown("### Шаг 2: Оракул помечает решение"))
        circuit2 = QuantumCircuit(qreg)
        circuit2.h(qreg) 
        # Матрица оракула (только вещественная часть)
        oracle_matrix = np.array([[tU[i, j].real for j in range(4)] for i in range(4)])
        circuit2.unitary(oracle_matrix, qreg, label='Oracle')
        circuit2.save_statevector()
        
        job = simulator.run(circuit2)
        result = job.result()
        statevector = result.get_statevector(0)
        display(plot_bloch_multivector(statevector))
        display(circuit2.draw(output='mpl'))

        # Визуализация матрицы оракула
        plt.figure(figsize=(5, 5))
        sns.heatmap([[float(x.real) for x in row] for row in tU], annot=True, cmap="YlOrRd")
        plt.title(f"Oracle Matrix for {f.__name__}")
        plt.show()

        # Шаг 3: Применяем диффузию (отдельная схема)
        display(Markdown("### Шаг 3: Усиливаем решение (Диффузия)"))
        circuit3 = QuantumCircuit(qreg)
        circuit3.h(qreg) 
        circuit3.unitary(oracle_matrix, qreg, label='Oracle')  # Шаг 2
        circuit3.h(qreg) 
        circuit3.x(qreg)
        circuit3.cz(qreg[0], qreg[1])
        circuit3.x(qreg)
        circuit3.h(qreg)
        circuit3.save_statevector()
        
        job = simulator.run(circuit3)
        result = job.result()
        statevector = result.get_statevector(0)
        display(plot_bloch_multivector(statevector))
        display(circuit3.draw(output='mpl'))

        # Визуализация матрицы диффузии
        plt.figure(figsize=(5, 5))
        sns.heatmap([[float(x.real) for x in row] for row in tD], annot=True, cmap="YlOrRd")
        plt.title(f"Diffusion Matrix for {f.__name__}")
        plt.show()

        # Финальная схема для измерений
        circuit_final = QuantumCircuit(qreg)
        circuit_final.h(qreg)
        circuit_final.unitary(oracle_matrix, qreg, label='Oracle')
        circuit_final.h(qreg)
        circuit_final.x(qreg)
        circuit_final.cz(qreg[0], qreg[1])
        circuit_final.x(qreg)
        circuit_final.h(qreg)
        circuit_final.measure_all()
        
        # Вероятности конечного состояния
        display(Markdown("### Финальные вероятности"))
       
        simulator_qasm = AerSimulator()
        job = simulator_qasm.run(circuit_final, shots=1024)
        result = job.result()
        counts = result.get_counts(0)
        display(plot_histogram(counts, title=f"Вероятности для {f.__name__}"))

        return D * (U * psy)

def f1(x1, x2):
    return x1 and x2

def f2(x1, x2):
    return not x1 and x2

if __name__ == "__main__":
    print("\n======= Задание 1 ======= ")
    print("\n=== Базовые операции === \n")
    
    q = Quantum.b0
    q2 = q
    print("Кубит |0> - ", end='')
    print_qubit(q2)
    
    print("Сумма |0> + |1>  = ", end='')
    print_qubit(Quantum.b0 + Quantum.b1)
    
    print("Тензорное умножение |0> * |1> - ", end='')
    print_qubit(Quantum.b0 * Quantum.b1)
    
    print("\nКубит |010> - ", end='')
    print_qubit(Quantum.b0 * Quantum.b1 * Quantum.b0)
    
    print("Кубит |0000> - ", end='')
    print_qubit(Quantum.b0 * Quantum.b0 * Quantum.b0 * Quantum.b0)
    
    print("\nГейты Паули - \n")
    print("X\n ", end='')
    print_gate(Quantum.PaulX)
    
    print("Y\n", end='')
    print_gate(Quantum.PaulY)
    
    print("Z\n", end='')
    print_gate(Quantum.PaulZ)
    
    print("\nX|0>- ", end='')
    print_qubit(Quantum.PaulX * Quantum.b0)
    
    print("\nCNOT\n", end='')
    print_gate(Quantum.CNOT)
    
    print("\nCNOT|01> - ", end='')
    print_qubit(Quantum.CNOT * (Quantum.b0 * Quantum.b1))
    
    print("\n======= Задание 2 ======= ")
    print("\n=== Оракулы === \n")
    print("Оракул постоянной функции \n", end='')
    print_gate(Quantum.Oracle_f1)
    
    print("Оракул XOR функции \n", end='')
    print_gate(Quantum.Oracle_fxor)
    
    print("Оракул XOR к |000> = ", end='')
    print_qubit(Quantum.Oracle_fxor * (Quantum.b0 * Quantum.b0 * Quantum.b0))
    
    print("Оракул 1 к |010> = ", end='')
    print_qubit(Quantum.Oracle_f1 * (Quantum.b0 * Quantum.b1 * Quantum.b0))
    
    print("\n======= Задание 3 ======= ")
    print("\n=== Алгоритм Гровера === \n")
    
    print("Подбор для \"X1 and X2\" = ", end='')
    ans1 = Quantum.Grover(f1)
    print("Ответ алгоритма Гровера - \"X1 and X2 == 1 при X1X2 = ", end='')
    if ans1[0] == Complex(1): print("00\" ", end='')
    if ans1[1] == Complex(1): print("01\" ", end='')
    if ans1[2] == Complex(1): print("10\" ", end='')
    if ans1[3] == Complex(1): print("11\" - верно ", end='')
    print_qubit(ans1)
    
    print("\nПодбор для \"(not X1) or X2\" = ", end='')
    ans2 = Quantum.Grover(f2)
    print("Ответ алгоритма Гровера - \"(not X1) or X2 == 1 при X1X2 = ", end='')
    if ans2[0] == Complex(1): print("00\" ", end='')
    if ans2[1] == Complex(1): print("01\" - верно ", end='')
    if ans2[2] == Complex(1): print("10\" ", end='')
    if ans2[3] == Complex(1): print("11\" ", end='')
    print_qubit(ans2)
