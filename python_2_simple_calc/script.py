# Визначаємо функції для кожної арифметичної операції

def add(num1, num2):
    """Функція для додавання двох чисел."""
    return num1 + num2


def subtract(num1, num2):
    """Функція для віднімання другого числа від першого."""
    return num1 - num2


def multiply(num1, num2):
    """Функція для множення двох чисел."""
    return num1 * num2


def divide(num1, num2):
    """Функція для ділення першого числа на друге.
    Обробляє випадок ділення на нуль.

    Повертає float або None, якщо ділення на нуль.
    """
    if num2 == 0:
        return None
    return num1 / num2


# Основна функція калькулятора
def calculator():
    """Запускає цикл роботи простого калькулятора."""
    print("Ласкаво просимо до простого калькулятора!")

    while True:  # Безкінечний цикл, доки користувач не вирішить вийти
        # --- Отримання першого числа ---
        while True:  # Внутрішній цикл для перевірки коректності вводу першого числа
            try:
                num1_str = input("Введіть перше число (або 'q' для виходу): ").strip()
                if num1_str.lower() == 'q':
                    print("Вихід. Дякуємо за використання калькулятора!")
                    return
                num1 = float(num1_str)
                break
            except ValueError:
                print("Помилка: Будь ласка, введіть дійсне число або 'q' для виходу.")

        # --- Отримання операції ---
        # Визначаємо доступні операції
        valid_operations = ['+', '-', '*', '/']
        operation = input(f"Виберіть операцію ({', '.join(valid_operations)}) або 'q' для виходу: ").strip()

        # Перевіряємо, чи введена операція є допустимою
        while True:
            if operation.lower() == 'q':
                print("Вихід. Дякуємо за використання калькулятора!")
                return
            if operation in valid_operations:
                break
            print(f"Недійсна операція. Будь ласка, виберіть одну з: {', '.join(valid_operations)} або 'q' для виходу.")
            operation = input(f"Виберіть операцію ({', '.join(valid_operations)}) або 'q' для виходу: ").strip()

        # --- Отримання другого числа ---
        while True:  # Внутрішній цикл для перевірки коректності вводу другого числа
            try:
                num2_str = input("Введіть друге число (або 'q' для виходу): ").strip()
                if num2_str.lower() == 'q':
                    print("Вихід. Дякуємо за використання калькулятора!")
                    return
                num2 = float(num2_str)
                break
            except ValueError:
                print("Помилка: Будь ласка, введіть дійсне число або 'q' для виходу.")

        # --- Виконання операції та виведення результату ---
        result = None  # Ініціалізуємо змінну для результату

        if operation == '+':
            result = add(num1, num2)
        elif operation == '-':
            result = subtract(num1, num2)
        elif operation == '*':
            result = multiply(num1, num2)
        elif operation == '/':
            result = divide(num1, num2)

        # Виводимо результат або повідомлення про помилку ділення на нуль
        if operation == '/' and result is None:
            print("Помилка: Ділення на нуль неможливе!")
        else:
            print(f"Результат: {num1} {operation} {num2} = {result}")
        print("-" * 20)  # Розділювач для кращої читабельності

        # --- Запит на продовження ---
        next_calculation = input("Хочете виконати ще одну операцію? (так/ні або 'q' для виходу): ").strip().lower()
        if next_calculation not in ('так', 'y', 'yes'):
            print("Дякуємо за використання калькулятора!")
            break  # Виходимо з основного циклу `while True`


# Запускаємо калькулятор, коли скрипт виконується
if __name__ == "__main__":
    calculator()
