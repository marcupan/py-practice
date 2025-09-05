import random  # Модуль для генерації випадкових елементів
import string  # Модуль, що містить корисні рядкові константи


def generate_password(length, use_uppercase, use_digits, use_symbols):
    """
    Генерує випадковий пароль заданої довжини та складності.

    Args:
      length (int): Бажана довжина пароля.
      use_uppercase (bool): Чи включати великі літери.
      use_digits (bool): Чи включати цифри.
      use_symbols (bool): Чи включати спеціальні символи.

    Returns:
      str: Згенерований пароль.
    """

    # Визначаємо базовий набір символів (малі літери завжди включені)
    character_pool = list(string.ascii_lowercase)  # list() для можливості змінювати список

    # Додаємо інші набори символів до пулу, якщо користувач їх вибрав
    if use_uppercase:
        character_pool.extend(list(string.ascii_uppercase))  # Додаємо великі літери
    if use_digits:
        character_pool.extend(list(string.digits))  # Додаємо цифри
    if use_symbols:
        # Додаємо стандартний набір символів пунктуації
        character_pool.extend(list(string.punctuation))

    # Перемішуємо пул символів для кращої випадковості (опціонально, але рекомендується)
    random.shuffle(character_pool)

    # Генеруємо пароль, випадково вибираючи символи з підготовленого пулу
    # random.choices() дозволяє вибирати кілька елементів зі списку (з можливістю повторень)
    # 'k=length' вказує, скільки символів потрібно вибрати
    password_list = random.choices(character_pool, k=length)

    # Об'єднуємо список символів в один рядок
    password = "".join(password_list)

    return password


# Основна частина програми для взаємодії з користувачем
if __name__ == "__main__":
    print("Генератор безпечних паролів")
    print("-" * 30)

    # Отримуємо бажану довжину пароля від користувача
    while True:  # Цикл для перевірки коректності вводу довжини
        try:
            password_length_str = input("Введіть бажану довжину пароля (наприклад, 12): ")
            password_length = int(password_length_str)
            if password_length <= 0:
                print("Довжина пароля повинна бути позитивним числом.")
            else:
                break  # Виходимо з циклу, якщо довжина коректна
        except ValueError:
            print("Будь ласка, введіть дійсне ціле число для довжини.")


    # Запитуємо користувача про використання різних типів символів
    # Функція для спрощення запиту так/ні
    def ask_yes_no(prompt):
        while True:
            answer = input(f"{prompt} (так/ні): ").lower().strip()
            if answer in ['так', 'т']:
                return True
            elif answer in ['ні', 'н']:
                return False
            else:
                print("Будь ласка, введіть 'так' або 'ні'.")


    include_uppercase = ask_yes_no("Включати великі літери (A-Z)?")
    include_digits = ask_yes_no("Включати цифри (0-9)?")
    include_symbols = ask_yes_no(f"Включати спеціальні символи ({string.punctuation})?")

    # Генеруємо пароль за допомогою нашої функції
    generated_password = generate_password(password_length, include_uppercase, include_digits, include_symbols)

    # Виводимо згенерований пароль
    print("-" * 30)
    print(f"Ваш згенерований пароль: {generated_password}")
    print("-" * 30)
