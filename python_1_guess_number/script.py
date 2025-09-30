import random  # Імпортуємо модуль random для генерації випадкових чисел


from typing import Optional

def guess_the_number_game(lower_bound: int = 1, upper_bound: int = 100, max_attempts: Optional[int] = None):
    """Функція для гри 'Вгадай число'.

    Підказки:
    - Введіть ціле число, або 'q' щоб вийти.
    - Число має бути між lower_bound та upper_bound (включно).

    Параметри:
    - lower_bound: нижня межа діапазону (включно)
    - upper_bound: верхня межа діапазону (включно)
    - max_attempts: максимальна кількість зарахованих спроб (None — без обмежень)
    """

    if lower_bound > upper_bound:
        lower_bound, upper_bound = upper_bound, lower_bound

    # Генеруємо випадкове число в заданому діапазоні
    secret_number = random.randint(lower_bound, upper_bound)

    attempts = 0  # Лічильник зарахованих спроб (тільки валідні числа в межах)
    guessed_correctly = False  # Прапорець, що показує, чи вгадав користувач число

    print("Вітаю у грі 'Вгадай число'!")
    msg_range = f"Я загадав число між {lower_bound} та {upper_bound}. Спробуйте вгадати!"
    if max_attempts is not None:
        msg_range += f" У вас є до {max_attempts} спроб."
    print(msg_range)

    # Починаємо цикл гри, який триває, доки користувач не вгадає число
    while not guessed_correctly:
        # Перевіряємо ліміт спроб перед новим введенням
        if max_attempts is not None and attempts >= max_attempts:
            print("Спроби закінчилися. Ви програли. Спробуйте ще раз!")
            break

        try:
            user_guess_str = input("Ваше припущення (або 'q' для виходу): ").strip()
            if user_guess_str.lower() == 'q':
                print("Вихід з гри. Дякуємо за спробу!")
                break

            # Перетворюємо введене значення на ціле число
            user_guess = int(user_guess_str)

            # Перевіряємо, чи вгадування знаходиться в допустимому діапазоні
            if user_guess < lower_bound or user_guess > upper_bound:
                print(f"Будь ласка, введіть число між {lower_bound} та {upper_bound}.")
                continue  # Не зараховуємо спробу

            # Збільшуємо лічильник спроб тільки для валідних вхідних даних у межах діапазону
            attempts += 1

            # Порівнюємо вгадування користувача із загаданим числом
            if user_guess < secret_number:
                print("Загадане число більше.")
            elif user_guess > secret_number:
                print("Загадане число менше.")
            else:
                # Якщо числа співпали, встановлюємо прапорець і виводимо вітальне повідомлення
                guessed_correctly = True
                print(f"Вітаю! Ви вгадали число {secret_number}!")
                print(f"Вам знадобилося {attempts} спроб.")

        except ValueError:
            # Обробка помилки, якщо користувач ввів не ціле число
            print("Будь ласка, введіть дійсне ціле число або 'q' для виходу.")


# Запускаємо функцію гри
if __name__ == "__main__":
    guess_the_number_game()
