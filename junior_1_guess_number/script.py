import random  # Імпортуємо модуль random для генерації випадкових чисел


def guess_the_number_game():
    """Функція для гри 'Вгадай число'."""

    # Встановлюємо діапазон для загадування числа
    lower_bound = 1
    upper_bound = 100
    # Генеруємо випадкове число в заданому діапазоні
    secret_number = random.randint(lower_bound, upper_bound)

    attempts = 0  # Лічильник спроб
    guessed_correctly = False  # Прапорець, що показує, чи вгадав користувач число

    print(f"Вітаю у грі 'Вгадай число'!")
    print(f"Я загадав число між {lower_bound} та {upper_bound}. Спробуйте вгадати!")

    # Починаємо цикл гри, який триває, доки користувач не вгадає число
    while not guessed_correctly:
        # Запитуємо вгадування у користувача
        # Використовуємо try-except для обробки випадків, коли введено не число
        try:
            user_guess_str = input("Ваше припущення: ")
            # Перетворюємо введене значення на ціле число
            user_guess = int(user_guess_str)

            # Збільшуємо лічильник спроб після кожного введення
            attempts += 1

            # Перевіряємо, чи вгадування знаходиться в допустимому діапазоні
            if user_guess < lower_bound or user_guess > upper_bound:
                print(f"Будь ласка, введіть число між {lower_bound} та {upper_bound}.")
                continue  # Переходимо до наступної ітерації циклу

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
            print("Будь ласка, введіть дійсне ціле число.")


# Запускаємо функцію гри
if __name__ == "__main__":
    guess_the_number_game()
