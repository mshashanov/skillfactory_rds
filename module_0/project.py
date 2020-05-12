import numpy as np

def randint(rand_size=None):
    return np.random.randint(1, 101, size=rand_size)

def game_core_v1(number):
    '''Просто угадываем на random, никак не используя информацию о больше или меньше.
       Функция принимает загаданное число и возвращает число попыток'''
    count = 0
    while True:
        count += 1
        predict = randint() # предполагаемое число
        if number == predict:
            return count # выход из цикла, если угадали

def game_core_v2(number):
    '''Сначала устанавливаем любое random число, а потом уменьшаем или увеличиваем его в зависимости от того, больше оно или меньше нужного.
       Функция принимает загаданное число и возвращает число попыток'''
    count = 0
    predict = randint()
    while number != predict:
        count += 1
        if number > predict:
            predict += 1
        elif number < predict:
            predict -= 1
    return count # выход из цикла, если угадали

def game_core_v3(number):
    '''Алгоритм с делением пополам диапазона поиска случайного числа'''
    count = 0
    predict = 50
    range = [1, 100]
    predict_ls = []
    while number != predict:
        count += 1

        if number > predict:
            range[0] = predict
        elif number < predict:
            range[1] = predict

        # Случай для крайних диапазонов
        if (range in [[1, 2], [99, 100]]):
            if (predict == range[0]):
                predict = range[1]
            elif (predict == range[1]):
                predict = range[0]
        # Общий случай
        else:
            predict = round(sum(range) / len(range))

        predict_ls.append(predict)

        # Проверка, если попадает в бесконечный цикл
        if (count > 10):
            print(f"number = {number}, range = {range}, predict = {predict}")
            if (count > 15):
                return 100

    return count # выход из цикла, если угадали

def score_game(game_core):
    '''Запускаем игру 1000 раз, чтобы узнать, как быстро игра угадывает число'''
    count_ls = []
    np.random.seed(1)  # фиксируем RANDOM SEED, чтобы ваш эксперимент был воспроизводим!
    random_array = randint(rand_size=1000)
    for number in random_array:
        count_ls.append(game_core(number))

    min_ls, max_ls = min(random_array), max(random_array)
    # print(f"random_array: min = {min_ls}, max = {max_ls}")
    min_ls, max_ls = min(count_ls), max(count_ls)
    # print(f"count_ls: min = {min_ls}, max = {max_ls}")

    score = int(np.mean(count_ls))
    name = game_core.__name__
    print(f"Ваш алгоритм \"{name}\" угадывает число в среднем за {score} попыток")
    return score

# запускаем
score_game(game_core_v1)
score_game(game_core_v2)
score_game(game_core_v3)
