#2.43
from ast import arg
from pyknow import *
import sys
import random

list_of_questions = [
    ["Какой у вас климат(жаркий/холодный/умеренный)?", #жаркий/холодный/умеренный
        #жаркий
        "Есть ли у вас гараж(да/нет)?" #да/нет
    ],
    ["Автомобиль нужен для дальних поездок(да/нет)?"], #да/нет
    ["Готовы ли вы переплачивать за экологию(да/нет)?", #да/нет
        #да
        "Есть ли у вас газоколонки в городе(да/нет)?", #да/нет
        "Есть ли у вас зарядные электростанции в городе(да/нет)?", #да/нет
        #нет
        "Есть ли у вас сеть бензоколонок в городе(да/нет)?" #да/нет
    ],
    ["Сколько у вас парковочного места?(мало/много)"], #мало/много
    ["Сколько вам нужно пассажиских мест?(1/2/4/5/8/16)"], #1/2/4/5/8/16
    ["Какая вам нужна проходимость?(высокая/низкая/средняя)"], #высокая/низкая/средняя
    ["Какая вам нужна вместимость багажа?(большая/небольшая)"], #большая/небольшая
    ["Машина нужна для роскоши?(да/нет)", #да/нет
        #да
        "Какая должна быть скорость?(высокая/низкая)" #высокая/низкая
    ],
    ["У вас большая семья?(да/нет)"], #да/нет
    ["На сколько сильно вам нужен автомомиль для поездок по городу?(сильно/не нужен/желательно)"], #сильно/не нужен/желательно
    ["Есть ли у вас домашние животные?(да/нет)"], #да/нет
    ["Транспорт нужен для работы?(да/нет)"] #да/нет
]

list_of_answers = [
    [["жаркий", "холодный", "умеренный"], ["да", "нет"]],
    [["да", "нет"]],
    [["да", "нет"], ["нет", "да"], ["нет", "да"], ["да", "нет"]],
    [["много", "мало"]],
    [["1", "2", "4", "5", "8", "16", "больше n", "больше n"]],
    [["высокая", "низкая", "средняя"]],
    [["большая", "небольшая"]],
    [["да", "нет"], ["высокая", "низкая"]],
    [["да", "нет"]],
    #[["большие", "небольшие"]],
    [["сильно", "не нужен", "желательно"]],
    [["да", "нет"]],
    [["да", "нет"]]
]

list_of_facts = [
    [["жаркий климат", "", ""], ["есть гараж", ""]],
    [["для дальних поездок", ""]],
    [["готовность переплачивать за экологию", ""], ["", "есть сеть газоколонок"], ["", "есть сеть зарядных станций"], ["есть сеть бензоколонок", ""]],
    [["много парковочного места", "мало парковочного места"]],
    [["1 место", "2 места", "4 места", "5 мест", "8 мест", "16 мест", "больше n", "больше n"]],
    [["высокая проходимость", "низкая проходимость", "средняя проходимость"]],
    [["большая вместимость багажа", "небольшая вместимость багажа"]],
    [["чтобы покрасоваться", ""], ["высокая максимальная скорость", "низкая максимальная скорость"]],
    [["большая семья", ""]],
    #[["большие", "небольшие"]],
    [["исключительно для поездок по городу", "для поездок на дачу", "в основном для поездок по городу"]],
    [["есть домашнее животное", ""]],
    [["для ведения бизнеса", ""]]
]

class PL(KnowledgeEngine):
    @Rule(OR(Fact('крупные габариты'), Fact('большая семья'), Fact('для поездок на дачу'), Fact('для перевозки крупногабаритных грузов'), Fact('есть домашнее животное'), Fact('для ведения бизнеса')))
    def bigBaggage(self):
        self.declare(Fact('большая вместимость багажа'))

    @Rule(OR(Fact('крупные габариты'), Fact('есть домашнее животное'), Fact('для ведения бизнеса'), Fact('большая семья')))
    def manySeats(self):
        self.declare(Fact('пассажирских сидений больше одного'))

    @Rule(Fact('мало парковочного места'))
    def smallCar(self):
        self.declare(Fact('небольшие габариты'))

    @Rule(Fact('много парковочного места'))
    def bigCar(self):
        self.declare(Fact('крупные габариты'))
        
    @Rule(Fact('чтобы покрасоваться'))
    def rich(self):
        self.declare(Fact('высокая максимальная скорость'))

    @Rule(Fact('в основном для поездок по городу'))
    def averagePassability(self):
        self.declare(Fact('средняя проходимость'))
        
    @Rule(Fact('средняя проходимость'))
    def roadNetwork(self):
        self.declare(Fact('привязан к дорожной сети'))
        
    @Rule(Fact('исключительно для поездок по городу'))
    def lowPassability(self):
        self.declare(Fact('низкая проходимость'))

    @Rule(Fact('для поездок на дачу'))
    def highPassability(self):
        self.declare(Fact('высокая проходимость'))

    @Rule(Fact('высокая проходимость'))
    def infrastuctureNonDepend(self):
        self.declare(Fact('не зависит от инфраструктуры'))
    
    @Rule(OR(Fact('жаркий климат'), AND(Fact('готовность переплачивать за экологию'), Fact('есть сеть зарядных станций'))))
    def electric(self):
        self.declare(Fact('электрический'))

    @Rule(Fact('готовность переплачивать за экологию'), Fact('есть сеть газоколонок'))
    def dizel(self):
        self.declare(Fact('газодизельный'))

    @Rule((Fact('есть сеть бензоколонок')))
    def gasoline(self):
        self.declare(Fact('бензиновый'))
    
    @Rule(Fact('жаркий климат'), Fact('есть гараж'))
    def openedRoof(self):
        self.declare(Fact('открытая крыша'))

    @Rule(Fact('для дальних поездок'))
    def longRide(self):
        self.declare(Fact('долго едет на одном баке'))

    @Rule(Fact('пассажирских сидений больше одного'))
    def manySeats2(self):
        self.declare(Fact('2 места'))

    @Rule(Fact('пассажирских сидений больше одного'))
    def manySeats4(self):
        self.declare(Fact('4 мест'))

    @Rule(Fact('пассажирских сидений больше одного'))
    def manySeats5(self):
        self.declare(Fact('5 мест'))

    @Rule(Fact('пассажирских сидений больше одного'))
    def manySeats8(self):
        self.declare(Fact('8 мест'))

    @Rule(Fact('пассажирских сидений больше одного'))
    def manySeats16(self):
        self.declare(Fact('16 мест'))

    @Rule(Fact('не зависит от инфраструктуры'),
          Fact('газодизельный'),
          Fact('большая вместимость багажа'),
          Fact('2 места'),
          Fact('долго едет на одном баке'),
                NOT(OR(
                    Fact('электрический'),
                    Fact('бензиновый'),
                    Fact('открытая крыша'),
                    Fact('небольшие габариты')
                )))
    def Truck(self):
        self.declare(Fact(pl = 'грузовики'))

    @Rule(Fact('не зависит от инфраструктуры'),
          Fact('долго едет на одном баке'),
          Fact('большая вместимость багажа'),
          Fact('5 мест'),
          Fact('бензиновый'),
                NOT(OR(
                    Fact('электрический'),
                    Fact('газодизельный'),
                    Fact('открытая крыша')
                )))
    def OffRoad(self):
        self.declare(Fact(pl = 'внедорожники'))

    @Rule(Fact('1 место'),
          Fact('бензиновый'),
          Fact('низкая проходимость'),
          Fact('высокая максимальная скорость'),
                NOT(OR(
                    Fact('электрический'),
                    Fact('газодизельный'),
                    Fact('пассажирских сидений больше одного')
                )))
    def Sport(self):
        self.declare(Fact(pl = 'спортивные'))

    @Rule(Fact('5 мест'),
          Fact('привязан к дорожной сети'),
          Fact('электрический'),
          Fact('небольшие габариты'),
            NOT(OR(
                Fact('газодизельный'),
                Fact('бензиновый'),
                Fact('открытая крыша')
            )))
    def ElectroCar(self):
        self.declare(Fact(pl = 'электромобили'))

    @Rule(Fact('долго едет на одном баке'),
          Fact('бензиновый'),
          Fact('2 места'),
          Fact('небольшие габариты'),
          Fact('привязан к дорожной сети'),
          Fact('высокая максимальная скорость'),
          Fact('открытая крыша'),
                NOT(OR(
                    Fact('электрический'),
                    Fact('газодизельный')
                )))
    def Cabriolet(self):
        self.declare(Fact(pl = 'кабриолеты'))
    
    @Rule(Fact('привязан к дорожной сети'),
          Fact('бензиновый'),
          Fact('8 мест'),
          Fact('большая вместимость багажа'),
          Fact('долго едет на одном баке'),
                NOT(OR(
                    Fact('электрический'),
                    Fact('газодизельный'),
                    Fact('открытая крыша'),
                    Fact('небольшие габариты')
                )))
    def Minivan(self):
        self.declare(Fact(pl = 'минивэны'))

    @Rule(Fact('привязан к дорожной сети'),
          Fact('бензиновый'),
          Fact('16 мест'),
          Fact('чтобы покрасоваться'),
                NOT(OR(
                    Fact('электрический'),
                    Fact('газодизельный'),
                    Fact('открытая крыша'),
                    Fact('небольшие габариты')
                )))
    def Limousine(self):
        self.declare(Fact(pl = 'лимузины'))

    @Rule(NOT(
            OR(
                Fact(pl = 'грузовики'),
                Fact(pl = 'внедорожники'),
                Fact(pl = 'спортивные'),
                Fact(pl = 'электромобили'),
                Fact(pl = 'кабриолеты'),
                Fact(pl = 'минивэны'),
                Fact(pl = 'лимузины')
            )
        )
    )
    def Nothing(self):
        self.declare(Fact(pl = 'Nothing'))


    @Rule(Fact(pl = MATCH.a))
    def print_result(self, a):
        if a == "Nothing":
            print('Извините, по вашим запросам ничего подобрать не получилось.')
        else:
            print('Рекомендуем выбрать - {0}'.format(a))
                    
    def factz(self,l):
        for x in l:
            self.declare(x)

def errorMessage():
    print("Использование:\n"
    "{0} --import <testFile>\tимпорт готовых фактов из файла\n"
    "{0} --runtime\t\tнабор фактов от руки. Пустая линия - конец ввода\n"
    "{0} --smart\t\tумный режим (пользователь должен ответить на несколько вопросов для получения результата)".format(sys.argv[0]))
    sys.exit(1)

def addFact(facts, string):
    if string != "":
        facts.append(Fact(string))


def workWithQuestion(facts, num = 0):
    if num == 0:
        num = random.randint(0, len(list_of_questions) - 1)
    level = 0
    while level < len(list_of_questions[num]):
        line = input(list_of_questions[num][level] + '\n')
        length = len(list_of_answers[num][level])
        while line not in list_of_answers[num][level]:
            line = input("Извините, я не могу вас понять. Попробуйте снова.\n")
        index = list_of_answers[num][level].index(line)
        if line == list_of_answers[num][level][0]:
            addFact(facts, list_of_facts[num][level][0])
            level += 1
        elif line == list_of_answers[num][level][length - 1] and level == 0 and len(list_of_questions[num]) > 1:
            level = len(list_of_questions[num]) - 1
        else:
            addFact(facts, list_of_facts[num][level][index])
            break
    list_of_questions.pop(num)
    list_of_answers.pop(num)
    list_of_facts.pop(num)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        errorMessage()
    facts = list()
    if sys.argv[1] == "--import":
        if len(sys.argv) < 3:
            errorMessage()
        try:
            test_file = open(sys.argv[2], 'r')
        except FileNotFoundError:
            print("Файл {0} не найден. Исключение типа FileNotFoundError.".format(sys.argv[2]))
            quit()
        temp = test_file.read().splitlines()
        for l in temp:
            facts.append(Fact(l))
        test_file.close()
    elif sys.argv[1] == "--runtime":
        while True:
            line = input()
            if not line:
                break
            facts.append(Fact(line))
    elif sys.argv[1] == "--smart":
        ans = input("Здравствуйте! Желаете выбрать автомобиль?\n")
        while ans not in ["да", "нет"]:
            ans = input("Извините, я не могу вас понять. Попробуйте снова.\n")
        if ans == "нет":
            print("До свидания.")
            quit()
        ans = input("Нужна ли помощь эксперта?\n")
        while ans not in ["да", "нет"]:
            ans = input("Извините, я не могу вас понять. Попробуйте снова.\n")
        if ans == "нет":
            ans = input("Введите название файла с уже написанными требованиями: ")
            try:
                test_file = open(ans, 'r')
            except FileNotFoundError:
                print("Файл {0} не найден. Исключение типа FileNotFoundError.".format(ans))
                quit()
            temp = test_file.read().splitlines()
            for l in temp:
                facts.append(Fact(l))
            test_file.close()
        else:
            while len(list_of_questions) > 0:
                workWithQuestion(facts)
    else:
        errorMessage()
    ex1 = PL()
    ex1.reset()
    ex1.factz(facts)
    ex1.run()