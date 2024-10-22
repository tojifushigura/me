1import pandas as pd

# Устанавливаем точность отображения данных в Pandas
pd.set_option("display.precision", 2)

# Загрузка данных
data = pd.read_csv('titanic_train.csv', index_col='PassengerId')

# Фильтрация и сортировка по значениям Fare
filtered_data = data[(data['Embarked'] == 'C') & (data['Fare'] > 200)].sort_values(by='Fare', ascending=False)

# Функция для определения возрастной категории
def age_category(age):
    if age < 30:
        return 1
    elif age < 55:
        return 2
    elif age >= 55:
        return 3

# Добавление возрастной категории в датасет
data['Age_category'] = data['Age'].apply(age_category)

# Подсчет мужчин и женщин
male_count = data[data['Sex'] == 'male'].shape[0]
female_count = data[data['Sex'] == 'female'].shape[0]

# Подсчет пассажиров второго класса
pclass_2_count = data['Pclass'].value_counts()[2]

# Медиана и стандартное отклонение Fare
fare_median = round(data['Fare'].median(), 2)
fare_std = round(data['Fare'].std(), 2)

# Средний возраст выживших и умерших
survived_avg_age = data[data['Survived'] == 1]['Age'].mean()
not_survived_avg_age = data[data['Survived'] == 0]['Age'].mean()

# Доля выживших среди молодежи и пожилых
young_survival_rate = data[(data['Age'] < 30) & (data['Survived'] == 1)].shape[0] / data[data['Age'] < 30].shape[0] * 100
old_survival_rate = data[(data['Age'] > 60) & (data['Survived'] == 1)].shape[0] / data[data['Age'] > 60].shape[0] * 100

# Доля выживших среди мужчин и женщин
male_survival_rate = data[(data['Sex'] == 'male') & (data['Survived'] == 1)].shape[0] / data[data['Sex'] == 'male'].shape[0] * 100
female_survival_rate = data[(data['Sex'] == 'female') & (data['Survived'] == 1)].shape[0] / data[data['Sex'] == 'female'].shape[0] * 100

# Наиболее популярное имя среди мужчин
def get_first_name(full_name):
    if '(' in full_name:  # для имен, включающих девичью фамилию
        name = full_name.split('(')[1].split(' ')[0]
    else:
        name = full_name.split(',')[1].split('.')[1].split(' ')[1]
    return name

most_common_male_name = data[data['Sex'] == 'male']['Name'].apply(get_first_name).value_counts().idxmax()

# Средний возраст по классам и полу
avg_age_by_class_and_sex = data.groupby(['Pclass', 'Sex'])['Age'].mean().unstack()

# Формирование выводов
statements = []
if avg_age_by_class_and_sex.loc[1, 'male'] > 40:
    statements.append("В среднем мужчины 1 класса старше 40 лет.")
if avg_age_by_class_and_sex.loc[1, 'female'] > 40:
    statements.append("В среднем женщины 1 класса старше 40 лет.")
if (avg_age_by_class_and_sex['male'] > avg_age_by_class_and_sex['female']).all():
    statements.append("Мужчины всех классов в среднем старше, чем женщины того же класса.")
if (avg_age_by_class_and_sex.loc[1].mean() > avg_age_by_class_and_sex.loc[2].mean() and
    avg_age_by_class_and_sex.loc[2].mean() > avg_age_by_class_and_sex.loc[3].mean()):
    statements.append("В среднем, пассажиры первого класса старше, чем пассажиры 2-го класса, которые старше, чем пассажиры 3-го класса.")

# Сбор данных для таблицы
summary_data = {
    'Показатель': [
        'Количество мужчин', 'Количество женщин',
        'Количество пассажиров второго класса',
        'Медиана Fare', 'Стандартное отклонение Fare',
        'Средний возраст выживших', 'Средний возраст умерших',
        'Доля выживших среди молодежи (<30)', 'Доля выживших среди пожилых (>60)',
        'Доля выживших среди мужчин', 'Доля выживших среди женщин',
        'Наиболее популярное имя среди мужчин'
    ],
    'Значение': [
        male_count, female_count, pclass_2_count,
        fare_median, fare_std,
        survived_avg_age, not_survived_avg_age,
        young_survival_rate, old_survival_rate,
        male_survival_rate, female_survival_rate,
        most_common_male_name
    ]
}

summary_df = pd.DataFrame(summary_data)

# Вывод итогов
print("Итоги анализа:")
print(summary_df)
print("\nУтверждения о среднем возрасте:")
for statement in statements:
    print(statement)
  
