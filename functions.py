# Библиотека функций для ноутбука 1-го этапа соревнования Моторика


# Импортируем библиотеки
import pandas as pd
import numpy as np

# графические библиотеки
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# библиотеки машинного обучения
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import os

def get_all_sensors_plot(id, X_train, plot_counter):
    """
    Функция построения диаграммы показания датчиков. Аргумент функции - 
    id - номер наблюдения;
    X_train - обучающая выборка;
    plot_counter - порядковый номер рисунка.
    """
    
    fig = px.line(data_frame=X_train[id].T)
    
    fig.update_layout(
        title=dict(text=f'Рис. {plot_counter}'+' - сигналы датчиков <br> наблюдения ' + str(id), x=.5, y=0.08, xanchor='center'), 
        xaxis_title_text = 'Время', 
        yaxis_title_text = 'Сигналы датчиков', # yaxis_range = [0, 3000],
        legend_title_text='Индекс <br>датчика',
        width=600, height=400,
        margin=dict(l=20, r=20, t=20, b=100),
    )

    #fig.show()

    # сохраним результат в папке figures. Если такой папки нет, то создадим её
    if not os.path.exists("figures"):
        os.mkdir("figures")

    fig.write_image(f'figures/fig_{plot_counter}.png', engine="kaleido")


def get_sensor_list(id, X_train, print_active=False, print_reliable=False):
    """
    Функция печати и импорта в память всех номеров датчиков
    Аргумент функции - номер наблюдения. 
    """
    
    df = pd.DataFrame(data = X_train[id], index = [s for s in range(X_train.shape[1])], 
                        columns = [s for s in range(X_train.shape[2])]
    )
    
    # Создадим список индексов активных и пассивных датчиков. Среднее значение сигнала не превышает 200 единиц.
    active_sensors, passive_sensors, reliable_sensors, unreliable_sensors  = list(), list(), list(), list()
    
    for i in range(X_train.shape[1]):
        # если средняя амплитуда превышает 200, то добавляем индекс в 'active_sensors'
        if df.iloc[i].mean() > 200:
            active_sensors.append(i)
                   
            # Если разница между абсолютными средними значениями за последние 15 сек и первые 60 сек превышает 200,
            # то датчики заносим в список надежных. Остальные датчики с малой амплитудой - в список ненадёжных. 
            if abs(df.iloc[i][0:49].mean() - df.iloc[i][85:].mean()) > 200:
                reliable_sensors.append(i)
            else:
                unreliable_sensors.append(i)
        else:
            passive_sensors.append(i)
  
      
    if print_active is True:
        print(f"Активные датчики наблюдения " + str(id) +": ", active_sensors)
        print(f"Пассивные датчики наблюдения " + str(id) +":", str(passive_sensors))
    elif print_reliable is True:
        print(f"Датчики с большой амплитудой, наблюдения " + str(id) +": ", reliable_sensors)
        print(f"Датчики с малой амплитудой, " + str(id) +": ", unreliable_sensors)  
    
    return active_sensors, passive_sensors, reliable_sensors, unreliable_sensors



def get_active_passive_sensors_plot(id, X_train, plot_counter):
    """
    Функция построения графика показаний активных и пассивных датчиков.
    Аргумент функции:
    id - номер наблюдения;
    X_train - обучающая выборка;
    plot_counter - порядковый номер рисунка.  
    """
        
    active_sensors, passive_sensors, reliable_sensors, unreliable_sensors = get_sensor_list(id, X_train) # списки сенсоров не печатаем

    
    df = pd.DataFrame(data = X_train[id], 
        index = [s for s in range(X_train.shape[1])], 
        columns = [s for s in range(X_train.shape[2])]
    )
    
    #get_sensor_list(id, False)

    df_3 = pd.DataFrame(X_train[id][active_sensors].T, columns=active_sensors)
    df_4 = pd.DataFrame(X_train[id][passive_sensors].T, columns=passive_sensors)

    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('активные датчики', 'пассивные датчики')
    )
    
    for i in df_3.columns: fig.add_trace(go.Scatter(x=df_3.index, y=df_3[i], name=str(df[i].name)), row=1, col=1)

    for i in df_4.columns: fig.add_trace(go.Scatter(x=df_4.index, y=df_4[i], name=str(df[i].name)), row=1, col=2)

    fig.update_layout(title={'text':f'Рис. {plot_counter}'+' - Активные и пассивные датчики наблюдения ' + str(id), 'x':0.5, 'y':0.05}
    )

    fig.update_layout(width=1200, height=400, legend_title_text ='Номер датчика',
                        xaxis_title_text  = 'Время',  yaxis_title_text = 'Сигнал датчика', yaxis_range=  [0, 3500], 
                        xaxis2_title_text = 'Время', yaxis2_title_text = 'Сигнал датчика', yaxis2_range= [0 , 200],
                        margin=dict(l=100, r=60, t=80, b=100), 
                        #showlegend=False # легенда загромождает картинку
    )

    #fig.show()
    fig.write_image(f'figures/fig_{plot_counter}.png', engine="kaleido")

def get_test_id(id, y_train):
    """
    #Функция отображения списка наблюдений.
    #Аргументы функции: список из номера теста (timestep) и класса жеста
    """
    
    samples = list(y_train[y_train['Class']==id].index)
    samples = pd.Series(data=samples, name=f'{str(id)}', index=[id]*len(samples))
    return samples


def get_sensors_in_all_tests_plot(id, arg2, X_train, y_train, plot_counter):
    """
    Функция вывода диаграммы показания отдельных датчиков во всех наблюдениях конкретного жеста
    Аргументом функции является строка - список датчиков
    """
    #функция отбора наблюдений в переменную 'samples' 
    samples = get_test_id(id, y_train)
    sensor_list = arg2 
   

    df_selected = pd.DataFrame(columns=range(X_train[id].shape[1]))
    for sample in samples:
        for sensor in sensor_list:
            df = pd.DataFrame(X_train[sample,sensor]).T
            df_selected = pd.merge(df_selected, df, how='outer') 

    # определим сколько графиков выводить
    len(arg2)
    if len(arg2)%2==0:

        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            f"датчик {arg2[0]}", f"датчик {arg2[1]}", 
            f"датчик {arg2[2]} ", f"датчик {arg2[3]}"),
            vertical_spacing=.15
        )

        df_1 = df_selected.iloc[0::len(sensor_list)].T
        df_2 = df_selected.iloc[1::len(sensor_list)].T
        df_3 = df_selected.iloc[2::len(sensor_list)].T
        df_4 = df_selected.iloc[3::len(sensor_list)].T

        for i in df_1.columns: 
            fig.add_trace(go.Scatter(x=df_1.index, y=df_1[i]), row=1, col=1)

        for i in df_2.columns:
            fig.add_trace(go.Scatter(x=df_2.index, y=df_2[i]), row=1, col=2)

        for i in df_3.columns:
            fig.add_trace(go.Scatter(x=df_3.index, y=df_3[i]), row=2, col=1)
            
        for i in df_4.columns:
            fig.add_trace(go.Scatter(x=df_4.index, y=df_4[i]), row=2, col=2)
            


        fig.update_layout(height=800, width=1000, #yaxis_type='log', 
                        title_text="Показания датчиков " + str(sensor_list) + " в наблюдениях жеста 0-1", title_xanchor='left', title_font=dict(size = 16),
                        xaxis_title_text  = 'Время', yaxis_title_text  = 'Сигнал датчика', yaxis_range =[0 ,3500], 
                        xaxis2_title_text = 'Время', yaxis2_title_text = 'Сигнал датчика', yaxis2_range=[0 ,3500], 
                        xaxis3_title_text = 'Время', yaxis3_title_text = 'Сигнал датчика', yaxis3_range=[0 ,3000], 
                        xaxis4_title_text = 'Время', yaxis4_title_text = 'Сигнал датчика', yaxis4_range=[0 ,3000],
                        margin=dict(l=60, r=60, t=50, b=80),  
                        showlegend=False # легенда загромождает картинку
        )
        
        fig.update_layout(title=dict(text=f'Рис. {plot_counter} - Сигнал датчиков во всех наблюдениях жеста {id}', x=0.5, y=0.01, xanchor='center')
        )

        fig.show()
        fig.write_image(f'figures/fig_{plot_counter}.png', engine="kaleido")


def data_preproc(df_data):
    "Функция преобразования выборки: разница показателей всех датчиков за"
    "первую и вторую половину времени активности по каждому объекту"
    
    sens_cols = []
    for i in range(df_data.shape[1]):
        sens_cols.append('sensor_' + str(i))

    
    df_feated = pd.DataFrame(columns=sens_cols)
    for i in range(df_data.shape[0]):
        sensors = pd.DataFrame(data=df_data[i], index=range(40), columns=range(60)).T
        df_feated.loc[i, sens_cols] = (sensors.iloc[30:].mean() - sensors.iloc[:30].mean()).values.reshape(-1) 
    
    return df_feated

def stratified_cross_valid(model, X_train, y_train, n, metric):
    "Функция оценки модели на кросс-валидации: вывод графика с результатами кросс-валидации,"
    "кривой достаточности данных и таблицы с результатами"
    
    cv_splitter = StratifiedKFold(n_splits = n)

    metric_table = pd.DataFrame()

    i = 0
                    
    cv_res = cross_validate(
        model, 
        X_train, 
        y_train, 
        scoring = metric, 
        n_jobs = -1, 
        cv = cv_splitter, 
        return_train_score = True, 
        verbose = 0)

    cv_train = cv_res['train_score'].mean()
    cv_test = cv_res['test_score'].mean()


    metric_table.loc[i, 'cv_train'] = cv_train
    metric_table.loc[i, 'cv_test'] = cv_test

                    
    metric_table['cv_dif'] = metric_table['cv_train'] - metric_table['cv_test']

    #результаты кросс-валидации
    fig = sns.pointplot(np.arange(n), cv_res['train_score'], color = 'r')
    fig = sns.pointplot(np.arange(n), cv_res['test_score'], color = 'g')
    fig.set_title('Результаты кросс-валидации', fontsize=16)
    fig.set_xlabel('Порядковый номер части совокупности')
    fig.set_ylabel('Показатель качества модели\n f-score')
    #plot.set_ylim(0.4, 1.1)
    fig.grid() # показать сетку
    plt.show()

    # кривая достаточности данных
    result = []

    s = len(X_train)
    p = len(X_train) // (n + 1)
    for i in np.arange(p, s - p + 1, p):
        model.fit(X_train.iloc[:i], y_train.iloc[:i])
        predict = model.predict(X_train.iloc[i:i+p])
        res = f1_score(y_train.iloc[i:i+p], predict,  average='macro')
        result.append(res)

    fig = sns.pointplot(np.arange(len(result)), result)
    fig.set_title('Кривая достаточности данных', fontsize=16)
    fig.set_xlabel('Итерация')
    fig.set_ylabel('Показатель качества  модели\n f-score')
    fig.set_ylim(0.4, 1.05)
    fig.grid() # показать сетку
    plt.show()
    print ()

    display(metric_table.sort_values(by ='cv_test'))
    print ()