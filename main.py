from fileinput import filename

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 1. Заголовок
st.write("В этом приложении ты можешь посмотреть данные о котировках компании Apple или визуализировать разные зависмости из исследования по чаевым.")

# 2. Выбор диапазона дат
st.sidebar.header("Котировки акций компании Apple")
start_date = st.sidebar.date_input("Начальная дата", pd.to_datetime("2000-01-01"))
end_date = st.sidebar.date_input("Конечная дата", pd.to_datetime("today"))

# 3. Кнопка для загрузки данных
if st.sidebar.button("Показать котировки"):
    # Загружаем данные о котировках
    apple_stock = yf.download("AAPL", start=start_date, end=end_date)

    if not apple_stock.empty:
        # Выводим таблицу с котировками
        st.subheader("Котировки акций компании Apple")
        st.write(apple_stock)

        # Выводим график
        st.subheader("График котировок акций Apple")
        st.line_chart(apple_stock['Close'])
    else:
        st.error("Выберете другой временной диапазон")

# Построение графиков по tips
st.sidebar.header("Визуализация данных по чаевым")
uploaded_file = st.sidebar.file_uploader('Загрузи CSV файл', type='csv')

if uploaded_file is not None:
    tips = pd.read_csv(uploaded_file)
else:
    path = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv'
    tips = pd.read_csv(path)

if st.sidebar.button("Визуализировать данные по чаевым"):
    st.subheader("Визуализация данных по чаевым")

    np.random.seed(0)
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2023-01-31')
    numb_dates = 244
    dates = pd.to_datetime(np.random.choice(pd.date_range(start_date, end_date), numb_dates))
    tips['time_order'] = dates

    fig = plt.figure(figsize=(12, 6))
    sns.lineplot(data=tips, x='time_order', y='tip', marker='o')
    plt.title('Динамика чаевых во времени в январе 2023 года')
    plt.xlabel('Дата')
    plt.ylabel('Чаевые')
    plt.xticks(rotation=45)
    plt.grid()
    st.pyplot(fig)
    plt.savefig('graph.png')
    with open('graph.png', "rb") as f:
        st.download_button(
            label="Скачать график",
            data=f,
            file_name='graph.png',
            mime="image/png",
            key=1
        )

    tot_bill_hist = sns.displot(tips['total_bill'], bins=30, kde=True, height=6, aspect=1.5)
    tot_bill_hist.fig.suptitle('Гистограмма общего счета', fontsize=16)
    tot_bill_hist.set_axis_labels('Общий счет', 'Частота')
    plt.grid()
    st.pyplot(tot_bill_hist)
    plt.savefig('graph.png')
    with open('graph.png', "rb") as f:
        st.download_button(
            label="Скачать график",
            data=f,
            file_name='graph.png',
            mime="image/png",
            key=2
        )

    tb_tip = sns.relplot(data=tips, x='total_bill', y='tip', hue=tips.index, style='sex', s=50, alpha=1, kind='scatter',
                         height=4, aspect=2)
    tb_tip.fig.suptitle('Связь между общим счетом и чаевыми', fontsize=16)
    tb_tip.set_axis_labels('Общий счет', 'Чаевые')
    tb_tip.set_xticklabels(rotation=45)
    plt.grid()
    st.pyplot(tb_tip)
    plt.savefig('graph.png')
    with open('graph.png', "rb") as f:
        st.download_button(
            label="Скачать график",
            data=f,
            file_name='graph.png',
            mime="image/png",
            key=3
        )

    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=tips, x='total_bill', y='tip', size='size', sizes=(20, 200), hue='time_order', alpha=1)
    plt.title('Связь между общим счётом, чаевыми и размером заказа', fontsize=16)
    plt.xlabel('Общий счёт', fontsize=14)
    plt.ylabel('Чаевые', fontsize=14)
    plt.legend(title='День/Размер заказа', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    st.pyplot(fig)
    plt.savefig('graph.png')
    with open('graph.png', "rb") as f:
        st.download_button(
            label="Скачать график",
            data=f,
            file_name='graph.png',
            mime="image/png",
            key=4
        )

    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(data=tips, x='day', y='total_bill', palette='Set1')
    plt.title('Связь между днем недели и размером счета', fontsize=16)
    plt.xlabel('День недели', fontsize=14)
    plt.ylabel('Размер счета', fontsize=14)
    plt.grid()
    st.pyplot(fig)
    plt.savefig('graph.png')
    with open('graph.png', "rb") as f:
        st.download_button(
            label="Скачать график",
            data=f,
            file_name='graph.png',
            mime="image/png",
            key=5
        )

    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=tips, x='tip', y='day', s=50, color='sex', hue='sex', alpha=1)
    plt.title('Чаевы по дням недели', fontsize=16)
    plt.xlabel('Чаевые', fontsize=14)
    plt.ylabel('День недели', fontsize=14)
    plt.legend(title='Пол посетителя')
    plt.grid()
    st.pyplot(fig)
    plt.savefig('graph.png')
    with open('graph.png', "rb") as f:
        st.download_button(
            label="Скачать график",
            data=f,
            file_name='graph.png',
            mime="image/png",
            key=6
        )

    fig = plt.figure(figsize=(6, 6))
    sns.boxplot(data=tips, x='day', y='total_bill', hue='time', palette='Set1')
    plt.title('Сумма счетов по дням недели и времени (Dinner/Lunch)', fontsize=16)
    plt.xlabel('День недели', fontsize=14)
    plt.ylabel('Сумма счетов', fontsize=14)
    plt.legend(title='Время')
    plt.grid()
    st.pyplot(fig)
    plt.savefig('graph.png')
    with open('graph.png', "rb") as f:
        st.download_button(
            label="Скачать график",
            data=f,
            file_name='graph.png',
            mime="image/png",
            key=7
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(data=tips[tips['time'] == 'Lunch'], x='tip', bins=20, ax=axes[0], color='lightblue')
    axes[0].set_title('Гистограмма чаевых на обед', fontsize=16)
    axes[0].set_xlabel('Чаевые', fontsize=14)
    axes[0].set_ylabel('Частота', fontsize=14)

    sns.histplot(data=tips[tips['time'] == 'Dinner'], x='tip', bins=20, ax=axes[1], color='lightgreen')
    axes[1].set_title('Гистограмма чаевых на ужин', fontsize=16)
    axes[1].set_xlabel('Чаевые', fontsize=14)
    axes[1].set_ylabel('Частота', fontsize=14)
    st.pyplot(fig)
    plt.savefig('graph.png')
    with open('graph.png', "rb") as f:
        st.download_button(
            label="Скачать график",
            data=f,
            file_name='graph.png',
            mime="image/png",
            key=8
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(data=tips[tips['sex'] == 'Male'], x='total_bill', y='tip', hue='smoker', style='smoker', ax=axes[0],
                    palette='Set1', markers=['o', 's'])
    axes[0].set_title('Размер счета и чаевые (Мужчины)', fontsize=16)
    axes[0].set_xlabel('Размер счета', fontsize=14)
    axes[0].set_ylabel('Чаевые', fontsize=14)

    sns.scatterplot(data=tips[tips['sex'] == 'Female'], x='total_bill', y='tip', hue='smoker', style='smoker',
                    ax=axes[1], palette='Set1', markers=['o', 's'])
    axes[1].set_title('Размер счета и чаевые (Женщины)', fontsize=16)
    axes[1].set_xlabel('Размер счета', fontsize=14)
    axes[1].set_ylabel('Чаевые', fontsize=14)
    st.pyplot(fig)
    plt.savefig('graph.png')
    with open('graph.png', "rb") as f:
        st.download_button(
            label="Скачать график",
            data=f,
            file_name='graph.png',
            mime="image/png"
            ,
            key=9
        )

    correlation_matrix = tips.corr(numeric_only=True)
    fig = plt.figure(figsize=(10, 8))

    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
                cbar_kws={"shrink": .8}, linewidths=0.5)
    plt.title('Тепловая карта зависимостей численных переменных', fontsize=16)
    st.pyplot(fig)
    plt.savefig('graph.png')
    with open('graph.png', "rb") as f:
        st.download_button(
            label="Скачать график",
            data=f,
            file_name='graph.png',
            mime="image/png",
            key=9
        )