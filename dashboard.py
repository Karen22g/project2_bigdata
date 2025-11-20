import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from pymongo import MongoClient
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
#from dotenv import load_dotenv
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
#from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.models import load_model
from itertools import combinations
from collections import Counter
from sklearn.model_selection import train_test_split

# --------------------------
# Config
# --------------------------
st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# --------------------------
# Utility functions
# --------------------------

client = MongoClient("mongodb://karenproject2:z9PkVdtemdbbrO38N9RO9wEKW0xenBwKw3M1HETe2eDjPI4Abkk8aeN7HUAIuJWdDNV1VAMLgx8oACDbU7EXFw==@karenproject2.mongo.cosmos.azure.com:10255/?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@karenproject2@")
db = client.sample_supplies
collection = db.sales

def crear_lags(df_prod, n_lags=14):
    df = df_prod.copy()

    # Crear 14 lags
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['total_quantity'].shift(i)

    # Eliminar filas con NaN por lags
    df = df.dropna()

    # Variables X finales: lags + month + is_weekend
    columnas_X = [f'lag_{i}' for i in range(1, n_lags + 1)] + ['date','month', 'is_weekend', "day_of_week", 'product']

    return df[['total_quantity'] + columnas_X]

def crear_df(collection):
    pipeline = [
    # Asegura que saleDate siempre sea Date real
    {
        "$addFields": {
            "saleDate": {
                "$cond": [
                    { "$eq": [ { "$type": "$saleDate" }, "string" ] },
                    { "$toDate": "$saleDate" },
                    "$saleDate"
                ]
            }
        }
    },

    # Filtrar los que NO tienen saleDate válido
    { "$match": { "saleDate": { "$ne": None } } },

    # Evitar explosión si items no existe o está vacío
    { "$unwind": { "path": "$items", "preserveNullAndEmptyArrays": False } },

    {
        "$group": {
            "_id": {
                "date": {
                    "$dateToString": {
                        "format": "%Y-%m-%d",
                        "date": "$saleDate"
                    }
                },
                "product": "$items.name"
            },
            "total_quantity": { "$sum": "$items.quantity" },
            "total_revenue": {
                "$sum": { "$multiply": ["$items.quantity", "$items.price"] }
            }
        }
    },

    { "$sort": { "_id.date": 1, "_id.product": 1 } }
    ]

    results = list(collection.aggregate(pipeline, allowDiskUse=True))
    df = pd.DataFrame(results)

    print(df.shape)

    # si aún así faltara _id, lo detectamos
    if "_id" not in df.columns:
        raise ValueError('⚠️ El pipeline regresó documentos sin _id. Revisa input data.')

    df = df.join(pd.json_normalize(df["_id"])).drop(columns=["_id"])
    return df

def preprocess(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['product','date']).reset_index(drop=True)
    #df['total_revenue'] = df['total_revenue'].apply(lambda x: float(x.to_decimal()))
    df['month'] = df['date'].dt.month
    #df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df

df_raw = crear_df(collection)
df = preprocess(df_raw)
df['total_revenue'] = df['total_revenue'].astype(float)
#df['total_revenue'] = df['total_revenue'].apply(lambda x: float(x.to_decimal()))

def crear_lags_df(df_prod, n_lags=14):
    d = df_prod.copy().reset_index(drop=True)
    for i in range(1, n_lags + 1):
        d[f'lag_{i}'] = d['total_quantity'].shift(i)
    d = d.dropna().reset_index(drop=True)
    lag_cols = [f'lag_{i}' for i in range(1, n_lags + 1)]
    cols = ['date','total_quantity','month','is_weekend'] + lag_cols
    return d[cols]

def plot_timeseries(ts, product):
    ts = ts.groupby(['product','date'])[['total_revenue','total_quantity']].sum().reset_index()
    dfp = ts[ts["product"] == product].sort_values("date").copy()

    # Evitar división por cero
    dfp["ingreso_por_unidad"] = dfp.apply(
        lambda row: row["total_revenue"] / row["total_quantity"]
        if row["total_quantity"] > 0 else 0,
        axis=1
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dfp["date"],
        y=dfp["ingreso_por_unidad"],
        mode="lines+markers",
        name="Ingreso por unidad",
        hovertemplate="<b>Fecha:</b> %{x|%Y-%m-%d}<br>" +
                      "<b>Ingreso total:</b> %{customdata[0]}<br>" +
                      "<b>Unidades:</b> %{customdata[1]}<br>" +
                      "<b>Ingreso/und:</b> %{y:.2f}<extra></extra>",
        customdata=dfp[["total_revenue", "total_quantity"]]
    ))

    fig.update_layout(
        title=f"Ingreso promedio por unidad a lo largo del tiempo – {product}",
        xaxis_title="Fecha",
        yaxis_title="Ingreso promedio por unidad ($)",
        template="plotly_white"
    )

    return fig

def build_seq_exog_from_df(df_lags, n_lags=14):
    lag_cols = [f'lag_{i}' for i in range(n_lags, 0, -1)]
    X_seq = df_lags[lag_cols].values.reshape((-1, n_lags, 1)).astype(float)
    X_exog = df_lags[['month','is_weekend']].values.astype(float)
    y = df_lags['total_quantity'].values.astype(float)
    return X_seq, X_exog, y

def combsproducts(collection):
    pipeline = [
        {
            "$addFields": {
                "item_names": {
                    "$map": {
                        "input": "$items",
                        "as": "item",
                        "in": "$$item.name"
                    }
                }
            }
        },
        {
            "$addFields": {
                "n_products": {"$size": {"$setUnion": ["$item_names", []]}}
            }
        },
        {
            "$project": {
                "saleDate": 1,
                "item_names": 1,
                "n_products": 1,
                "_id": 0
            }
        }
    ]

    results = list(collection.aggregate(pipeline, allowDiskUse=True))
    df = pd.DataFrame(results)

        # Flatten todas las combinaciones de productos
    pairs = []
    for items in df['item_names']:
        # combinaciones de 2 productos
        pairs.extend(combinations(sorted(items), 2))

    pair_counts = Counter(pairs)
    pair_df = pd.DataFrame(pair_counts.items(), columns=['pair','count'])
    pair_df[['product_1','product_2']] = pd.DataFrame(pair_df['pair'].tolist(), index=pair_df.index)
    pair_df.drop(columns='pair', inplace=True)

    # Probabilidad conjunta (aparece en la misma venta / total de ventas)
    total_sales = len(df)
    pair_df['probability'] = pair_df['count'] / total_sales

    return pair_df

st.sidebar.header("Filtros")

# --- Filtro por tipo de producto ---
productos = df['product'].dropna().unique().tolist()
producto_selected = st.sidebar.multiselect(
    "Seleccionar producto(s):",
    productos,
    default=productos  # todos por defecto
)

# --- Filtro de tiempo ---
time_filter = st.sidebar.selectbox(
    "Rango de tiempo:",
    [
        "Últimos 6 meses",
        "Último mes",
        "Últimas 2 semanas",
        "Última semana",
        "Personalizado"
    ]
)

# --- Aplicación del filtro temporal ---
df_raw['date'] = pd.to_datetime(df['date'])
today = df_raw['date'].max()  # o pd.Timestamp.today()

if time_filter == "Últimos 6 meses":
    start_date = today - pd.DateOffset(months=6)
elif time_filter == "Último mes":
    start_date = today - pd.DateOffset(months=1)
elif time_filter == "Últimas 2 semanas":
    start_date = today - pd.Timedelta(days=14)
elif time_filter == "Última semana":
    start_date = today - pd.Timedelta(days=7)
elif time_filter == "Personalizado":
    start_date = st.sidebar.date_input("Fecha inicio", today - pd.DateOffset(months=1))
    end_date = st.sidebar.date_input("Fecha fin", today)
else:
    start_date = df['date'].min()
    end_date = today

# Si no es custom, end_date es hoy
if time_filter != "Personalizado":
    end_date = today

# --- Aplicar filtros al DataFrame ---
df_filtered = df[
    (df['product'].isin(producto_selected)) &
    (df['date'] >= pd.to_datetime(start_date)) &
    (df['date'] <= pd.to_datetime(end_date))
]
print(df['total_revenue'].sum())

st.write(f"📌 Mostrando datos desde **{start_date}** hasta **{end_date}**")

tab1, tab2, tab3 = st.tabs(["Sales💰", "Clients👥", "Forecasting📈"])

with tab1:
    st.sidebar.markdown(f"**Filas:** {df_filtered.shape[0]} &nbsp;&nbsp; **Productos:** {df_filtered['product'].nunique()}")

    # Layout main
    st.title("📊 Office Sales Inc - Sales Department Report")
    col1, col2 = st.columns([2,1])

    st.subheader("KPIs")
    total_units = int(df_filtered['total_quantity'].sum())

    total_rev = float(df_filtered['total_revenue'].sum())
    total_rev_day = float(df_filtered['total_revenue'].sum()/df_filtered.groupby('date').ngroups)
    tickets = df_filtered.shape[0]
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Unidades vendidas", f"{total_units}")
    k2.metric("Ingreso total", f"${total_rev:,.2f}")
    k3.metric("Ingreso por día", f"${total_rev_day:,.2f}")
    k4.metric("Dinero promedio por venta", f"${total_rev/tickets:,.2f}")

    # Combinar la serie de tiempo de unidades vendidas de cierto producto vs total de ingreso

    col1, col2 = st.columns([1, 1])

    with col1:
        ts_rev = df_filtered.groupby('date')['total_revenue'].sum().reset_index()
        fig_rev = px.line(ts_rev, x='date', y='total_revenue',
                        title='Ingreso total', height=400)
        fig_rev.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_rev, use_container_width=True)

    with col2:
        ts_tickets = df_filtered.groupby('date').size().rename('tickets').reset_index()
        fig_tickets = px.line(ts_tickets, x='date', y='tickets',
                            title='Tickets vendidos por día', height=400)
        fig_tickets.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_tickets, use_container_width=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        # Pie chart: participación de ingresos por producto
        rev_by_prod = df_filtered.groupby("product")["total_revenue"].sum().reset_index()

        fig_pie = px.pie(
            rev_by_prod,
            values="total_revenue",
            names="product",
            title="Participación de ingresos por producto",
            hole=0.3  # donut style; quítalo si lo quieres en ponqué sólido
        )

        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(template="plotly_white")

        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Selección de producto
        product_list = df_filtered["product"].unique()
        product = st.selectbox("Selecciona un producto", product_list)

        # Selección de tipo de visualización
        viz_options = [
            "Ingreso por día",
            "Boxplot de precio",
            "Correlación precio vs unidades",
            "Unidades vendidas por día"
        ]
        selected_viz = st.selectbox("Selecciona tipo de visualización", viz_options)

        # Mostrar gráfica según selección
        if selected_viz == "Ingreso por día":
            ts = df_filtered[df_filtered["product"]==product].groupby("date")["total_revenue"].sum().reset_index()
            fig = px.line(ts, x="date", y="total_revenue", title=f"Ingreso por día - {product}")
            st.plotly_chart(fig, use_container_width=True)

        elif selected_viz == "Boxplot de precio":
            fig = px.box(df_filtered[df_filtered["product"]==product],
                        y="total_revenue", points="outliers",
                        title=f"Distribución de ingreso (precio aproximado) - {product}")
            st.plotly_chart(fig, use_container_width=True)

        elif selected_viz == "Correlación precio vs unidades":
            dfp = df_filtered[df_filtered["product"]==product]
            dfp = dfp.assign(price_per_unit=dfp["total_revenue"]/dfp["total_quantity"])
            fig = px.scatter(dfp, x="total_quantity", y="price_per_unit",
                            title=f"Precio por unidad vs unidades vendidas - {product}",
                            labels={"total_quantity":"Unidades vendidas","price_per_unit":"Ingreso / unidad"})
            st.plotly_chart(fig, use_container_width=True)

        elif selected_viz == "Unidades vendidas por día":
            ts = df_filtered[df_filtered["product"]==product].groupby("date")["total_quantity"].sum().reset_index()
            fig = px.line(ts, x="date", y="total_quantity", title=f"Unidades vendidas por día - {product}")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Contar cuántas transacciones hay en total
    total_transactions = collection.count_documents({})

    # Pipeline para contar en cuántas transacciones aparece cada producto
    pipeline = [
        {"$unwind": "$items"},
        {"$group": {"_id": "$items.name", "num_transactions": {"$sum": 1}}}
    ]

    results = list(collection.aggregate(pipeline, allowDiskUse=True))
    dfp = pd.DataFrame(results)

    # Calcular la probabilidad de aparición de cada producto
    dfp["total_transactions"] = total_transactions
    dfp["probability"] = dfp["num_transactions"] / dfp["total_transactions"]
    dfp = dfp.rename(columns={"_id": "product"})

    # Sidebar: seleccionar métrica
    metric_sel = st.selectbox(
        "Qué quieres visualizar por producto:",
        options=["Ingreso total ($)", "Unidades vendidas", 'Probabilidad de compra (%)']
    )

    # Agregar por producto
    agg = df_filtered.groupby('product').agg(
        total_revenue=('total_revenue', 'sum'),
        total_units=('total_quantity', 'sum')
    ).reset_index()

    # Ordenar según la métrica seleccionada
    if metric_sel == "Ingreso total ($)":
        agg = agg.sort_values('total_revenue', ascending=False)
        y = 'total_revenue'
        y_label = "Ingreso total ($)"
    if metric_sel == 'Probabilidad de compra (%)':
        agg = agg.merge(dfp[['product','probability']], on='product', how='left')
        agg = agg.sort_values('probability', ascending=False)
        agg['probability'] = agg['probability'] * 100  # convertir a porcentaje
        y = 'probability'
        y_label = "Probabilidad de compra (%)"
    else:
        agg = agg.sort_values('total_units', ascending=False)
        y = 'total_units'
        y_label = "Unidades vendidas"

    # Gráfico de barras
    fig = px.bar(
        agg,
        x='product',
        y=y,
        color=y,
        title=f"Productos según {y_label}",
        text=y,
        template='plotly_white'
    )

    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    fig.update_layout(yaxis_title=y_label, xaxis_title="Producto", uniformtext_minsize=8, uniformtext_mode='hide')

    st.plotly_chart(fig, use_container_width=True)

    combsproducts_df = combsproducts(collection)

    # Tomar los top 7 productos asociados para cada product_1
    top_pairs = combsproducts_df.sort_values(['product_1', 'probability'], ascending=[True, False]) \
                    .groupby('product_1').head(10)

    # Gráfico de barras faceteado por product_1
    fig = px.bar(
        top_pairs,
        x='product_2',
        y='probability',
        text='count',
        facet_col='product_1',
        facet_col_wrap=3,  # 3 gráficos por fila
        labels={'product_2':'Producto asociado', 'probability':'Probabilidad conjunta'},
        title='Productos que se compran juntos (Top asociados por producto)',
    )

    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def dfc(collection):
    pipeline = [
        # 1. Desenrollar items
        {"$unwind": "$items"},

        # 2. Agregar total por item
        {"$addFields": {"item_total": {"$multiply": ["$items.quantity", "$items.price"]}}},

        # 3. Agrupar por venta para sumar totales y capturar cliente
        {"$group": {
            "_id": "$_id",
            "saleDate": {"$first": "$saleDate"},
            "customer": {"$first": "$customer"},
            "customer_gender": {"$first": "$customer.gender"},
            "customer_age": {"$first": "$customer.age"},
            "customer_satisfaction": {"$first": "$customer.satisfaction"},
            "couponUsed": {"$first": "$couponUsed"},
            "storeLocation": {"$first": "$storeLocation"},
            "purchaseMethod": {"$first": "$purchaseMethod"},
            "total_revenue": {"$sum": "$item_total"},
            "items": {"$push": "$items"}
        }},

        # 4. Ordenar por fecha (opcional)
        {"$sort": {"saleDate": 1}}
    ]

    # Ejecutar pipeline
    results = list(collection.aggregate(pipeline, allowDiskUse=True))

    # Convertir a DataFrame
    df_clients = pd.DataFrame(results)

    # Opcional: aplanar la columna de items si quieres analizar los productos individuales
    df_clients["num_items"] = df_clients["items"].apply(lambda x: len(x))
    #df_clients['total_revenue'] = df_clients['total_revenue'].apply(lambda x: float(x.to_decimal()))
    df_clients['total_revenue'] = df_clients['total_revenue'].astype(float)

    return df_clients

df_clients = dfc(collection)
df_clients = dfc(collection)

df_clients['saleDate'] = pd.to_datetime(
    df_clients['saleDate'],
    format='ISO8601'
).dt.normalize()

df_clients['saleDate'] = df_clients['saleDate'].dt.tz_localize(None)

#df_clients['saleDate'] = pd.to_datetime(df_clients['saleDate']).dt.normalize()  # fecha sin hora
print(df_clients['total_revenue'].sum())

df1 = df_clients[
    (df_clients['saleDate'] >= pd.to_datetime(start_date)) &
    (df_clients['saleDate'] <= pd.to_datetime(end_date))
]

with tab2:
    st.header("📊 Análisis general de clientes")

    # --- Selección de tienda ---
    tiendas = ['Todas'] + list(df1['storeLocation'].unique())
    sel_store = st.selectbox("Selecciona una tienda", tiendas)

    if sel_store != 'Todas':
        df_store = df1[df1['storeLocation'] == sel_store]
    else:
        df_store = df1.copy()

    # --- Métricas principales ---
    total_rev_store = df_store['total_revenue'].sum()
    unique_clients = df_store['customer'].apply(lambda x: x['email']).nunique()
    avg_satisfaction = df_store['customer'].apply(lambda x: x['satisfaction']).mean()
    total_tickets = df_store.shape[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Ingresos totales", f"${total_rev_store:,.2f}")
    col2.metric("👥 Clientes únicos", f"{unique_clients}")
    col3.metric("⭐ Satisfacción promedio", f"{avg_satisfaction:.2f}")
    col4.metric("🛒 Compras totales (tickets)", f"{total_tickets}")

    st.markdown("---")

    # --- Gráficos de método de compra, género, cupones ---
    col1, col2, col3 = st.columns(3)
    
    purchase_counts = df_store['purchaseMethod'].value_counts()
    fig_pm = px.pie(purchase_counts, names=purchase_counts.index, values=purchase_counts.values,
                    title="Método de compra")
    col1.plotly_chart(fig_pm, use_container_width=True)

    gender_counts = df_store['customer'].apply(lambda x: x['gender']).value_counts()
    fig_gender = px.pie(gender_counts, names=gender_counts.index, values=gender_counts.values,
                        title="Compras por género")
    col2.plotly_chart(fig_gender, use_container_width=True)

    coupon_counts = df_store['couponUsed'].value_counts()
    fig_coupon = px.pie(coupon_counts, names=coupon_counts.index, values=coupon_counts.values,
                        title="Uso de cupones")
    col3.plotly_chart(fig_coupon, use_container_width=True)

    # --- Gráficos por rango de edad y satisfacción ---
    col1, col2 = st.columns(2)
    
    age_counts = pd.cut(df_store['customer'].apply(lambda x: x['age']),
                        bins=[0,18,25,35,45,55,65,100],
                        labels=["<18","18-25","26-35","36-45","46-55","56-65","65+"]).value_counts().sort_index()
    fig_age = px.bar(age_counts, x=age_counts.index, y=age_counts.values,
                     labels={'x':'Rango de edad','y':'Número de compras'},
                     title="Compras por rango de edad")
    col1.plotly_chart(fig_age, use_container_width=True)

    satisfaction_counts = df_store['customer'].apply(lambda x: x['satisfaction']).value_counts().sort_index()
    fig_sat = px.bar(satisfaction_counts, x=satisfaction_counts.index, y=satisfaction_counts.values,
                     labels={'x':'Nivel de satisfacción','y':'Número de compras'},
                     title="Nivel de satisfacción de clientes")
    col2.plotly_chart(fig_sat, use_container_width=True)

    st.markdown("---")
    
    # --- Comportamiento de un cliente ---
    st.header("👤 Comportamiento de un cliente")
    
    # Crear una columna temporal con el email del cliente
    df_store['customer_email'] = df_store['customer'].apply(lambda x: x['email'])
    df_store['customer_revenue'] = df_store['total_revenue']

    # Ordenar clientes por total de ingresos
    emails = df_store.sort_values('customer_revenue', ascending=False)['customer_email'].unique()

    # Selectbox para escoger un cliente
    sel_email = st.selectbox("Selecciona un cliente", emails)

    # Filtrar df por cliente seleccionado
    df_client = df_store[df_store['customer_email'] == sel_email]
    
    if len(df_client) > 0:
        # Métricas del cliente
        total_purchases = df_client.shape[0]
        product_counts = Counter([item['name'] for sublist in df_client['items'] for item in sublist])
        most_bought_product = max(product_counts, key=product_counts.get)
        store_most = df_client['storeLocation'].mode()[0]
        avg_spent = df_client['total_revenue'].mean()
        day_most = df_client['saleDate'].dt.day_name().mode()[0]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🛒 Compras totales", total_purchases)
        col2.metric("📦 Producto más comprado", most_bought_product)
        col3.metric("🏬 Tienda más visitada", store_most)
        col4.metric("💰 Gasto promedio por compra", f"${avg_spent:,.2f}")
        col5.metric("📅 Día con más compras", day_most)

        st.markdown("---")

        # Historial de compras del cliente
        df_hist = df_client.explode('items')
        df_items = pd.DataFrame(df_hist['items'].tolist(), index=df_hist.index)
#        df_items['price'] = df_items['price'].apply(lambda x: float(x.to_decimal()))
        df_items['price'] = df_items['price'].astype(float)
        df_items['saleDate'] = df_hist['saleDate']
        df_items['revenue'] = df_items['quantity'] * df_items['price'].astype(float)

        print(df_items)

        # Serie de tiempo: gasto por día
        ts = df_items.groupby('saleDate')['revenue'].sum().reset_index()
        fig_hist = px.line(
            ts,
            x='saleDate',
            y='revenue',
            title="Historial de compras del cliente (serie de tiempo)",
            labels={'saleDate':'Fecha', 'revenue':'Monto gastado ($)'},
            markers = True
        )
        st.plotly_chart(fig_hist, use_container_width=True)

def df_models(collection):
    pipeline = [
    {"$unwind": "$items"},

    # Añadir el campo "hour" antes del group
    {
        "$addFields": {
            "hour": {
                "$hour": {
                    "$dateFromString": {
                        "dateString": "$saleDate",
                        "onError": None,
                        "onNull": None
                    }
                }
            }
        }
    },

    {
        "$group": {
            "_id": {
                "date": {
                    "$dateToString": {
                        "format": "%Y-%m-%d",
                        "date": { "$dateFromString": { "dateString": "$saleDate" }}
                    }
                },
                "product": "$items.name"
            },
            "total_quantity": {"$sum": "$items.quantity"},
            "total_revenue": {
                "$sum": {
                    "$multiply": ["$items.quantity", "$items.price"]
                }
            },
            # Puedes sumar todas las horas si quieres
            # o quedártelas como lista
            "hours": { "$addToSet": "$hour" }
        }
    },

    {"$sort": {"_id.date": 1, "_id.product": 1}}
    ]
    results = collection.aggregate(pipeline)
    df = pd.DataFrame(results)
    df = df.join(pd.json_normalize(df["_id"]))
    df = df.drop(columns=["_id"])

    print(df.columns)
    return df

cols = ['total_quantity', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8', 'lag_9', 'lag_10', 'lag_11', 'lag_12', 'lag_13',
        'lag_14', 'month', 'is_weekend', 'day_of_week']

def process_train(df, cols):
    #df["hour"] = pd.to_datetime(df["time"], format="%H:%M:%S").dt.hour
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(["product", "date"])

    # Nueva información temporal
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    df_aux = (
        df#(columns=["hour"])  # quitamos "hour"
        .groupby(["date", "product"], as_index=False)
        .agg({
            "total_quantity": "sum",
            "month": "first",
            "day_of_week": "first",
            "is_weekend": "first"
        })
    )

    df_prod = crear_lags(df_aux)

    #st.dataframe(df_prod)

    # Fecha límite = última fecha - 14 días
    last_date = df_prod['date'].max()
    cutoff_date = last_date - pd.Timedelta(days=14)

    # Train = hasta cutoff_date
    train_df = df_prod[df_prod['date'] <= cutoff_date]

    # Test = después del cutoff_date
    test_df  = df_prod[df_prod['date'] > cutoff_date]

    # Features y target
    X_train = train_df[cols].drop("total_quantity", axis=1)
    y_train = train_df["total_quantity"]

    X_test  = test_df[cols].drop("total_quantity", axis=1)
    y_test  = test_df["total_quantity"]

    print(f"Train: {len(X_train)} filas")
    print(f"Test (últimas 2 semanas): {len(X_test)} filas")

    return df_aux, X_train, y_train, X_test, y_test

with tab3:

    st.header("📈 Forecasting de demanda - Próximos 7 días")

    # Selección de producto
    product_list = ['backpack', 'binder', 'envelopes', 'laptop', 'notepad', 'pens',
       'printer paper']
    sel_product = st.selectbox("Selecciona un producto para pronosticar", product_list)

    df = df_models(collection)
    #st.dataframe(df[df['product']==sel_product])

    df, X_train, y_train, X_test, y_test = process_train(df[df['product']==sel_product], cols)
    
    # Ruta del modelo
    model_path = f"best_model_{sel_product}.pkl"

    # Cargar modelo
    try:
        model = joblib.load(model_path)
        forecast_14 = model.predict(X_test)
        st.success(f"Modelo cargado para: {sel_product}")
    except FileNotFoundError:
        st.warning(f"No hay modelo disponible para {sel_product}")
        st.stop()

    st.subheader("Pronóstico de unidades vendidas para próxima semana")
    
    l = len(y_test)

    # Mostrar tabla de pronóstico
    df_forecast = pd.DataFrame({
        'Dates': df[df['product']==sel_product].tail(l)['date'].values,
        "Real sales next week": y_test.values,
        "Sales next week": forecast_14
    })

    col1, col2 = st.columns(2)

    with col1:
        df_forecast = df_forecast.sort_values("Dates").reset_index(drop=True)

        st.subheader("📋 Resultados del forecast")
        st.dataframe(df_forecast.style.format({
            "Real sales next week": "{:.2f}",
            "Sales next week": "{:.2f}"
        }))

    with col2:

        # Gráfica comparativa
        st.subheader("📊 Comparación: Real vs Pronosticado")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_forecast["Dates"], df_forecast["Real sales next week"], label="Real", linewidth=2)
        ax.plot(df_forecast["Dates"], df_forecast["Sales next week"], label="Forecast", linestyle="--", linewidth=2)

        ax.set_xlabel("Fecha")
        ax.set_ylabel("Cantidad")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)