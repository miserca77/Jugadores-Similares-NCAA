
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

@st.cache_data
def cargar_datos():
    df = pd.read_excel('sr_ncaa_adjusted.xlsx')
    df.loc[df.duplicated(subset=['Player'], keep=False), 'Player'] += '_' + df.groupby('Player').cumcount().astype(str)
    df['Player'] = df['Player'].str.strip()
    df['Player'] = df['Player'].str.replace("'", "", regex=False)
    df['Player'] = df['Player'].str.replace(",", "", regex=False)
    df['Player_limpio'] = df['Player'].str.lower()
    return df

sr_ncaa_adjusted = cargar_datos()
columnas_disponibles = sr_ncaa_adjusted.select_dtypes(include=['number']).columns.tolist()
posiciones_disponibles = sorted(sr_ncaa_adjusted['Position'].dropna().unique())

st.title("🔍 Buscador de Jugadores Similares - NCAA")

jugador_seleccionado = st.selectbox("Selecciona un jugador", sorted(sr_ncaa_adjusted["Player"].unique()))

metricas_seleccionadas = st.multiselect(
    "Selecciona las métricas a usar para calcular similitud (deja vacío para usar todas)",
    columnas_disponibles,
    default=columnas_disponibles
)

posiciones_filtradas = st.multiselect(
    "Selecciona las posiciones a incluir (deja vacío para incluir todas)",
    posiciones_disponibles,
    default=posiciones_disponibles
)

num_similares = st.slider("Número de jugadores similares a mostrar", min_value=1, max_value=20, value=5)

if st.button("🔎 Buscar jugadores similares"):
    try:
        # Filtrar posiciones
        if posiciones_filtradas:
            df_filtrado = sr_ncaa_adjusted[sr_ncaa_adjusted["Position"].isin(posiciones_filtradas)].copy()
        else:
            df_filtrado = sr_ncaa_adjusted.copy()
        # 🔄 Ordenar por nombre de jugador para garantizar coherencia en y, X, etc.    
        df_filtrado = df_filtrado.sort_values(by="Player").reset_index(drop=True)
            
        # Preparar matrices
        y = df_filtrado["Player"].values
        metricas_ordenadas = sorted(metricas_seleccionadas)
        X = df_filtrado[metricas_ordenadas].values
        X_std = StandardScaler().fit_transform(X)

        # Caso A: pocas métricas → usar distancia euclídea
        if len(metricas_seleccionadas) <= 3:
            from sklearn.metrics.pairwise import euclidean_distances

            dist_matrix = euclidean_distances(X_std)
            idx = np.where(y == jugador_seleccionado)[0][0]
            distancias = dist_matrix[idx]
            indices_ordenados = np.argsort(distancias)[1:num_similares+1]

            jugadores_similares = y[indices_ordenados]
            factores = distancias[indices_ordenados]

            df_similares = pd.DataFrame({
                "Jugador similar": jugadores_similares,
                "Distancia euclídea": factores
            })

        # Caso B: muchas métricas → usar PCA + correlación
        else:
            pca = PCA(n_components=X_std.shape[1]-1, random_state=42)
            X_pca = pca.fit_transform(X_std)
            expl = pca.explained_variance_ratio_

            # Calculamos número mínimo de componentes para explicar al menos 95% de la varianza
            explained_var_cumsum = np.cumsum(expl)
            num_componentes = np.searchsorted(explained_var_cumsum, 0.95) + 1

            columns_pca = [f"PCA{i+1}" for i in range(num_componentes)]
            df_pca = pd.DataFrame(data=X_pca[:, :num_componentes], columns=columns_pca, index=y)

            # Mostrar varianza explicada acumulada
            st.write(f"🔍 Se usan {num_componentes} componentes principales para explicar el 95% de la varianza acumulada.")

            corr_matrix = df_pca.T.corr(method='pearson')

            if jugador_seleccionado not in corr_matrix.index:
                st.error("El jugador no se encuentra en la matriz de correlación.")
                st.stop()

            row = corr_matrix.loc[jugador_seleccionado]
            similares = row.drop(jugador_seleccionado).nlargest(num_similares)

            df_similares = pd.DataFrame({
                "Jugador similar": similares.index,
                "Factor de correlación": similares.values
            })

        # Mostrar resultados
        jugador_base = sr_ncaa_adjusted[sr_ncaa_adjusted["Player"] == jugador_seleccionado]
        jugadores_similares_df = sr_ncaa_adjusted[sr_ncaa_adjusted["Player"].isin(df_similares["Jugador similar"])]
        resultado_final = pd.concat([jugador_base, jugadores_similares_df])

        st.subheader(f"🎯 Jugadores más similares a: {jugador_seleccionado}")
        st.dataframe(df_similares)

        st.subheader("📋 Datos de los jugadores encontrados:")
        st.dataframe(resultado_final.reset_index(drop=True))

        # Mostrar gráfico PCA solo si se aplicó
        if len(metricas_seleccionadas) > 3:
            st.subheader("📈 Varianza explicada acumulada (PCA)")
            fig, ax = plt.subplots()
            ax.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
            ax.set_xlabel('Número de componentes')
            ax.set_ylabel('Varianza explicada')
            ax.grid(True)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
