
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

@st.cache_data
def cargar_datos():
    df = pd.read_excel('sr_ncaa_adjusted.xlsx', engine='openpyxl')
    df.loc[df.duplicated(subset=['Player'], keep=False), 'Player'] += '_' + df.groupby('Player').cumcount().astype(str)
    df['Player'] = df['Player'].str.replace("'", "", regex=False)
    df['Player'] = df['Player'].str.replace(",", "", regex=False)
    df['Player'] = df['Player'].str.strip()
    df['Player_limpio'] = df['Player'].str.lower()
    return df

sr_ncaa_adjusted = cargar_datos()

# 1. Selección de columnas numéricas disponibles
columnas_disponibles = sr_ncaa_adjusted.select_dtypes(include=['number']).columns.tolist()
metricas_seleccionadas = st.multiselect(
    "Selecciona las métricas a usar para calcular similitud (deja vacío para usar todas)",
    columnas_disponibles,
    default=columnas_disponibles
)
metricas_ordenadas = [col.strip() for col in metricas_seleccionadas]
metricas_ordenadas = [col for col in metricas_ordenadas if col]
metricas_ordenadas = sorted(metricas_ordenadas)
st.write("🧪 Columnas utilizadas:", metricas_ordenadas)

# 2. Selección de posiciones a elegir
posiciones_disponibles = sorted(sr_ncaa_adjusted['Position'].dropna().unique())
posiciones_filtradas = st.multiselect(
    "Selecciona las posiciones a incluir (deja vacío para incluir todas)",
    posiciones_disponibles,
    default=posiciones_disponibles
)
if posiciones_filtradas:
    df_filtrado = sr_ncaa_adjusted[sr_ncaa_adjusted["Position"].isin(posiciones_filtradas)].copy()
else:
    df_filtrado = sr_ncaa_adjusted[sr_ncaa_adjusted["Position"].isin(posiciones_disponibles)].copy()
df_filtrado = df_filtrado.sort_values(by="Player").reset_index(drop=True)

# 3. Selección del nombre del jugador base
jugadores_filtrados = sorted(df_filtrado["Player"].unique())
jugador_seleccionado = st.selectbox("Selecciona un jugador base", jugadores_filtrados)

# 4. Selección del número de similares y cálculo
num_similares = st.slider("Número de jugadores similares a mostrar", min_value=1, max_value=20, value=5)

if st.button("🔎 Buscar jugadores similares"):
    try:
        y = df_filtrado["Player"].values
        X = df_filtrado[metricas_ordenadas].values
        X_std = StandardScaler().fit_transform(X)

        idx = np.where(y == jugador_seleccionado)[0][0]
        jugador_original = y[idx]

        if len(metricas_ordenadas) <= 3:
            from sklearn.metrics.pairwise import euclidean_distances
            dist_matrix = euclidean_distances(X_std)
            distancias = dist_matrix[idx]
            indices_ordenados = np.argsort(distancias)[1:num_similares+1]

            df_similares = pd.DataFrame({
                "Jugador similar": y[indices_ordenados],
                "Distancia euclídea": distancias[indices_ordenados]
            })

        else:
            pca = PCA(n_components=X_std.shape[1]-1, random_state=42)
            X_pca = pca.fit_transform(X_std)
            expl = pca.explained_variance_ratio_
            explained_var_cumsum = np.cumsum(expl)
            num_componentes = np.searchsorted(explained_var_cumsum, 0.95) + 1

            columns_pca = [f"PCA{i+1}" for i in range(num_componentes)]
            df_pca = pd.DataFrame(X_pca[:, :num_componentes], columns=columns_pca, index=y)

            # --- ORDENAR EL ÍNDICE ---
            df_pca = df_pca.sort_index()
            jugador_original = jugador_original  # ya está en y, que es el índice de df_pca

            # --- MATRIZ DE CORRELACIÓN ROBUSTA Y ORDENADA ---
            corr_matrix = df_pca.T.corr(method='pearson')
            corr_matrix = corr_matrix.reindex(index=df_pca.index, columns=df_pca.index)

            if jugador_original not in corr_matrix.index:
                st.error(f"❌ El jugador '{jugador_original}' no está en la matriz de correlación.")
                st.write("Índices disponibles:", list(corr_matrix.index[:5]))
                st.stop()

            row = corr_matrix.loc[jugador_original]
            similares = row.drop(jugador_original).nlargest(num_similares)

            df_similares = pd.DataFrame({
                "Jugador similar": similares.index,
                "Factor de correlación": similares.values
            })

        # Mostrar resultados
        jugador_base = sr_ncaa_adjusted[sr_ncaa_adjusted["Player"] == jugador_original]
        jugadores_similares_df = sr_ncaa_adjusted[sr_ncaa_adjusted["Player"].isin(df_similares["Jugador similar"])]
        resultado_final = pd.concat([jugador_base, jugadores_similares_df])

        st.subheader(f"🎯 Jugadores más similares a: {jugador_original}")
        st.dataframe(df_similares)

        st.subheader("📋 Datos de los jugadores encontrados:")
        st.dataframe(resultado_final.reset_index(drop=True))

        if len(metricas_ordenadas) > 3:
            st.subheader("📈 Varianza explicada acumulada (PCA)")
            fig, ax = plt.subplots()
            ax.plot(explained_var_cumsum, marker='o')
            ax.set_xlabel('Número de componentes')
            ax.set_ylabel('Varianza explicada')
            ax.grid(True)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
