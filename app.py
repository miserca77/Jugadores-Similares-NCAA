
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

@st.cache_data
def cargar_datos():
    df = pd.read_excel('sr_ncaa_adjusted.xlsx')
    df.loc[df.duplicated(subset=['Player'], keep=False), 'Player'] += '_' + df.groupby('Player').cumcount().astype(str)
    df['Player'] = df['Player'].str.strip()
    df['Player'] = df['Player'].str.replace("'", "", regex=False)
    df['Player'] = df['Player'].str.replace(",", "", regex=False)
    return df

sr_ncaa_adjusted = cargar_datos()
columnas_disponibles = sr_ncaa_adjusted.select_dtypes(include=['number']).columns.tolist()
posiciones_disponibles = sorted(sr_ncaa_adjusted['Position'].dropna().unique()) if 'Position' in sr_ncaa_adjusted.columns else []

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

num_similares = st.slider("Número de jugadores similares a mostrar", min_value=1, max_value=50, value=10)

if st.button("🔎 Buscar jugadores similares"):
    try:
        if posiciones_filtradas:
            df_filtrado = sr_ncaa_adjusted[sr_ncaa_adjusted["Position"].isin(posiciones_filtradas)].copy()
        else:
            df_filtrado = sr_ncaa_adjusted.copy()

        df_filtrado = df_filtrado.sort_values(by="Player").reset_index(drop=True)

        y = df_filtrado["Player"].values
        metricas_ordenadas = sorted(metricas_seleccionadas) if metricas_seleccionadas else sorted(columnas_disponibles)
        X = df_filtrado[metricas_ordenadas].values
        X_std = StandardScaler().fit_transform(X)

        if len(metricas_ordenadas) <= 5:
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

        else:
            pca = PCA(n_components=X_std.shape[1]-1, random_state=42)
            X_pca = pca.fit_transform(X_std)
            expl = pca.explained_variance_ratio_
            explained_var_cumsum = np.cumsum(expl)
            num_componentes = np.searchsorted(explained_var_cumsum, 0.95) + 1

            st.write(f"🔍 Se usan {num_componentes} componentes principales para explicar el 95% de la varianza acumulada.")

            columns_pca = [f"PCA{i+1}" for i in range(num_componentes)]
            df_pca = pd.DataFrame(data=X_pca[:, :num_componentes], columns=columns_pca, index=y).sort_index()

            corr_matrix = df_pca.T.corr(method='pearson')
            corr_matrix = corr_matrix.reindex(index=df_pca.index, columns=df_pca.index)

            if jugador_seleccionado not in corr_matrix.index:
                st.error("El jugador no se encuentra en la matriz de correlación.")
                st.stop()

            row = corr_matrix.loc[jugador_seleccionado]
            similares = row.drop(jugador_seleccionado).nlargest(num_similares)

            df_similares = pd.DataFrame({
                "Jugador similar": similares.index,
                "Factor de correlación": similares.values
            })

            # Gráfico de varianza acumulada
            fig, ax = plt.subplots()
            ax.plot(explained_var_cumsum, marker='o')
            ax.set_xlabel('Número de componentes')
            ax.set_ylabel('Varianza explicada acumulada')
            ax.grid(True)
            st.pyplot(fig)

        # Preparar dataframes base y similares
        jugador_base = sr_ncaa_adjusted[sr_ncaa_adjusted["Player"] == jugador_seleccionado]
        jugadores_similares_df = sr_ncaa_adjusted[sr_ncaa_adjusted["Player"].isin(df_similares["Jugador similar"])]
        resultado_final = pd.concat([jugador_base, jugadores_similares_df])

        st.subheader(f"🎯 Jugadores más similares a: {jugador_seleccionado}")
        st.dataframe(df_similares)

        # Mostrar tabla con métricas seleccionadas (sin normalizar)
        st.subheader("📋 Datos de los jugadores con métricas seleccionadas:")
        columnas_a_mostrar = ["Player"]
        if "Position" in resultado_final.columns:
            columnas_a_mostrar.append("Position")
        columnas_a_mostrar += [col for col in metricas_ordenadas if col in resultado_final.columns]
        resultado_a_mostrar = resultado_final[columnas_a_mostrar]
        st.dataframe(resultado_a_mostrar.reset_index(drop=True))

        # Mostrar tabla con todas las métricas completas (sin normalizar)
        st.subheader("📋 Datos completos (todas las métricas) de los jugadores encontrados:")
        st.dataframe(resultado_final.reset_index(drop=True))

        # Exportar CSV - Formato ancho sin normalizar incluyendo columna Position
        columnas_a_descargar = [col for col in resultado_final.columns if col not in ["School", "Conference"]]

        if "Position" not in columnas_a_descargar and "Position" in resultado_final.columns:
            columnas_a_descargar.append("Position")

        orden = []
        for c in ["Player", "Position"]:
            if c in columnas_a_descargar:
                orden.append(c)
        resto_cols = [c for c in columnas_a_descargar if c not in orden]
        columnas_a_descargar = orden + resto_cols

        resultado_a_descargar = resultado_final[columnas_a_descargar]

        csv_buffer = resultado_a_descargar.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
        st.download_button(
            label="⬇️ Descargar todos los datos en CSV (Power BI ready)",
            data=csv_buffer,
            file_name="jugadores_similares_completo.csv",
            mime="text/csv"
        )

        # Exportar CSV - Formato largo (tidy) solo con métricas seleccionadas
        if metricas_ordenadas:
            columnas_largas = ["Player"] + [col for col in metricas_ordenadas if col in resultado_a_descargar.columns]
            df_largo = resultado_a_descargar[columnas_largas].melt(
                id_vars=["Player"],
                var_name="Metrica",
                value_name="Valor"
            )
            csv_largo = df_largo.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
            st.download_button(
                label="⬇️ Descargar formato largo (Player / Metrica / Valor)",
                data=csv_largo,
                file_name="jugadores_similares_seleccionado.csv",
                mime="text/csv"
            )

        # Exportar datos normalizados para radar en formato largo
        if metricas_ordenadas:
            scaler = MinMaxScaler()
            df_norm = resultado_final[["Player"] + metricas_ordenadas].copy()
            df_norm[metricas_ordenadas] = scaler.fit_transform(df_norm[metricas_ordenadas])

            df_norm_largo = df_norm.melt(id_vars=["Player"], var_name="Metrica", value_name="Valor")
            df_norm_largo = df_norm_largo.rename(columns={"Player": "Jugador"})

            csv_norm = df_norm_largo.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
            st.download_button(
                label="📊 Descargar datos NORMALIZADOS (0-1) para gráfico radar (formato largo)",
                data=csv_norm,
                file_name="jugadores_similares_radar_normalizado.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
