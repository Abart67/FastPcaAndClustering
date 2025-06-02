import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from plotly import graph_objects as go
from plotly.graph_objects import Scatter
from scipy.cluster.hierarchy import linkage, fcluster
import os


def upload_management(uploaded_file, use_example_data):
    # Gestion des conflits entre les deux sources de données
    if uploaded_file is not None and use_example_data:
        st.warning("Vous devez d'abord supprimer le fichier chargé pour utiliser des données d'exemple.")
        use_example_data = False  # Réinitialisation de l'option si un fichier est chargé

    if use_example_data and uploaded_file:
        st.warning("Vous devez désactiver l'option 'Utiliser des données d'exemple' pour charger un fichier.")
        uploaded_file = None  # Réinitialisation du fichier si l’option est activée

    # Chargement des données en fonction de l'option sélectionnée
    if uploaded_file is not None:
        try:
            separator = st.text_input("Séparateur du fichier (par défaut : ,)", value=",")
            df = pd.read_csv(uploaded_file, sep=separator)
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")
            st.stop()
    elif use_example_data:
        separator = st.text_input("Séparateur du fichier (par défaut : ,)", value=",")
        try:
            path = os.path.join("data", "pays.csv")
            df = pd.read_csv(path, sep=separator)
        except Exception as e:
            st.error(f"Erreur lors du chargement des données d'exemple : {e}")
            st.stop()
    else:
        st.warning("Veuillez choisir un fichier ou activer l'option pour utiliser des données d'exemple.")
        st.stop()

    return df



def data_preparation(df):
    var_all = df.columns
    col_index, col_misdata = st.columns([3, 2])

    with col_index:
        # Sélection de la colonne représentant les observations
        observation_column = st.selectbox(
            "Choix de la colonne des observations :",
            options=var_all,
            help="Cette colonne sera exclue de l'ACP et utilisée pour identifier les observations."
        )
        df.index = df[observation_column]

    with col_misdata:
        miss_data_management = st.selectbox(
            "Traitement des valeurs manquantes :",
            ["Remplacer par 0", "Remplacer par la moyenne"]
        )

        numeric_columns = df.select_dtypes(include=['number']).columns
        if miss_data_management == "Remplacer par 0":
            df[numeric_columns] = df[numeric_columns].fillna(0)
        elif miss_data_management == "Remplacer par la moyenne":
            df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.mean()))

    # Aperçu des données
    st.write("**Aperçu des données :**")
    st.dataframe(df.head(5))

    return df, observation_column


def data_description(df, observation_column, analysis_choice):

    if analysis_choice == "Résumé du jeu de données":
        num_quantitative = df.drop(columns=observation_column).select_dtypes(include=['number']).shape[1]
        num_qualitative = df.drop(columns=observation_column).select_dtypes(include=['object', 'category', 'bool']).shape[1]

        st.markdown(
            f"""
            <style>
            .cards-container {{
                display: flex;
                justify-content: space-between;
                gap: 10px;
                margin-top: 20px;
            }}
            .card {{
                background-color: #f9f9f9;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                flex: 1;
                text-align: center;
            }}
            .card-title {{
                font-size: 18px;
                font-weight: bold;
                color: #333;
            }}
            .card-value {{
                font-size: 26px;
                font-weight: bold;
                color:rgb(255, 60, 0);
            }}
            </style>

            <div class="cards-container">
                <div class="card">
                    <div class="card-title">Nombre d'observations</div>
                    <div class="card-value">{df.shape[0]}</div>
                </div>
                <div class="card">
                    <div class="card-title">Variables quantitatives</div>
                    <div class="card-value">{num_quantitative}</div>
                </div>
                <div class="card">
                    <div class="card-title">Variables qualitatives</div>
                    <div class="card-value">{num_qualitative}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif analysis_choice == "Analyse des variables quantitatives":
        quantitative_df = df.drop(columns=observation_column).select_dtypes(include=['number'])

        if not quantitative_df.empty:
            summary = pd.DataFrame({
                "Type": quantitative_df.dtypes,
                "Valeurs manquantes": quantitative_df.isna().sum(),
                "Moyenne": quantitative_df.mean(),
                "Médiane": quantitative_df.median(),
                "Min": quantitative_df.min(),
                "Max": quantitative_df.max()
            })
            st.dataframe(summary)
        else:
            st.info("Aucune variable quantitative dans les données.")

    elif analysis_choice == "Analyse des variables qualitatives":
        qualitative_df = df.drop(columns=observation_column).select_dtypes(include=['object', 'category', 'bool'])

        if not qualitative_df.empty:
            summary = pd.DataFrame({
                "Type": qualitative_df.dtypes,
                "Valeurs manquantes": qualitative_df.isna().sum(),
                "Catégorie la plus fréquente": qualitative_df.mode().iloc[0]
            })
            st.dataframe(summary)
        else:
            st.info("Aucune variable qualitative dans les données.")



def hcpca_and_results(
    df, df_actives_ind,
    var_active_selected,
    observation_column,
    n_components,
    clusters_n,
    cluster_names,
    component_x,
    component_y,
    contribution_threashold,
    var_selected_supp=None,
    ind_selected_supp=None,
    ind_active_selected=None
):

    # --- Standardisation des données ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_actives_ind[var_active_selected])

    # --- Réalisation de l'ACP ---
    # Initialiser l'ACP avec le nombre ajusté de composantes
    pca = PCA(n_components=n_components)

    # Effectuer l'ACP
    X_pca = pca.fit_transform(X_scaled)

    # Création d'un DataFrame pour l'ACP
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    pca_df[observation_column] = df_actives_ind[observation_column]

    # Récupérer l'inertia de chaque composante principale
    inertia = pca.explained_variance_ratio_

    # Calculer la contribution des observations à chaque composante principale
    contributions = np.abs(X_pca)  # Valeurs absolues des composantes principales
    contributions_df = pd.DataFrame(contributions, columns=[f"PC{i+1}" for i in range(n_components)])
    contributions_df[observation_column] = df_actives_ind[observation_column]


    # Classification
    linkage_matrix = linkage(X_pca, method="ward")
    clusters = fcluster(linkage_matrix, clusters_n, criterion="maxclust")
    pca_df["Cluster"] = clusters

    # Appliquer les nouveaux noms de clusters
    pca_df["Cluster"] = pca_df["Cluster"].map(cluster_names)

    # --- Filtrer les observations par contribution --- 
    df_filtered_by_contributions = pca_df[
        (contributions_df[component_x] >= contribution_threashold) &
        (contributions_df[component_y] >= contribution_threashold)
    ]

    # Associer le nom des clusters au individus du dataframe actif
    df_actives_ind["Cluster"] = list(pca_df["Cluster"])


    numeric_columns = df_actives_ind.select_dtypes(include="number").columns.difference(["Cluster"])
    cluster_mean = df_actives_ind.groupby("Cluster")[numeric_columns].mean()


    #if var_selected_supp:
        # Standardisation des variables supplémentaires
    #    scaler_supp = StandardScaler()
    #    X_supp = scaler_supp.fit_transform(df[var_selected_supp])

        # Calculer les corrélations des variables supplémentaires avec les composantes principales
    #    correlations_supp = np.dot(X_supp.T, X_scaled) / (len(df) - 1)

        # Projeter les variables supplémentaires sur les composantes principales
    #    projections_supp = np.dot(correlations_supp, pca.components_.T)

    # Filtrer les lignes correspondant aux individus supplémentaires
    if ind_selected_supp:
        df_supp = df[df[observation_column].isin(ind_selected_supp)]
        X_supp_ind = scaler.transform(df_supp[var_active_selected])
        coord_supp_ind = pd.DataFrame(
            np.dot(X_supp_ind, pca.components_.T),
            index=df_supp[observation_column].values,
            columns=[f"PC{i+1}" for i in range(n_components)]
        )
    else:
        coord_supp_ind = pd.DataFrame()

    ### Calcul des metriques
    if ind_active_selected is None:
        ind_active_selected = df_actives_ind[observation_column].tolist()
    # Contributions brutes
    contributions_raw = (X_pca**2) / (len(ind_active_selected) * pca.explained_variance_)
    # Somme des contributions brutes pour chaque composante
    contributions_raw_sum = np.sum(contributions_raw, axis=0)
    # Contributions normalisées
    contributions_active_ind = pd.DataFrame(
        100 * contributions_raw / contributions_raw_sum,
        index=df_actives_ind[observation_column].values,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )


    # Coordonnées des individus actifs
    coord_active_ind = pd.DataFrame(
        X_pca,
        index=df_actives_ind[observation_column].values,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )


    return (
    df_filtered_by_contributions,
    var_selected_supp,
    ind_selected_supp,
    coord_supp_ind,
    contributions_active_ind,
    coord_active_ind,
    pca,
    inertia,
    cluster_mean
    )



def show_pca_graph(
    graph_selected,
    graph_title,
    df_filtered_by_contributions,
    df_actives_ind,
    observation_column,
    component_x,
    component_y,
    labels_show,
    coord_supp_ind,
    var_active_selected,
    pca,
    inertia,
    composantes
):
        # --- Graphique ACP ---
    if graph_selected == "Plan factoriel":
        fig_acp = px.scatter(
            df_filtered_by_contributions,
            x=component_x,
            y=component_y,
            color="Cluster",
            text= "noms_observations" if labels_show else None,
            labels={"color": "Cluster"},
            color_discrete_sequence=px.colors.qualitative.Bold,
            width=700,
            height=700,
            title=graph_title
        )

        # Positionner les labels au-dessus des points
        fig_acp.update_traces(
            textposition="top center",  # Position centrée au-dessus du point
            marker=dict(size=6, symbol="circle")  # Style pour les individus actifs
        )
        
            # --- Ajouter les individus supplémentaires ---
        if not coord_supp_ind.empty:
            fig_acp.add_trace(
                px.scatter(
                    coord_supp_ind,
                    x=component_x,
                    y=component_y,
                    text=coord_supp_ind.index,  # Labels des individus supplémentaires
                ).data[0]  # Ajoute les points comme une nouvelle trace
            )

            # Mettre à jour le style des individus supplémentaires
            fig_acp.data[-1].marker.update(size=8, symbol="cross", color="red")  # Points rouges en croix
            fig_acp.data[-1].update(textposition="top center", name="Individus supplémentaires")



        # Ajouter des lignes verticales et horizontales
        fig_acp.add_vline(x=0, line=dict(color="black", dash="dash", width=2))  # Ligne verticale à x=0
        fig_acp.add_hline(y=0, line=dict(color="black", dash="dash", width=2))  # Ligne horizontale à y=0

        # Personnaliser le quadrillage et définir le fond
        fig_acp.update_layout(
            xaxis=dict(
                showgrid=True,       # Afficher les lignes de grille sur l'axe x
                gridcolor='lightgray', # Couleur des lignes de grille
                gridwidth=0.5,        # Largeur des lignes de grille
                title=f"{component_x} (inertia: {inertia[composantes.index(component_x)]*100:.2f}%)"
            ),
            yaxis=dict(
                showgrid=True,       # Afficher les lignes de grille sur l'axe y
                gridcolor='lightgray', # Couleur des lignes de grille
                gridwidth=0.5,        # Largeur des lignes de grille
                dtick=1,               # Définir le pas de traçage de l'axe des y à 1
                title=f"{component_y} (inertia: {inertia[composantes.index(component_y)]*100:.2f}%)"
            ),
            plot_bgcolor="rgba(230, 230, 230, 0.5)"  # Ajouter un fond grisé transparent
        )

        st.plotly_chart(fig_acp, use_container_width=False)

    # --- Cercle des corrélations ---
    elif graph_selected == "Cercle des corrélations":
        # Récupérer les indices des composantes sélectionnées
        index_x = composantes.index(component_x)
        index_y = composantes.index(component_y)
        
        # Calcul des corrélations
        correlations = pca.components_.T * np.sqrt(pca.explained_variance_)*0.98

        circle_radius = 1.0

        # Points pour dessiner le cercle
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = circle_radius * np.cos(theta)
        circle_y = circle_radius * np.sin(theta)

        # Initialisation du graphique
        fig_cercle = go.Figure()

        # Ajouter le cercle
        fig_cercle.add_trace(go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            line=dict(color="black", dash="solid"),
            name="Cercle de corrélation"
        ))

        # Ajouter les vecteurs des variables actives
        for i, var in enumerate(var_active_selected):
            fig_cercle.add_trace(go.Scatter(
                x=[0, correlations[i, index_x]],  # Corrélation avec la composante X sélectionnée
                y=[0, correlations[i, index_y]],  # Corrélation avec la composante Y sélectionnée
                mode="lines+markers+text",
                line=dict(color="rgba(0, 94, 255, 0.8)", width=2),
                marker=dict(size=6),
                text=[None, var],  # Nom de la variable au bout de la flèche
                textposition="top center" if correlations[i, index_y]>=0 else "bottom center",
                textfont=dict(
                    family="Arial",  # Police
                    size=12,         # Taille de la police
                    color="black"),
                hovertemplate=(
                    f"<b>{var}</b><br>"  # Nom de la variable
                    f"Corrélation avec {component_x}: {correlations[i, index_x]:.3f}<br>"  # Corrélation X
                    f"Corrélation avec {component_y}: {correlations[i, index_y]:.3f}<extra></extra>"  # Corrélation Y
                ),
                name=var
            ))

        # Ajouter des axes (lignes horizontales et verticales)
        fig_cercle.add_hline(y=0, line=dict(color="black", dash="dash"))
        fig_cercle.add_vline(x=0, line=dict(color="black", dash="dash"))

        # Personnalisation du graphique
        fig_cercle.update_layout(
            title="Cercle des corrélations",
            xaxis=dict(
                title=component_x,
                range=[-1.1, 1.1],
                showgrid=True,
                gridcolor="lightgray"
            ),
            yaxis=dict(
                title=component_y,
                range=[-1.1, 1.1],
                showgrid=True,
                gridcolor="lightgray"
            ),
            plot_bgcolor="rgba(230, 230, 230, 0.5)",
            showlegend=False,
            width=700,
            height=800
        )

        st.plotly_chart(fig_cercle, use_container_width=True)

            
        # --- Barplot des inerties ---
    elif graph_selected == "Barplot des inerties":     
        # Calculer l'inertia cumulée
        cumulative_inertia = np.cumsum(inertia) * 100  # Convertir en pourcentage

        # Création du barplot avec la courbe
        fig_barplot = go.Figure()

        # Ajouter les barres pour l'inertia individuelle
        fig_barplot.add_trace(go.Bar(
            x=[f"PC{i+1}" for i in range(len(inertia))],
            y=[val * 100 for val in inertia],
            name="inertie individuelle",
            text=[f"{val * 100:.2f}%" for val in inertia],
            textposition="outside",
            marker_color="rgba(0, 94, 255, 0.75)",
        ))

        # Ajouter la courbe pour l'inertia cumulée
        fig_barplot.add_trace(go.Scatter(
            x=[f"PC{i+1}" for i in range(len(inertia))],
            y=cumulative_inertia,
            name="inertie cumulée",
            mode="lines+markers",
            line=dict(color="rgba(247, 63, 12, 0.8)", width=2),
            marker=dict(size=6)
        ))

        total_inertia = round(cumulative_inertia[len(cumulative_inertia)-1],2)
        # Mise en page du graphique
        fig_barplot.update_layout(
            title=f"Graphe de l'inertie expliquée par composante ({total_inertia}%)",
            xaxis_title="Composantes principales",
            yaxis_title="inertie (%)",
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            plot_bgcolor="rgba(230, 230, 230, 0.5)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            width=700,
            height=650
        )

        # Affichage du graphique dans Streamlit
        st.plotly_chart(fig_barplot, use_container_width=True)






def table_manager(tabs, table_selectionnee,df_actives_ind, var_active_selected, pca, n_components,
                  coord_active_ind, coord_supp_ind,
                  contributions_active_ind, cluster_mean):
    

            # Affichage des tables en fonction de la sélection
        if table_selectionnee == "VAR. Corrélations entre variables":
            corr_matrix = df_actives_ind[var_active_selected].corr()
            st.write("**Corrélations entre variables :**")
            for i in range(len(corr_matrix)):
                corr_matrix.iloc[i, i] = np.nan
            st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm",vmin=-1, vmax=1).format("{:.2f}"))

        elif table_selectionnee == "VAR. Corrélations des variables avec les composantes principales":
            cor_var_comps = pd.DataFrame(
                pca.components_.T * np.sqrt(pca.explained_variance_)*0.98,
                index=var_active_selected,
                columns=[f"PC{i+1}" for i in range(n_components)]
            )
            st.write("**Corrélations des variables avec les composantes principales :**")
            st.dataframe(cor_var_comps.style.background_gradient(cmap="coolwarm",vmin=-1, vmax=1).format("{:.2f}"))
        
        elif table_selectionnee == "VAR. Contributions des variables actives à chaque composante":
            contributions_variables = pd.DataFrame(
                pca.components_ ** 2,
                index=[f"PC{i+1}" for i in range(n_components)],
                columns=var_active_selected
            ).T
            st.write("**Contributions des variables actives à chaque composante :**")
            st.dataframe(contributions_variables.style.format("{:.2f}").background_gradient(cmap="coolwarm"))


        elif table_selectionnee == "IND. Coordonnées des individus":
            st.write("**Coordonnées des individus actifs**")
            st.dataframe(coord_active_ind.style.format("{:.2f}").background_gradient(cmap="coolwarm"))
            st.write("**Coordonnées des individus supplémentaires**")
            st.dataframe(coord_supp_ind.style.format("{:.2f}").background_gradient(cmap="coolwarm"))


        elif table_selectionnee == "IND. Contributions des individus à chaque composante":
            st.write("**Contributions des individus à chaque composante**")
            st.dataframe(contributions_active_ind)

        elif table_selectionnee == "CLUST. Description des groupes":
            st.write("**Description des groupes (moyenne par variables)**")
            st.dataframe(cluster_mean.T.style.format("{:.2f}").background_gradient(cmap="coolwarm",axis=1))