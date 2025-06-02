import streamlit as st
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import fpce_functions as af



#st.set_page_config(page_title="Stream PCA", layout="wide")
st.set_page_config(page_title="Fast PCA & Clustering Explorer")


# Inject custom CSS to adjust sidebar width
st.markdown(
    """
    <style>
    /* Set the sidebar width */
    [data-testid="stSidebar"] {
        min-width: 400px; /* Minimum width of the sidebar */
        max-width: 400px; /* Maximum width of the sidebar */
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.sidebar.header("Paramètres")

tabs = st.tabs(["Chargement des données", "Graphiques", "Tables"])

with tabs[0]:
    st.subheader("**Chargement des données :**")
    # Chargement d'un fichier CSV par l'utilisateur
    uploaded_file = st.file_uploader("Chargez un fichier CSV", type="csv", key="uploaded_file")

    # État initial de l'option "Utiliser des données d'exemple"
    use_example_data = st.checkbox(
        "Utiliser des données d'exemple ?", 
        key="use_example_data", 
        value=False, 
        disabled=uploaded_file is not None
    )



    # Chargement et préparation du jeu de données
    df = af.upload_management(uploaded_file, use_example_data)
    var_all = df.columns
    col_index, col_misdata = st.columns([3,2])
    df, observation_column = af.data_preparation(df)


    # Description du jeu de données
    st.subheader("Description du jeu de données")
    analysis_choice = st.selectbox(
        "Choisissez le type d'analyse :",
        ["Résumé du jeu de données", "Analyse des variables quantitatives", "Analyse des variables qualitatives"]
    )
    af.data_description(df, observation_column, analysis_choice)
    



# --- Sélection des variables actives et indicatives ---
with st.sidebar.expander("Paramètres de l'ACP", expanded=True):

    variables_numeriques = df.select_dtypes(include=[np.number]).columns

    # Option "Tout sélectionner" pour les variables actives
    if st.checkbox("Choix variables actives (tout sélectionner)", value=True):
        var_active_selected = [
            var for var in variables_numeriques if var != observation_column
        ]
    else:
        var_active_selected = st.multiselect(
            "Variables actives :",
            options=[var for var in variables_numeriques if var != observation_column],
            default=[var for var in variables_numeriques if var != observation_column]
        )

    # Exclusion mutuelle : variables disponibles pour les indicatives
    variables_supplementaires_disponibles = [
        var for var in var_all if var not in var_active_selected and var != observation_column
    ]

    # Option "Tout sélectionner" pour les variables supplémentaires
    if st.checkbox("Choix variables supplémentaires (tout sélectionner)", value=True):
        var_selected_supp = variables_supplementaires_disponibles
    else:
        var_selected_supp = st.multiselect(
            "Variables supplémentaires :",
            options=variables_supplementaires_disponibles,
            default=variables_supplementaires_disponibles,
        )

    liste_individus = df[observation_column].unique()


    if st.checkbox("Choix individus actifs (tout sélectionner)", value=True):
        ind_active_selected = [
            ind for ind in liste_individus
        ]
    else:
        ind_active_selected = st.multiselect(
            "individus actifs :",
            options=[ind for ind in liste_individus],
            default=[ind for ind in liste_individus]
        )

    # Exclusion mutuelle : individus supplémentaires
    individus_supplementaires_disponibles = [
        ind for ind in liste_individus if ind not in ind_active_selected
    ]

    # Option "Tout sélectionner" 
    if st.checkbox("Choix individus supplémentaires (tout sélectionner)", value=True):
        ind_selected_supp = individus_supplementaires_disponibles
    else:
        ind_selected_supp = st.multiselect(
            "Variables supplémentaires :",
            options=individus_supplementaires_disponibles,
            default=individus_supplementaires_disponibles,
        )

    # Si aucune variable active n'est sélectionnée, afficher un message d'erreur
    if len(var_active_selected) < 2:
        st.error("Veuillez sélectionner au moins deux variables actives.")
        st.stop()

    # Choix du nombre de components principales
    max_components = min(len(var_active_selected), len(df))
    n_components = st.slider(
        "Nombre de components principales",
        min_value=1,
        max_value=max_components,
        value=max_components,
        step=1
    )

    clusters_n = st.slider("Nombre de clusters", 2, 10, 4)


with st.sidebar.expander("Filtre et affichage du biplot", expanded=False):

    # Sélection des components X et Y
    components = [f"PC{i+1}" for i in range(n_components)]
    component_x = st.selectbox("Composante X", components, index=0)
    component_y = st.selectbox("Composante Y", components, index=1)


        # Option pour afficher tous les labels
    labels_show = st.checkbox("Afficher les labels des individus", value=True)

    # Filtre des contributions des observations
    contribution_threashold = st.slider(
        "Filtrer les contributions (composante X)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,  # Valeur par défaut
        step=0.01,
        help="Seules les observations dont la contribution à la composante principale est supérieure ou égale au seuil seront affichées."
    )

    cluster_names = {}
    for cluster_id in range(1, clusters_n + 1):
        cluster_names[cluster_id] = st.text_input(
            f"Nom du Cluster {cluster_id}",
            value=f"Cluster {cluster_id}"
        )

df_actives_ind = df[df[observation_column].isin(ind_active_selected)]

(
df_filtered_by_contributions,
var_selected_supp,
ind_selected_supp,
coord_supp_ind,
contributions_active_ind,
coord_active_ind,
pca,
inertia,cluster_mean) =  af.hcpca_and_results(
    df, df_actives_ind,
    var_active_selected,
    observation_column,
    n_components,
    clusters_n,
    cluster_names,
    component_x,
    component_y,
    contribution_threashold,
    var_selected_supp=var_selected_supp,
    ind_selected_supp=ind_selected_supp,
    ind_active_selected=ind_active_selected)



with tabs[1]:

    col1, col2 = st.columns([5,2])

    with col1:
        # Menu déroulant pour sélectionner le graphique
        graph_selected = st.selectbox(
            "Choisissez le graphique à afficher :",
            options=["Plan factoriel", "Cercle des corrélations", "Barplot des inerties"],
            index=0  # Par défaut, le Biplot est sélectionné
        )
        
        # Associer un titre par défaut à chaque type de graphique
        titres_par_defaut = {
            "Plan factoriel": "Plan factoriel des individus",
            "Cercle des corrélations": "Cercle des corrélations",
            "Barplot des inerties": "Barplot des inerties"
        }

    # Zone de texte pour personnaliser le titre
    with col2:
        graph_title = st.text_input(
            "Saisir un titre au graphique",
            value=titres_par_defaut[graph_selected]
        )
    

    # Créer une nouvelle colonne 'Nom' dans df_filtered_by_contributions
    df_filtered_by_contributions['noms_observations'] = df_filtered_by_contributions.index.map(
        lambda idx: df_actives_ind[observation_column].get(idx, None)
    )

    af.show_pca_graph(
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
    components
)



# --- Onglet pour les tables ---
with tabs[2]:
    st.subheader("Tables des résultats")
    # Menu déroulant pour sélectionner la table
    table_selectionnee = st.selectbox(
        "Choisissez la table à afficher :",
        options=[
            "VAR. Corrélations entre variables",
            "VAR. Corrélations des variables avec les composantes principales",
            "VAR. Contributions des variables actives à chaque composante",
            "IND. Coordonnées des individus",
            "IND. Contributions des individus à chaque composante",

            "CLUST. Description des groupes"
            

        ],key="selectbox_table_manager"
    )


    af.table_manager(tabs, table_selectionnee,df_actives_ind, var_active_selected, pca, n_components,
                    coord_active_ind, coord_supp_ind,
                    contributions_active_ind,cluster_mean)
