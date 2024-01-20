import streamlit as st


st.title('Times Series on eFX Trading')
st.subheader('Research project PI2 for BNY Mellon')

# Image 1
image_url1 = "https://i.ibb.co/3h2KPmf/Logo-esilv-png-blanc.png"  # Remplacez ceci par l'URL de votre première image
image_url2 = "https://i.ibb.co/kKKSDMg/bny-mellon-logo-0.png"  # Remplacez ceci par l'URL de votre deuxième image

# Ajouter un séparateur pour plus de clarté
# Afficher les images côte à côte
col1, col2 = st.columns(2)

# Image 1 dans la première colonne
col1.image(image_url1, use_column_width=True)

# Image 2 dans la deuxième colonne
col2.image(image_url2, use_column_width=True)
st.markdown("---")

# Footer avec les noms des contributeurs
st.markdown("Project realised by :")
st.markdown("- BONFIGLIOLI Margherita")
st.markdown("- OUSTRAIN Edgar")
st.markdown("- THOMAS Victor")
st.markdown("- JOUVE Thomas")
st.markdown("- MIDAVAINE Théo")