import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Rekomendasi Menu Kimbogi", layout="wide")
st.title("🍣 Sistem Rekomendasi Paket Menu Kimbogi")
st.write("Aplikasi analisis aturan asosiasi untuk penentuan paket bundling makanan.")

# Parameter di Sidebar
st.sidebar.header("Pengaturan Parameter Apriori")
min_supp = st.sidebar.slider("Minimum Support", 0.05, 0.50, 0.15, step=0.01)
min_conf = st.sidebar.slider("Minimum Confidence", 0.10, 1.00, 0.60, step=0.05)

# Load Data Matriks
@st.cache_data
def load_data():
    return pd.read_pickle('basket_matrix.pkl')

basket_sets = load_data()

# Proses Apriori
frequent_itemsets = apriori(basket_sets, min_support=min_supp, use_colnames=True)

if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)

    if not rules.empty:
        rules = rules.sort_values(by='lift', ascending=False)

        # Format Tampilan Aturan
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

        st.subheader("🔥 Rekomendasi Kombinasi Paket Menu (Association Rules)")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']], use_container_width=True)
    else:
        st.warning("Tidak ditemukan aturan asosiasi yang memenuhi syarat Minimum Confidence.")
else:
    st.warning("Tidak ada kombinasi menu yang memenuhi syarat Minimum Support.")
