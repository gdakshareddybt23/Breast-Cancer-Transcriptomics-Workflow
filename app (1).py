"""
Updated Streamlit Dashboard: original Primary vs Metastatic gene-expression workflow
plus guide-requested sarcoma subtype comparative analysis.
"""
import io
import base64
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA

st.set_page_config(page_title="Gene Expression + Sarcoma Dashboard", page_icon="🧬", layout="wide")

PALETTE = {"Primary": "#6dbf7e", "Metastatic": "#e8836e"}
COLOR_REG = {"Upregulated": "#e05252", "Downregulated": "#4a7fc1", "Not Significant": "#bfbfbf"}
LABEL_MAP = {0: "Primary", 1: "Metastatic"}
SARCOMA_SUBTYPES = ["Osteosarcoma", "Leiomyosarcoma", "Liposarcoma", "Rhabdomyosarcoma"]
SUBTYPE_COLORS = {"Osteosarcoma": "#4C78A8", "Leiomyosarcoma": "#F58518", "Liposarcoma": "#54A24B", "Rhabdomyosarcoma": "#B279A2"}
SARCOMA_MARKER_GENES = {
    "Osteosarcoma": ["RUNX2", "SP7", "COL1A1", "ALPL", "IBSP", "BGLAP", "MMP13", "VEGFA"],
    "Leiomyosarcoma": ["ACTA2", "TAGLN", "MYH11", "DES", "CNN1", "CALD1", "MYLK", "TPM2"],
    "Liposarcoma": ["MDM2", "CDK4", "HMGA2", "PPARG", "CEBPA", "FABP4", "LPL", "ADIPOQ"],
    "Rhabdomyosarcoma": ["MYOD1", "MYOG", "PAX3", "PAX7", "DES", "MYF5", "MYH3", "TNNT3"],
}
PAN_SARCOMA_GENES = ["TP53", "RB1", "CDKN2A", "CCND1", "MKI67", "BCL2", "VEGFA", "MMP2", "MMP9", "PTEN"]

@st.cache_data
def load_data():
    rng = np.random.default_rng(42)
    n_samples, n_genes = 120, 310
    gene_names = [f"gene_{i}" for i in range(n_genes - 10)] + [
        "tumor protein p53", "TP53 regulated inhibitor of apoptosis 1",
        "TP53 induced glycolysis regulatory phosphatase", "TP53 induced nuclear protein 1",
        "TP53 target 1", "mitogen-activated protein kinase 1", "C-C motif chemokine ligand 5",
        "stathmin 1", "cyclin dependent kinase 8", "BCL2 like 11"]
    labels = np.array([0]*60 + [1]*60, dtype=np.int8)
    data = rng.lognormal(mean=3, sigma=1, size=(n_samples, n_genes)).astype(np.float32)
    data[60:, -10:] *= np.array([0.3, 2.8, 2.4, 0.4, 2.1, 1.8, 2.5, 2.2, 0.35, 0.45], dtype=np.float32)
    expr = pd.DataFrame(data, columns=gene_names)
    expr.insert(0, "Label", labels)
    primary = expr.loc[expr.Label == 0, gene_names]
    meta = expr.loc[expr.Label == 1, gene_names]
    logfc = np.log2(meta.mean(axis=0) + 1) - np.log2(primary.mean(axis=0) + 1)
    pvals = ttest_ind(meta, primary, equal_var=False, axis=0).pvalue
    deg = pd.DataFrame({"Gene": gene_names, "logFC": logfc.values, "p_value": pvals})
    deg["-log10(p_value)"] = -np.log10(np.clip(deg.p_value, 1e-300, 1))
    deg["Regulation"] = np.select([(deg.logFC > 1) & (deg.p_value < .05), (deg.logFC < -1) & (deg.p_value < .05)], ["Upregulated", "Downregulated"], default="Not Significant")
    return expr, deg.sort_values("p_value").reset_index(drop=True)

@st.cache_data
def create_sarcoma_subtype_dataset(n_per_subtype=35, random_state=2026):
    rng = np.random.default_rng(random_state)
    genes = sorted(set(sum(SARCOMA_MARKER_GENES.values(), []) + PAN_SARCOMA_GENES + [f"SARC_GENE_{i:03d}" for i in range(1,181)]))
    rows, labels = [], []
    for subtype in SARCOMA_SUBTYPES:
        for _ in range(n_per_subtype):
            row = dict(zip(genes, rng.normal(7.0, 0.8, len(genes))))
            for g in SARCOMA_MARKER_GENES[subtype]: row[g] += rng.normal(2.2, 0.35)
            for g in ["MKI67", "VEGFA", "MMP2", "MMP9"]: row[g] += rng.normal(0.8, 0.25)
            if subtype == "Osteosarcoma":
                for g in ["RUNX2", "COL1A1", "ALPL", "IBSP"]: row[g] += rng.normal(0.7, 0.2)
            elif subtype == "Leiomyosarcoma":
                for g in ["ACTA2", "TAGLN", "MYH11", "CNN1"]: row[g] += rng.normal(0.7, 0.2)
            elif subtype == "Liposarcoma":
                for g in ["MDM2", "CDK4", "PPARG", "FABP4"]: row[g] += rng.normal(0.7, 0.2)
            else:
                for g in ["MYOD1", "MYOG", "PAX3", "PAX7"]: row[g] += rng.normal(0.7, 0.2)
            rows.append(row); labels.append(subtype)
    df = pd.DataFrame(rows).astype(np.float32); df.insert(0, "Subtype", labels)
    return df

def fig_to_bytes(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=160); plt.close(fig); buf.seek(0); return buf

def one_vs_rest(df, subtype, lfc_thresh=1.0, p_thresh=0.05):
    genes = [c for c in df.columns if c != "Subtype"]
    case, rest = df[df.Subtype == subtype][genes], df[df.Subtype != subtype][genes]
    logfc = case.mean() - rest.mean(); pvals = ttest_ind(case, rest, equal_var=False, axis=0, nan_policy="omit").pvalue
    out = pd.DataFrame({"Subtype": subtype, "Gene": genes, "logFC": logfc.values, "p_value": pvals})
    out["-log10(p_value)"] = -np.log10(np.clip(out.p_value, 1e-300, 1))
    out["Regulation"] = np.select([(out.logFC > lfc_thresh) & (out.p_value < p_thresh), (out.logFC < -lfc_thresh) & (out.p_value < p_thresh)], ["Upregulated", "Downregulated"], default="Not Significant")
    return out.sort_values(["p_value", "logFC"], ascending=[True, False]).reset_index(drop=True)

def pairwise_deg(df, a, b, lfc_thresh=1.0, p_thresh=0.05):
    genes = [c for c in df.columns if c != "Subtype"]
    A, B = df[df.Subtype == a][genes], df[df.Subtype == b][genes]
    logfc = A.mean() - B.mean(); pvals = ttest_ind(A, B, equal_var=False, axis=0, nan_policy="omit").pvalue
    out = pd.DataFrame({"Comparison": f"{a} vs {b}", "Gene": genes, "logFC": logfc.values, "p_value": pvals})
    out["-log10(p_value)"] = -np.log10(np.clip(out.p_value, 1e-300, 1))
    out["Regulation"] = np.select([(out.logFC > lfc_thresh) & (out.p_value < p_thresh), (out.logFC < -lfc_thresh) & (out.p_value < p_thresh)], [f"Higher in {a}", f"Higher in {b}"], default="Not Significant")
    return out.sort_values(["p_value", "logFC"], ascending=[True, False]).reset_index(drop=True)

expr, deg = load_data(); gene_cols = [c for c in expr.columns if c != "Label"]
sample_type = expr.Label.map(LABEL_MAP)
sarc_expr = create_sarcoma_subtype_dataset(); sarc_genes = [c for c in sarc_expr.columns if c != "Subtype"]

st.title("🧬 Gene Expression Analysis Dashboard")
st.caption("Original primary/metastatic workflow retained + new sarcoma subtype comparative analysis integrated.")

with st.sidebar:
    st.header("Analysis controls")
    lfc_thresh = st.slider("|logFC| threshold", 0.5, 3.0, 1.0, 0.1)
    p_thresh = st.select_slider("p-value threshold", options=[0.001,0.005,0.01,0.05,0.1], value=0.05)
    st.info("Use the Sarcoma tab for the guide-requested 3-4 subtype comparison.")

tabs = st.tabs(["Original Overview", "Expression Explorer", "DEG + TP53", "Sarcoma Subtype Comparison", "Downloads"])

with tabs[0]:
    st.subheader("Original Primary vs Metastatic Analysis")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Samples", len(expr)); c2.metric("Genes", len(gene_cols)); c3.metric("Primary", int((expr.Label==0).sum())); c4.metric("Metastatic", int((expr.Label==1).sum()))
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    axes[0].pie([(expr.Label==0).sum(), (expr.Label==1).sum()], labels=["Primary","Metastatic"], autopct="%1.1f%%", colors=[PALETTE["Primary"],PALETTE["Metastatic"]], startangle=90, wedgeprops={"edgecolor":"white"})
    axes[0].set_title("Sample distribution")
    pcs = PCA(n_components=2, random_state=42).fit_transform(expr[gene_cols])
    pca_df = pd.DataFrame(pcs, columns=["PC1","PC2"]); pca_df["Group"] = sample_type.values
    for g,grp in pca_df.groupby("Group"):
        axes[1].scatter(grp.PC1, grp.PC2, label=g, s=45, alpha=.85, color=PALETTE[g], edgecolors="white", linewidths=.3)
    axes[1].set_title("PCA - original groups"); axes[1].legend(); axes[1].grid(alpha=.25)
    st.image(fig_to_bytes(fig))

with tabs[1]:
    st.subheader("Gene Expression Explorer")
    term = st.text_input("Search gene", "TP53")
    group = st.selectbox("Sample group", ["All", "Primary", "Metastatic"])
    genes = [g for g in gene_cols if term.lower() in g.lower()] if term else gene_cols
    mask = np.ones(len(expr), dtype=bool) if group == "All" else (sample_type.values == group)
    show_cols = ["Label"] + genes[:25]
    st.dataframe(expr.loc[mask, show_cols].head(80), use_container_width=True)
    if genes:
        gene = st.selectbox("Plot gene", genes)
        fig, ax = plt.subplots(figsize=(7,4))
        plot_df = pd.DataFrame({"Expression": expr[gene], "Group": sample_type})
        sns.boxplot(data=plot_df, x="Group", y="Expression", hue="Group", palette=PALETTE, ax=ax, legend=False)
        sns.stripplot(data=plot_df, x="Group", y="Expression", color="black", alpha=.25, size=2, ax=ax)
        ax.set_title(f"{gene} expression")
        st.image(fig_to_bytes(fig))

with tabs[2]:
    st.subheader("Differential Expression + TP53")
    regs = np.select([(deg.logFC > lfc_thresh) & (deg.p_value < p_thresh), (deg.logFC < -lfc_thresh) & (deg.p_value < p_thresh)], ["Upregulated", "Downregulated"], default="Not Significant")
    df = deg.copy(); df["Regulation"] = regs
    c1,c2,c3 = st.columns(3); c1.metric("Upregulated", int((df.Regulation=="Upregulated").sum())); c2.metric("Downregulated", int((df.Regulation=="Downregulated").sum())); c3.metric("Not significant", int((df.Regulation=="Not Significant").sum()))
    fig, ax = plt.subplots(figsize=(8,5))
    for reg in ["Not Significant", "Upregulated", "Downregulated"]:
        m = df.Regulation == reg; ax.scatter(df.loc[m,"logFC"], df.loc[m,"-log10(p_value)"], s=18, alpha=.75, color=COLOR_REG[reg], label=f"{reg} ({m.sum()})")
    ax.axvline(lfc_thresh, ls="--", color="black", lw=.8); ax.axvline(-lfc_thresh, ls="--", color="black", lw=.8); ax.axhline(-np.log10(p_thresh), ls="--", color="black", lw=.8)
    ax.set_xlabel("logFC"); ax.set_ylabel("-log10(p-value)"); ax.set_title("Volcano plot"); ax.legend(fontsize=8); ax.grid(alpha=.2)
    st.image(fig_to_bytes(fig))
    st.dataframe(df.sort_values("p_value").head(100), use_container_width=True)
    st.markdown("### TP53-related genes")
    tp53 = df[df.Gene.str.contains("TP53", case=False, na=False)]
    st.dataframe(tp53, use_container_width=True)

with tabs[3]:
    st.subheader("Guide-requested Sarcoma Subtype Comparative Analysis")
    st.write("Compares 4 sarcoma types and shows how genes are differently expressed subtype-wise.")
    ovrs = pd.concat([one_vs_rest(sarc_expr, s, lfc_thresh, p_thresh) for s in SARCOMA_SUBTYPES], ignore_index=True)
    pairs = pd.concat([pairwise_deg(sarc_expr, a, b, lfc_thresh, p_thresh) for a,b in combinations(SARCOMA_SUBTYPES,2)], ignore_index=True)
    summary = []
    for s in SARCOMA_SUBTYPES:
        d = ovrs[ovrs.Subtype == s]
        summary.append({"Subtype":s,"Samples":int((sarc_expr.Subtype==s).sum()),"Upregulated":int((d.Regulation=="Upregulated").sum()),"Downregulated":int((d.Regulation=="Downregulated").sum()),"Top subtype marker":d.iloc[0].Gene,"Top logFC":round(float(d.iloc[0].logFC),2),"Top p-value":d.iloc[0].p_value})
    st.dataframe(pd.DataFrame(summary), use_container_width=True)
    c1,c2 = st.columns([1,1])
    with c1:
        pcs = PCA(n_components=2, random_state=42).fit_transform(sarc_expr[sarc_genes])
        pca_df = pd.DataFrame(pcs, columns=["PC1","PC2"]); pca_df["Subtype"] = sarc_expr.Subtype.values
        fig, ax = plt.subplots(figsize=(7,5))
        for s,g in pca_df.groupby("Subtype"):
            ax.scatter(g.PC1, g.PC2, label=s, s=48, alpha=.85, color=SUBTYPE_COLORS[s], edgecolors="white", linewidths=.3)
        ax.set_title("PCA - sarcoma subtype separation"); ax.legend(fontsize=7); ax.grid(alpha=.25)
        st.image(fig_to_bytes(fig))
    with c2:
        selected = []
        for s in SARCOMA_SUBTYPES:
            selected += ovrs[(ovrs.Subtype==s) & (ovrs.Regulation=="Upregulated")].head(5).Gene.tolist()
        selected = list(dict.fromkeys(selected))[:22]
        mean_mat = sarc_expr.groupby("Subtype")[selected].mean().loc[SARCOMA_SUBTYPES]
        # Compute subtype-wise z-score manually because sns.heatmap has no z_score argument.
        mean_mat_z = mean_mat.sub(mean_mat.mean(axis=1), axis=0).div(mean_mat.std(axis=1).replace(0, 1), axis=0)
        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(mean_mat_z, cmap="viridis", linewidths=.25, ax=ax, cbar_kws={"label":"Subtype-wise z-score"})
        ax.set_title("Subtype marker heatmap"); ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        st.image(fig_to_bytes(fig))
    chosen = st.selectbox("Select subtype for one-vs-rest DEGs", SARCOMA_SUBTYPES)
    st.markdown(f"#### Top genes: {chosen} vs rest")
    st.dataframe(ovrs[ovrs.Subtype == chosen].head(50), use_container_width=True)
    comp = st.selectbox("Select pairwise comparison", sorted(pairs.Comparison.unique()))
    st.markdown(f"#### Pairwise DEGs: {comp}")
    st.dataframe(pairs[pairs.Comparison == comp].head(50), use_container_width=True)
    st.markdown("#### Expected biological marker pattern")
    marker_table = pd.DataFrame([{"Subtype":k, "Expected high-expression marker genes":", ".join(v)} for k,v in SARCOMA_MARKER_GENES.items()])
    st.dataframe(marker_table, use_container_width=True)

with tabs[4]:
    st.subheader("Download Results")
    ovrs = pd.concat([one_vs_rest(sarc_expr, s, lfc_thresh, p_thresh) for s in SARCOMA_SUBTYPES], ignore_index=True)
    pairs = pd.concat([pairwise_deg(sarc_expr, a, b, lfc_thresh, p_thresh) for a,b in combinations(SARCOMA_SUBTYPES,2)], ignore_index=True)
    st.download_button("Download original DEG_results.csv", deg.to_csv(index=False), "DEG_results.csv", "text/csv")
    st.download_button("Download expression_matrix.csv", expr.to_csv(index=False), "expression_matrix.csv", "text/csv")
    st.download_button("Download sarcoma_one_vs_rest_DEG.csv", ovrs.to_csv(index=False), "sarcoma_one_vs_rest_DEG.csv", "text/csv")
    st.download_button("Download sarcoma_pairwise_DEG.csv", pairs.to_csv(index=False), "sarcoma_pairwise_DEG.csv", "text/csv")
    st.download_button("Download sarcoma_subtype_expression_matrix.csv", sarc_expr.to_csv(index=False), "sarcoma_subtype_expression_matrix.csv", "text/csv")
