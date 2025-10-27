import marimo

__generated_with = "0.12.5"
app = marimo.App(width="full", app_title="Teleworkability Index Explorer")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import altair as alt
    from pathlib import Path
    return Path, alt, mo, np, pd


@app.cell
def __(mo):
    mo.md(
        r"""
        # Teleworkability Index (ψ) Explorer
        
        **Interactive exploration of occupation-level teleworkability predictions**
        
        This tool allows you to explore the teleworkability index constructed using O*NET features 
        and ORS survey labels. The index predicts the fraction of work that can be done from home 
        for 873 occupations.
        
        ---
        """
    )
    return


@app.cell
def __(Path, pd):
    # Load the full predictions
    data_path = Path(__file__).parent.parent / "results" / "full_occupation_predictions.csv"
    df = pd.read_csv(data_path, index_col=0)
    
    # Load metrics
    metrics_path = Path(__file__).parent.parent / "results" / "model_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    
    # Add occupation names (if available)
    try:
        ref_path = Path(__file__).parent.parent / "data" / "onet_data" / "processed" / "reference" / "OCCUPATION_DATA.csv"
        occ_ref = pd.read_csv(ref_path)
        df = df.merge(occ_ref[['ONET_SOC_CODE', 'TITLE']], left_index=True, right_on='ONET_SOC_CODE', how='left')
        df.set_index('ONET_SOC_CODE', inplace=True)
    except:
        df['TITLE'] = df.index
    
    # Create categories
    df['Category'] = pd.cut(df['Predicted'], 
                            bins=[0, 0.1, 0.3, 0.5, 0.7, 1.0],
                            labels=['Very Low (0-10%)', 'Low (10-30%)', 
                                   'Medium (30-50%)', 'High (50-70%)', 
                                   'Very High (70-100%)'])
    return data_path, df, metrics, metrics_path, occ_ref, ref_path


@app.cell
def __(df, metrics, mo):
    mo.md(
        f"""
        ## Model Performance Summary
        
        **Overall Statistics:**
        - Total Occupations: {len(df):,}
        - Labeled (from ORS): {df['is_labeled'].sum():,}
        - Predicted (unlabeled): {(df['is_labeled'] == 0).sum():,}
        
        **Training Metrics:**
        - F1 Score (Zero Classification): {metrics[metrics['split'] == 'train']['f1'].values[0]:.3f}
        - MAE (Non-Zero): {metrics[metrics['split'] == 'train']['mae_non_zero'].values[0]:.3f}
        - Correlation: {metrics[metrics['split'] == 'train']['corr'].values[0]:.3f}
        - R²: {metrics[metrics['split'] == 'train']['r2'].values[0]:.3f}
        """
    )
    return


@app.cell
def __(df, mo):
    # Interactive filters
    mo.md("## Filter Occupations")
    
    search = mo.ui.text(placeholder="Search occupation title...", label="Search")
    
    category_filter = mo.ui.dropdown(
        options=['All'] + df['Category'].dropna().unique().tolist(),
        value='All',
        label="Teleworkability Category"
    )
    
    show_labeled = mo.ui.checkbox(value=True, label="Show labeled (ORS)")
    show_unlabeled = mo.ui.checkbox(value=True, label="Show predicted")
    
    mo.hstack([search, category_filter, show_labeled, show_unlabeled])
    return category_filter, search, show_labeled, show_unlabeled


@app.cell
def __(category_filter, df, search, show_labeled, show_unlabeled):
    # Apply filters
    filtered_df = df.copy()
    
    # Search filter
    if search.value:
        filtered_df = filtered_df[
            filtered_df['TITLE'].str.contains(search.value, case=False, na=False)
        ]
    
    # Category filter
    if category_filter.value != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == category_filter.value]
    
    # Labeled/unlabeled filter
    if not show_labeled.value:
        filtered_df = filtered_df[filtered_df['is_labeled'] == 0]
    if not show_unlabeled.value:
        filtered_df = filtered_df[filtered_df['is_labeled'] == 1]
    
    n_filtered = len(filtered_df)
    return filtered_df, n_filtered


@app.cell
def __(filtered_df, mo, n_filtered):
    mo.md(f"### Filtered Results: {n_filtered:,} occupations")
    return


@app.cell
def __(alt, filtered_df, mo):
    # Distribution histogram
    chart = alt.Chart(filtered_df.reset_index()).mark_bar().encode(
        x=alt.X('Predicted:Q', bin=alt.Bin(maxbins=30), title='Teleworkability Index (ψ)'),
        y=alt.Y('count()', title='Number of Occupations'),
        color=alt.Color('is_labeled:N', 
                       scale=alt.Scale(domain=[0, 1], range=['#FF6B6B', '#4ECDC4']),
                       legend=alt.Legend(title='Source', labelExpr="datum.value == 1 ? 'Labeled (ORS)' : 'Predicted'"))
    ).properties(
        width=700,
        height=400,
        title='Distribution of Teleworkability Index'
    ).interactive()
    
    mo.ui.altair_chart(chart)
    return (chart,)


@app.cell
def __(filtered_df, mo):
    # Top 10 most teleworkable
    top_10 = filtered_df.nlargest(10, 'Predicted')[['TITLE', 'Predicted', 'is_labeled']]
    top_10['Predicted'] = (top_10['Predicted'] * 100).round(1).astype(str) + '%'
    top_10['Source'] = top_10['is_labeled'].map({1: 'ORS Survey', 0: 'Predicted'})
    top_10 = top_10[['TITLE', 'Predicted', 'Source']]
    top_10.columns = ['Occupation', 'Teleworkability', 'Source']
    
    mo.md("### 🏆 Top 10 Most Teleworkable Occupations")
    mo.ui.table(top_10.reset_index(drop=True))
    return top_10,


@app.cell
def __(filtered_df, mo):
    # Bottom 10 least teleworkable
    bottom_10 = filtered_df.nsmallest(10, 'Predicted')[['TITLE', 'Predicted', 'is_labeled']]
    bottom_10['Predicted'] = (bottom_10['Predicted'] * 100).round(1).astype(str) + '%'
    bottom_10['Source'] = bottom_10['is_labeled'].map({1: 'ORS Survey', 0: 'Predicted'})
    bottom_10 = bottom_10[['TITLE', 'Predicted', 'Source']]
    bottom_10.columns = ['Occupation', 'Teleworkability', 'Source']
    
    mo.md("### 📌 Bottom 10 Least Teleworkable Occupations")
    mo.ui.table(bottom_10.reset_index(drop=True))
    return bottom_10,


@app.cell
def __(filtered_df, mo):
    # Summary statistics
    summary = filtered_df['Predicted'].describe()
    
    mo.md(
        f"""
        ### 📊 Summary Statistics (Filtered Data)
        
        - **Mean**: {summary['mean']:.3f}
        - **Median**: {summary['50%']:.3f}
        - **Std Dev**: {summary['std']:.3f}
        - **Min**: {summary['min']:.3f}
        - **Max**: {summary['max']:.3f}
        """
    )
    return (summary,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        
        **About this tool:**
        
        - The teleworkability index ranges from 0 (not teleworkable) to 1 (fully teleworkable)
        - Predictions are based on a two-stage Random Forest model
        - Stage 1: Classify whether an occupation can be done remotely (>0%)
        - Stage 2: Predict the teleworkability fraction for non-zero occupations
        
        **Data sources:**
        - O*NET Database (skills, abilities, work context, activities)
        - ORS Survey (labeled teleworkability data for 370 occupations)
        
        **Citation:** Valdes-Bobes, M. & Lukianova, A. (2025). "Why Remote Work Stuck: A Structural Decomposition"
        """
    )
    return


if __name__ == "__main__":
    app.run()
