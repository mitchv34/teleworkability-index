import marimo

__generated_with = "0.12.5"
app = marimo.App(width="full", app_title="Feature Importance Explorer")


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
        # 🔍 Feature Importance Explorer
        
        **Understanding what drives teleworkability predictions**
        
        This tool explores which O*NET features (skills, abilities, work context, activities) 
        are most important for predicting teleworkability. We examine both stages of the model:
        
        1. **Classifier**: Which features predict whether work can be done remotely
        2. **Regressor**: Which features predict how much work can be done remotely
        
        ---
        """
    )
    return


@app.cell
def __(Path, pd):
    # Load feature importance data
    clf_path = Path(__file__).parent.parent / "results" / "classifier_feature_importance.csv"
    reg_path = Path(__file__).parent.parent / "results" / "regressor_feature_importance.csv"
    
    clf_importance = pd.read_csv(clf_path)
    reg_importance = pd.read_csv(reg_path)
    
    # Extract feature category from element_id (first letter: 1=Skills, 2=Abilities, 4=Work Activities)
    def categorize_feature(element_id):
        if pd.isna(element_id):
            return 'Unknown'
        first_digit = str(element_id)[0]
        if first_digit == '1':
            return 'Skills'
        elif first_digit == '2':
            return 'Abilities'
        elif first_digit == '4':
            return 'Work Activities'
        else:
            return 'Work Context'
    
    clf_importance['category'] = clf_importance['element_id'].apply(categorize_feature)
    reg_importance['category'] = reg_importance['element_id'].apply(categorize_feature)
    return categorize_feature, clf_importance, clf_path, reg_importance, reg_path


@app.cell
def __(mo):
    # Model selection
    model_choice = mo.ui.radio(
        options=['Classifier (Zero/Non-Zero)', 'Regressor (Teleworkability Amount)'],
        value='Classifier (Zero/Non-Zero)',
        label="Select Model Stage"
    )
    
    model_choice
    return (model_choice,)


@app.cell
def __(clf_importance, model_choice, reg_importance):
    # Select appropriate importance data
    importance_df = clf_importance if 'Classifier' in model_choice.value else reg_importance
    return (importance_df,)


@app.cell
def __(importance_df, mo):
    # Category filter
    categories = ['All'] + sorted(importance_df['category'].unique().tolist())
    
    category_filter = mo.ui.dropdown(
        options=categories,
        value='All',
        label="Feature Category"
    )
    
    # Top N slider
    top_n = mo.ui.slider(
        start=5,
        stop=30,
        value=15,
        step=5,
        label="Number of Features to Show",
        show_value=True
    )
    
    mo.hstack([category_filter, top_n])
    return categories, category_filter, top_n


@app.cell
def __(category_filter, importance_df, top_n):
    # Apply filters
    filtered_imp = importance_df.copy()
    
    if category_filter.value != 'All':
        filtered_imp = filtered_imp[filtered_imp['category'] == category_filter.value]
    
    # Get top N by MDI importance
    top_features = filtered_imp.nlargest(top_n.value, 'mdi_importance')
    return filtered_imp, top_features


@app.cell
def __(alt, mo, top_features):
    # MDI importance chart
    mdi_chart = alt.Chart(top_features).mark_bar().encode(
        y=alt.Y('element_name:N', sort='-x', title='Feature'),
        x=alt.X('mdi_importance:Q', title='Mean Decrease in Impurity (MDI)'),
        color=alt.Color('category:N', legend=alt.Legend(title='Category')),
        tooltip=['element_name:N', 'mdi_importance:Q', 'category:N', 'direction_label:N']
    ).properties(
        width=700,
        height=max(300, len(top_features) * 20),
        title='Feature Importance (MDI)'
    )
    
    mo.md("### 📊 Mean Decrease in Impurity (MDI)")
    mo.ui.altair_chart(mdi_chart)
    return (mdi_chart,)


@app.cell
def __(alt, mo, top_features):
    # Permutation importance chart
    perm_chart = alt.Chart(top_features).mark_bar().encode(
        y=alt.Y('element_name:N', sort='-x', title='Feature'),
        x=alt.X('permutation_importance:Q', title='Permutation Importance'),
        color=alt.Color('category:N', legend=alt.Legend(title='Category')),
        tooltip=['element_name:N', 'permutation_importance:Q', 'category:N', 'direction_label:N']
    ).properties(
        width=700,
        height=max(300, len(top_features) * 20),
        title='Permutation Importance'
    )
    
    mo.md("### 🔀 Permutation Importance")
    mo.md("*How much does model performance drop when this feature is randomly shuffled?*")
    mo.ui.altair_chart(perm_chart)
    return (perm_chart,)


@app.cell
def __(mo, top_features):
    # Direction analysis
    mo.md("### ↕️ Feature Direction")
    mo.md("*Does this feature increase or decrease teleworkability?*")
    
    direction_table = top_features[['element_name', 'category', 'direction_sign', 'direction_label']].copy()
    direction_table.columns = ['Feature', 'Category', 'Sign', 'Direction']
    direction_table['Sign'] = direction_table['Sign'].map({1: '↑', -1: '↓', 0: '↔'})
    
    mo.ui.table(direction_table.reset_index(drop=True))
    return direction_table,


@app.cell
def __(importance_df, mo):
    # Category breakdown
    category_stats = importance_df.groupby('category').agg({
        'mdi_importance': ['mean', 'sum', 'count']
    }).round(4)
    category_stats.columns = ['Mean Importance', 'Total Importance', 'Feature Count']
    category_stats = category_stats.sort_values('Total Importance', ascending=False)
    
    mo.md("### 📈 Importance by Category")
    mo.ui.table(category_stats)
    return category_stats,


@app.cell
def __(clf_importance, mo, reg_importance):
    # Key insights
    top_clf = clf_importance.nlargest(1, 'mdi_importance').iloc[0]
    top_reg = reg_importance.nlargest(1, 'mdi_importance').iloc[0]
    
    mo.md(
        f"""
        ### 💡 Key Insights
        
        **Classifier (Can work be done remotely?):**
        - Most important feature: **{top_clf['element_name']}**
        - Direction: {top_clf['direction_label']}
        - MDI Importance: {top_clf['mdi_importance']:.4f}
        
        **Regressor (How much work can be done remotely?):**
        - Most important feature: **{top_reg['element_name']}**
        - Direction: {top_reg['direction_label']}
        - MDI Importance: {top_reg['mdi_importance']:.4f}
        """
    )
    return top_clf, top_reg


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        
        ## Understanding Feature Importance
        
        **Two Measures of Importance:**
        
        1. **MDI (Mean Decrease in Impurity)**: 
           - How much does this feature reduce uncertainty when making splits in the tree?
           - Built into Random Forest training
           - Fast to compute but can be biased toward high-cardinality features
        
        2. **Permutation Importance**:
           - How much does model performance drop if we shuffle this feature's values?
           - More reliable but slower to compute
           - Directly measures impact on prediction quality
        
        **Direction Interpretation:**
        
        For the **Classifier**:
        - ↑ "More likely zero" = Feature makes remote work LESS feasible
        - ↓ "Less likely zero" = Feature makes remote work MORE feasible
        
        For the **Regressor**:
        - ↑ "Increases teleworkability" = Higher values → more remote work possible
        - ↓ "Decreases teleworkability" = Higher values → less remote work possible
        
        **Feature Categories:**
        
        - **Skills**: Developed capacities (e.g., Programming, Critical Thinking)
        - **Abilities**: Enduring attributes (e.g., Oral Expression, Manual Dexterity)
        - **Work Context**: Physical and social conditions (e.g., Face-to-Face, Outdoors)
        - **Work Activities**: General work behaviors (e.g., Analyzing Data, Operating Vehicles)
        
        **Example Interpretations:**
        
        - High importance of "Face-to-Face Discussions" with negative direction → Jobs requiring face-to-face work are less teleworkable
        - High importance of "Working with Computers" with positive direction → Computer-intensive jobs are more teleworkable
        - High importance of "Manual Dexterity" with negative direction → Jobs requiring physical manipulation are less teleworkable
        
        **Citation:** Valdes-Bobes, M. & Lukianova, A. (2025). "Why Remote Work Stuck: A Structural Decomposition"
        """
    )
    return


if __name__ == "__main__":
    app.run()
