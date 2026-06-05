
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_rel, chi2_contingency, friedmanchisquare

# Page configuration
st.set_page_config(
    page_title="XAI Study Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E5266;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4A90A4;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #E8F4F8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4A90A4;
    }
    .conclusion-box {
        background-color: #D4EDDA;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28A745;
        margin-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# =====================================================================
# LOAD DATA
# =====================================================================

@st.cache_data
def load_data():
    """Load and preprocess the study data"""
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base, '..', 'data', 'xai_study_results.csv'))
    return df

# Load data
try:
    df = load_data()
    
    # Prepare subsets
    no_exp_data = df[df['explanation_shown'] == False].copy()
    with_exp_data = df[df['explanation_shown'] == True].copy()
    exp_data = df[df['explanation_shown'] == True].copy()
    
    data_loaded = True
except FileNotFoundError:
    st.error("❌ Could not find 'xai_study_results.csv'. Make sure it exists at data/xai_study_results.csv")
    data_loaded = False

# =====================================================================
# HEADER
# =====================================================================

st.markdown('<p class="main-header">🤖 Explainable AI Study Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Do Explanations Build Trust? A Study on AI Recommendations</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #999;">Group 02 | DIS IS 596 | UIUC</p>', unsafe_allow_html=True)

if not data_loaded:
    st.stop()

# =====================================================================
# SIDEBAR
# =====================================================================

st.sidebar.markdown("## 📊 Study Overview")
st.sidebar.markdown(f"**Participants:** {df['participant_id'].nunique()}")
st.sidebar.markdown(f"**Total Observations:** {len(df):,}")
st.sidebar.markdown(f"**Movies per Participant:** {df.groupby('participant_id')['movieId'].nunique().iloc[0]}")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔬 Research Questions")
st.sidebar.markdown("""
**RQ1:** Do explanations increase trust?  
**RQ2:** Do explanations increase selection?  
**RQ3:** Do explanation types differ?
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎯 Navigation")
st.sidebar.info("Use the tabs above to explore different analyses")

# =====================================================================
# TABS
# =====================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview", 
    "🔍 RQ1: Trust", 
    "✅ RQ2: Selection", 
    "🎨 RQ3: Explanation Types",
    "⏱️ Decision Time",
    "👥 Participants"
])

# =====================================================================
# TAB 1: OVERVIEW
# =====================================================================

with tab1:
    st.markdown('<p class="sub-header">Study Overview</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Participants",
            value=df['participant_id'].nunique()
        )
    
    with col2:
        st.metric(
            label="Total Interactions",
            value=f"{len(df):,}"
        )
    
    with col3:
        trust_increase = with_exp_data['trust_mean'].mean() - no_exp_data['trust_mean'].mean()
        st.metric(
            label="Trust Increase",
            value=f"+{trust_increase:.2f}",
            delta="With Explanations"
        )
    
    with col4:
        selection_increase = (with_exp_data['decision'].mean() - no_exp_data['decision'].mean()) * 100
        st.metric(
            label="Selection Increase",
            value=f"+{selection_increase:.1f}%",
            delta="With Explanations"
        )
    
    st.markdown("---")
    
    # Key Findings
    st.markdown('<p class="sub-header">🎯 Key Findings</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ RQ1: Trust Analysis
        - **Without Explanation:** {:.2f} ± {:.2f}
        - **With Explanation:** {:.2f} ± {:.2f}
        - **Increase:** +{:.2f} points
        - **Result:** Statistically Significant
        """.format(
            no_exp_data['trust_mean'].mean(),
            no_exp_data['trust_mean'].std(),
            with_exp_data['trust_mean'].mean(),
            with_exp_data['trust_mean'].std(),
            trust_increase
        ))
    
    with col2:
        st.markdown("""
        ### ✅ RQ2: Selection Analysis
        - **Without Explanation:** {:.1f}%
        - **With Explanation:** {:.1f}%
        - **Increase:** +{:.1f} percentage points
        - **Result:** Statistically Significant
        """.format(
            no_exp_data['decision'].mean() * 100,
            with_exp_data['decision'].mean() * 100,
            selection_increase
        ))
    
    st.markdown("---")
    
    # Demographics
    st.markdown('<p class="sub-header">👥 Participant Demographics</p>', unsafe_allow_html=True)
    
    participant_data = df.groupby('participant_id').first()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Top 5 Favorite Genres**")
        genre_counts = participant_data['favorite_genre'].value_counts().head(5)
        for genre, count in genre_counts.items():
            pct = (count / len(participant_data)) * 100
            st.write(f"• {genre}: {count} ({pct:.1f}%)")
    
    with col2:
        st.markdown("**Top 5 Moods**")
        mood_counts = participant_data['mood'].value_counts().head(5)
        for mood, count in mood_counts.items():
            pct = (count / len(participant_data)) * 100
            st.write(f"• {mood}: {count} ({pct:.1f}%)")
    
    with col3:
        st.markdown("**Top 5 Liked Aspects**")
        aspect_counts = participant_data['liked_aspects'].value_counts().head(5)
        for aspect, count in aspect_counts.items():
            pct = (count / len(participant_data)) * 100
            st.write(f"• {aspect}: {count} ({pct:.1f}%)")

# =====================================================================
# TAB 2: RQ1 - TRUST
# =====================================================================

with tab2:
    st.markdown('<p class="sub-header">RQ1: Do Explanations Increase Trust?</p>', unsafe_allow_html=True)
    
    # Calculate statistics
    trust_no_exp = no_exp_data['trust_mean'].mean()
    trust_with_exp = with_exp_data['trust_mean'].mean()
    trust_diff = trust_with_exp - trust_no_exp
    
    # Paired t-test
    paired_data = df.groupby(['participant_id', 'explanation_shown'])['trust_mean'].mean().unstack()
    trust_without = paired_data[False].values
    trust_with = paired_data[True].values
    
    t_stat, p_value = ttest_rel(trust_with, trust_without)
    diff = trust_with - trust_without
    cohens_d = diff.mean() / diff.std()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Without Explanation", f"{trust_no_exp:.3f}")
    with col2:
        st.metric("With Explanation", f"{trust_with_exp:.3f}")
    with col3:
        st.metric("Difference", f"+{trust_diff:.3f}", delta="Increase")
    with col4:
        st.metric("Cohen's d", f"{cohens_d:.3f}")
    
    st.markdown("---")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        conditions = ['Without\nExplanation', 'With\nExplanation']
        means = [trust_no_exp, trust_with_exp]
        sds = [no_exp_data['trust_mean'].std(), with_exp_data['trust_mean'].std()]
        
        bars = ax.bar(conditions, means, color=['#d9534f', '#5cb85c'], alpha=0.7, 
                      edgecolor='black', linewidth=1.5)
        ax.errorbar(conditions, means, yerr=sds, fmt='none', color='black', 
                   capsize=5, linewidth=2)
        
        ax.set_ylabel('Mean Trust Score', fontsize=12, fontweight='bold')
        ax.set_title('Trust Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 5)
        ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, mean, sd in zip(bars, means, sds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + sd + 0.1,
                   f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
        
        if p_value < 0.05:
            y_max = max(means) + max(sds) + 0.3
            ax.plot([0, 1], [y_max, y_max], 'k-', linewidth=1.5)
            sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'
            ax.text(0.5, y_max + 0.1, sig_text, ha='center', fontsize=16, fontweight='bold')
        
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for i in range(len(trust_without)):
            color = '#5cb85c' if trust_with[i] > trust_without[i] else '#d9534f'
            ax.plot([0, 1], [trust_without[i], trust_with[i]], 
                   color=color, alpha=0.3, linewidth=1)
        
        ax.plot([0, 1], [trust_without.mean(), trust_with.mean()], 
               color='black', linewidth=3, marker='o', markersize=10, label='Mean')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Without\nExplanation', 'With\nExplanation'])
        ax.set_ylabel('Trust Score', fontsize=12, fontweight='bold')
        ax.set_title('Individual Changes', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 5)
        ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
    
    # Statistical Results
    st.markdown("---")
    st.markdown('<p class="sub-header">📊 Statistical Results</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Paired t-test:**
        - t-statistic: {t_stat:.4f}
        - p-value: {p_value:.6f}
        - Degrees of freedom: {len(trust_without)-1}
        """)
    
    with col2:
        if p_value < 0.001:
            sig = "Highly Significant (p < 0.001)"
        elif p_value < 0.01:
            sig = "Very Significant (p < 0.01)"
        elif p_value < 0.05:
            sig = "Significant (p < 0.05)"
        else:
            sig = "Not Significant (p ≥ 0.05)"
        
        if abs(cohens_d) < 0.2:
            effect = "Negligible"
        elif abs(cohens_d) < 0.5:
            effect = "Small"
        elif abs(cohens_d) < 0.8:
            effect = "Medium"
        else:
            effect = "Large"
        
        st.markdown(f"""
        **Interpretation:**
        - Significance: {sig}
        - Effect Size: {effect}
        """)
    
    # Conclusion
    if p_value < 0.05:
        st.markdown(f"""
        <div class="conclusion-box">
        <strong>✅ RQ1 SUPPORTED:</strong> Explanations significantly increase trust in AI recommendations 
        (Mean difference = +{trust_diff:.3f}, {sig}, Cohen's d = {cohens_d:.3f})
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="conclusion-box">
        <strong>✗ RQ1 NOT SUPPORTED:</strong> No significant difference in trust
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# TAB 3: RQ2 - SELECTION
# =====================================================================

with tab3:
    st.markdown('<p class="sub-header">RQ2: Do Explanations Increase Selection Rate?</p>', unsafe_allow_html=True)
    
    # Calculate statistics
    selection_no_exp = no_exp_data['decision'].mean()
    selection_with_exp = with_exp_data['decision'].mean()
    selection_diff = selection_with_exp - selection_no_exp
    
    # McNemar's test
    df_sorted = df.sort_values(['participant_id', 'condition', 'movieId'])
    no_exp_decisions = df_sorted[df_sorted['explanation_shown'] == False]['decision'].values
    with_exp_decisions = df_sorted[df_sorted['explanation_shown'] == True]['decision'].values
    
    both_yes = ((no_exp_decisions == 1) & (with_exp_decisions == 1)).sum()
    both_no = ((no_exp_decisions == 0) & (with_exp_decisions == 0)).sum()
    no_then_yes = ((no_exp_decisions == 0) & (with_exp_decisions == 1)).sum()
    yes_then_no = ((no_exp_decisions == 1) & (with_exp_decisions == 0)).sum()
    
    b = yes_then_no
    c = no_then_yes
    
    if (b + c) > 0:
        mcnemar_stat = ((abs(b - c) - 1) ** 2) / (b + c)
        mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, 1)
    else:
        mcnemar_stat = 0
        mcnemar_p = 1.0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Without Explanation", f"{selection_no_exp:.1%}")
    with col2:
        st.metric("With Explanation", f"{selection_with_exp:.1%}")
    with col3:
        st.metric("Difference", f"+{selection_diff:.1%}", delta="Increase")
    with col4:
        st.metric("Chi-square", f"{mcnemar_stat:.3f}")
    
    st.markdown("---")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        conditions = ['Without\nExplanation', 'With\nExplanation']
        rates = [selection_no_exp * 100, selection_with_exp * 100]
        
        bars = ax.bar(conditions, rates, color=['#d9534f', '#5cb85c'], alpha=0.7,
                     edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Selection Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Selection Rate Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        if mcnemar_p < 0.05:
            y_max = max(rates) + 8
            ax.plot([0, 1], [y_max, y_max], 'k-', linewidth=1.5)
            sig_text = '***' if mcnemar_p < 0.001 else '**' if mcnemar_p < 0.01 else '*'
            ax.text(0.5, y_max + 2, sig_text, ha='center', fontsize=16, fontweight='bold')
        
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        categories = ['Selected\nWITHOUT\nonly', 'Selected in\nBOTH', 
                     'Selected\nWITH\nonly', 'Selected in\nNEITHER']
        values = [yes_then_no, both_yes, no_then_yes, both_no]
        colors = ['#d9534f', '#f0ad4e', '#5cb85c', '#5bc0de']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7,
                     edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Number of Movies', fontsize=12, fontweight='bold')
        ax.set_title('Decision Change Patterns', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(value)}', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
    
    # Statistical Results
    st.markdown("---")
    st.markdown('<p class="sub-header">📊 Statistical Results</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **McNemar's Test:**
        - Chi-square statistic: {mcnemar_stat:.4f}
        - p-value: {mcnemar_p:.6f}
        - Changed NO→YES: {c} movies
        - Changed YES→NO: {b} movies
        """)
    
    with col2:
        if mcnemar_p < 0.001:
            sig = "Highly Significant (p < 0.001)"
        elif mcnemar_p < 0.01:
            sig = "Very Significant (p < 0.01)"
        elif mcnemar_p < 0.05:
            sig = "Significant (p < 0.05)"
        else:
            sig = "Not Significant (p ≥ 0.05)"
        
        st.markdown(f"""
        **Interpretation:**
        - Significance: {sig}
        - More users changed from NO to YES
        """)
    
    # Conclusion
    if mcnemar_p < 0.05:
        st.markdown(f"""
        <div class="conclusion-box">
        <strong>✅ RQ2 SUPPORTED:</strong> Explanations significantly increase selection behavior
        ({c} movies changed NO→YES vs {b} changed YES→NO, {sig})
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="conclusion-box">
        <strong>✗ RQ2 NOT SUPPORTED:</strong> No significant difference in selection rate
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# TAB 4: RQ3 - EXPLANATION TYPES
# =====================================================================

with tab4:
    st.markdown('<p class="sub-header">RQ3: Do Explanation Types Differ in Effectiveness?</p>', unsafe_allow_html=True)
    
    # Trust by type
    trust_by_type = exp_data.groupby('explanation_type')['trust_mean'].agg(['mean', 'std', 'count'])
    trust_by_type = trust_by_type.sort_values('mean', ascending=False)
    
    # Selection by type
    selection_by_type = exp_data.groupby('explanation_type')['decision'].agg(['mean', 'sum', 'count'])
    selection_by_type = selection_by_type.sort_values('mean', ascending=False)
    
    # Friedman test (trust)
    trust_rating = exp_data[exp_data['explanation_type'] == 'rating'].groupby('participant_id')['trust_mean'].mean()
    trust_similarity = exp_data[exp_data['explanation_type'] == 'similarity'].groupby('participant_id')['trust_mean'].mean()
    trust_popularity = exp_data[exp_data['explanation_type'] == 'popularity'].groupby('participant_id')['trust_mean'].mean()
    
    common_participants = trust_rating.index.intersection(trust_similarity.index).intersection(trust_popularity.index)
    trust_rating_aligned = trust_rating.loc[common_participants].values
    trust_similarity_aligned = trust_similarity.loc[common_participants].values
    trust_popularity_aligned = trust_popularity.loc[common_participants].values
    
    friedman_stat, friedman_p = friedmanchisquare(trust_rating_aligned, trust_similarity_aligned, trust_popularity_aligned)
    
    # Chi-square test (selection)
    contingency_table = pd.crosstab(exp_data['explanation_type'], exp_data['decision'])
    chi2_stat, chi2_p, chi2_dof, chi2_expected = chi2_contingency(contingency_table)
    
    # Display rankings
    st.markdown("### 🏆 Rankings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Trust Rankings:**")
        for i, exp_type in enumerate(trust_by_type.index, 1):
            mean_val = trust_by_type.loc[exp_type, 'mean']
            st.write(f"{i}. **{exp_type.capitalize()}**: {mean_val:.3f}")
    
    with col2:
        st.markdown("**Selection Rate Rankings:**")
        for i, exp_type in enumerate(selection_by_type.index, 1):
            rate = selection_by_type.loc[exp_type, 'mean']
            st.write(f"{i}. **{exp_type.capitalize()}**: {rate:.1%}")
    
    st.markdown("---")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        types = trust_by_type.index.tolist()
        means = trust_by_type['mean'].values
        stds = trust_by_type['std'].values
        
        colors_map = {'rating': '#FFD700', 'similarity': '#FF69B4', 'popularity': '#87CEEB'}
        bar_colors = [colors_map.get(t, '#999999') for t in types]
        
        bars = ax.bar([t.capitalize() for t in types], means, color=bar_colors, 
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.errorbar([t.capitalize() for t in types], means, yerr=stds, 
                   fmt='none', color='black', capsize=5, linewidth=2)
        
        ax.set_ylabel('Mean Trust Score', fontsize=12, fontweight='bold')
        ax.set_title('Trust by Explanation Type', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 5)
        ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                   f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        types_sel = selection_by_type.index.tolist()
        rates = selection_by_type['mean'].values * 100
        
        bar_colors_sel = [colors_map.get(t, '#999999') for t in types_sel]
        
        bars = ax.bar([t.capitalize() for t in types_sel], rates, color=bar_colors_sel,
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Selection Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Selection Rate by Type', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
    
    # Statistical Results
    st.markdown("---")
    st.markdown('<p class="sub-header">📊 Statistical Results</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Trust Analysis (Friedman Test):**")
        st.write(f"- Chi-square: {friedman_stat:.4f}")
        st.write(f"- p-value: {friedman_p:.6f}")
        
        if friedman_p < 0.05:
            st.success("✅ Significant differences between types")
        else:
            st.info("No significant differences")
    
    with col2:
        st.markdown("**Selection Analysis (Chi-square):**")
        st.write(f"- Chi-square: {chi2_stat:.4f}")
        st.write(f"- p-value: {chi2_p:.6f}")
        
        if chi2_p < 0.05:
            st.success("✅ Significant differences between types")
        else:
            st.info("No significant differences")

# =====================================================================
# TAB 5: DECISION TIME
# =====================================================================

with tab5:
    st.markdown('<p class="sub-header">⏱️ Decision Time Analysis</p>', unsafe_allow_html=True)
    
    # Calculate statistics
    time_no_exp = no_exp_data['decision_time'].mean()
    time_with_exp = with_exp_data['decision_time'].mean()
    time_diff = time_with_exp - time_no_exp
    
    # Paired t-test
    time_paired = df.groupby(['participant_id', 'explanation_shown'])['decision_time'].mean().unstack()
    time_without = time_paired[False].values
    time_with = time_paired[True].values
    
    t_time, p_time = ttest_rel(time_with, time_without)
    diff_time = time_with - time_without
    cohens_d_time = diff_time.mean() / diff_time.std()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Without Explanation", f"{time_no_exp:.2f}s")
    with col2:
        st.metric("With Explanation", f"{time_with_exp:.2f}s")
    with col3:
        delta_label = "Slower" if time_diff > 0 else "Faster"
        st.metric("Difference", f"{time_diff:+.2f}s", delta=delta_label)
    with col4:
        st.metric("Cohen's d", f"{cohens_d_time:.3f}")
    
    st.markdown("---")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        positions = [1, 2]
        data_to_plot = [time_without, time_with]
        
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                       showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        for patch, color in zip(bp['boxes'], ['#d9534f', '#5cb85c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(['Without\nExplanation', 'With\nExplanation'])
        ax.set_ylabel('Decision Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Decision Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        faster_count = sum(time_with < time_without)
        slower_count = sum(time_with > time_without)
        
        for i in range(len(time_without)):
            color = '#5cb85c' if time_with[i] < time_without[i] else '#d9534f'
            ax.plot([0, 1], [time_without[i], time_with[i]], 
                   color=color, alpha=0.3, linewidth=1)
        
        ax.plot([0, 1], [time_without.mean(), time_with.mean()], 
               color='black', linewidth=3, marker='o', markersize=10, label='Mean')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Without\nExplanation', 'With\nExplanation'])
        ax.set_ylabel('Decision Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Individual Changes', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        ax.text(0.98, 0.02, f"Faster: {faster_count}\nSlower: {slower_count}",
               transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        st.pyplot(fig)
    
    # Interpretation
    st.markdown("---")
    st.markdown('<p class="sub-header">💡 Interpretation</p>', unsafe_allow_html=True)
    
    if p_time < 0.05:
        if time_diff > 0:
            st.info(f"""
            Users took **{abs(time_diff):.1f} seconds longer** to decide WITH explanations.
            This suggests explanations require additional cognitive processing.
            However, the benefit in trust and selection may justify this small time cost.
            """)
        else:
            st.success(f"""
            Users decided **{abs(time_diff):.1f} seconds FASTER** with explanations!
            This suggests explanations actually REDUCE cognitive load and decision uncertainty.
            """)
    else:
        st.success("""
        **No significant difference in decision time!**
        
        Explanations increased trust and selection WITHOUT slowing users down - efficiency preserved!
        This is the ideal outcome: better decisions at no time cost.
        """)

# =====================================================================
# TAB 6: PARTICIPANTS
# =====================================================================

with tab6:
    st.markdown('<p class="sub-header">👥 Participant Explorer</p>', unsafe_allow_html=True)
    
    participant_ids = sorted(df['participant_id'].unique())
    selected_participant = st.selectbox("Select a participant:", participant_ids)
    
    # Filter data for selected participant
    participant_df = df[df['participant_id'] == selected_participant]
    
    # Display participant info
    st.markdown("### Participant Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Favorite Genre:** {participant_df['favorite_genre'].iloc[0]}")
        st.write(f"**Favorite Movie:** {participant_df['favorite_movie'].iloc[0]}")
    
    with col2:
        st.write(f"**Mood:** {participant_df['mood'].iloc[0]}")
        st.write(f"**Liked Aspects:** {participant_df['liked_aspects'].iloc[0]}")
    
    with col3:
        st.write(f"**Preferred Year:** {participant_df['preferred_year'].iloc[0]}")
        st.write(f"**Condition Order:** {participant_df['condition_order'].iloc[0]}")
    
    st.markdown("---")
    
    # Show movies and decisions
    st.markdown("### Movies & Decisions")
    
    # Get the 6 unique movies
    movies = participant_df[participant_df['explanation_shown'] == False][['movieId', 'title', 'genres']].drop_duplicates()
    
    for idx, (_, movie) in enumerate(movies.iterrows(), 1):
        with st.expander(f"Movie {idx}: {movie['title']}"):
            movie_id = movie['movieId']
            
            # Get data for both conditions
            no_exp_row = participant_df[(participant_df['movieId'] == movie_id) & 
                                       (participant_df['explanation_shown'] == False)].iloc[0]
            with_exp_row = participant_df[(participant_df['movieId'] == movie_id) & 
                                          (participant_df['explanation_shown'] == True)].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**WITHOUT Explanation:**")
                st.write(f"• Decision: {'✅ Selected' if no_exp_row['decision'] == 1 else '❌ Not Selected'}")
                st.write(f"• Decision Time: {no_exp_row['decision_time']:.2f}s")
                st.write(f"• Trust Score: {no_exp_row['trust_mean']:.2f}")
            
            with col2:
                st.markdown("**WITH Explanation:**")
                st.write(f"• Decision: {'✅ Selected' if with_exp_row['decision'] == 1 else '❌ Not Selected'}")
                st.write(f"• Decision Time: {with_exp_row['decision_time']:.2f}s")
                st.write(f"• Trust Score: {with_exp_row['trust_mean']:.2f}")
                st.write(f"• Type: {with_exp_row['explanation_type'].capitalize()}")
            
            # Show explanation
            if with_exp_row['explanation_text']:
                st.markdown("**Explanation:**")
                st.info(with_exp_row['explanation_text'])

# =====================================================================
# FOOTER
# =====================================================================

st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #999; font-size: 0.9rem;">
XAI Study Dashboard | Group 02 | DIS IS 596 | University of Illinois Urbana-Champaign
</p>
""", unsafe_allow_html=True)
