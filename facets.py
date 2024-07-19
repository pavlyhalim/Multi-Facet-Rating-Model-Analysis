import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
import warnings

warnings.filterwarnings('ignore')

class MFRM:
    def __init__(self, data, facets):
        self.data = data
        self.facets = facets
        self.n_facets = len(facets)
        self.facet_levels = {facet: self.data[facet].unique() for facet in facets}
        self.n_levels = {facet: len(levels) for facet, levels in self.facet_levels.items()}
        self.params = None
        self.se = None
        self.fit_stats = None
        self.max_score = int(self.data['Max_Score'].max())

    def initialize_params(self):
        return np.random.randn(sum(self.n_levels.values()))

    def mfrm_probability(self, params):
        logits = np.zeros(len(self.data))
        start = 0
        for facet in self.facets:
            end = start + self.n_levels[facet]
            facet_params = params[start:end]
            logits += facet_params[self.data[facet].cat.codes.values]
            start = end
        return expit(logits)

    def neg_log_likelihood(self, params):
        probs = self.mfrm_probability(params)
        probs = np.clip(probs, 1e-15, 1-1e-15) 
        return -np.sum(self.data['Score'] * np.log(probs) + (self.data['Max_Score'] - self.data['Score']) * np.log(1 - probs))

    def estimate_params(self, max_iter=1000, convergence_threshold=1e-6):
        params = self.initialize_params()
        for _ in range(max_iter):
            old_params = params.copy()
            result = minimize(self.neg_log_likelihood, params, method='L-BFGS-B')
            params = result.x
            if np.max(np.abs(params - old_params)) < convergence_threshold:
                break
        self.params = params
        self.calculate_se()

    def calculate_se(self):
        hessian = self.calculate_hessian()
        if np.linalg.det(hessian) == 0:
            print("Warning: Hessian is singular, cannot calculate standard errors.")
            self.se = np.full(len(self.params), np.nan)
        else:
            self.se = np.sqrt(np.diag(np.linalg.inv(hessian)))

    def calculate_hessian(self):
        epsilon = 1e-5
        n_params = len(self.params)
        hessian = np.zeros((n_params, n_params))
        for i in range(n_params):
            for j in range(i, n_params):
                params_plus_i = self.params.copy()
                params_minus_i = self.params.copy()
                params_plus_i[i] += epsilon
                params_minus_i[i] -= epsilon
                
                if i == j:
                    d2f = (self.neg_log_likelihood(params_plus_i) - 2*self.neg_log_likelihood(self.params) + self.neg_log_likelihood(params_minus_i)) / (epsilon**2)
                else:
                    params_plus_j = self.params.copy()
                    params_plus_ij = self.params.copy()
                    params_plus_j[j] += epsilon
                    params_plus_ij[i] += epsilon
                    params_plus_ij[j] += epsilon
                    
                    d2f = (self.neg_log_likelihood(params_plus_ij) - self.neg_log_likelihood(params_plus_i) - self.neg_log_likelihood(params_plus_j) + self.neg_log_likelihood(self.params)) / (epsilon**2)
                
                hessian[i, j] = d2f
                hessian[j, i] = d2f
        
        return hessian
    

    def calculate_fit_statistics(self):
        expected_scores = self.mfrm_probability(self.params) * self.data['Max_Score']
        residuals = self.data['Score'] - expected_scores
        std_residuals = residuals / np.sqrt(expected_scores * (1 - expected_scores/self.data['Max_Score']))
        
        self.infit = np.mean(std_residuals**2, axis=0)
        self.outfit = np.mean((residuals / (expected_scores * (1 - expected_scores/self.data['Max_Score'])))**2, axis=0)
        
        self.fit_stats = {
            'Infit': self.infit,
            'Outfit': self.outfit,
            'Point-measure correlation': np.corrcoef(self.data['Score'], expected_scores)[0, 1]
        }

    def plot_wright_map(self):
        plt.figure(figsize=(12, 8))
        start = 0
        for i, facet in enumerate(self.facets):
            end = start + self.n_levels[facet]
            facet_params = self.params[start:end]
            plt.subplot(1, self.n_facets, i+1)
            plt.hist(facet_params, bins=20, orientation='horizontal')
            plt.title(facet)
            plt.xlabel('Frequency')
            plt.ylabel('Measure (logits)')
            start = end
        plt.tight_layout()
        plt.savefig('wright_map.png')
        plt.close()

    def plot_category_probability_curves(self):
        plt.figure(figsize=(10, 6))
        theta = np.linspace(-5, 5, 100)
        for k in range(self.max_score + 1):
            p = expit(k * theta - np.sum([expit(j * theta) for j in range(k)]))
            plt.plot(theta, p, label=f'Category {k}')
        plt.xlabel('Ability - Difficulty')
        plt.ylabel('Probability')
        plt.title('Category Probability Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig('category_probability_curves.png')
        plt.close()

    def plot_category_information_functions(self):
        plt.figure(figsize=(10, 6))
        theta = np.linspace(-5, 5, 100)
        for k in range(self.max_score + 1):
            p = expit(k * theta - np.sum([expit(j * theta) for j in range(k)]))
            info = k**2 * p * (1 - p)
            plt.plot(theta, info, label=f'Category {k}')
        plt.xlabel('Ability - Difficulty')
        plt.ylabel('Information')
        plt.title('Category Information Functions')
        plt.legend()
        plt.grid(True)
        plt.savefig('category_information_functions.png')
        plt.close()

    def plot_facet_vertical_rulers(self):
        plt.figure(figsize=(12, 8))
        facets = self.facets
        for i, facet in enumerate(facets):
            start = sum(self.n_levels[f] for f in facets[:i])
            end = start + self.n_levels[facet]
            facet_params = self.params[start:end]
            
            plt.subplot(1, len(facets) + 1, i + 1)
            plt.scatter(np.zeros_like(facet_params), facet_params, marker='|')
            plt.title(facet)
            plt.ylim(min(self.params) - 1, max(self.params) + 1)
            plt.axis('off')
            plt.ylabel('Measure (logits)')
        
        plt.subplot(1, len(facets) + 1, len(facets) + 1)
        plt.yticks(np.arange(int(min(self.params)) - 1, int(max(self.params)) + 2))
        plt.title('Measure')
        plt.ylim(min(self.params) - 1, max(self.params) + 1)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('facet_vertical_rulers.png')
        plt.close()

    def plot_bias_interaction(self):
        facet1, facet2 = self.facets[:2] 
        bias = self.data.groupby([facet1, facet2])['Score'].mean().unstack()
        plt.figure(figsize=(12, 8))
        sns.heatmap(bias, annot=True, cmap='coolwarm', center=0)
        plt.title(f'Bias Interaction: {facet1} x {facet2}')
        plt.tight_layout()
        plt.savefig('bias_interaction.png')
        plt.close()

    def calculate_reliability(self):
        var_true = np.var(self.params[:self.n_levels[self.facets[0]]])
        var_error = np.mean(self.se[:self.n_levels[self.facets[0]]]**2)
        reliability = var_true / (var_true + var_error)
        return np.clip(reliability, 0, 1) 

    def calculate_separation(self):
        reliability = self.calculate_reliability()
        separation = np.sqrt(reliability / (1 - reliability))
        return separation if not np.isinf(separation) else np.nan

    def print_measurement_report(self, facet, title):
        print(f"\nTable {title}")
        print("=" * 80)
        print(f"{'ID':8s} {'Score':8s} {'Count':8s} {'Measure':8s} {'Model SE':8s} {'Infit':8s} {'Outfit':8s}")
        print(f"{'':8s} {'':8s} {'':8s} {'':8s} {'':8s} {'MnSq':8s} {'MnSq':8s}")
        print("-" * 80)
        
        start = sum(self.n_levels[f] for f in self.facets[:self.facets.index(facet)])
        end = start + self.n_levels[facet]
        facet_params = self.params[start:end]
        facet_se = self.se[start:end]
        
        for i, (level, param, se) in enumerate(zip(self.facet_levels[facet], facet_params, facet_se)):
            facet_data = self.data[self.data[facet] == level]
            score = facet_data['Score'].sum()
            count = len(facet_data)
            
            expected_scores = self.mfrm_probability(self.params)[facet_data.index] * facet_data['Max_Score']
            residuals = facet_data['Score'] - expected_scores
            std_residuals = residuals / np.sqrt(expected_scores * (1 - expected_scores/facet_data['Max_Score']))
            
            infit = np.mean(std_residuals**2)
            outfit = np.mean((residuals / (expected_scores * (1 - expected_scores/facet_data['Max_Score'])))**2)
            
            level_str = str(level)[:8]
            print(f"{level_str:<8s} {score:8d} {count:8d} {param:8.2f} {se:8.2f} {infit:8.2f} {outfit:8.2f}")
        
        print("-" * 80)
        print(f"{'Mean':8s} {'-':>8s} {'-':>8s} {np.mean(facet_params):8.2f} {np.mean(facet_se):8.2f} {np.mean(self.infit):8.2f} {np.mean(self.outfit):8.2f}")
        print(f"{'S.D.':8s} {'-':>8s} {'-':>8s} {np.std(facet_params):8.2f} {np.std(facet_se):8.2f} {np.std(self.infit):8.2f} {np.std(self.outfit):8.2f}")
        print("=" * 80)
        
        reliability = self.calculate_reliability()
        separation = self.calculate_separation()
        rmse = np.mean(facet_se)
        adj_sd = np.sqrt(max(np.var(facet_params) - rmse**2, 0))
        
        print(f"Model, Populn: RMSE {rmse:.2f} Adj (True) S.D. {adj_sd:.2f} Separation {separation:.2f} Reliability {reliability:.2f}")
        print(f"Model, Sample: RMSE {rmse:.2f} Adj (True) S.D. {adj_sd:.2f} Separation {separation:.2f} Reliability {reliability:.2f}")
        print(f"Model, Fixed (all same) chi-square: {np.sum((facet_params / facet_se)**2):.1f} d.f.: {len(facet_params)} significance (probability): {1 - stats.chi2.cdf(np.sum((facet_params / facet_se)**2), len(facet_params)):.4f}")
        
    def print_category_statistics(self):
        print("\nTable 5. Category Statistics.")
        print("=" * 80)
        print(f"{'Category':10s} {'Score':10s} {'Count':10s} {'Cum.':10s} {'Prob.':10s} {'Exp.':10s} {'OUTFIT':10s}")
        print(f"{'Label':10s} {'Value':10s} {'Used':10s} {'%':10s} {'Meas.':10s} {'Meas.':10s} {'MnSq':10s}")
        print("-" * 80)
        
        total_count = len(self.data)
        cumulative_count = 0
        expected_scores = self.mfrm_probability(self.params) * self.data['Max_Score']
        
        for score in range(self.max_score + 1):
            count = np.sum(self.data['Score'] == score)
            cumulative_count += count
            prob = count / total_count
            
            mask = self.data['Score'] == score
            if np.sum(mask) > 0:
                expected_measure = np.mean(expected_scores[mask])
                outfit = np.mean(((self.data['Score'][mask] - expected_scores[mask]) / 
                                  np.sqrt(expected_scores[mask] * (1 - expected_scores[mask]/self.data['Max_Score'][mask])))**2)
            else:
                expected_measure = np.nan
                outfit = np.nan
            
            print(f"{score:10d} {score:10d} {count:10d} {cumulative_count/total_count*100:10.1f} {prob:10.2f} {expected_measure:10.2f} {outfit:10.2f}")
        
        print("-" * 80)

    def print_unexpected_responses(self):
        print("\nTable 6. Unexpected Responses (arranged by fN).")
        print("=" * 80)
        print(f"{'Examinee':10s} {'Rater':10s} {'Criterion':10s} {'Score':10s} {'Exp.':10s} {'Resid.':10s} {'StRes':10s}")
        print("-" * 80)
        
        expected_scores = self.mfrm_probability(self.params) * self.data['Max_Score']
        residuals = self.data['Score'] - expected_scores
        std_residuals = residuals / np.sqrt(expected_scores * (1 - expected_scores/self.data['Max_Score']))
        
        unexpected = np.abs(std_residuals) > 2 
        unexpected_data = self.data[unexpected].sort_values('Score', ascending=False)
        
        for _, row in unexpected_data.iterrows():
            examinee = str(row['Examinee'])[:10]
            rater = str(row['Rater'])[:10]
            criterion = str(row['Criterion'])[:10]
            score = row['Score']
            exp_score = expected_scores.loc[_]
            resid = residuals.loc[_]
            st_res = std_residuals.loc[_]
            
            print(f"{examinee:<10s} {rater:<10s} {criterion:<10s} {score:10d} {exp_score:10.2f} {resid:10.2f} {st_res:10.2f}")
        
        print("-" * 80)

    def print_bias_interaction(self):
        print("\nTable 7. Bias/Interaction Report (arranged by fN).")
        print("=" * 80)
        print(f"{'Examinee':10s} {'Rater':10s} {'Obs.':10s} {'Exp.':10s} {'Bias':10s} {'SE':10s} {'t':10s}")
        print("-" * 80)
        
        expected_scores = self.mfrm_probability(self.params) * self.data['Max_Score']
        bias = self.data['Score'] - expected_scores
        
        for examinee in self.facet_levels['Examinee']:
            for rater in self.facet_levels['Rater']:
                mask = (self.data['Examinee'] == examinee) & (self.data['Rater'] == rater)
                if np.sum(mask) > 0:
                    obs = np.mean(self.data.loc[mask, 'Score'])
                    exp = np.mean(expected_scores[mask])
                    bias_value = obs - exp
                    se = np.std(bias[mask]) / np.sqrt(np.sum(mask))
                    t = bias_value / se if se != 0 else np.nan
                    
                    examinee_str = str(examinee)[:10]
                    rater_str = str(rater)[:10]
                    
                    print(f"{examinee_str:<10s} {rater_str:<10s} {obs:10.2f} {exp:10.2f} {bias_value:10.2f} {se:10.2f} {t:10.2f}")
        
        print("-" * 80)

    def run_irt_analysis(self):
        print("\nTable 8. IRT Analysis Results")
        print("====================")

        irt_data = self.data.copy()
        
        def irt_likelihood(params):
            n_examinees = len(self.facet_levels['Examinee'])
            n_raters = len(self.facet_levels['Rater'])
            n_criteria = len(self.facet_levels['Criterion'])
            
            examinee_abilities = params[:n_examinees]
            rater_severities = params[n_examinees:n_examinees+n_raters]
            criteria_difficulties = params[n_examinees+n_raters:]
            
            examinee_idx = irt_data['Examinee'].cat.codes
            rater_idx = irt_data['Rater'].cat.codes
            criteria_idx = irt_data['Criterion'].cat.codes
            
            logits = (examinee_abilities[examinee_idx] - 
                      rater_severities[rater_idx] - 
                      criteria_difficulties[criteria_idx])
            
            prob = expit(logits)
            likelihood = (irt_data['Score'] * np.log(prob) + 
                          (self.max_score - irt_data['Score']) * np.log(1 - prob))
            return -np.sum(likelihood)

        initial_params = np.zeros(len(self.facet_levels['Examinee']) + 
                                  len(self.facet_levels['Rater']) + 
                                  len(self.facet_levels['Criterion']))
        result = minimize(irt_likelihood, initial_params, method='L-BFGS-B')

        n_examinees = len(self.facet_levels['Examinee'])
        n_raters = len(self.facet_levels['Rater'])
        
        examinee_abilities = result.x[:n_examinees]
        rater_severities = result.x[n_examinees:n_examinees+n_raters]
        criteria_difficulties = result.x[n_examinees+n_raters:]

        # Create and print IRT results table
        irt_results = pd.DataFrame({
            'Facet': ['Examinee']*n_examinees + ['Rater']*n_raters + ['Criterion']*len(self.facet_levels['Criterion']),
            'Element': (list(self.facet_levels['Examinee']) + 
                        list(self.facet_levels['Rater']) + 
                        list(self.facet_levels['Criterion'])),
            'Measure': np.concatenate([examinee_abilities, rater_severities, criteria_difficulties])
        })
        
        print("\nIRT Measures:")
        print(irt_results.to_string(index=False))

        # Plot Item Characteristic Curves
        plt.figure(figsize=(10, 6))
        theta = np.linspace(-4, 4, 100)
        
        for criterion, difficulty in zip(self.facet_levels['Criterion'], criteria_difficulties):
            prob = expit(theta - difficulty)
            plt.plot(theta, prob, label=criterion)

        plt.title("Item Characteristic Curves")
        plt.xlabel("Theta (Ability)")
        plt.ylabel("Probability")
        plt.legend(title="Criteria", loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('irt_icc.png')
        plt.close()
        
    def run_analysis(self):
        self.estimate_params()
        self.calculate_fit_statistics()
        self.plot_wright_map()
        self.plot_category_probability_curves()
        self.plot_category_information_functions()
        self.plot_facet_vertical_rulers()
        self.plot_bias_interaction()
        
        print("Advanced MFRM Analysis Results")
        print("==============================")
        
        self.print_measurement_report('Rater', "1. Rater Measurement Report (arranged by MN)")
        self.print_measurement_report('Examinee', "2. Examinee Measurement Report (arranged by MN)")
        self.print_measurement_report('Criterion', "3. Criteria Measurement Report (arranged by MN)")
        self.print_category_statistics()
        self.print_unexpected_responses()
        self.print_bias_interaction()
        self.run_irt_analysis()


if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    df_long = df.melt(id_vars=['Examinee', 'Rater'], 
                      value_vars=['Content', 'Organization', 'Language use'], 
                      var_name='Criterion', value_name='Score')
    df_long['Max_Score'] = 6 

    for col in ['Examinee', 'Rater', 'Criterion']:
        df_long[col] = df_long[col].astype('category')

    mfrm = MFRM(df_long, facets=['Examinee', 'Rater', 'Criterion'])
    mfrm.run_analysis()

    rater_scores = df.pivot(index='Examinee', columns='Rater', values='Overall score')

    kappas = []
    for i in range(len(rater_scores.columns)):
        for j in range(i+1, len(rater_scores.columns)):
            rater1 = rater_scores.iloc[:, i].dropna()
            rater2 = rater_scores.iloc[:, j].dropna()
            common_index = rater1.index.intersection(rater2.index)
            if len(common_index) > 1:
                kappa = cohen_kappa_score(rater1[common_index], rater2[common_index])
                kappas.append(kappa)
    mean_cohen_kappa = np.mean(kappas) if kappas else np.nan

    print("\nAdditional Analyses")
    print("===================")
    print("\nInter-rater Reliability:")
    print(f"Mean Cohen's Kappa: {mean_cohen_kappa:.3f}")

    criteria_corr = df[['Content', 'Organization', 'Language use', 'Overall score']].corr()
    print("\nCorrelations between Criteria:")
    print(criteria_corr)

    scale_usage = df_long.groupby(['Rater', 'Score'])['Score'].count().unstack(fill_value=0)
    scale_usage_pct = scale_usage.div(scale_usage.sum(axis=1), axis=0)
    print("\nGrading Scale Usage Tendencies:")
    print(scale_usage_pct)

    plt.figure(figsize=(12, 8))
    scale_usage_pct.T.plot(kind='bar', stacked=True)
    plt.title('Grading Scale Usage Tendencies')
    plt.xlabel('Score')
    plt.ylabel('Percentage')
    plt.legend(title='Rater', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('scale_usage.png')
    plt.close()

    print("\nAnalysis complete. Visualizations saved as PNG files.")