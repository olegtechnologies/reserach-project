"""
Core Banking Application Upgrade Management System
Using Machine Learning to Predict Upgrade Success
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


class CoreBankingDataGenerator:
    """
    Generate realistic synthetic core banking application data for ML model training
    
    This generator creates data based on:
    - Industry best practices and patterns
    - Nigerian banking sector characteristics
    - Realistic correlations between features
    - Domain expert knowledge encoded as rules
    """
    
    def __init__(self, n_samples=5000, seed=42):
        self.n_samples = n_samples
        np.random.seed(seed)
        
        # Nigerian bank tier distribution (based on CBN classification)
        self.bank_tiers = {
            'tier1': 0.15,  # Large banks (GTBank, Access, UBA, etc.)
            'tier2': 0.35,  # Medium banks
            'tier3': 0.35,  # Small banks
            'microfinance': 0.15  # Microfinance institutions
        }
        
    def generate_dataset(self):
        """
        Generate realistic synthetic banking upgrade data with Nigerian context
        """
        # Generate bank tiers first (needed for correlations)
        bank_tier = np.random.choice(
            list(self.bank_tiers.keys()), 
            self.n_samples, 
            p=list(self.bank_tiers.values())
        )
        
        data = {
            # Bank Classification
            'bank_tier': bank_tier,
            
            # System Characteristics (correlated with bank tier)
            'current_version': np.random.choice(['v8.2', 'v8.5', 'v9.0', 'v9.2', 'v10.1'], self.n_samples),
            'target_version': np.random.choice(['v9.0', 'v9.2', 'v10.1', 'v10.5', 'v11.0'], self.n_samples),
            'version_jump_size': np.random.choice(['minor', 'major_1x', 'major_2x'], self.n_samples, p=[0.4, 0.4, 0.2]),
            'system_age_months': self._generate_correlated_int(bank_tier, 6, 120, tier_influence=-0.3),
            'last_upgrade_months_ago': self._generate_correlated_int(bank_tier, 1, 48, tier_influence=-0.2),
            'code_customization_percent': self._generate_correlated_float(bank_tier, 10, 80, tier_influence=-0.3),
            'third_party_integrations': self._generate_correlated_int(bank_tier, 3, 50, tier_influence=0.5),
            
            # Performance Metrics (larger banks handle more volume)
            'avg_transaction_volume': self._generate_correlated_int(bank_tier, 10000, 500000, tier_influence=0.6),
            'peak_transaction_volume': self._generate_correlated_int(bank_tier, 50000, 1000000, tier_influence=0.6),
            'system_uptime_percent': self._generate_correlated_float(bank_tier, 95.0, 99.99, tier_influence=0.4),
            'avg_response_time_ms': self._generate_correlated_int(bank_tier, 50, 2000, tier_influence=-0.5),
            'incident_count_last_6months': self._generate_correlated_int(bank_tier, 0, 15, tier_influence=-0.4),
            'unplanned_downtime_hours_last_year': self._generate_correlated_int(bank_tier, 0, 50, tier_influence=-0.5),
            
            # Infrastructure Model (Nigerian Context - tier 1 banks more likely cloud)
            'infrastructure_model': self._generate_infrastructure_model(bank_tier),
            'server_count': self._generate_correlated_int(bank_tier, 2, 50, tier_influence=0.7),
            'database_size_gb': self._generate_correlated_int(bank_tier, 100, 50000, tier_influence=0.8),
            'concurrent_users': self._generate_correlated_int(bank_tier, 50, 10000, tier_influence=0.7),
            'network_bandwidth_mbps': self._generate_correlated_int(bank_tier, 100, 10000, tier_influence=0.6),
            'power_stability_score': self._generate_power_stability(bank_tier),  # Nigerian infrastructure challenge
            
            # Deployment Strategy (tier 1 banks use more advanced strategies)
            'deployment_strategy': self._generate_deployment_strategy(bank_tier),
            'deployment_automation_level': self._generate_correlated_float(bank_tier, 0, 100, tier_influence=0.5),
            
            # Project Management (tier 1 banks more agile)
            'pm_methodology': self._generate_pm_methodology(bank_tier),
            'critical_incidents_last_year': self._generate_correlated_int(bank_tier, 0, 8, tier_influence=-0.4),
            
            # Testing & Preparation (tier 1 banks better prepared)
            'test_environment_available': self._generate_binary_correlated(bank_tier, base_prob=0.8, tier_influence=0.15),
            'backup_verified': self._generate_binary_correlated(bank_tier, base_prob=0.9, tier_influence=0.08),
            'rollback_plan_exists': self._generate_binary_correlated(bank_tier, base_prob=0.85, tier_influence=0.12),
            'staff_training_completed': self._generate_binary_correlated(bank_tier, base_prob=0.7, tier_influence=0.2),
            'early_stage_testing_completed': self._generate_binary_correlated(bank_tier, base_prob=0.75, tier_influence=0.18),
            
            # Nigerian Regulatory Compliance (CBN) - tier 1 banks more compliant
            'cbn_compliance_verified': self._generate_binary_correlated(bank_tier, base_prob=0.85, tier_influence=0.12),
            'data_localization_compliant': self._generate_binary_correlated(bank_tier, base_prob=0.8, tier_influence=0.15),
            'cyber_security_framework_updated': self._generate_binary_correlated(bank_tier, base_prob=0.75, tier_influence=0.18),
            'bcp_dr_plan_tested': self._generate_binary_correlated(bank_tier, base_prob=0.7, tier_influence=0.2),  # Business Continuity/Disaster Recovery
            
            # Customer Impact Metrics (correlated with bank performance)
            'customer_satisfaction_score': self._generate_correlated_float(bank_tier, 60, 95, tier_influence=0.3),
            'customer_complaints_last_quarter': self._generate_correlated_poisson(bank_tier, base_lambda=15, tier_influence=-0.4),
            'digital_banking_adoption_percent': self._generate_correlated_float(bank_tier, 40, 90, tier_influence=0.4),
            'service_disruption_tolerance_hours': self._generate_correlated_int(bank_tier, 1, 12, tier_influence=-0.3),
            
            # Business Factors
            'upgrade_window_hours': self._generate_correlated_int(bank_tier, 2, 72, tier_influence=0.3),
            'business_critical_period': np.random.choice([0, 1], self.n_samples, p=[0.7, 0.3]),
            'regulatory_compliance_required': np.random.choice([0, 1], self.n_samples, p=[0.6, 0.4]),
            
            # Resource Availability (tier 1 banks have more resources)
            'dedicated_team_size': self._generate_correlated_int(bank_tier, 2, 20, tier_influence=0.6),
            'vendor_support_available': self._generate_binary_correlated(bank_tier, base_prob=0.7, tier_influence=0.2),
            'budget_allocated_usd': self._generate_correlated_int(bank_tier, 10000, 5000000, tier_influence=0.8),
            'external_consultant_engaged': self._generate_binary_correlated(bank_tier, base_prob=0.6, tier_influence=0.25),
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable based on realistic business logic
        df['upgrade_success'] = self._calculate_success_probability(df)
        
        # Create additional derived features
        df['upgrade_risk_score'] = self._calculate_risk_score(df)
        df['system_health_score'] = self._calculate_health_score(df)
        
        return df
    
    def _calculate_success_probability(self, df):
        """Calculate upgrade success based on multiple factors including Nigerian context"""
        
        success_prob = np.ones(len(df)) * 0.5
        
        # Positive factors - Technical Preparation
        success_prob += df['test_environment_available'] * 0.12
        success_prob += df['backup_verified'] * 0.10
        success_prob += df['rollback_plan_exists'] * 0.09
        success_prob += df['staff_training_completed'] * 0.07
        success_prob += df['vendor_support_available'] * 0.06
        success_prob += df['early_stage_testing_completed'] * 0.08
        success_prob += (df['system_uptime_percent'] > 99.0) * 0.04
        
        # Positive factors - Nigerian Regulatory Compliance
        success_prob += df['cbn_compliance_verified'] * 0.09
        success_prob += df['data_localization_compliant'] * 0.06
        success_prob += df['cyber_security_framework_updated'] * 0.05
        success_prob += df['bcp_dr_plan_tested'] * 0.07
        
        # Positive factors - Deployment Strategy
        success_prob += (df['deployment_strategy'] == 'canary') * 0.08
        success_prob += (df['deployment_strategy'] == 'blue_green') * 0.08
        success_prob += (df['deployment_automation_level'] > 70) * 0.06
        
        # Positive factors - Infrastructure
        success_prob += (df['infrastructure_model'] == 'hybrid_cloud') * 0.07
        success_prob += (df['power_stability_score'] > 85) * 0.05
        
        # Positive factors - Project Management
        success_prob += (df['pm_methodology'].isin(['agile', 'scrum', 'kanban'])) * 0.05
        
        # Negative factors - Technical Risks
        success_prob -= (df['incident_count_last_6months'] > 5) * 0.12
        success_prob -= (df['code_customization_percent'] > 50) * 0.09
        success_prob -= (df['business_critical_period'] == 1) * 0.07
        success_prob -= (df['version_jump_size'] == 'major_2x') * 0.11
        success_prob -= (df['upgrade_window_hours'] < 8) * 0.09
        success_prob -= (df['unplanned_downtime_hours_last_year'] > 20) * 0.07
        
        # Negative factors - Customer Impact
        success_prob -= (df['customer_satisfaction_score'] < 70) * 0.06
        success_prob -= (df['customer_complaints_last_quarter'] > 25) * 0.05
        success_prob -= (df['service_disruption_tolerance_hours'] < 4) * 0.06
        
        # Negative factors - Infrastructure Challenges
        success_prob -= (df['power_stability_score'] < 75) * 0.08
        success_prob -= (df['network_bandwidth_mbps'] < 500) * 0.04
        
        # Add some randomness
        success_prob += np.random.normal(0, 0.05, len(df))
        
        # Clip between 0 and 1
        success_prob = np.clip(success_prob, 0, 1)
        
        # Convert to binary outcome
        return (success_prob > 0.5).astype(int)
    
    def _calculate_risk_score(self, df):
        """Calculate overall risk score for the upgrade including Nigerian context"""
        
        risk = np.zeros(len(df))
        
        # Technical Risk
        risk += df['code_customization_percent'] / 10
        risk += df['third_party_integrations'] / 5
        risk += df['incident_count_last_6months'] * 5
        risk += (df['version_jump_size'] == 'major_2x') * 20
        risk += (1 - df['test_environment_available']) * 15
        risk += (1 - df['backup_verified']) * 25
        risk += (1 - df['early_stage_testing_completed']) * 12
        
        # Regulatory & Compliance Risk
        risk += (1 - df['cbn_compliance_verified']) * 18
        risk += (1 - df['data_localization_compliant']) * 12
        risk += (1 - df['cyber_security_framework_updated']) * 10
        
        # Infrastructure Risk
        risk += (100 - df['power_stability_score']) / 5
        risk += (df['infrastructure_model'] == 'on_premise') * 8
        
        # Deployment Risk
        risk += (df['deployment_strategy'] == 'big_bang') * 15
        risk += (100 - df['deployment_automation_level']) / 10
        
        # Customer Risk
        risk += (100 - df['customer_satisfaction_score']) / 5
        risk += df['customer_complaints_last_quarter'] / 2
        
        return np.clip(risk, 0, 100)
    
    def _calculate_health_score(self, df):
        """Calculate current system health score"""
        
        health = np.ones(len(df)) * 50
        
        health += df['system_uptime_percent'] / 2
        health -= df['avg_response_time_ms'] / 50
        health -= df['incident_count_last_6months'] * 3
        health -= df['unplanned_downtime_hours_last_year'] / 2
        
        return np.clip(health, 0, 100)
    
    def _generate_correlated_int(self, bank_tier, min_val, max_val, tier_influence=0.5):
        """Generate integer values correlated with bank tier
        Note: Microfinance includes digital fintechs (high resources) and traditional MFBs (low resources)"""
        tier_scores = {'tier1': 1.0, 'tier2': 0.7, 'tier3': 0.4, 'microfinance': 0.2}
        
        values = np.zeros(len(bank_tier))
        for i, tier in enumerate(bank_tier):
            if tier == 'microfinance':
                # 50% digital fintechs (score 0.7), 50% traditional MFBs (score 0.2)
                tier_score = np.random.choice([0.7, 0.2], p=[0.50, 0.50])
            else:
                tier_score = tier_scores[tier]
            # Adjust range based on tier
            if tier_influence > 0:
                # Positive correlation: higher tier = higher value
                adjusted_min = min_val + (max_val - min_val) * tier_score * tier_influence * 0.3
                adjusted_max = min_val + (max_val - min_val) * (tier_score * tier_influence + (1 - tier_influence))
            else:
                # Negative correlation: higher tier = lower value
                tier_influence = abs(tier_influence)
                adjusted_min = min_val + (max_val - min_val) * ((1 - tier_score) * tier_influence)
                adjusted_max = max_val - (max_val - min_val) * tier_score * tier_influence * 0.3
            
            values[i] = np.random.randint(int(adjusted_min), int(adjusted_max) + 1)
        
        return values.astype(int)
    
    def _generate_correlated_float(self, bank_tier, min_val, max_val, tier_influence=0.5):
        """Generate float values correlated with bank tier
        Note: Microfinance includes digital fintechs (high performance) and traditional MFBs (low performance)"""
        tier_scores = {'tier1': 1.0, 'tier2': 0.7, 'tier3': 0.4, 'microfinance': 0.2}
        
        values = np.zeros(len(bank_tier))
        for i, tier in enumerate(bank_tier):
            if tier == 'microfinance':
                # 50% digital fintechs (score 0.7), 50% traditional MFBs (score 0.2)
                tier_score = np.random.choice([0.7, 0.2], p=[0.50, 0.50])
            else:
                tier_score = tier_scores[tier]
            # Adjust range based on tier
            if tier_influence > 0:
                adjusted_min = min_val + (max_val - min_val) * tier_score * tier_influence * 0.3
                adjusted_max = min_val + (max_val - min_val) * (tier_score * tier_influence + (1 - tier_influence))
            else:
                tier_influence = abs(tier_influence)
                adjusted_min = min_val + (max_val - min_val) * ((1 - tier_score) * tier_influence)
                adjusted_max = max_val - (max_val - min_val) * tier_score * tier_influence * 0.3
            
            values[i] = np.random.uniform(adjusted_min, adjusted_max)
        
        return values
    
    def _generate_correlated_poisson(self, bank_tier, base_lambda=15, tier_influence=0.5):
        """Generate Poisson-distributed values correlated with bank tier"""
        tier_scores = {'tier1': 1.0, 'tier2': 0.7, 'tier3': 0.4, 'microfinance': 0.2}
        
        values = np.zeros(len(bank_tier))
        for i, tier in enumerate(bank_tier):
            tier_score = tier_scores[tier]
            # Adjust lambda based on tier (negative influence = better tier has lower lambda)
            if tier_influence < 0:
                adjusted_lambda = base_lambda * (1 - tier_score * abs(tier_influence))
            else:
                adjusted_lambda = base_lambda * (tier_score * tier_influence + (1 - tier_influence))
            
            values[i] = np.random.poisson(max(1, adjusted_lambda))
        
        return values.astype(int)
    
    def _generate_binary_correlated(self, bank_tier, base_prob=0.5, tier_influence=0.2):
        """Generate binary values (0/1) correlated with bank tier
        Note: Microfinance includes digital fintechs (high tech) and traditional MFBs (low tech)"""
        tier_scores = {'tier1': 1.0, 'tier2': 0.7, 'tier3': 0.4, 'microfinance': 0.2}
        
        values = np.zeros(len(bank_tier))
        for i, tier in enumerate(bank_tier):
            if tier == 'microfinance':
                # 50% digital fintechs (score 0.7), 50% traditional MFBs (score 0.2)
                tier_score = np.random.choice([0.7, 0.2], p=[0.50, 0.50])
            else:
                tier_score = tier_scores[tier]
            # Adjust probability based on tier
            adjusted_prob = min(0.98, base_prob + (tier_score - 0.5) * tier_influence)
            values[i] = 1 if np.random.random() < adjusted_prob else 0
        
        return values.astype(int)
    
    def _generate_infrastructure_model(self, bank_tier):
        """Generate infrastructure model based on bank tier (larger banks more likely cloud)"""
        models = []
        for tier in bank_tier:
            if tier == 'tier1':
                # Tier 1 banks: more likely hybrid/public cloud
                model = np.random.choice(['on_premise', 'hybrid_cloud', 'private_cloud', 'public_cloud'],
                                        p=[0.15, 0.50, 0.25, 0.10])
            elif tier == 'tier2':
                # Tier 2 banks: balanced
                model = np.random.choice(['on_premise', 'hybrid_cloud', 'private_cloud', 'public_cloud'],
                                        p=[0.30, 0.40, 0.25, 0.05])
            elif tier == 'tier3':
                # Tier 3 banks: more on-premise
                model = np.random.choice(['on_premise', 'hybrid_cloud', 'private_cloud', 'public_cloud'],
                                        p=[0.50, 0.30, 0.15, 0.05])
            else:  # microfinance
                # Microfinance: Mixed (Digital fintechs cloud-native, traditional MFBs on-premise)
                # OPay, Kuda, PalmPay, FairMoney = public/hybrid cloud; Traditional MFBs = on-premise
                model = np.random.choice(['on_premise', 'hybrid_cloud', 'private_cloud', 'public_cloud'],
                                        p=[0.40, 0.30, 0.10, 0.20])
            models.append(model)
        
        return np.array(models)
    
    def _generate_deployment_strategy(self, bank_tier):
        """Generate deployment strategy based on bank tier (larger banks use advanced strategies)"""
        strategies = []
        for tier in bank_tier:
            if tier == 'tier1':
                # Tier 1 banks: prefer canary/blue-green
                strategy = np.random.choice(['big_bang', 'canary', 'blue_green', 'rolling'],
                                          p=[0.10, 0.35, 0.40, 0.15])
            elif tier == 'tier2':
                # Tier 2 banks: balanced
                strategy = np.random.choice(['big_bang', 'canary', 'blue_green', 'rolling'],
                                          p=[0.20, 0.30, 0.30, 0.20])
            elif tier == 'tier3':
                # Tier 3 banks: more big bang
                strategy = np.random.choice(['big_bang', 'canary', 'blue_green', 'rolling'],
                                          p=[0.35, 0.25, 0.25, 0.15])
            else:  # microfinance
                # Microfinance: Mixed (Fintechs use advanced CI/CD, traditional MFBs use big bang)
                strategy = np.random.choice(['big_bang', 'canary', 'blue_green', 'rolling'],
                                          p=[0.35, 0.25, 0.25, 0.15])
            strategies.append(strategy)
        
        return np.array(strategies)
    
    def _generate_pm_methodology(self, bank_tier):
        """Generate PM methodology based on bank tier (larger banks more agile)"""
        methodologies = []
        for tier in bank_tier:
            if tier == 'tier1':
                # Tier 1 banks: mostly agile/scrum/kanban
                methodology = np.random.choice(['waterfall', 'agile', 'scrum', 'kanban'],
                                             p=[0.10, 0.35, 0.35, 0.20])
            elif tier == 'tier2':
                # Tier 2 banks: balanced
                methodology = np.random.choice(['waterfall', 'agile', 'scrum', 'kanban'],
                                             p=[0.20, 0.30, 0.30, 0.20])
            elif tier == 'tier3':
                # Tier 3 banks: more waterfall
                methodology = np.random.choice(['waterfall', 'agile', 'scrum', 'kanban'],
                                             p=[0.35, 0.25, 0.25, 0.15])
            else:  # microfinance
                # Microfinance: mostly waterfall
                methodology = np.random.choice(['waterfall', 'agile', 'scrum', 'kanban'],
                                             p=[0.50, 0.20, 0.20, 0.10])
            methodologies.append(methodology)
        
        return np.array(methodologies)
    
    def _generate_power_stability(self, bank_tier):
        """Generate power stability score (Nigerian context - location-based variation)"""
        tier_scores = {'tier1': 1.0, 'tier2': 0.7, 'tier3': 0.4, 'microfinance': 0.2}
        
        values = np.zeros(len(bank_tier))
        for i, tier in enumerate(bank_tier):
            tier_score = tier_scores[tier]
            # Tier 1 banks in better locations (Lagos, Abuja) with generators/UPS
            # Base score varies, tier 1 can afford better backup power
            base_score = np.random.uniform(60, 75)  # Nigerian baseline
            tier_bonus = tier_score * 25  # Up to +25 for tier 1
            values[i] = min(100, base_score + tier_bonus + np.random.normal(0, 5))
        
        return values


class CoreBankingUpgradePredictor:
    """Machine Learning model to predict upgrade success and provide recommendations"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.label_encoders = {}
        
    def prepare_features(self, df, is_training=True):
        """Prepare features for machine learning"""
        
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['current_version', 'target_version', 'version_jump_size', 
                          'infrastructure_model', 'deployment_strategy', 'pm_methodology']
        
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col + '_encoded'] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                df_processed[col + '_encoded'] = self.label_encoders[col].transform(df_processed[col])
        
        # Select features for modeling
        if is_training:
            self.feature_cols = [col for col in df_processed.columns 
                                if col not in ['upgrade_success', 'current_version', 
                                              'target_version', 'version_jump_size',
                                              'infrastructure_model', 'deployment_strategy', 
                                              'pm_methodology', 'bank_tier']]
        
        X = df_processed[self.feature_cols]
        
        # Scale numerical features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return pd.DataFrame(X_scaled, columns=self.feature_cols, index=X.index)
    
    def train(self, X_train, y_train):
        """Train the Random Forest model with hyperparameter tuning"""
        
        print("Training Random Forest Classifier...")
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(rf, param_grid, cv=5, 
                                   scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        
        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Best Cross-Validation ROC-AUC Score: {grid_search.best_score_:.4f}")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Failure', 'Success']))
        
        print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred):.4f}")
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Failure', 'Success'],
                   yticklabels=['Failure', 'Success'])
        plt.title('Confusion Matrix - Upgrade Success Prediction')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        print("\nConfusion matrix saved as 'confusion_matrix.png'")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Upgrade Success Prediction')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300)
        print("ROC curve saved as 'roc_curve.png'")
        
        return y_pred, y_pred_proba
    
    def feature_importance_analysis(self, feature_names):
        """Analyze and visualize feature importance"""
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        print("\nTop 15 Most Important Features:")
        for i in range(min(15, len(indices))):
            print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_n = 20
        top_indices = indices[:top_n]
        
        plt.barh(range(top_n), importances[top_indices], color='steelblue')
        plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importance - Core Banking Upgrade Prediction')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        print("\nFeature importance plot saved as 'feature_importance.png'")
    
    def predict_upgrade_success(self, upgrade_data):
        """Predict success probability for a new upgrade scenario"""
        
        X = self.prepare_features(upgrade_data, is_training=False)
        
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        result = {
            'success_prediction': 'SUCCESS' if prediction == 1 else 'FAILURE',
            'success_probability': probability[1],
            'failure_probability': probability[0],
            'confidence': max(probability),
            'recommendation': self._generate_recommendation(probability[1], upgrade_data)
        }
        
        return result
    
    def _generate_recommendation(self, success_prob, upgrade_data):
        """Generate actionable recommendations based on prediction"""
        
        if success_prob >= 0.80:
            return "PROCEED: High probability of success. Ensure all preparation steps are complete."
        elif success_prob >= 0.60:
            return "PROCEED WITH CAUTION: Moderate success probability. Review risk mitigation strategies."
        elif success_prob >= 0.40:
            return "DELAY: Consider improving preparation - more testing, training, or resources."
        else:
            return "DO NOT PROCEED: High risk of failure. Significant improvements needed before upgrade."


def main():
    """Main execution function"""
    
    print("="*80)
    print("CORE BANKING APPLICATION UPGRADE MANAGEMENT SYSTEM")
    print("Machine Learning Model for Predicting Upgrade Success")
    print("="*80)
    
    # Step 1: Generate synthetic data
    print("\n[Step 1] Generating synthetic banking upgrade dataset...")
    generator = CoreBankingDataGenerator(n_samples=5000)
    df = generator.generate_dataset()
    
    print(f"Dataset generated: {len(df)} records")
    print(f"Success rate: {df['upgrade_success'].mean()*100:.2f}%")
    print(f"\nDataset shape: {df.shape}")
    print("\nFirst few records:")
    print(df.head())
    
    # Save dataset
    df.to_csv('banking_upgrade_dataset.csv', index=False)
    print("\nDataset saved to 'banking_upgrade_dataset.csv'")
    
    # Step 2: Prepare data for modeling
    print("\n[Step 2] Preparing data for machine learning...")
    predictor = CoreBankingUpgradePredictor()
    
    X = predictor.prepare_features(df, is_training=True)
    y = df['upgrade_success']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Step 3: Train model
    print("\n[Step 3] Training machine learning model...")
    predictor.train(X_train, y_train)
    
    # Step 4: Evaluate model
    print("\n[Step 4] Evaluating model performance...")
    y_pred, y_pred_proba = predictor.evaluate(X_test, y_test)
    
    # Step 5: Feature importance analysis
    print("\n[Step 5] Analyzing feature importance...")
    predictor.feature_importance_analysis(X.columns.tolist())
    
    # Step 6: Example prediction
    print("\n" + "="*80)
    print("EXAMPLE: PREDICTING A NEW UPGRADE SCENARIO")
    print("="*80)
    
    # Create example upgrade scenario
    example_scenario = pd.DataFrame([{
        'bank_tier': 'tier2',
        'current_version': 'v9.0',
        'target_version': 'v10.1',
        'system_age_months': 24,
        'last_upgrade_months_ago': 12,
        'avg_transaction_volume': 250000,
        'peak_transaction_volume': 500000,
        'system_uptime_percent': 99.5,
        'avg_response_time_ms': 200,
        'infrastructure_model': 'hybrid_cloud',
        'server_count': 10,
        'database_size_gb': 5000,
        'concurrent_users': 2000,
        'network_bandwidth_mbps': 1000,
        'power_stability_score': 85.0,
        'incident_count_last_6months': 2,
        'critical_incidents_last_year': 0,
        'unplanned_downtime_hours_last_year': 5,
        'code_customization_percent': 30.0,
        'third_party_integrations': 15,
        'version_jump_size': 'major_1x',
        'deployment_strategy': 'blue_green',
        'deployment_automation_level': 75.0,
        'pm_methodology': 'agile',
        'test_environment_available': 1,
        'backup_verified': 1,
        'rollback_plan_exists': 1,
        'staff_training_completed': 1,
        'early_stage_testing_completed': 1,
        'cbn_compliance_verified': 1,
        'data_localization_compliant': 1,
        'cyber_security_framework_updated': 1,
        'bcp_dr_plan_tested': 1,
        'customer_satisfaction_score': 85.0,
        'customer_complaints_last_quarter': 10,
        'digital_banking_adoption_percent': 75.0,
        'service_disruption_tolerance_hours': 6,
        'upgrade_window_hours': 24,
        'business_critical_period': 0,
        'regulatory_compliance_required': 1,
        'dedicated_team_size': 8,
        'vendor_support_available': 1,
        'budget_allocated_usd': 500000,
        'external_consultant_engaged': 1,
        'upgrade_risk_score': 25.0,
        'system_health_score': 85.0
    }])
    
    result = predictor.predict_upgrade_success(example_scenario)
    
    print("\nUpgrade Scenario Details:")
    print(f"  Current Version: {example_scenario['current_version'].values[0]}")
    print(f"  Target Version: {example_scenario['target_version'].values[0]}")
    print(f"  Infrastructure: {example_scenario['infrastructure_model'].values[0].replace('_', ' ').title()}")
    print(f"  Deployment Strategy: {example_scenario['deployment_strategy'].values[0].replace('_', '-').title()}")
    print(f"  System Uptime: {example_scenario['system_uptime_percent'].values[0]:.2f}%")
    print(f"  CBN Compliance: {'Verified' if example_scenario['cbn_compliance_verified'].values[0] else 'Pending'}")
    print(f"  Customer Satisfaction: {example_scenario['customer_satisfaction_score'].values[0]:.1f}%")
    
    print("\n" + "-"*60)
    print("PREDICTION RESULTS:")
    print("-"*60)
    print(f"Prediction: {result['success_prediction']}")
    print(f"Success Probability: {result['success_probability']*100:.2f}%")
    print(f"Failure Probability: {result['failure_probability']*100:.2f}%")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"\nRecommendation: {result['recommendation']}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated Files:")
    print("  1. banking_upgrade_dataset.csv - Training dataset")
    print("  2. confusion_matrix.png - Model confusion matrix")
    print("  3. roc_curve.png - ROC curve analysis")
    print("  4. feature_importance.png - Feature importance chart")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
