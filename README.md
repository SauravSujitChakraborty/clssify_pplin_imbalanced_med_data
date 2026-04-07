# clssify_pplin_imbalanced_med_data

This project was made by me in Sept'25, preserved and published on Apr 6,'26.

THEORY:

1. The Biological Theory: Epistasis
   
==> ​Most basic AI models look for "Linear" relationships (e.g., age increases with risk). However, genetics is often Non-Linear.
​
==> Epistatic Interaction: In this code, the disease only manifests if Gene_A=1 AND Gene_B=1.

​==> If a patient has only one of these, they are a carrier but remain healthy.

==> ​This is a difficult logic for simple models like Logistic Regression to find, but it is exactly what Gradient Boosting is designed to map. 

​ 2. The Statistical Challenge: Class Imbalance

==> ​The data is Stochastically Generated to be highly imbalanced:

==> ​Gene_B Prevalence: 5%

​==> Resulting Disease Prevalence: ~0.5% (Rare Event)

==> ​In medical AI, accuracy is a trap. If 99.5% of people are healthy, a model that simply guesses "Healthy" every time is 99.5% accurate but has no practical use. This pipeline addresses this by using:

==> ​Stratified Splitting: Ensuring that the few cases of disease are distributed evenly between training and testing.

==> ​Probability Calibration: Using predict_proba to give a risk percentage (e.g., 97.49%) rather than just a "Yes/No" label.

​3. The Technical Engine: Use of Histogram

==> ​The model used is a Histogram-based Gradient Boosting Classifier.

==> ​Working: It builds an ensemble of decision trees. Each new tree focuses specifically on the "mistakes" made by the previous trees.

==> Reason for it to be Histogram based: It bins continuous features (like Protein_Level) into discrete integer bins. This reduces the number of decision thresholds the model uses uses to divide them into groups which the computer has to calculate, making it significantly faster and more memory-efficient than standard Gradient Boosting. Decision thresholds are like conditional statements. For every feature in your data, the model tests various values to see which one reduces uncertainty (loss) the most.

4. Evaluation Theory: The Confusion Matrix
   
==> ​Because the data is imbalanced, we ignore accuracy and focus on the Confusion Matrix.

==> ​Question: Can the model find ALL the sick people?

​==> Precision: When the model says someone is sick, how often is it right?

​==> The Goal: In medicine, we usually prefer a "False Positive" (testing a healthy person further) over a "False Negative" (missing a sick person).

Quantitative Application(Probability)

1. Generating the Features (Binomial)

==> When we create the dataset, we use $np.random.choice([0, 1], p=[0.90, 0.10])$. This is a Bernoulli Trial (a single-event Binomial Distribution) and uses pseudorandom values, which is controlled by Stochastic Sampling since it is a stochastic process.

==> It decides whether a patient has a mutation (1) or not (0) based on a fixed probability.

==> This is used to create the "Environment," but it is not the AI's "intelligence."

2. Finding out P (Gradient Boosting)
   
==> The AI finds out the probability P (the Risk Probability) using the Logit Link Function within the Gradient Boosting trees.

==> The model doesn't know we used a Binomial distribution to make the data.

==> It looks at the resulting patterns and calculates P by minimizing a Log-Loss function.

==> It basically asks: "What is the likelihood of the 'Disease' label given that Gene_A and Gene_B are both 1?"

4. The Epistatic Rule (Deterministic)
   
==> The "Ground Truth" P in my code is actually Binary (0 or 1) because I used a hard-coded logic: (A == 1) & (B == 1).
If the rule is met, P=1.
If not, P=0.

==> The Gradient Booster tries to approximate this. So, in my output, Patient_001 has a probability of 97.49%—the model is nearly certain, so the patient follows my hidden rule.

==> When my model runs $predict_proba$, it is performing this calculation under the hood:

 $$ P(y=1) = \frac{1}{1 + e^{-(\text{Raw Score})}} $$
 
1. The Probability Mass Function (PMF)
   
==> The probability of a single patient having a specific mutation is modeled by the following formula:

 $$ P(X = k) = p^k (1-p)^{1-k} $$
 
where:
$X$: The random variable (the Gene status).
$k$: The outcome (1 for mutated, 0 for normal).
$p$: The probability of success (e.g., 0.10 for Gene_A).
$1-p$: The probability of failure (0.90 for Gene_A).
