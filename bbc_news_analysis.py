import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the dataset
df = pd.read_csv('bbc_news.csv')

# 1. Dataset Exploration
print("\n=== Dataset Info ===")
print(df.info())
print("\n=== First 5 Rows ===")
print(df.head())

# Extract categories from the 'link' column
df['category'] = df['link'].str.extract(r'https://www.bbc.co.uk/news/([^/]+)')

# 2. Enhanced Class Distribution Visualization
plt.figure(figsize=(14, 8))

# Get value counts and sort
category_counts = df['category'].value_counts().sort_values(ascending=True)

# Create horizontal bar plot
ax = sns.barplot(x=category_counts.values, y=category_counts.index, palette='viridis')

# Add count labels on the bars
for i, v in enumerate(category_counts.values):
    ax.text(v + 5, i, str(v), color='black', fontweight='bold')

# Calculate and add percentage labels
total = len(df)
for i, (category, count) in enumerate(category_counts.items()):
    percentage = f'({(count/total)*100:.1f}%)'
    ax.text(count + 5, i, percentage, va='center', color='#555555')

plt.title('Distribution of News Categories', fontsize=16, pad=20)
plt.xlabel('Number of Articles', fontsize=12, labelpad=10)
plt.ylabel('Category', fontsize=12, labelpad=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add grid lines for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Remove spines for cleaner look
for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed metrics
print("\n=== Category Distribution Metrics ===")
print(f"Total number of articles: {total}")
print("\nNumber of articles per category:")
print(category_counts)
print("\nPercentage distribution:")
print(round((category_counts / total) * 100, 2))

# 3. Text Preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back to string
    return ' '.join(tokens)

# Apply preprocessing to the description column
df['processed_text'] = df['description'].apply(preprocess_text)

# 4. Feature Engineering
# Combine title and description for better feature representation
df['combined_text'] = df['title'] + ' ' + df['description']
df['processed_combined'] = df['combined_text'].apply(preprocess_text)

# 5. Vectorization
# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df['processed_combined'])

# Count Vectorizer
count_vectorizer = CountVectorizer(max_features=5000)
X_count = count_vectorizer.fit_transform(df['processed_combined'])

# 6. Train-Test Split (80-20)
X = X_tfidf  # Using TF-IDF features
le = LabelEncoder()
y = le.fit_transform(df['category'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Dimensionality Reduction
# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_train.toarray())

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
X_tsne = tsne.fit_transform(X_train.toarray())

# Enhanced Dimensionality Reduction Plots
plt.figure(figsize=(20, 8))

# Create a custom color palette
palette = sns.color_palette('husl', n_colors=len(le.classes_))

# PCA Plot
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=y_train, 
                     cmap='husl', 
                     alpha=0.7,
                     s=60,
                     edgecolor='w',
                     linewidth=0.5)

# Add explained variance ratio
explained_var = pca.explained_variance_ratio_
plt.title(f'PCA - News Categories\nExplained Variance: {explained_var[0]:.2f}, {explained_var[1]:.2f}', 
          fontsize=14, pad=15)
plt.xlabel('Principal Component 1', fontsize=12, labelpad=10)
plt.ylabel('Principal Component 2', fontsize=12, labelpad=10)

# Add grid
plt.grid(True, linestyle='--', alpha=0.6)

# t-SNE Plot
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                     c=y_train, 
                     cmap='husl',
                     alpha=0.7,
                     s=60,
                     edgecolor='w',
                     linewidth=0.5)

plt.title('t-SNE - News Categories', fontsize=14, pad=15)
plt.xlabel('t-SNE 1', fontsize=12, labelpad=10)
plt.ylabel('t-SNE 2', fontsize=12, labelpad=10)

# Add grid
plt.grid(True, linestyle='--', alpha=0.6)

# Create a single legend for both plots
handles, labels = scatter.legend_elements()
plt.legend(handles, le.classes_, 
           title='Categories',
           bbox_to_anchor=(1.05, 1),
           loc='upper left',
           borderaxespad=0.,
           frameon=True,
           framealpha=0.9,
           edgecolor='#333333')

plt.tight_layout()
plt.savefig('dimensionality_reduction.png', dpi=300, bbox_inches='tight')
plt.show()

# Print explained variance for PCA
print("\n=== PCA Explained Variance ===")
print(f"Explained variance ratio (first 2 components): {explained_var[0]:.4f}, {explained_var[1]:.4f}")
print(f"Total explained variance: {sum(explained_var[:2]):.4f}")

# 8. Model Building
# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Model Evaluation
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n=== {model_name} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, 
                yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.show()

# Evaluate models
evaluate_model(y_test, y_pred_nb, "Naive Bayes")
evaluate_model(y_test, y_pred_lr, "Logistic Regression")

# 9. Clustering and Topic-wise t-SNE
# Let's use KMeans for clustering
from sklearn.cluster import KMeans

# Number of clusters = number of unique categories
n_clusters = len(le.classes_)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_tfidf)

# Add cluster labels to the dataframe
df['cluster'] = cluster_labels

# Enhanced t-SNE Clustering Plot
plt.figure(figsize=(14, 10))

# Create scatter plot with better styling
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                     c=cluster_labels, 
                     cmap='tab20',
                     alpha=0.8,
                     s=80,
                     edgecolor='w',
                     linewidth=0.7)

plt.title('t-SNE Clustering of News Articles', fontsize=16, pad=20)
plt.xlabel('t-SNE 1', fontsize=12, labelpad=10)
plt.ylabel('t-SNE 2', fontsize=12, labelpad=10)

# Add grid
plt.grid(True, linestyle='--', alpha=0.3)

# Add cluster centers if needed
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.8, marker='X')

# Add colorbar with better styling
cbar = plt.colorbar(scatter, pad=0.02, aspect=40)
cbar.set_label('Cluster', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)

# Add silhouette score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X_tfidf, cluster_labels)
plt.text(0.02, 0.98, f'Silhouette Score: {silhouette_avg:.3f}',
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
         verticalalignment='top',
         fontsize=10)

plt.tight_layout()
plt.savefig('tsne_clustering.png', dpi=300, bbox_inches='tight')
plt.show()

# Print clustering metrics
print("\n=== Clustering Metrics ===")
print(f"Number of clusters: {n_clusters}")
print(f"Silhouette Score: {silhouette_avg:.4f}")
print("\nCluster sizes:")
print(df['cluster'].value_counts().sort_index())

# Print top terms per cluster
print("\n=== Top Terms per Cluster ===")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names_out()

for i in range(n_clusters):
    print(f"\nCluster {i}:")
    print(f"Most common category: {df[df['cluster'] == i]['category'].value_counts().idxmax()}")
    print("Top terms:", ", ".join([terms[ind] for ind in order_centroids[i, :10]]))
