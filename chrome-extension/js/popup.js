const API_URL_KEY = 'apiUrl';
const THEME_KEY = 'theme';
const DEFAULT_API_URL = 'http://localhost:8000';

let currentFilter = 'all';
let allPredictions = [];

// Initialisation
document.addEventListener('DOMContentLoaded', async () => {
    await loadSettings();
    setupEventListeners();
});

// Charger les paramètres
async function loadSettings() {
    const { apiUrl, theme } = await chrome.storage.sync.get([API_URL_KEY, THEME_KEY]);
    
    document.getElementById('apiUrl').value = apiUrl || DEFAULT_API_URL;
    
    if (theme === 'dark') {
        document.body.setAttribute('data-theme', 'dark');
        document.getElementById('toggleTheme').textContent = '☀️';
    }
}

// Configurer les écouteurs d'événements
function setupEventListeners() {
    document.getElementById('saveApiUrl').addEventListener('click', saveApiUrl);
    document.getElementById('analyzeBtn').addEventListener('click', analyzeComments);
    document.getElementById('toggleTheme').addEventListener('click', toggleTheme);
    document.getElementById('refreshBtn').addEventListener('click', () => location.reload());
    document.getElementById('copyResults').addEventListener('click', copyResults);
    document.getElementById('exportCSV').addEventListener('click', exportCSV);
    
    // Filtres
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentFilter = e.target.dataset.filter;
            displayComments(allPredictions);
        });
    });
}

// Sauvegarder l'URL de l'API
async function saveApiUrl() {
    const apiUrl = document.getElementById('apiUrl').value.trim();
    
    if (!apiUrl) {
        showError('Veuillez entrer une URL valide');
        return;
    }
    
    await chrome.storage.sync.set({ [API_URL_KEY]: apiUrl });
    showMessage('URL sauvegardée avec succès!');
}

// Basculer le thème
async function toggleTheme() {
    const currentTheme = document.body.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.body.setAttribute('data-theme', newTheme === 'dark' ? 'dark' : '');
    document.getElementById('toggleTheme').textContent = newTheme === 'dark' ? '☀️' : '🌙';
    
    await chrome.storage.sync.set({ [THEME_KEY]: newTheme });
}

// Analyser les commentaires
async function analyzeComments() {
    try {
        showLoading(true);
        hideError();
        
        // Extraire les commentaires de la page
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        
        if (!tab.url.includes('youtube.com/watch')) {
            throw new Error('Veuillez ouvrir une vidéo YouTube');
        }
        
        const response = await chrome.tabs.sendMessage(tab.id, { action: 'extractComments' });
        
        if (!response.success || response.comments.length === 0) {
            throw new Error('Aucun commentaire trouvé. Scrollez pour charger plus de commentaires.');
        }
        
        console.log(`${response.comments.length} commentaires extraits`);
        
        // Envoyer à l'API
        const { apiUrl } = await chrome.storage.sync.get(API_URL_KEY);
        const url = apiUrl || DEFAULT_API_URL;
        
        const apiResponse = await fetch(`${url}/predict_batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                comments: response.comments.map(c => ({ text: c.text }))
            })
        });
        
        if (!apiResponse.ok) {
            throw new Error(`Erreur API: ${apiResponse.status}`);
        }
        
        const data = await apiResponse.json();
        
        // Afficher les résultats
        displayResults(data);
        
    } catch (error) {
        console.error('Erreur:', error);
        showError(error.message);
    } finally {
        showLoading(false);
    }
}

// Afficher les résultats
function displayResults(data) {
    allPredictions = data.predictions;
    
    // Statistiques
    displayStatistics(data.statistics, data.total_comments);
    
    // Commentaires
    displayComments(data.predictions);
    
    // Afficher les sections
    document.getElementById('statistics').classList.remove('hidden');
    document.getElementById('filters').classList.remove('hidden');
    document.getElementById('commentsList').classList.remove('hidden');
    document.getElementById('extraActions').classList.remove('hidden');
}

// Afficher les statistiques
function displayStatistics(stats, total) {
    document.getElementById('positivePercent').textContent = `${stats.positive_percentage}%`;
    document.getElementById('neutralPercent').textContent = `${stats.neutral_percentage}%`;
    document.getElementById('negativePercent').textContent = `${stats.negative_percentage}%`;
    
    const positiveCount = Math.round(total * stats.positive_percentage / 100);
    const neutralCount = Math.round(total * stats.neutral_percentage / 100);
    const negativeCount = Math.round(total * stats.negative_percentage / 100);
    
    document.getElementById('positiveCount').textContent = positiveCount;
    document.getElementById('neutralCount').textContent = neutralCount;
    document.getElementById('negativeCount').textContent = negativeCount;
    
    // Graphique
    createChart(stats);
}

// Créer le graphique
function createChart(stats) {
    const canvas = document.getElementById('sentimentChart');
    const ctx = canvas.getContext('2d');
    
    if (window.sentimentChart) {
        window.sentimentChart.destroy();
    }
    
    window.sentimentChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Positifs', 'Neutres', 'Négatifs'],
            datasets: [{
                data: [
                    stats.positive_percentage,
                    stats.neutral_percentage,
                    stats.negative_percentage
                ],
                backgroundColor: ['#2ecc71', '#95a5a6', '#e74c3c'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Afficher les commentaires
function displayComments(predictions) {
    const container = document.getElementById('commentsList');
    container.innerHTML = '';
    
    const filtered = predictions.filter(p => {
        if (currentFilter === 'all') return true;
        return p.sentiment.toLowerCase() === currentFilter;
    });
    
    if (filtered.length === 0) {
        container.innerHTML = '<p style="text-align: center; padding: 20px;">Aucun commentaire dans cette catégorie</p>';
        return;
    }
    
    filtered.forEach(prediction => {
        const div = document.createElement('div');
        div.className = `comment-item ${prediction.sentiment.toLowerCase()}`;
        
        div.innerHTML = `
            <div class="comment-header">
                <span class="comment-sentiment">${getSentimentEmoji(prediction.sentiment)} ${prediction.sentiment}</span>
                <span class="comment-confidence">${(prediction.confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="comment-text">${truncateText(prediction.text, 150)}</div>
        `;
        
        container.appendChild(div);
    });
}

// Obtenir l'emoji du sentiment
function getSentimentEmoji(sentiment) {
    const emojis = {
        'Positif': '😊',
        'Neutre': '😐',
        'Négatif': '😞'
    };
    return emojis[sentiment] || '❓';
}

// Tronquer le texte
function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// Copier les résultats
async function copyResults() {
    const stats = {
        total: allPredictions.length,
        positifs: allPredictions.filter(p => p.sentiment === 'Positif').length,
        neutres: allPredictions.filter(p => p.sentiment === 'Neutre').length,
        négatifs: allPredictions.filter(p => p.sentiment === 'Négatif').length
    };
    
    const text = `
YouTube Sentiment Analysis
==========================
Total des commentaires: ${stats.total}
Positifs: ${stats.positifs} (${((stats.positifs/stats.total)*100).toFixed(1)}%)
Neutres: ${stats.neutres} (${((stats.neutres/stats.total)*100).toFixed(1)}%)
Négatifs: ${stats.négatifs} (${((stats.négatifs/stats.total)*100).toFixed(1)}%)
    `.trim();
    
    await navigator.clipboard.writeText(text);
    showMessage('Résultats copiés!');
}

// Exporter en CSV
function exportCSV() {
    const csv = [
        ['Texte', 'Sentiment', 'Confiance'],
        ...allPredictions.map(p => [
            `"${p.text.replace(/"/g, '""')}"`,
            p.sentiment,
            p.confidence.toFixed(4)
        ])
    ].map(row => row.join(',')).join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sentiment-analysis-${Date.now()}.csv`;
    a.click();
    showMessage('CSV exporté!');
}// Afficher/masquer le chargement
function showLoading(show) {
document.getElementById('loading').classList.toggle('hidden', !show);
document.getElementById('analyzeBtn').disabled = show;
}// Afficher une erreur
function showError(message) {
const errorDiv = document.getElementById('errorMessage');
errorDiv.textContent = `❌ ${message}`;
errorDiv.classList.remove('hidden');
}// Masquer l'erreur
function hideError() {
document.getElementById('errorMessage').classList.add('hidden');
}// Afficher un message temporaire
function showMessage(message) {
const errorDiv = document.getElementById('errorMessage');
errorDiv.style.background = '#d4edda';
errorDiv.style.color = '#155724';
errorDiv.style.borderColor = '#155724';
errorDiv.textContent = `✅ ${message}`;
errorDiv.classList.remove('hidden');setTimeout(() => {
    errorDiv.classList.add('hidden');
    errorDiv.style.background = '';
    errorDiv.style.color = '';
    errorDiv.style.borderColor = '';
}, 3000);
}
    