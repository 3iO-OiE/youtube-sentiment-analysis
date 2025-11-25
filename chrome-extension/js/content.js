// Fonction pour extraire les commentaires YouTube
function extractComments() {
    const comments = [];
    
    // Sélecteur pour les commentaires YouTube
    const commentElements = document.querySelectorAll('ytd-comment-thread-renderer');
    
    commentElements.forEach((element, index) => {
        try {
            // Extraire le texte du commentaire
            const commentTextElement = element.querySelector('#content-text');
            
            if (commentTextElement) {
                const text = commentTextElement.innerText.trim();
                
                if (text && text.length > 0) {
                    comments.push({
                        id: `comment_${index}`,
                        text: text,
                        author: element.querySelector('#author-text')?.innerText.trim() || 'Unknown',
                        likes: element.querySelector('#vote-count-middle')?.innerText.trim() || '0'
                    });
                }
            }
        } catch (error) {
            console.error('Erreur extraction commentaire:', error);
        }
    });
    
    return comments;
}

// Écouter les messages du popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'ping') {
        sendResponse({ success: true });
        return true;
    }
    
    if (request.action === 'extractComments') {
        const comments = extractComments();
        sendResponse({
            success: true,
            comments: comments,
            count: comments.length
        });
    }
    return true;
});

console.log('YouTube Sentiment Analyzer - Content Script chargé');