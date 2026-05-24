document.addEventListener('DOMContentLoaded', () => {
    const statusPanel = document.getElementById('status-panel');
    const statusText = document.getElementById('status-text');

    async function checkStatus() {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            
            if (data.status === 'Alarm') {
                statusPanel.className = 'status-alarm';
                statusText.textContent = 'Alarm';
            } else {
                statusPanel.className = 'status-normal';
                statusText.textContent = 'Normal';
            }
        } catch (error) {
            console.error('Error fetching status:', error);
            statusPanel.className = 'status-alarm'; // Fallback to alarm style on error just in case
            statusText.textContent = 'Связь потеряна';
        }
    }

    // Poll the server every 1 second
    setInterval(checkStatus, 1000);
});
