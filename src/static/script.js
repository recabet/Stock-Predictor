document.getElementById('predictionForm').addEventListener('submit', function (event) {
    event.preventDefault();

    const stockName = document.getElementById('stockName').value;
    const days = document.getElementById('minutes').value;
    const interval = document.getElementById('interval').value;

    fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            stock_name: stockName,
            minutes: days,
            interval: interval
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.predicted_price && data.predicted_price.length > 0) {
            const result = data.predicted_price
                .map((price, index) => `${interval[1]}${index + 1}: $${price.toFixed(2)}`)
                .join('<br>');
            document.getElementById('result').innerHTML = `<strong>Predicted Prices:</strong><br>${result}`;
        } else if (data.error) {
            document.getElementById('result').innerText = `Error: ${data.error}`;
        } else {
            document.getElementById('result').innerText = 'No predictions returned.';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerText = 'An error occurred while fetching the prediction.';
    });
});
