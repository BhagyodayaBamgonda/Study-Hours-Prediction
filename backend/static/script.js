// Prediction form
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const pastScore = Number(document.getElementById('past_score').value);
    const targetScore = Number(document.getElementById('target_score').value);
    const lastWeekHours = Number(document.getElementById('last_week_hours').value);

    const data = {
        past_score: pastScore,
        subjects: Number(document.getElementById('subjects').value),
        last_week_hours: lastWeekHours,
        stress_level: Number(document.getElementById('stress_level').value),
        sleep_hours: Number(document.getElementById('sleep_hours').value),
        target_score: targetScore
    };

    try {
        const response = await fetch("/predict", {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Server error');

        const result = await response.json();
        const predicted = result.recommended_study_hours;

        let totalStudyHours;

        // Condition: if target < past, subtract abs(prediction), else add prediction
        if (targetScore < pastScore) {
            totalStudyHours = lastWeekHours - Math.abs(predicted);
        } else {
            totalStudyHours = lastWeekHours + predicted;
        }

        document.getElementById('result').textContent = totalStudyHours.toFixed(2);
    } catch (error) {
        const errorDiv = document.getElementById('errorMessage');
        errorDiv.textContent = 'Error fetching prediction. Make sure Flask server is running.';
        errorDiv.style.display = 'block';
        console.error(error);
    }
});

// Visualization button
document.getElementById('visualizeBtn').addEventListener('click', async () => {
    const chart = document.getElementById('chart');
    chart.style.display = 'block';

    try {
        // Add timestamp to avoid browser caching
        chart.src = 'http://127.0.0.1:5000/visualize?' + new Date().getTime();
    } catch (error) {
        alert("Error loading chart. Make sure Flask backend is running.");
        console.error(error);
    }
});
