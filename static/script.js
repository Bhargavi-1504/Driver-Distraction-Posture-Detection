const form = document.getElementById('upload-form');
const loader = document.getElementById('loader');
const resultDiv = document.getElementById('result');
const predictionText = document.getElementById('prediction-text');
const resultImage = document.getElementById('result-image');

form.addEventListener('submit', function(e) {
    e.preventDefault();

    loader.classList.remove('hidden');
    resultDiv.classList.add('hidden');

    const fileInput = document.getElementById('file-input');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    fetch('/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loader.classList.add('hidden');
        predictionText.innerText = data.prediction;
        resultImage.src = data.plot_img;
        resultDiv.classList.remove('hidden');
    })
    .catch(err => {
        loader.classList.add('hidden');
        alert('Error processing image.');
        console.error(err);
    });
});

function resetPage() {
    resultDiv.classList.add('hidden');
    document.getElementById('file-input').value = '';
}
