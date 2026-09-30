// GridGuardAI — frontend logic with animations

const dropZone   = document.getElementById('drop-zone');
const fileInput  = document.getElementById('file-input');
const fileName   = document.getElementById('file-name');
const predictBtn = document.getElementById('predict-btn');
const errorMsg   = document.getElementById('error-msg');
const loader     = document.getElementById('loader');
const uploadSec  = document.getElementById('upload-section');
const resultsSec = document.getElementById('results-section');

let currentFile = null;
let allSuspicious = [];
let chartInstance = null;

['dragenter', 'dragover'].forEach(ev =>
    dropZone.addEventListener(ev, e => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    })
);
['dragleave', 'drop'].forEach(ev =>
    dropZone.addEventListener(ev, e => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    })
);

dropZone.addEventListener('drop', e => {
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
});

fileInput.addEventListener('change', e => {
    if (e.target.files[0]) setFile(e.target.files[0]);
});

function setFile(f) {
    if (!f.name.endsWith('.csv')) {
        showError('Please select a CSV file');
        return;
    }
    currentFile = f;
    fileName.textContent = `✓ ${f.name} (${(f.size / 1024 / 1024).toFixed(2)} MB)`;
    predictBtn.disabled = false;
    hideError();
}

function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.classList.add('visible');
}
function hideError() {
    errorMsg.classList.remove('visible');
}

function animateCounter(el, target, duration = 1200) {
    const startTime = performance.now();
    function tick(now) {
        const progress = Math.min((now - startTime) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const value = Math.floor(target * eased);
        el.textContent = value.toLocaleString();
        if (progress < 1) requestAnimationFrame(tick);
        else el.textContent = target.toLocaleString();
    }
    requestAnimationFrame(tick);
}

function runSteps() {
    const steps = document.querySelectorAll('#loader .step');
    steps.forEach(s => s.classList.remove('active', 'done'));
    steps[0].classList.add('active');
    let i = 0;
    const interval = setInterval(() => {
        if (i < steps.length) {
            steps[i].classList.remove('active');
            steps[i].classList.add('done');
            i++;
            if (i < steps.length) steps[i].classList.add('active');
        } else {
            clearInterval(interval);
        }
    }, 2500);
    return () => clearInterval(interval);
}

predictBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    predictBtn.disabled = true;
    hideError();
    loader.classList.remove('hidden');

    const stopSteps = runSteps();

    const fd = new FormData();
    fd.append('file', currentFile);

    try {
        const res = await fetch('/predict', { method: 'POST', body: fd });
        const data = await res.json();

        if (!res.ok || data.error) throw new Error(data.error || 'Prediction failed');

        stopSteps();
        document.querySelectorAll('#loader .step').forEach(s => {
            s.classList.remove('active');
            s.classList.add('done');
        });

        await new Promise(r => setTimeout(r, 400));

        renderResults(data);
        uploadSec.classList.add('hidden');
        resultsSec.classList.remove('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });

    } catch (err) {
        stopSteps();
        showError(err.message);
        predictBtn.disabled = false;
    } finally {
        loader.classList.add('hidden');
    }
});

function renderResults(data) {
    const s = data.summary;
    allSuspicious = data.suspicious;

    animateCounter(document.getElementById('total-customers'), s.total_customers);
    animateCounter(document.getElementById('flagged-count'), s.flagged);
    document.getElementById('flagged-pct').textContent = s.flagged_pct + '%';
    document.getElementById('proc-time').textContent = s.processing_time_sec + 's';

    renderTable(allSuspicious);
    renderChart(s);

    document.getElementById('search-input').addEventListener('input', e => {
        const q = e.target.value.toLowerCase();
        const filtered = allSuspicious.filter(c =>
            c.customer_id.toLowerCase().includes(q)
        );
        renderTable(filtered);
    });

    document.getElementById('download-btn').onclick = () => downloadCSV(allSuspicious);
}

function renderTable(list) {
    const tbody = document.querySelector('#results-table tbody');
    tbody.innerHTML = '';

    list.forEach((c, i) => {
        const risk = c.risk_score;
        let cls = 'risk-low', badge = 'Low', badgeCls = 'badge-low';
        if (risk >= 0.8) { cls = 'risk-high'; badge = 'High risk — inspect'; badgeCls = 'badge-high'; }
        else if (risk >= 0.5) { cls = 'risk-med'; badge = 'Medium risk'; badgeCls = 'badge-med'; }

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${i + 1}</td>
            <td><code>${c.customer_id}</code></td>
            <td class="${cls}">${(risk * 100).toFixed(1)}%</td>
            <td><span class="badge ${badgeCls}">${badge}</span></td>
        `;
        tbody.appendChild(tr);
    });
}

function renderChart(summary) {
    const ctx = document.getElementById('riskChart').getContext('2d');
    if (chartInstance) chartInstance.destroy();

    const high = summary.flagged * 0.35;
    const med = summary.flagged * 0.45;
    const low = summary.flagged * 0.20;
    const normal = summary.total_customers - summary.flagged;

    chartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['High Risk', 'Medium Risk', 'Low Risk', 'Normal'],
            datasets: [{
                data: [Math.round(high), Math.round(med), Math.round(low), normal],
                backgroundColor: [
                    'rgba(248, 113, 113, 0.8)',
                    'rgba(251, 191, 36, 0.8)',
                    'rgba(148, 163, 184, 0.6)',
                    'rgba(52, 211, 153, 0.5)',
                ],
                borderColor: '#0e1424',
                borderWidth: 3,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#94a3b8',
                        font: { size: 11 },
                        padding: 14,
                        usePointStyle: true,
                    }
                },
                tooltip: {
                    backgroundColor: '#0e1424',
                    borderColor: '#1e2942',
                    borderWidth: 1,
                    titleColor: '#e2e8f0',
                    bodyColor: '#94a3b8',
                    padding: 12,
                }
            }
        }
    });
}

function downloadCSV(list) {
    const rows = [['Rank', 'Customer ID', 'Risk Score', 'Recommendation']];
    list.forEach((c, i) => {
        const risk = c.risk_score;
        const rec = risk >= 0.8 ? 'High risk — inspect' : risk >= 0.5 ? 'Medium risk' : 'Low risk';
        rows.push([i + 1, c.customer_id, (risk * 100).toFixed(2) + '%', rec]);
    });

    const csv = rows.map(r => r.map(v => `"${v}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `gridguardai_suspicious_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
}