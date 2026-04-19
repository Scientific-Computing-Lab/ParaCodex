/**
 * Paracodex Pipeline Web Application
 */

// ============================================================
// STATE
// ============================================================
const state = {
    currentJobId: null,
    currentPath: null,
    selectedPath: null,
    activeJobs: new Map(),
    eventSource: null,
    activeTab: 'pipeline',
    tracesLoaded: false,
    skillsLoaded: false,
};

// ============================================================
// DOM ELEMENTS
// ============================================================
const elements = {
    // Form
    pipelineForm:      document.getElementById('pipelineForm'),
    modeToggles:       document.querySelectorAll('input[name="inputMode"]'),
    customInputGroup:  document.getElementById('customInputGroup'),
    parbenchInputGroup: document.getElementById('parbenchInputGroup'),
    sourceDirectory:   document.getElementById('sourceDirectory'),
    parbenchSpec:      document.getElementById('parbenchSpec'),
    fromApi:           document.getElementById('fromApi'),
    toApi:             document.getElementById('toApi'),
    model:             document.getElementById('model'),
    commandPreview:    document.getElementById('commandPreviewText'),
    startBtn:          document.getElementById('startBtn'),
    browseBtn:         document.getElementById('browseBtn'),
    browseSpecBtn:     document.getElementById('browseSpecBtn'),

    // Job Status
    jobStatusCard:  document.getElementById('jobStatusCard'),
    statusBadge:    document.getElementById('statusBadge'),
    jobId:          document.getElementById('jobId'),
    jobTranslation: document.getElementById('jobTranslation'),
    jobSource:      document.getElementById('jobSource'),
    killJobBtn:     document.getElementById('killJobBtn'),

    // Stage
    stageDescription: document.getElementById('stageDescription'),

    // Log panel
    logPanel:       document.getElementById('logPanel'),
    logPanelHeader: document.getElementById('logPanelHeader'),
    logPanelBody:   document.getElementById('logPanelBody'),
    logLineCount:   document.getElementById('logLineCount'),

    // Artifacts
    artifactCard:          document.getElementById('artifactCard'),
    artifactsList:         document.getElementById('artifactsList'),
    artifactViewerTitle:   document.getElementById('artifactViewerTitle'),
    artifactViewerContent: document.getElementById('artifactViewerContent'),

    // Results
    jobResults:    document.getElementById('jobResults'),
    resultWorkdir: document.getElementById('resultWorkdir'),
    resultOutput:  document.getElementById('resultOutput'),

    // Modals
    viewJobsBtn:     document.getElementById('viewJobsBtn'),
    jobsModal:       document.getElementById('jobsModal'),
    jobsList:        document.getElementById('jobsList'),
    closeJobsModal:  document.getElementById('closeJobsModal'),
    closeJobsList:   document.getElementById('closeJobsList'),

    directoryModal:   document.getElementById('directoryModal'),
    breadcrumb:       document.getElementById('breadcrumb'),
    directoryList:    document.getElementById('directoryList'),
    closeModal:       document.getElementById('closeModal'),
    cancelBrowse:     document.getElementById('cancelBrowse'),
    selectDirectory:  document.getElementById('selectDirectory'),

    performanceModal:   document.getElementById('performanceModal'),
    performanceContent: document.getElementById('performanceContent'),
    analyzeBtn:         document.getElementById('analyzeBtn'),

    parbenchVerifyBtn:    document.getElementById('parbenchVerifyBtn'),
    parbenchResultsPanel: document.getElementById('parbenchResultsPanel'),
    parbenchResultsBody:  document.getElementById('parbenchResultsBody'),

    jobTabsContainer: document.getElementById('jobTabsContainer'),
    jobTabs:          document.getElementById('jobTabs'),

    toastContainer: document.getElementById('toastContainer'),
};

// ============================================================
// TOAST NOTIFICATIONS (replaces alert())
// ============================================================
function showToast(message, type = 'error', duration = 5000) {
    const icons = { error: '❌', success: '✅', warning: '⚠️' };
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || '💬'}</span>
        <span class="toast-message">${escapeHtml(message)}</span>
        <button class="toast-close" aria-label="Dismiss">×</button>
    `;

    const dismiss = () => {
        toast.classList.add('removing');
        toast.addEventListener('animationend', () => toast.remove(), { once: true });
    };

    toast.querySelector('.toast-close').addEventListener('click', dismiss);
    elements.toastContainer.appendChild(toast);

    if (duration > 0) setTimeout(dismiss, duration);
}

// ============================================================
// INITIALIZATION
// ============================================================
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const configResp = await fetch('/api/config');
        const config = await configResp.json();
        const defaultDir = config.default_source_dir || '~/';
        state.currentPath = defaultDir;
        elements.sourceDirectory.value = defaultDir;
    } catch {
        state.currentPath = './';
        elements.sourceDirectory.value = './';
    }
    initEventListeners();
    updateCommandPreview();
    checkForRunningJobs();
});

async function checkForRunningJobs() {
    await fetchActiveJobs();

    if (state.activeJobs.size > 0) {
        for (const [jobId, job] of state.activeJobs) {
            if (job.status === 'running') {
                state.currentJobId = jobId;
                await loadJobArtifacts(job);
                monitorJob(jobId);
                break;
            }
        }

        if (!state.currentJobId) {
            const firstJobId = state.activeJobs.keys().next().value;
            if (firstJobId) {
                state.currentJobId = firstJobId;
                await loadJobArtifacts(state.activeJobs.get(firstJobId));
            }
        }
    }
}

function initEventListeners() {
    elements.pipelineForm.addEventListener('submit', handleFormSubmit);

    elements.modeToggles.forEach(t => t.addEventListener('change', handleModeToggle));
    document.querySelectorAll('input[name="engine"]').forEach(r =>
        r.addEventListener('change', handleEngineToggle)
    );
    elements.sourceDirectory.addEventListener('input', updateCommandPreview);
    elements.parbenchSpec.addEventListener('input', updateCommandPreview);
    elements.fromApi.addEventListener('change', updateCommandPreview);
    elements.toApi.addEventListener('change', updateCommandPreview);
    elements.model.addEventListener('input', updateCommandPreview);
    document.getElementById('supervisorToggle')?.addEventListener('change', updateCommandPreview);
    document.getElementById('baselineToggle')?.addEventListener('change', updateCommandPreview);

    elements.browseBtn.addEventListener('click', () => openDirectoryBrowser(false));
    elements.browseSpecBtn.addEventListener('click', () => openDirectoryBrowser(true));

    elements.killJobBtn.addEventListener('click', handleKillJob);

    // Directory modal
    elements.closeModal.addEventListener('click', closeDirectoryBrowser);
    elements.cancelBrowse.addEventListener('click', closeDirectoryBrowser);
    elements.selectDirectory.addEventListener('click', selectDirectory);
    elements.directoryModal.querySelector('.modal-backdrop').addEventListener('click', closeDirectoryBrowser);

    // Jobs modal
    elements.viewJobsBtn.addEventListener('click', () => switchTab('history'));
    elements.closeJobsModal.addEventListener('click', closeJobsModal);
    elements.closeJobsList.addEventListener('click', closeJobsModal);

    // Performance modal close buttons
    document.querySelectorAll('.close-modal').forEach(btn => {
        btn.addEventListener('click', function () {
            this.closest('.modal').style.display = 'none';
        });
    });

    // Main tab bar
    document.querySelectorAll('.main-tab').forEach(btn =>
        btn.addEventListener('click', () => switchTab(btn.dataset.tab))
    );

    // History tab filters
    let historyDebounce;
    document.getElementById('historySearch').addEventListener('input', () => {
        clearTimeout(historyDebounce);
        historyDebounce = setTimeout(loadHistory, 300);
    });
    document.getElementById('historyStatusFilter').addEventListener('change', loadHistory);
    document.getElementById('historyRefreshBtn').addEventListener('click', loadHistory);

    // Skill modal close
    document.getElementById('closeSkillModal').addEventListener('click', () => {
        document.getElementById('skillModal').style.display = 'none';
    });
    document.getElementById('skillModal').querySelector('.modal-backdrop')
        .addEventListener('click', () => {
            document.getElementById('skillModal').style.display = 'none';
        });

    document.getElementById('closeJobDetailModal')?.addEventListener('click', () => {
        document.getElementById('jobDetailModal').style.display = 'none';
    });
    document.getElementById('closeResultsModal')?.addEventListener('click', () => {
        document.getElementById('resultsModal').style.display = 'none';
    });
    document.querySelector('#resultsModal .modal-backdrop')?.addEventListener('click', () => {
        document.getElementById('resultsModal').style.display = 'none';
    });
    document.querySelectorAll('.job-detail-tab').forEach(btn =>
        btn.addEventListener('click', () => switchJobDetailTab(btn.dataset.tab))
    );

    // Performance analysis
    if (elements.analyzeBtn) {
        elements.analyzeBtn.addEventListener('click', handleAnalyzePerformance);
    }

    // ParBench verification
    if (elements.parbenchVerifyBtn) {
        elements.parbenchVerifyBtn.addEventListener('click', handleParbenchVerify);
    }

    // Log panel toggle
    elements.logPanelHeader.addEventListener('click', () => {
        elements.logPanel.classList.toggle('collapsed');
    });

    // Close modals on backdrop click
    window.addEventListener('click', e => {
        if (e.target.classList.contains('modal')) {
            e.target.style.display = 'none';
        }
    });
}

// ============================================================
// ENGINE TOGGLE
// ============================================================
function getActiveEngine() {
    const checked = document.querySelector('input[name="engine"]:checked');
    return checked ? checked.value : null;
}

function handleEngineToggle(e) {
    const engine = e.target.value;

    document.querySelectorAll('.engine-toggle').forEach(el => {
        el.classList.remove('active-opencode', 'active-codex');
    });
    const activeLabel = e.target.closest('.engine-toggle');
    if (activeLabel) activeLabel.classList.add(`active-${engine}`);

    // Update model field hint and placeholder
    const modelHint = document.getElementById('modelHint');
    const modelInput = elements.model;
    if (engine === 'codex') {
        if (modelHint) modelHint.textContent = 'Codex model, e.g. o3, gpt-4o (optional)';
        modelInput.placeholder = 'Default (CODEX_MODEL env var)';
        // Swap datalist to Codex models
        const dl = document.getElementById('modelOptions');
        if (dl) dl.innerHTML = `
            <option value="o3">
            <option value="o4-mini">
            <option value="gpt-4o">
            <option value="gpt-4.1">
        `;
    } else {
        if (modelHint) modelHint.textContent = 'opencode model, e.g. anthropic/claude-sonnet-4-5 (optional)';
        modelInput.placeholder = 'Default (OPENCODE_MODEL env var)';
        const dl = document.getElementById('modelOptions');
        if (dl) dl.innerHTML = `
            <option value="anthropic/claude-sonnet-4-5">
            <option value="anthropic/claude-opus-4-5">
            <option value="anthropic/claude-haiku-4-5">
            <option value="openai/gpt-4o">
            <option value="google/gemini-1.5-pro-002">
            <option value="google/gemini-2.0-flash">
        `;
    }

    updateCommandPreview();
}

// ============================================================
// MODE TOGGLE
// ============================================================
function getActiveMode() {
    const checked = document.querySelector('input[name="inputMode"]:checked');
    return checked ? checked.value : 'custom';
}

function handleModeToggle(e) {
    const isParbench = e.target.value === 'parbench';

    document.querySelectorAll('.mode-toggle').forEach(el => {
        el.classList.remove('active-custom', 'active-parbench');
    });
    const activeLabel = e.target.closest('.mode-toggle');
    if (activeLabel) activeLabel.classList.add(isParbench ? 'active-parbench' : 'active-custom');

    elements.customInputGroup.style.display  = isParbench ? 'none' : 'block';
    elements.parbenchInputGroup.style.display = isParbench ? 'block' : 'none';

    elements.sourceDirectory.required = !isParbench;
    elements.parbenchSpec.required    = isParbench;

    const fromApiSelect = elements.fromApi;
    const fromApiBadge  = document.getElementById('fromApiAutoBadge');
    const fromApiHint   = document.getElementById('fromApiHint');

    if (isParbench) {
        fromApiSelect.style.display = 'none';
        fromApiSelect.disabled = true;
        fromApiSelect.required = false;
        if (fromApiBadge) fromApiBadge.style.display = 'block';
        if (fromApiHint)  fromApiHint.textContent = 'Read automatically from spec';
    } else {
        fromApiSelect.style.display = '';
        fromApiSelect.disabled = false;
        fromApiSelect.required = true;
        if (fromApiBadge) fromApiBadge.style.display = 'none';
        if (fromApiHint)  fromApiHint.textContent = 'Current implementation';
        if (!fromApiSelect.value) fromApiSelect.value = 'serial';
    }

    const descEl = document.getElementById('modeDescription');
    if (descEl) {
        descEl.textContent = isParbench
            ? '🔬 Use a ParBench JSON spec file — the source API and files to translate are read automatically.'
            : '📂 Specify a local directory of source code to translate manually.';
    }

    updateCommandPreview();
}

// ============================================================
// COMMAND PREVIEW
// ============================================================
function updateCommandPreview() {
    const mode   = getActiveMode();
    const engine = getActiveEngine();
    const model  = elements.model.value;
    let cmd = '';

    if (mode === 'custom') {
        const sourceDir = elements.sourceDirectory.value || state.currentPath || './';
        const fromApi   = elements.fromApi.value || 'serial';
        const toApi     = elements.toApi.value   || 'omp';
        cmd = `python setup_pipeline_workdir.py -s ${sourceDir} -f ${fromApi} -t ${toApi}`;
    } else {
        const specFile = elements.parbenchSpec.value || 'spec.json';
        const toApi    = elements.toApi.value || 'omp';
        cmd = `python setup_pipeline_workdir.py --parbench_spec ${specFile} -t ${toApi}`;
    }

    const baseline  = document.getElementById('baselineToggle')?.checked;
    const supervise = document.getElementById('supervisorToggle')?.checked;
    if (model)     cmd += ` --model ${model}`;
    if (engine)    cmd += ` --engine ${engine}`;
    if (baseline)  cmd += ' --baseline';
    if (supervise) cmd += ' --supervise';
    cmd += ' --yes';

    elements.commandPreview.textContent = cmd;
}

// ============================================================
// FORM SUBMISSION
// ============================================================
async function handleFormSubmit(e) {
    e.preventDefault();

    const mode = getActiveMode();
    const formData = {
        to_api: elements.toApi.value,
        model: elements.model.value,
    };
    const activeEngine = getActiveEngine();
    if (activeEngine) formData.engine = activeEngine;
    formData.baseline  = document.getElementById('baselineToggle')?.checked  || false;
    formData.supervise = document.getElementById('supervisorToggle')?.checked || false;

    if (mode === 'custom') {
        formData.source_directory = elements.sourceDirectory.value;
        formData.from_api = elements.fromApi.value;
        if (!formData.source_directory || !formData.from_api || !formData.to_api) {
            showToast('Please fill in all required fields', 'error');
            return;
        }
    } else {
        formData.parbench_spec = elements.parbenchSpec.value;
        if (!formData.parbench_spec || !formData.to_api) {
            showToast('Please fill in all required fields', 'error');
            return;
        }
    }

    elements.startBtn.disabled = true;
    elements.startBtn.innerHTML = '<span class="btn-icon">⏳</span> Starting...';

    try {
        const response = await fetch('/api/start-pipeline', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData),
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Failed to start pipeline');

        state.currentJobId = data.job_id;

        addJobTab(data.job_id, {
            id: data.job_id,
            status: 'running',
            from_api: mode === 'custom' ? formData.from_api : 'parbench',
            to_api: formData.to_api,
            model: formData.model,
            engine: formData.engine,
            source_directory: mode === 'custom' ? formData.source_directory : formData.parbench_spec,
        });

        if (mode === 'parbench') {
            formData.from_api = 'ParBench-Auto';
            formData.source_directory = formData.parbench_spec;
        }
        showJobStatus(data.job_id, formData);
        monitorJob(data.job_id);
        startLogStream(data.job_id);

        showToast('Pipeline started successfully', 'success', 3000);
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        elements.startBtn.disabled = false;
        elements.startBtn.innerHTML = '<span class="btn-icon">🚀</span> Start Pipeline';
    }
}

// ============================================================
// LIVE LOG STREAMING (SSE)
// ============================================================
function startLogStream(jobId) {
    if (state.eventSource) {
        state.eventSource.close();
        state.eventSource = null;
    }

    // Expand the log panel when a new job starts
    elements.logPanel.classList.remove('collapsed');
    elements.logPanelBody.innerHTML = '<div class="log-spinner">Connecting...</div>';
    elements.logLineCount.textContent = '';

    let lineCount = 0;
    let firstLine = true;

    const es = new EventSource(`/api/logs/${jobId}`);
    state.eventSource = es;

    es.onmessage = (event) => {
        let payload;
        try { payload = JSON.parse(event.data); } catch { return; }

        if (payload.error && !payload.done) {
            appendLog({ timestamp: now(), message: `Error: ${payload.error}` }, 'log-error');
            return;
        }

        if (payload.done) {
            es.close();
            state.eventSource = null;
            return;
        }

        if (firstLine) {
            elements.logPanelBody.innerHTML = '';
            firstLine = false;
        }

        lineCount++;
        elements.logLineCount.textContent = `(${lineCount} lines)`;

        const cls = classifyLogLine(payload.message);
        appendLog(payload, cls);
    };

    es.onerror = () => {
        es.close();
        state.eventSource = null;
    };
}

function now() {
    return new Date().toTimeString().slice(0, 8);
}

function classifyLogLine(msg) {
    if (!msg) return '';
    const m = msg.toLowerCase();
    if (m.includes('error') || m.includes('failed') || m.includes('✗')) return 'log-error';
    if (m.includes('warning') || m.includes('warn'))                      return 'log-warn';
    if (m.includes('success') || m.includes('✓') || m.includes('completed')) return 'log-success';
    return '';
}

function appendLog(entry, extraClass = '') {
    const div = document.createElement('div');
    div.className = `log-entry ${extraClass}`.trim();
    div.innerHTML = `
        <span class="log-timestamp">${escapeHtml(entry.timestamp || now())}</span>
        <span class="log-message">${escapeHtml(entry.message || '')}</span>
    `;
    elements.logPanelBody.appendChild(div);
    // Auto-scroll to bottom
    elements.logPanelBody.scrollTop = elements.logPanelBody.scrollHeight;
}

// ============================================================
// MULTI-JOB MANAGEMENT
// ============================================================
async function fetchActiveJobs() {
    try {
        const response = await fetch('/api/active-jobs');
        const data = await response.json();
        if (data.jobs) {
            // On page load, only restore jobs that are still running.
            // Completed/failed/killed jobs from previous sessions should not
            // reappear as open tabs after a refresh.
            data.jobs
                .filter(job => job.status === 'running')
                .forEach(job => state.activeJobs.set(job.id, job));
            renderJobTabs();
            pollActiveJobs();
        }
    } catch { /* non-fatal */ }
}

function renderJobTabs() {
    if (state.activeJobs.size === 0) {
        elements.jobTabsContainer.style.display = 'none';
        return;
    }
    elements.jobTabsContainer.style.display = 'block';

    let html = '';
    state.activeJobs.forEach((job, jobId) => {
        const isActive    = jobId === state.currentJobId;
        const translation = `${job.from_api}→${job.to_api}`;
        const shortId     = jobId.slice(0, 8);
        html += `
            <div class="job-tab ${isActive ? 'active' : ''}" data-job-id="${jobId}" onclick="switchToJob('${jobId}')">
                <div class="job-tab-label">
                    <span title="${jobId}">${translation} <span style="color:var(--text-tertiary);font-size:0.7rem;">${shortId}</span></span>
                    <span class="status-badge ${job.status}">${job.status}</span>
                </div>
                <span class="close-tab" onclick="event.stopPropagation(); closeJobTab('${jobId}')" title="Close tab">×</span>
            </div>
        `;
    });
    elements.jobTabs.innerHTML = html;
}

window.switchToJob = async function (jobId) {
    state.currentJobId = jobId;
    const job = state.activeJobs.get(jobId);
    if (!job) return;
    await loadJobArtifacts(job);
    renderJobTabs();
};

window.closeJobTab = function (jobId) {
    state.activeJobs.delete(jobId);
    if (state.currentJobId === jobId) {
        if (state.activeJobs.size > 0) {
            switchToJob(state.activeJobs.keys().next().value);
        } else {
            elements.jobStatusCard.style.display = 'none';
            elements.artifactCard.style.display  = 'none';
            elements.jobTabsContainer.style.display = 'none';
            state.currentJobId = null;
        }
    }
    renderJobTabs();
};

async function pollActiveJobs() {
    for (const [jobId, job] of state.activeJobs) {
        if (job.status !== 'running') continue;
        try {
            const response = await fetch(`/api/job-progress/${jobId}`);
            const data = await response.json();
            const updated = { ...job, ...data, status: data.job.status };
            state.activeJobs.set(jobId, updated);

            if (jobId === state.currentJobId) {
                updateJobStatus(updated.status, updated);
                if (data.current_stage) updateStageDisplay(data.current_stage, data.stages);
                if (data.artifacts)     updateArtifactsList(data.artifacts);
                if (updated.status !== 'running') fetchJobDetails(jobId);
            }
        } catch { /* non-fatal */ }
    }
    renderJobTabs();
}

function addJobTab(jobId, jobData) {
    state.activeJobs.set(jobId, jobData);
    renderJobTabs();
}

// ============================================================
// JOB STATUS DISPLAY
// ============================================================
function initializeStageStepper() {
    const stageStepper = document.getElementById('stageStepper');
    if (!stageStepper) return;

    const stages = [
        { id: 'analysis',    icon: '🔍', label: 'Analysis' },
        { id: 'translation', icon: '🔄', label: 'Translation' },
        { id: 'optimization',icon: '⚡', label: 'Optimization' },
        { id: 'supervision', icon: '✅', label: 'Verify' },
    ];

    stageStepper.innerHTML = stages.map((stage, i) =>
        `<div class="stage-step" data-stage="${stage.id}">
            <div class="stage-icon">${stage.icon}</div>
            <div class="stage-label">${stage.label}</div>
        </div>` +
        (i < stages.length - 1 ? '<div class="stage-connector"></div>' : '')
    ).join('');
}

function showJobStatus(jobId, formData) {
    elements.jobStatusCard.style.display = 'block';
    elements.jobStatusCard.scrollIntoView({ behavior: 'smooth' });

    // Show truncated job ID with full ID as tooltip
    elements.jobId.textContent = jobId.slice(0, 8) + '…';
    elements.jobId.title = jobId;

    let translationText = `${formData.from_api} → ${formData.to_api}`;
    if (formData.model) translationText += ` (${formData.model})`;
    if (formData.engine) translationText += ` [${formData.engine}]`;
    elements.jobTranslation.textContent = translationText;
    elements.jobSource.textContent = formData.source_directory;

    initializeStageStepper();
    updateJobStatus('running', formData);

    elements.killJobBtn.style.display = 'inline-flex';

    document.querySelectorAll('.stage-step').forEach(s => s.classList.remove('active', 'completed'));
    document.querySelectorAll('.stage-connector').forEach(c => c.classList.remove('active'));
    elements.stageDescription.textContent = 'Initializing pipeline...';

    resetArtifactsPanel();
    elements.jobResults.style.display = 'none';
}

function resetArtifactsPanel() {
    elements.artifactsList.innerHTML = `
        <div class="artifact-item empty">
            <span class="artifact-icon">📝</span>
            <span class="artifact-name">Waiting for artifacts...</span>
        </div>`;
    elements.artifactViewerTitle.textContent = 'Select an artifact to view';
    elements.artifactViewerContent.innerHTML = `
        <div class="artifact-placeholder">
            <div class="placeholder-icon">📋</div>
            <div class="placeholder-text">Select an artifact from the sidebar to view its contents</div>
        </div>`;
}

function updateJobStatus(status, jobData) {
    elements.statusBadge.textContent = status;
    elements.statusBadge.className   = `status-badge ${status}`;

    elements.killJobBtn.style.display = status === 'running' ? 'inline-flex' : 'none';
    if (status === 'running') {
        elements.killJobBtn.disabled = false;
        elements.killJobBtn.innerHTML = '<span>⏹</span> Kill Job';
    }

    elements.analyzeBtn && (elements.analyzeBtn.style.display = status === 'completed' ? 'inline-flex' : 'none');

    if (elements.parbenchVerifyBtn) {
        const showVerify = status === 'completed' && jobData && jobData.parbench_spec;
        elements.parbenchVerifyBtn.style.display = showVerify ? 'inline-flex' : 'none';
    }
}

// ============================================================
// JOB MONITORING
// ============================================================
let progressInterval = null;

function monitorJob(jobId) {
    if (progressInterval) clearInterval(progressInterval);
    updateAllJobsProgress();
    progressInterval = setInterval(updateAllJobsProgress, 2000);
}

async function updateAllJobsProgress() {
    await pollActiveJobs();

    const anyRunning = [...state.activeJobs.values()].some(j => j.status === 'running');
    if (!anyRunning && state.activeJobs.size > 0) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
}

function updateStageDisplay(currentStage) {
    const stageSteps      = document.querySelectorAll('.stage-step');
    const stageConnectors = document.querySelectorAll('.stage-connector');
    let stageIndex = 0;
    let foundCurrent = false;

    stageSteps.forEach((step, index) => {
        const stageId = step.dataset.stage;
        if (stageId === currentStage) {
            step.classList.add('active');
            step.classList.remove('completed');
            foundCurrent = true;
            stageIndex = index;
        } else if (!foundCurrent) {
            step.classList.add('completed');
            step.classList.remove('active');
        } else {
            step.classList.remove('active', 'completed');
        }
    });

    stageConnectors.forEach((connector, index) => {
        connector.classList.toggle('active', index < stageIndex);
    });

    const descriptions = {
        analysis:    'Analyzing source code structure and patterns...',
        translation: 'Translating code to target API...',
        optimization:'Optimizing code (Step 1 & 2)...',
        supervision: 'Verifying correctness and performance...',
    };
    elements.stageDescription.textContent = descriptions[currentStage] || 'Processing...';
}

function updateArtifactsList(artifacts) {
    const artifactCard = document.getElementById('artifactCard');

    if (!artifacts || artifacts.length === 0) {
        elements.artifactsList.innerHTML = `
            <div class="artifact-item empty">
                <span class="artifact-icon">📝</span>
                <span class="artifact-name">Waiting for artifacts...</span>
            </div>`;
        if (artifactCard) artifactCard.style.display = 'none';
        return;
    }

    if (artifactCard) artifactCard.style.display = 'block';

    const artifactIcons = { analysis: '🔍', plan: '📋', report: '📊', document: '📄' };

    elements.artifactsList.innerHTML = artifacts.map(a => `
        <div class="artifact-item" data-path="${escapeAttr(a.path)}" data-name="${escapeAttr(a.name)}">
            <span class="artifact-icon">${artifactIcons[a.type] || '📄'}</span>
            <span class="artifact-name" title="${escapeAttr(a.name)}">${escapeHtml(a.name)}</span>
        </div>
    `).join('');

    document.querySelectorAll('.artifact-item:not(.empty)').forEach(item => {
        item.addEventListener('click', () => {
            const path = item.dataset.path;
            const name = item.dataset.name;
            if (path && name) viewArtifact(path, name);
        });
    });
}

async function viewArtifact(artifactPath, artifactName) {
    document.querySelectorAll('.artifact-item').forEach(item => {
        item.classList.toggle('active', item.dataset.path === artifactPath);
    });

    elements.artifactViewerTitle.textContent = artifactName;
    elements.artifactViewerContent.innerHTML = `
        <div class="artifact-placeholder"><div class="placeholder-text">Loading...</div></div>`;

    try {
        const url = `/api/artifact/${state.currentJobId}/${encodeURIComponent(artifactPath)}`;
        const response = await fetch(url);
        const data = await response.json();

        if (data.error) {
            elements.artifactViewerContent.innerHTML = `
                <div class="artifact-placeholder"><div class="placeholder-text">Error: ${escapeHtml(data.error)}</div></div>`;
            showToast(`Failed to load artifact: ${data.error}`, 'error');
            return;
        }

        elements.artifactViewerContent.innerHTML = renderMarkdown(data.content);
    } catch (error) {
        elements.artifactViewerContent.innerHTML = `
            <div class="artifact-placeholder"><div class="placeholder-text">Failed to load artifact</div></div>`;
        showToast(`Error loading artifact: ${error.message}`, 'error');
    }
}

// ============================================================
// MARKDOWN RENDERER (XSS-safe)
// ============================================================
function renderMarkdown(raw) {
    // Escape first, then apply markdown transforms on the safe string
    let html = escapeHtml(raw)
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm,  '<h2>$1</h2>')
        .replace(/^# (.+)$/gm,   '<h1>$1</h1>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g,    '<em>$1</em>')
        .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
        .replace(/`([^`]+)`/g,    '<code>$1</code>')
        .replace(/^\s*[-*]\s+(.+)$/gm, '<li>$1</li>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');

    return `<div class="markdown-content"><p>${html}</p></div>`;
}

// ============================================================
// UTILITY: HTML ESCAPING
// ============================================================
function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function escapeAttr(str) {
    return escapeHtml(str);
}

// ============================================================
// JOB DETAILS
// ============================================================
async function fetchJobDetails(jobId) {
    try {
        const response = await fetch(`/api/job-status/${jobId}`);
        const job = await response.json();
        if (job.workdir || job.output_dir) {
            elements.resultWorkdir.textContent = job.workdir || 'N/A';
            elements.resultOutput.textContent  = job.output_dir || 'N/A';
            elements.jobResults.style.display  = 'block';
        }
    } catch { /* non-fatal */ }
}

// ============================================================
// DIRECTORY BROWSER
// ============================================================
let directoryBrowserSelectsFiles = false;

function openDirectoryBrowser(selectsFiles = false) {
    directoryBrowserSelectsFiles = selectsFiles;
    elements.directoryModal.style.display = 'flex';
    elements.directoryModal.classList.add('active');

    let startPath = state.currentPath || './';
    if (selectsFiles && elements.parbenchSpec.value) {
        const parts = elements.parbenchSpec.value.split('/');
        parts.pop();
        if (parts.length > 0) startPath = parts.join('/') || '/';
    } else if (!selectsFiles && elements.sourceDirectory.value) {
        startPath = elements.sourceDirectory.value;
    }

    state.currentPath = startPath;
    state.selectedPath = startPath;

    const titleEl = elements.directoryModal.querySelector('.modal-title');
    const btnEl   = elements.selectDirectory;
    if (selectsFiles) {
        titleEl.textContent = 'Browse JSON Spec Files';
        btnEl.textContent   = 'Select File';
    } else {
        titleEl.textContent = 'Browse Directories';
        btnEl.textContent   = 'Select Directory';
    }

    loadDirectory(state.currentPath);
}

function closeDirectoryBrowser() {
    elements.directoryModal.style.display = 'none';
    elements.directoryModal.classList.remove('active');
}

function selectDirectory() {
    if (state.selectedPath) {
        if (directoryBrowserSelectsFiles) {
            elements.parbenchSpec.value = state.selectedPath;
        } else {
            elements.sourceDirectory.value = state.selectedPath;
        }
        updateCommandPreview();
        closeDirectoryBrowser();
    }
}

async function loadDirectory(path) {
    elements.directoryList.innerHTML = '<div class="directory-loading">Loading...</div>';
    try {
        const response = await fetch(`/api/browse-directory?path=${encodeURIComponent(path)}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Failed to load directory');
        elements.breadcrumb.textContent = data.current_path;
        state.currentPath = data.current_path;
        renderDirectoryItems(data.items, data.parent);
    } catch (error) {
        elements.directoryList.innerHTML = `<div class="directory-loading text-error">Error: ${escapeHtml(error.message)}</div>`;
    }
}

function renderDirectoryItems(items, parent) {
    elements.directoryList.innerHTML = '';
    if (parent) elements.directoryList.appendChild(createDirectoryItem('..', parent, true));

    const dirs  = items.filter(i => i.is_dir);
    let   files = items.filter(i => !i.is_dir);
    if (directoryBrowserSelectsFiles) {
        files = files.filter(i => i.name.toLowerCase().endsWith('.json'));
    } else {
        files = [];
    }

    dirs.forEach(d => elements.directoryList.appendChild(createDirectoryItem(d.name, d.path, true)));
    files.forEach(f => elements.directoryList.appendChild(createDirectoryItem(f.name, f.path, false)));

    if (!dirs.length && !files.length && !parent) {
        elements.directoryList.innerHTML = '<div class="directory-loading">Empty directory</div>';
    }
}

function createDirectoryItem(name, path, isDir) {
    // Selectable = can be chosen as the final value (highlighted on click)
    const isSelectable = directoryBrowserSelectsFiles ? !isDir : (isDir && name !== '..');
    // Disabled = truly unclickable (files when browsing for a directory, not navigable)
    const isDisabled = !isDir && !directoryBrowserSelectsFiles;

    const div = document.createElement('div');
    div.className = `directory-item${isDisabled ? ' directory-item-disabled' : ''}`;
    if (path === state.selectedPath && isSelectable) div.classList.add('selected');

    const icon = isDir ? (name === '..' ? '⤴️' : '📁') : '📄';
    div.innerHTML = `<span class="directory-icon">${icon}</span><span class="directory-name">${escapeHtml(name)}</span>`;

    div.addEventListener('click', () => {
        if (isDir) {
            state.selectedPath = path;
            loadDirectory(path);
        } else if (directoryBrowserSelectsFiles) {
            document.querySelectorAll('.directory-item').forEach(el => el.classList.remove('selected'));
            div.classList.add('selected');
            state.selectedPath = path;
        }
    });

    return div;
}

// ============================================================
// JOBS LIST MODAL
// ============================================================
async function openJobsList() {
    elements.jobsModal.style.display = 'flex';
    elements.jobsModal.classList.add('active');
    await loadJobsList();
}

function closeJobsModal() {
    elements.jobsModal.style.display = 'none';
    elements.jobsModal.classList.remove('active');
}

async function loadJobsList() {
    elements.jobsList.innerHTML = '<div class="directory-loading">Loading jobs...</div>';
    try {
        const response = await fetch('/api/jobs');
        const jobs = await response.json();

        if (!jobs.length) {
            elements.jobsList.innerHTML = '<div class="directory-loading">No jobs found</div>';
            return;
        }

        elements.jobsList.innerHTML = '';
        jobs.forEach(job => elements.jobsList.appendChild(createJobItem(job)));
    } catch (error) {
        elements.jobsList.innerHTML = `<div class="directory-loading text-error">Error: ${escapeHtml(error.message)}</div>`;
    }
}

function createJobItem(job) {
    const item = document.createElement('div');
    item.className = 'job-item';
    item.innerHTML = `
        <div class="job-item-header">
            <div class="job-item-id" title="${escapeAttr(job.id)}">ID: ${escapeHtml(job.id.slice(0, 8))}…</div>
            <div class="status-badge ${job.status}">${job.status}</div>
        </div>
        <div class="job-item-body">
            <div class="job-item-field">
                <div class="job-item-label">Translation</div>
                <div class="job-item-value">${escapeHtml(job.from_api)} → ${escapeHtml(job.to_api)}</div>
            </div>
            <div class="job-item-field">
                <div class="job-item-label">Created</div>
                <div class="job-item-value">${new Date(job.created_at).toLocaleString()}</div>
            </div>
            <div class="job-item-field">
                <div class="job-item-label">Source</div>
                <div class="job-item-value">${escapeHtml(job.source_directory || '')}</div>
            </div>
            ${job.workdir ? `<div class="job-item-field"><div class="job-item-label">Workdir</div><div class="job-item-value">${escapeHtml(job.workdir)}</div></div>` : ''}
        </div>
    `;

    item.addEventListener('click', async () => {
        await loadJobArtifacts(job);
        closeJobsModal();
    });

    return item;
}

async function loadJobArtifacts(job) {
    state.currentJobId = job.id;

    elements.jobStatusCard.style.display = 'block';
    elements.jobStatusCard.scrollIntoView({ behavior: 'smooth' });

    elements.jobId.textContent = job.id.slice(0, 8) + '…';
    elements.jobId.title = job.id;

    let translationText = `${job.from_api} → ${job.to_api}`;
    if (job.model) translationText += ` (${job.model})`;
    if (job.engine) translationText += ` [${job.engine}]`;
    elements.jobTranslation.textContent = translationText;
    elements.jobSource.textContent = job.source_directory || '';

    updateJobStatus(job.status, job);
    initializeStageStepper();

    document.querySelectorAll('.stage-step').forEach(s => s.classList.remove('active', 'completed'));
    document.querySelectorAll('.stage-connector').forEach(c => c.classList.remove('active'));

    try {
        const response = await fetch(`/api/job-progress/${job.id}`);
        const data = await response.json();

        if (data.artifacts && data.artifacts.length > 0) {
            updateArtifactsList(data.artifacts);
            elements.stageDescription.textContent = `Viewing artifacts for ${job.status} job`;
        } else {
            elements.artifactsList.innerHTML = `
                <div class="artifact-item empty">
                    <span class="artifact-icon">📝</span>
                    <span class="artifact-name">No artifacts found</span>
                </div>`;
            elements.stageDescription.textContent = 'No artifacts available for this job';
        }

        if (job.workdir || job.output_dir) {
            elements.resultWorkdir.textContent = job.workdir || 'N/A';
            elements.resultOutput.textContent  = job.output_dir || 'N/A';
            elements.jobResults.style.display  = 'block';
        } else {
            elements.jobResults.style.display = 'none';
        }
    } catch {
        elements.artifactsList.innerHTML = `
            <div class="artifact-item empty">
                <span class="artifact-icon">⚠️</span>
                <span class="artifact-name">Error loading artifacts</span>
            </div>`;
    }
}

// ============================================================
// KILL JOB
// ============================================================
async function handleKillJob() {
    const jobId = state.currentJobId || elements.jobId.title;
    if (!jobId) {
        showToast('No active job selected', 'error');
        return;
    }

    // Inline confirmation using a toast-style approach
    if (!window.confirm('Kill this job? This cannot be undone.')) return;

    elements.killJobBtn.disabled = true;
    elements.killJobBtn.innerHTML = '<span>⏳</span> Killing...';

    try {
        const response = await fetch(`/api/kill-job/${jobId}`, { method: 'POST' });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Failed to kill job');

        updateJobStatus('killed', null);
        if (state.eventSource) { state.eventSource.close(); state.eventSource = null; }
        showToast('Job killed', 'warning', 3000);
    } catch (error) {
        showToast(error.message, 'error');
        elements.killJobBtn.disabled = false;
        elements.killJobBtn.innerHTML = '<span>⏹</span> Kill Job';
    }
}

// ============================================================
// PARBENCH VERIFICATION
// ============================================================
async function handleParbenchVerify() {
    if (!state.currentJobId) return;

    const btn = elements.parbenchVerifyBtn;
    if (btn.disabled) return;
    const originalContent = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner" style="width:14px;height:14px;border-width:2px;display:inline-block;margin-right:8px;"></span> Verifying...';

    elements.parbenchResultsPanel.style.display = 'block';
    elements.parbenchResultsBody.innerHTML = `
        <div style="text-align:center; padding: var(--space-xl) 0;">
            <div class="spinner"></div>
            <div style="margin-top: var(--space-md); color: var(--text-muted);">
                Running ParBench verification…<br>
                <small>This may take a few minutes.</small>
            </div>
        </div>`;

    try {
        const res = await fetch(`/api/parbench-verify/${state.currentJobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config: 'correctness' }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Verification failed');
        renderParbenchResults(data);
    } catch (err) {
        elements.parbenchResultsBody.innerHTML = `
            <div style="text-align:center; color: var(--danger-color); padding: var(--space-xl) 0;">
                <div style="font-size: 2rem; margin-bottom: 1rem;">⚠️</div>
                <div>${escapeHtml(err.message || String(err))}</div>
            </div>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalContent;
    }
}

function renderParbenchResults(data) {
    let html = `
        <div style="display:flex; gap:var(--space-md); margin-bottom:var(--space-lg);">
            <div style="flex:1; background:var(--bg-surface); padding:var(--space-md); border-radius:var(--radius-md); text-align:center; border:1px solid var(--border-color);">
                <div style="font-size:2rem; font-weight:700; color:var(--primary-color);">${data.summary.total}</div>
                <div style="color:var(--text-muted); font-size:0.875rem;">Kernels Evaluated</div>
            </div>
            <div style="flex:1; background:var(--bg-surface); padding:var(--space-md); border-radius:var(--radius-md); text-align:center; border:1px solid var(--border-color);">
                <div style="font-size:2rem; font-weight:700; color:${data.summary.passed === data.summary.total ? 'var(--success-color)' : 'var(--warning-color)'};">${data.summary.passed}</div>
                <div style="color:var(--text-muted); font-size:0.875rem;">Passed</div>
            </div>
        </div>
        <div style="display:flex; flex-direction:column; gap:var(--space-md);">`;

    data.results.forEach(r => {
        const isPass = r.status === 'pass';
        const icon   = isPass ? '✅' : (r.status === 'timeout' ? '⏱️' : '❌');
        const color  = isPass ? 'var(--success-color)' : 'var(--danger-color)';
        html += `
            <div style="border:1px solid var(--border-color); border-radius:var(--radius-md); overflow:hidden;">
                <div style="padding:var(--space-md); background:var(--bg-surface); display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border-color);">
                    <strong style="font-family:var(--font-mono);">${escapeHtml(r.kernel)}</strong>
                    <span style="color:${color}; font-weight:600; display:flex; align-items:center; gap:6px;">${icon} ${escapeHtml(r.status.toUpperCase())}</span>
                </div>`;

        if (r.verification) {
            html += `<div style="padding:var(--space-md); font-size:0.875rem; color:var(--text-muted);">
                <div><strong>Strategy:</strong> ${escapeHtml(r.verification.strategy_used)}</div>
                <div style="margin-top:4px;"><strong>Details:</strong> ${escapeHtml(r.verification.details || 'None')}</div>
            </div>`;
        } else if (r.error || r.stderr) {
            html += `<div style="padding:var(--space-md); font-size:0.875rem; color:var(--danger-color); font-family:var(--font-mono); white-space:pre-wrap; overflow-x:auto;">${escapeHtml(r.error || r.stderr)}</div>`;
        }

        if (r.metrics && r.metrics.length > 0) {
            html += `<div style="padding:var(--space-sm) var(--space-md); background:rgba(0,0,0,0.1); border-top:1px solid var(--border-color); font-size:0.875rem;">
                <strong>Metrics:</strong>
                <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:6px;">
                    ${r.metrics.map(m => `<span style="background:var(--bg-elevated); padding:2px 8px; border-radius:4px; font-family:var(--font-mono); border:1px solid var(--border-color);">${escapeHtml(m.name)}: ${escapeHtml(String(m.value))} ${escapeHtml(m.unit)}</span>`).join('')}
                </div>
            </div>`;
        }
        html += '</div>';
    });

    html += '</div>';
    elements.parbenchResultsBody.innerHTML = html;
}

// ============================================================
// PERFORMANCE ANALYSIS
// ============================================================
async function handleAnalyzePerformance() {
    if (!state.currentJobId) return;

    elements.performanceModal.style.display = 'flex';
    elements.performanceContent.innerHTML = `
        <div style="text-align:center; padding:40px;">
            <div class="spinner" style="width:32px;height:32px;border-width:3px; margin:0 auto;"></div>
            <p style="margin-top:16px; color:var(--text-secondary);">Running nsys profiling…</p>
        </div>`;

    try {
        const response = await fetch(`/api/analyze-performance/${state.currentJobId}`, { method: 'POST' });
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        renderPerformanceResults(data);
    } catch (error) {
        elements.performanceContent.innerHTML = `
            <div style="color:var(--danger-color); padding:var(--space-lg);">
                <strong>Analysis Failed:</strong> ${escapeHtml(error.message)}
            </div>`;
    }
}

function renderPerformanceResults(data) {
    const orig  = data.original   || {};
    const trans = data.translated || {};
    const fmt   = ms => (ms != null ? ms.toFixed(3) + ' ms' : '–');

    let html = '';

    if (data.speedup) {
        const speedup = parseFloat(data.speedup).toFixed(2);
        const faster  = speedup >= 1.0;
        const color   = faster ? 'var(--success-color)' : 'var(--danger-color)';
        html += `
            <div style="background:${faster ? 'hsla(142,71%,45%,0.1)' : 'hsla(0,84%,60%,0.1)'}; color:${color}; border:1px solid ${color}; padding:var(--space-lg); border-radius:var(--radius-md); text-align:center; margin-bottom:var(--space-lg);">
                <div style="font-size:2.5rem; font-weight:700;">${speedup}×</div>
                <div>${faster ? 'Translation is faster' : 'Original is faster'}</div>
            </div>`;
    } else {
        html += `
            <div style="background:hsla(45,100%,51%,0.1); color:var(--warning-color); border:1px solid var(--warning-color); padding:var(--space-lg); border-radius:var(--radius-md); text-align:center; margin-bottom:var(--space-lg);">
                <div style="font-weight:700;">Comparison Unavailable</div>
                <div style="font-size:0.875rem; margin-top:4px;">Could not calculate speedup – missing profiling data</div>
            </div>`;
    }

    html += `
        <table style="width:100%; border-collapse:collapse;">
            <thead>
                <tr style="background:var(--bg-tertiary);">
                    <th style="padding:10px; text-align:left; border:1px solid var(--border-color);">Metric</th>
                    <th style="padding:10px; text-align:left; border:1px solid var(--border-color);">Original (${escapeHtml(orig.api || '–')})</th>
                    <th style="padding:10px; text-align:left; border:1px solid var(--border-color);">Translated (${escapeHtml(trans.api || '–')})</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding:10px; border:1px solid var(--border-color);"><strong>GPU Time</strong></td>
                    <td style="padding:10px; border:1px solid var(--border-color); font-family:monospace;">${fmt(orig.gpu_time_ms)}</td>
                    <td style="padding:10px; border:1px solid var(--border-color); font-family:monospace;">${fmt(trans.gpu_time_ms)}</td>
                </tr>
                <tr>
                    <td style="padding:10px; border:1px solid var(--border-color);">Method</td>
                    <td style="padding:10px; border:1px solid var(--border-color);">${escapeHtml(orig.method || '–')}</td>
                    <td style="padding:10px; border:1px solid var(--border-color);">${escapeHtml(trans.method || '–')}</td>
                </tr>
                <tr>
                    <td style="padding:10px; border:1px solid var(--border-color);">Status</td>
                    <td style="padding:10px; border:1px solid var(--border-color);">${orig.success  ? '✅ Success' : '❌ Failed'}</td>
                    <td style="padding:10px; border:1px solid var(--border-color);">${trans.success ? '✅ Success' : '❌ Failed'}</td>
                </tr>
            </tbody>
        </table>`;

    elements.performanceContent.innerHTML = html;
}

// ============================================================
// TAB SWITCHING
// ============================================================
function switchTab(name) {
    state.activeTab = name;
    document.querySelectorAll('.main-tab').forEach(btn =>
        btn.classList.toggle('active', btn.dataset.tab === name)
    );
    document.querySelectorAll('.tab-panel').forEach(panel =>
        panel.style.display = panel.id === `tab-${name}` ? 'block' : 'none'
    );
    if (name === 'dashboard') loadDashboard();
    if (name === 'history')   loadHistory();
    if (name === 'traces' && !state.tracesLoaded) { loadTracesList(); state.tracesLoaded = true; }
    if (name === 'skills'  && !state.skillsLoaded) { loadSkills();    state.skillsLoaded = true; }
}

// ============================================================
// DASHBOARD TAB
// ============================================================
async function loadDashboard() {
    const el = document.getElementById('dashboardContent');
    el.innerHTML = '<div class="tab-loading">Loading...</div>';
    try {
        const res  = await fetch('/api/dashboard/stats');
        const data = await res.json();
        el.innerHTML = renderDashboard(data);
    } catch (e) {
        el.innerHTML = `<div class="tab-loading" style="color:var(--danger-color)">Failed to load: ${escapeHtml(e.message)}</div>`;
    }
}

function renderDashboard(d) {
    const sc = d.status_counts || {};
    const running   = sc.running   || 0;
    const completed = sc.completed || 0;
    const failed    = sc.failed    || 0;
    const killed    = sc.killed    || 0;

    let html = `<div class="dash-stats-grid">
        <div class="dash-stat-card primary"><div class="stat-value">${d.total}</div><div class="stat-label">Total Jobs</div></div>
        <div class="dash-stat-card success"><div class="stat-value">${completed}</div><div class="stat-label">Completed</div></div>
        <div class="dash-stat-card danger"><div class="stat-value">${failed}</div><div class="stat-label">Failed</div></div>
        <div class="dash-stat-card warning"><div class="stat-value">${running}</div><div class="stat-label">Running</div></div>
        <div class="dash-stat-card"><div class="stat-value">${d.success_rate}%</div><div class="stat-label">Success Rate</div></div>
    </div>`;

    if (d.pairs && d.pairs.length) {
        const maxCount = Math.max(...d.pairs.map(p => p.total), 1);
        html += `<div class="dash-section-title">Translation Pairs</div>
        <table class="dash-pairs-table"><thead><tr>
            <th>Pair</th><th>Total</th><th>Completed</th><th>Rate</th><th style="min-width:120px;"></th>
        </tr></thead><tbody>`;
        for (const p of d.pairs) {
            const rate = p.total > 0 ? Math.round(p.succeeded / p.total * 100) : 0;
            const w = Math.round(p.total / maxCount * 100);
            html += `<tr>
                <td><code style="font-size:0.8125rem;">${escapeHtml(p.pair)}</code></td>
                <td>${p.total}</td>
                <td>${p.succeeded}</td>
                <td>${rate}%</td>
                <td><div class="dash-bar-track"><div class="dash-bar-fill${rate >= 70 ? ' success' : ''}" style="width:${w}%"></div></div></td>
            </tr>`;
        }
        html += '</tbody></table>';
    }

    if (d.recent && d.recent.length) {
        html += `<div class="dash-section-title" style="margin-top:var(--space-xl);">Recent Jobs</div>
        <table class="dash-recent-table"><thead><tr>
            <th>Date</th><th>Kernel</th><th>Translation</th><th>Engine</th><th>Status</th><th>Duration</th>
        </tr></thead><tbody>`;
        for (const j of d.recent) {
            const badge  = statusBadgeHtml(j.status);
            const dur    = j.duration_seconds != null ? formatDuration(j.duration_seconds) : '—';
            const date   = j.created_at ? j.created_at.slice(0, 16).replace('T', ' ') : '—';
            const kernel = escapeHtml(j.kernel_name || '—');
            const jDash = escapeAttr(JSON.stringify(j));
            html += `<tr style="cursor:pointer;" onclick="openJobDetail(${jDash})" title="View job details">
                <td style="color:var(--text-tertiary);white-space:nowrap;">${escapeHtml(date)}</td>
                <td style="font-size:0.8rem;">${kernel}</td>
                <td><code style="font-size:0.8rem;">${escapeHtml(j.from_api)}→${escapeHtml(j.to_api)}</code></td>
                <td style="color:var(--text-tertiary);font-size:0.8rem;">${escapeHtml(j.engine || 'codex')}</td>
                <td>${badge}</td>
                <td style="color:var(--text-tertiary);">${dur}</td>
            </tr>`;
        }
        html += '</tbody></table>';
    }

    return html;
}

// ============================================================
// HISTORY TAB
// ============================================================
async function loadHistory() {
    const tbody = document.getElementById('historyTbody');
    const empty = document.getElementById('historyEmpty');
    tbody.innerHTML = '<tr><td colspan="7" class="tab-loading">Loading...</td></tr>';
    empty.style.display = 'none';

    const search = document.getElementById('historySearch').value;
    const status = document.getElementById('historyStatusFilter').value;
    const params = new URLSearchParams({ search, status, limit: 100 });

    try {
        const res  = await fetch(`/api/history?${params}`);
        const data = await res.json();
        renderHistoryTable(data.jobs || []);
    } catch (e) {
        tbody.innerHTML = `<tr><td colspan="7" style="color:var(--danger-color);padding:var(--space-lg);">${escapeHtml(e.message)}</td></tr>`;
    }
}

function renderHistoryTable(jobs) {
    const tbody = document.getElementById('historyTbody');
    const empty = document.getElementById('historyEmpty');

    if (!jobs.length) {
        tbody.innerHTML = '';
        empty.style.display = 'block';
        return;
    }
    empty.style.display = 'none';

    tbody.innerHTML = jobs.map(j => {
        const date    = j.created_at ? j.created_at.slice(0, 16).replace('T', ' ') : '—';
        const dur     = j.duration_seconds != null ? formatDuration(j.duration_seconds) : '—';
        const kernel  = escapeHtml(j.kernel_name || '—');
        const pair    = `${escapeHtml(j.from_api || '?')}→${escapeHtml(j.to_api || '?')}`;
        const supBadge      = j.supervise ? ' <span title="Ran with supervisor" style="font-size:0.65rem;background:hsla(142,71%,45%,0.15);color:var(--success-color);border:1px solid hsla(142,71%,45%,0.3);border-radius:8px;padding:1px 5px;">✅ sup</span>' : '';
        const baselineBadge = j.baseline  ? ' <span title="Baseline mode (single-session)" style="font-size:0.65rem;background:hsla(210,90%,50%,0.12);color:#4da6ff;border:1px solid hsla(210,90%,50%,0.3);border-radius:8px;padding:1px 5px;">⚡ base</span>' : '';
        const eng     = escapeHtml(j.engine || 'codex');
        const badge   = statusBadgeHtml(j.status);
        const canView = true;
        const jStr    = escapeAttr(JSON.stringify(j));
        return `<tr>
            <td style="white-space:nowrap;color:var(--text-tertiary);font-size:0.8125rem;">${escapeHtml(date)}</td>
            <td><span style="font-size:0.8125rem;font-weight:500;">${kernel}</span></td>
            <td><code style="font-size:0.8rem;">${pair}</code></td>
            <td style="font-size:0.8rem;color:var(--text-tertiary);">${eng}${baselineBadge}${supBadge}</td>
            <td>${badge}</td>
            <td style="color:var(--text-tertiary);">${dur}</td>
            <td>
                <div class="history-actions">
                    <button class="btn-xs btn-xs-primary" onclick="openJobDetail(${jStr})">👁 View</button>
                    <button class="btn-xs btn-xs-primary" onclick="rerunJob(${jStr})">↺ Re-run</button>
                    <button class="btn-xs btn-xs-danger" onclick="deleteJob('${escapeAttr(j.id)}')">🗑</button>
                </div>
            </td>
        </tr>`;
    }).join('');
}

function rerunJob(job) {
    // Switch to pipeline tab and pre-fill the form
    switchTab('pipeline');
    const isParbench = !!job.parbench_spec;

    // Set mode
    const modeRadio = document.querySelector(`input[name="inputMode"][value="${isParbench ? 'parbench' : 'custom'}"]`);
    if (modeRadio) { modeRadio.checked = true; modeRadio.dispatchEvent(new Event('change')); }

    if (isParbench) {
        elements.parbenchSpec.value = job.parbench_spec || '';
    } else {
        elements.sourceDirectory.value = job.source_directory || '';
    }

    if (elements.fromApi && job.from_api) elements.fromApi.value = job.from_api;
    if (elements.toApi   && job.to_api)   elements.toApi.value   = job.to_api;
    elements.model.value = job.model || '';

    // Set engine radio
    if (job.engine) {
        const engRadio = document.querySelector(`input[name="engine"][value="${job.engine}"]`);
        if (engRadio) { engRadio.checked = true; engRadio.dispatchEvent(new Event('change')); }
    } else {
        document.querySelectorAll('input[name="engine"]').forEach(r => r.checked = false);
    }

    // Restore baseline and supervisor toggles
    const baselineEl = document.getElementById('baselineToggle');
    if (baselineEl) baselineEl.checked = !!job.baseline;
    const supEl = document.getElementById('supervisorToggle');
    if (supEl) supEl.checked = !!job.supervise;

    updateCommandPreview();
    elements.pipelineForm.scrollIntoView({ behavior: 'smooth' });
    showToast(`Form pre-filled from job ${job.id.slice(0, 8)}`, 'success', 3000);
}

async function downloadLog(jobId) {
    try {
        const res = await fetch(`/api/logs-download/${jobId}`);
        if (!res.ok) {
            const d = await res.json();
            showToast(d.error || 'Log not available', 'warning', 5000);
            return;
        }
        const blob = await res.blob();
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href = url; a.download = `paracodex-${jobId.slice(0, 8)}.log`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (e) {
        showToast(e.message, 'error');
    }
}

async function openJobArtifacts(job) {
    // Switch to pipeline tab and load this job's artifacts
    switchTab('pipeline');
    state.activeJobs.set(job.id, job);
    state.currentJobId = job.id;
    renderJobTabs();
    await loadJobArtifacts(job);
    elements.jobTabsContainer.style.display = 'block';
}

async function deleteJob(jobId) {
    if (!window.confirm('Delete this job record? This cannot be undone.')) return;
    try {
        const res = await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Delete failed');
        showToast('Job deleted', 'success', 3000);
        loadHistory();
    } catch (e) {
        showToast(e.message, 'error');
    }
}

// ============================================================
// JOB DETAIL MODAL
// ============================================================
let _jdCurrentJobId = null;
let _jdCurrentJob = null;  // full job object from /api/job-detail

async function openJobDetail(job) {
    _jdCurrentJobId = job.id || job.job_id;
    const modal = document.getElementById('jobDetailModal');
    if (!modal) return;

    // Set title and meta
    const kernel = job.kernel_name || '—';
    const pair   = `${job.from_api || '?'} → ${job.to_api || '?'}`;
    document.getElementById('jobDetailTitle').textContent = `${kernel}  ·  ${pair}`;
    const dur = job.duration_seconds != null ? formatDuration(job.duration_seconds) : null;
    document.getElementById('jobDetailMeta').innerHTML =
        `<span>${escapeHtml(job.id ? job.id.slice(0, 8) : '—')}</span>` +
        `  ·  ${statusBadgeHtml(job.status)}` +
        `  ·  <span>${escapeHtml(job.engine || 'codex')}</span>` +
        (dur ? `  ·  <span>${dur}</span>` : '');

    // Hide action buttons until data loads
    const perfBtn   = document.getElementById('jdPerfBtn');
    const verifyBtn = document.getElementById('jdVerifyBtn');
    if (perfBtn)   perfBtn.style.display   = 'none';
    if (verifyBtn) verifyBtn.style.display = 'none';

    // Reset to artifacts tab
    switchJobDetailTab('artifacts');
    modal.style.display = 'flex';

    // Load detail from API
    try {
        const res  = await fetch(`/api/job-detail/${_jdCurrentJobId}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        _jdCurrentJob = data.job;
        _renderJdArtifacts(data.artifacts || [], _jdCurrentJobId);
        _renderJdTraces(data.traces || []);

        // Show action buttons for completed jobs
        const j = data.job || {};
        if (j.status === 'completed') {
            if (perfBtn && j.workdir && j.output_dir) perfBtn.style.display = 'inline-flex';
            if (verifyBtn && j.parbench_spec)         verifyBtn.style.display = 'inline-flex';
        }
    } catch (e) {
        document.getElementById('jdArtifactsList').innerHTML =
            `<div class="artifact-item empty"><span class="artifact-name" style="color:var(--danger-color);">${escapeHtml(e.message)}</span></div>`;
    }
}

async function _jdRunPerformance() {
    const btn = document.getElementById('jdPerfBtn');
    const titleEl = document.getElementById('resultsModalTitle');
    const bodyEl  = document.getElementById('resultsModalBody');
    const modal   = document.getElementById('resultsModal');
    if (!_jdCurrentJobId || !modal) return;
    btn.disabled = true;
    btn.textContent = '⏳ Running…';
    titleEl.textContent = '📊 Performance Analysis';
    bodyEl.innerHTML = '⏳ Running performance analysis — this may take a minute…';
    modal.style.display = 'flex';
    try {
        const res  = await fetch(`/api/analyze-performance/${_jdCurrentJobId}`, { method: 'POST' });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        const orig  = data.original   || {};
        const trans = data.translated || {};
        const fmt   = ms => (ms != null ? ms.toFixed(3) + ' ms' : '—');
        let html = '';
        if (data.speedup != null) {
            const speedup = parseFloat(data.speedup).toFixed(2);
            const faster  = speedup >= 1.0;
            const color   = faster ? 'var(--success-color)' : 'var(--danger-color)';
            html += `<div style="background:${faster ? 'hsla(142,71%,45%,0.1)' : 'hsla(0,84%,60%,0.1)'};color:${color};border:1px solid ${color};padding:var(--space-lg);border-radius:var(--radius-md);text-align:center;margin-bottom:var(--space-lg);">
                <div style="font-size:2.5rem;font-weight:700;">${speedup}×</div>
                <div>${faster ? 'Translation is faster' : 'Original is faster'}</div>
            </div>`;
        } else {
            html += `<div style="background:hsla(45,100%,51%,0.1);color:var(--warning-color);border:1px solid var(--warning-color);padding:var(--space-lg);border-radius:var(--radius-md);text-align:center;margin-bottom:var(--space-lg);">
                <div style="font-weight:700;">Comparison Unavailable</div>
                <div style="font-size:0.875rem;margin-top:4px;">Could not calculate speedup — missing profiling data</div>
            </div>`;
        }
        html += `<table style="width:100%;border-collapse:collapse;">
            <thead><tr style="background:var(--bg-tertiary);">
                <th style="padding:10px;text-align:left;border:1px solid var(--border-color);">Metric</th>
                <th style="padding:10px;text-align:left;border:1px solid var(--border-color);">Original (${escapeHtml(orig.api || '–')})</th>
                <th style="padding:10px;text-align:left;border:1px solid var(--border-color);">Translated (${escapeHtml(trans.api || '–')})</th>
            </tr></thead>
            <tbody>
                <tr>
                    <td style="padding:10px;border:1px solid var(--border-color);"><strong>GPU Time</strong></td>
                    <td style="padding:10px;border:1px solid var(--border-color);font-family:monospace;">${fmt(orig.gpu_time_ms)}</td>
                    <td style="padding:10px;border:1px solid var(--border-color);font-family:monospace;">${fmt(trans.gpu_time_ms)}</td>
                </tr>
                <tr>
                    <td style="padding:10px;border:1px solid var(--border-color);">Method</td>
                    <td style="padding:10px;border:1px solid var(--border-color);">${escapeHtml(orig.method || '–')}</td>
                    <td style="padding:10px;border:1px solid var(--border-color);">${escapeHtml(trans.method || '–')}</td>
                </tr>
                <tr>
                    <td style="padding:10px;border:1px solid var(--border-color);">Status</td>
                    <td style="padding:10px;border:1px solid var(--border-color);">${orig.success  ? '✅ Success' : '❌ Failed'}</td>
                    <td style="padding:10px;border:1px solid var(--border-color);">${trans.success ? '✅ Success' : '❌ Failed'}</td>
                </tr>
            </tbody>
        </table>`;
        bodyEl.innerHTML = html;
    } catch (e) {
        bodyEl.innerHTML = `<span style="color:var(--danger-color);">Performance analysis failed: ${escapeHtml(e.message)}</span>`;
    } finally {
        btn.disabled = false;
        btn.textContent = '📊 Performance';
    }
}

async function _jdRunParbenchVerify() {
    const btn     = document.getElementById('jdVerifyBtn');
    const titleEl = document.getElementById('resultsModalTitle');
    const bodyEl  = document.getElementById('resultsModalBody');
    const modal   = document.getElementById('resultsModal');
    if (!_jdCurrentJobId || !modal) return;
    btn.disabled = true;
    btn.textContent = '⏳ Verifying…';
    titleEl.textContent = '✅ ParBench Verification';
    bodyEl.innerHTML = '⏳ Running verification…';
    modal.style.display = 'flex';
    const preStyle = 'style="background:var(--bg-primary);border:1px solid var(--border-color);border-radius:4px;padding:8px;font-size:0.75rem;white-space:pre-wrap;word-break:break-word;max-height:400px;overflow-y:auto;"';
    try {
        const res  = await fetch(`/api/parbench-verify/${_jdCurrentJobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config: 'correctness' }),
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        const sum = data.summary || {};
        let html = `<div style="display:flex;gap:var(--space-md);margin-bottom:var(--space-lg);">
            <div style="flex:1;background:var(--bg-surface);padding:var(--space-md);border-radius:var(--radius-md);text-align:center;border:1px solid var(--border-color);">
                <div style="font-size:2rem;font-weight:700;color:var(--primary-color);">${sum.total}</div>
                <div style="color:var(--text-muted);font-size:0.875rem;">Total</div>
            </div>
            <div style="flex:1;background:var(--bg-surface);padding:var(--space-md);border-radius:var(--radius-md);text-align:center;border:1px solid var(--border-color);">
                <div style="font-size:2rem;font-weight:700;color:${sum.passed === sum.total ? 'var(--success-color)' : 'var(--warning-color)'};">${sum.passed}</div>
                <div style="color:var(--text-muted);font-size:0.875rem;">Passed</div>
            </div>
            <div style="flex:1;background:var(--bg-surface);padding:var(--space-md);border-radius:var(--radius-md);text-align:center;border:1px solid var(--border-color);">
                <div style="font-size:2rem;font-weight:700;color:${sum.failed > 0 ? 'var(--danger-color)' : 'var(--text-tertiary)'};">${sum.failed}</div>
                <div style="color:var(--text-muted);font-size:0.875rem;">Failed</div>
            </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:var(--space-md);">`;
        for (const r of (data.results || [])) {
            const isPass = r.status === 'pass';
            const icon   = isPass ? '✅' : (r.status === 'timeout' ? '⏱️' : '❌');
            const color  = isPass ? 'var(--success-color)' : 'var(--danger-color)';
            html += `<div style="border:1px solid var(--border-color);border-radius:var(--radius-md);overflow:hidden;">
                <div style="padding:var(--space-md);background:var(--bg-surface);display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border-color);">
                    <strong style="font-family:var(--font-mono);">${escapeHtml(r.kernel || '')}</strong>
                    <span style="color:${color};font-weight:600;">${icon} ${escapeHtml(r.status.toUpperCase())}</span>
                </div>`;
            if (r.stdout) {
                html += `<div style="padding:var(--space-sm) var(--space-md);">
                    <div style="font-size:0.75rem;color:var(--text-tertiary);margin-bottom:4px;">stdout</div>
                    <pre ${preStyle}>${escapeHtml(r.stdout)}</pre>
                </div>`;
            }
            if (r.stderr) {
                html += `<div style="padding:var(--space-sm) var(--space-md);">
                    <div style="font-size:0.75rem;color:var(--text-tertiary);margin-bottom:4px;">stderr</div>
                    <pre ${preStyle}>${escapeHtml(r.stderr)}</pre>
                </div>`;
            }
            html += '</div>';
        }
        html += '</div>';
        bodyEl.innerHTML = html;
    } catch (e) {
        bodyEl.innerHTML = `<span style="color:var(--danger-color);">Verification failed: ${escapeHtml(e.message)}</span>`;
    } finally {
        btn.disabled = false;
        btn.textContent = '✅ Verify';
    }
}

function switchJobDetailTab(name) {
    document.querySelectorAll('.job-detail-tab').forEach(btn =>
        btn.classList.toggle('active', btn.dataset.tab === name)
    );
    document.querySelectorAll('.job-detail-pane').forEach(pane =>
        pane.style.display = pane.id === `jd-pane-${name}` ? (name === 'artifacts' || name === 'traces' ? 'flex' : 'block') : 'none'
    );
    if (name === 'info') _renderJdInfo(_jdCurrentJob);
}

function _renderJdArtifacts(artifacts, jobId) {
    const list = document.getElementById('jdArtifactsList');
    if (!artifacts.length) {
        list.innerHTML = '<div class="artifact-item empty"><span class="artifact-name">No artifacts found</span></div>';
        return;
    }
    const icons = { analysis: '🔍', plan: '📋', report: '📊', document: '📄' };
    list.innerHTML = artifacts.map(a => `
        <div class="artifact-item" data-path="${escapeAttr(a.path)}" data-name="${escapeAttr(a.name)}">
            <span class="artifact-icon">${icons[a.type] || '📄'}</span>
            <span class="artifact-name" title="${escapeAttr(a.name)}">${escapeHtml(a.name)}</span>
        </div>
    `).join('');
    list.querySelectorAll('.artifact-item:not(.empty)').forEach(item => {
        item.addEventListener('click', () => _viewJdArtifact(item.dataset.path, item.dataset.name, jobId));
    });
}

async function _viewJdArtifact(path, name, jobId) {
    document.querySelectorAll('#jdArtifactsList .artifact-item').forEach(i =>
        i.classList.toggle('active', i.dataset.path === path)
    );
    document.getElementById('jdArtifactTitle').textContent = name;
    const content = document.getElementById('jdArtifactContent');
    content.innerHTML = '<div class="artifact-placeholder"><div class="placeholder-text">Loading...</div></div>';
    try {
        const res  = await fetch(`/api/artifact/${jobId}/${encodeURIComponent(path)}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        content.innerHTML = renderMarkdown(data.content);
    } catch (e) {
        content.innerHTML = `<div class="artifact-placeholder"><div class="placeholder-text" style="color:var(--danger-color);">${escapeHtml(e.message)}</div></div>`;
    }
}

function _renderJdTraces(traces) {
    const list = document.getElementById('jdTracesList');
    if (!traces.length) {
        list.innerHTML = '<div class="traces-sidebar-header">Traces</div><div class="tab-loading">No traces for this job.</div>';
        return;
    }
    list.innerHTML = '<div class="traces-sidebar-header">Traces</div>' +
        traces.map(t => {
            const kb = t.size > 0 ? `${(t.size/1024).toFixed(1)} KB` : '';
            const displayName = t.label || ('📄 ' + t.id.slice(0, 22) + '…');
            return `<div class="trace-item" data-id="${escapeAttr(t.id)}"
                onclick="_loadJdTrace('${escapeAttr(t.id)}', this)">
                <div class="trace-item-name">${escapeHtml(displayName)}</div>
                <div class="trace-item-meta">${kb}</div>
            </div>`;
        }).join('');
}

async function _loadJdTrace(traceId, itemEl) {
    document.querySelectorAll('#jdTracesList .trace-item').forEach(el => el.classList.remove('active'));
    if (itemEl) itemEl.classList.add('active');
    const viewer = document.getElementById('jdTracesViewer');
    viewer.innerHTML = '<div class="tab-loading">Loading...</div>';
    try {
        const res  = await fetch(`/api/traces/${encodeURIComponent(traceId)}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        let html = '';
        if (data.meta?.cwd) html += `<div class="traces-meta">cwd: ${escapeHtml(data.meta.cwd)}</div>`;
        html += '<div class="trace-events">';
        for (const ev of (data.events || [])) html += renderTraceEvent(ev);
        if (data.truncated) html += `<div style="text-align:center;color:var(--text-tertiary);font-size:0.8125rem;padding:var(--space-md);">— truncated at 300 events —</div>`;
        html += '</div>';
        viewer.innerHTML = html;
    } catch (e) {
        viewer.innerHTML = `<div class="tab-loading" style="color:var(--danger-color);">${escapeHtml(e.message)}</div>`;
    }
}

function _renderJdInfo(job) {
    const el = document.getElementById('jdInfoContent');
    if (!el || !job) return;
    const row = (label, val, mono) => val
        ? `<div style="display:flex;gap:8px;align-items:flex-start;font-size:0.875rem;">
             <span style="min-width:130px;color:var(--text-tertiary);flex-shrink:0;">${label}</span>
             <span style="${mono ? 'font-family:monospace;word-break:break-all;' : ''}">${escapeHtml(String(val))}</span>
           </div>`
        : '';
    const fmtDate = s => s ? new Date(s).toLocaleString() : null;
    el.innerHTML = [
        row('Kernel', job.kernel_name),
        row('Translation', job.from_api && job.to_api ? `${job.from_api} → ${job.to_api}` : null),
        row('Mode', job.baseline ? '⚡ Baseline (single-session)' : 'Standard (multi-step)'),
        row('Engine', job.engine || 'codex'),
        row('Model', job.model),
        row('Status', job.status),
        row('Created', fmtDate(job.created_at)),
        row('Completed', fmtDate(job.completed_at)),
        row('Exit code', job.exit_code != null ? String(job.exit_code) : null),
        row('Error', job.error),
        '<hr style="border:none;border-top:1px solid var(--border-color);margin:8px 0;">',
        row('Work dir', job.workdir, true),
        row('Output dir', job.output_dir, true),
        row('ParBench spec', job.parbench_spec, true),
    ].filter(Boolean).join('');
}

// ============================================================
// TRACES TAB
// ============================================================
async function loadTracesList() {
    const el = document.getElementById('tracesList');
    try {
        const res  = await fetch('/api/traces/by-job');
        const data = await res.json();
        if (!data.jobs || !data.jobs.length) {
            el.innerHTML = '<div class="tab-loading">No traces yet.</div>';
            return;
        }
        el.innerHTML = data.jobs.map((job, i) => {
            const kernel = escapeHtml(job.kernel_name || job.job_id.slice(0, 8));
            const pair   = `${escapeHtml(job.from_api || '?')}→${escapeHtml(job.to_api || '?')}`;
            const date   = job.created_at ? job.created_at.slice(0, 16).replace('T', ' ') : '';
            const statusCls = job.status === 'completed' ? 'success' : job.status === 'failed' ? 'danger' : '';
            const tracesHtml = job.traces.map(t => {
                const kb = t.size > 0 ? ` · ${(t.size/1024).toFixed(1)} KB` : '';
                const displayName = t.label || ('📄 ' + t.id.slice(0, 20) + '…');
                return `<div class="trace-item trace-item-child" data-id="${escapeAttr(t.id)}"
                    onclick="loadTrace('${escapeAttr(t.id)}', this)">
                    <div class="trace-item-name">${escapeHtml(displayName)}</div>
                    <div class="trace-item-meta">${escapeHtml(t.filename.split('-')[1] || '')}${kb}</div>
                </div>`;
            }).join('');
            return `<div class="trace-job-group">
                <div class="trace-job-header" onclick="this.parentElement.classList.toggle('expanded')">
                    <div class="trace-job-title">
                        <span class="trace-job-kernel">${kernel}</span>
                        <code class="trace-job-pair">${pair}</code>
                    </div>
                    <div class="trace-job-meta">
                        <span class="trace-status-dot ${statusCls}"></span>
                        <span>${date}</span>
                        <span class="trace-job-count">${job.traces.length}</span>
                        <span class="trace-job-chevron">▶</span>
                    </div>
                </div>
                <div class="trace-job-traces">${tracesHtml}</div>
            </div>`;
        }).join('');
    } catch (e) {
        el.innerHTML = `<div class="tab-loading" style="color:var(--danger-color);">${escapeHtml(e.message)}</div>`;
    }
}

async function loadTrace(traceId, itemEl) {
    document.querySelectorAll('.trace-item').forEach(el => el.classList.remove('active'));
    if (itemEl) itemEl.classList.add('active');

    const viewer = document.getElementById('tracesViewer');
    viewer.innerHTML = '<div class="tab-loading">Loading trace...</div>';

    try {
        const res  = await fetch(`/api/traces/${encodeURIComponent(traceId)}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        let html = '';
        if (data.meta) {
            html += `<div class="traces-meta">ID: ${escapeHtml(data.meta.id || traceId)}`;
            if (data.meta.cwd) html += ` · cwd: ${escapeHtml(data.meta.cwd)}`;
            html += `</div>`;
        }

        html += '<div class="trace-events">';
        for (const ev of (data.events || [])) {
            html += renderTraceEvent(ev);
        }
        if (data.truncated) {
            html += `<div style="text-align:center;color:var(--text-tertiary);font-size:0.8125rem;padding:var(--space-md);">— truncated at 300 events —</div>`;
        }
        html += '</div>';

        viewer.innerHTML = html;
    } catch (e) {
        viewer.innerHTML = `<div class="tab-loading" style="color:var(--danger-color);">${escapeHtml(e.message)}</div>`;
    }
}

function renderTraceEvent(ev) {
    switch (ev.type) {
        case 'user':
            return `<div class="trace-event trace-event-user">${escapeHtml(ev.text)}</div>`;
        case 'assistant':
            return `<div class="trace-event trace-event-assistant">${renderMarkdown(ev.text)}</div>`;
        case 'reasoning':
            return `<div class="trace-event trace-event-reasoning">💭 ${escapeHtml(ev.text)}</div>`;
        case 'tool_call':
            return `<div class="trace-event trace-event-tool"><details>
                <summary>🔧 ${escapeHtml(ev.name)}</summary>
                <pre>${escapeHtml(ev.args)}</pre>
            </details></div>`;
        case 'tool_result':
            return `<div class="trace-event trace-event-result"><details>
                <summary>↩ result</summary>
                <pre>${escapeHtml(ev.output)}</pre>
            </details></div>`;
        default:
            return '';
    }
}

// ============================================================
// SKILLS TAB
// ============================================================
async function loadSkills() {
    const grid = document.getElementById('skillsGrid');
    try {
        const res  = await fetch('/api/skills');
        const data = await res.json();
        if (!data.skills || !data.skills.length) {
            grid.innerHTML = '<div class="tab-loading">No skills found.</div>';
            return;
        }
        grid.innerHTML = data.skills.map(s => {
            const badges = [];
            if (s.compatibility) badges.push(`<span class="skill-badge">${escapeHtml(s.compatibility)}</span>`);
            if (s.has_examples)   badges.push('<span class="skill-badge green">examples</span>');
            if (s.has_scripts)    badges.push('<span class="skill-badge green">scripts</span>');
            return `<div class="skill-card" onclick="openSkillModal(${escapeAttr(JSON.stringify(s))})">
                <div class="skill-card-name">${escapeHtml(s.name)}</div>
                <div class="skill-card-desc">${escapeHtml(s.description || 'No description')}</div>
                <div class="skill-card-badges">${badges.join('')}</div>
            </div>`;
        }).join('');
    } catch (e) {
        grid.innerHTML = `<div class="tab-loading" style="color:var(--danger-color);">${escapeHtml(e.message)}</div>`;
    }
}

function openSkillModal(skill) {
    document.getElementById('skillModalTitle').textContent = skill.name;
    document.getElementById('skillModalBody').innerHTML = renderMarkdown(skill.body || '');
    document.getElementById('skillModal').style.display = 'flex';
}

// ============================================================
// SHARED HELPERS (dashboard + history)
// ============================================================
function statusBadgeHtml(status) {
    const map = {
        completed: ['var(--success-color)',  '✓ Completed'],
        failed:    ['var(--danger-color)',   '✗ Failed'],
        running:   ['var(--primary-400)',    '⟳ Running'],
        killed:    ['var(--text-tertiary)',  '⏹ Killed'],
    };
    const [color, label] = map[status] || ['var(--text-tertiary)', status];
    return `<span style="color:${color};font-size:0.8125rem;font-weight:500;">${escapeHtml(label)}</span>`;
}

function formatDuration(seconds) {
    if (seconds == null || seconds < 0) return '—';
    if (seconds < 60)  return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}
