/**
 * TestWise Frontend Application Controller
 * Handles REST API communication, state management, and SVG chart rendering.
 */

(function () {
  'use strict';

  // State Store
  const state = {
    config: null,
    currentSessionId: null,
    currentEpisode: null,
    planData: null,
    evalData: null,
    curveData: null,
    selectedPlanNode: null,
    currentView: 'dashboard',
  };

  // Hypotheses and Investigation Colors
  const COLORS = {
    H1: '#0EA5E9',
    H2: '#F43F5E',
    H3: '#10B981',
    H4: '#8B5CF6',
    default: '#64748B',
  };

  const POLICY_NAMES_MAP = {
    fixed_order: 'Fixed Order',
    cheapest_first: 'Cheapest First',
    greedy_information_gain: 'Greedy Info Gain',
    contingent_optimal: 'Contingent Optimal',
  };

  // API Client Functions
  async function apiGet(endpoint) {
    const res = await fetch(endpoint);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'API request failed');
    }
    return res.json();
  }

  async function apiPost(endpoint, body = {}) {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'API request failed');
    }
    return res.json();
  }

  // Initialization
  async function init() {
    setupNavigation();
    setupEventListeners();

    try {
      state.config = await apiGet('/api/config');
      populateModelSpec();
      updateQuickStats();

      // Start initial episode
      await startNewEpisode();

      // Solve initial plan for preview
      solveContingentPlan();

      // Fetch algorithm lab curves
      fetchBudgetCurves();
    } catch (err) {
      console.error('Initialization error:', err);
      showToast('Backend connection error: ' + err.message, 'error');
    }
  }

  // Navigation Logic
  function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach((btn) => {
      btn.addEventListener('click', () => {
        const targetView = btn.dataset.view;
        switchView(targetView);
      });
    });

    document.getElementById('btn-banner-view-eval')?.addEventListener('click', () => {
      switchView('evaluation');
    });

    document.getElementById('btn-view-all-history')?.addEventListener('click', () => {
      switchView('workbench');
    });
  }

  function switchView(viewName) {
    state.currentView = viewName;
    document.querySelectorAll('.nav-item').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.view === viewName);
    });

    document.querySelectorAll('.view-section').forEach((sec) => {
      sec.classList.remove('active');
    });

    const activeSec = document.getElementById(`view-${viewName}`);
    if (activeSec) activeSec.classList.add('active');

    // Update Header
    const titles = {
      dashboard: ['Executive Command Center', 'Probabilistic reasoning under resource constraints'],
      workbench: ['Investigation Workbench', 'Sequential evidence collection and Bayesian belief updates'],
      'plan-inspector': ['Contingent-Plan Inspector', 'Outcome-dependent Bellman decision tree visualization'],
      'algorithm-lab': ['Algorithm Lab', 'Empirical comparison and theoretical trade-offs between policies'],
      evaluation: ['Evaluation & Comparison', 'Batch performance analysis across identical synthetic cases'],
      'ai-lab': ['AI Concepts Lab', 'Bayesian Networks (Exact DAG Enumeration) & Propositional Logic (AST Entailment)'],
      settings: ['Model Specification', 'Underlying hypothesis space and likelihood matrices'],
    };

    const [t, sub] = titles[viewName] || ['TestWise', 'Workbench'];
    document.getElementById('page-title').textContent = t;
    document.getElementById('page-subtitle').textContent = sub;

    // Render charts for the active view
    if (viewName === 'dashboard') {
      renderOverviewBeliefChart();
    } else if (viewName === 'workbench') {
      renderFullWorkbench();
    } else if (viewName === 'plan-inspector') {
      renderPlanTree(state.planData);
    } else if (viewName === 'algorithm-lab') {
      renderBudgetCurves(state.curveData);
    } else if (viewName === 'ai-lab') {
      initAiLab();
    }
  }

  // Setup Event Listeners
  function setupEventListeners() {
    // Episode Config & Controls
    document.getElementById('btn-start-episode')?.addEventListener('click', () => startNewEpisode());
    document.getElementById('btn-banner-new-episode')?.addEventListener('click', () => {
      switchView('dashboard');
      startNewEpisode();
    });

    document.getElementById('btn-config-reset')?.addEventListener('click', resetConfigForm);
    document.getElementById('btn-step-policy')?.addEventListener('click', () => stepPolicy());
    document.getElementById('btn-reset-episode')?.addEventListener('click', () => resetEpisode());

    // Workbench Full Controls
    document.getElementById('btn-full-step-policy')?.addEventListener('click', () => stepPolicy());
    document.getElementById('btn-full-run-complete')?.addEventListener('click', () => runToCompletion());
    document.getElementById('btn-full-reset')?.addEventListener('click', () => resetEpisode());

    // Workbench Tabs
    document.querySelectorAll('.panel-tab').forEach((tab) => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.panel-tab').forEach((t) => t.classList.remove('active'));
        tab.classList.add('active');

        const tabKey = tab.dataset.tab;
        document.querySelectorAll('.tab-content').forEach((tc) => tc.classList.add('hidden'));
        document.getElementById(`tab-${tabKey}`)?.classList.remove('hidden');
      });
    });

    // Plan Inspector Controls
    document.getElementById('btn-solve-plan-preview')?.addEventListener('click', () => solveContingentPlan());
    document.getElementById('btn-solve-full-plan')?.addEventListener('click', () => solveContingentPlan());

    // Evaluation Controls
    document.getElementById('btn-run-evaluation')?.addEventListener('click', () => runEvaluation());
    document.getElementById('btn-export-csv')?.addEventListener('click', () => exportEvaluationCSV());
    document.getElementById('eval-episode-filter-policy')?.addEventListener('change', () => filterEvaluationEpisodes());

    // Algorithm Lab Controls
    document.getElementById('btn-refresh-curves')?.addEventListener('click', () => fetchBudgetCurves());
  }

  // Episode Management
  async function startNewEpisode() {
    const seed = parseInt(document.getElementById('cfg-seed').value, 10) || 42;
    const budget = parseFloat(document.getElementById('cfg-budget').value) || 10.0;
    const policy = document.getElementById('cfg-policy').value || 'greedy_information_gain';
    const threshold = parseFloat(document.getElementById('cfg-threshold').value) || 0.90;

    try {
      const ep = await apiPost('/api/episode/start', {
        seed,
        budget,
        policy,
        confidence_threshold: threshold,
      });

      state.currentSessionId = ep.session_id;
      state.currentEpisode = ep;

      updateEpisodeUI();
      showToast(`Episode started with seed ${seed} under ${POLICY_NAMES_MAP[policy] || policy}.`);
    } catch (err) {
      showToast('Failed to start episode: ' + err.message, 'error');
    }
  }

  async function stepPolicy() {
    if (!state.currentSessionId) return;
    try {
      const ep = await apiPost('/api/episode/step', { session_id: state.currentSessionId });
      state.currentEpisode = ep;
      updateEpisodeUI();

      if (ep.is_terminal) {
        showToast(`Episode concluded: ${ep.termination_reason} (${ep.stop_reason || ''})`);
      }
    } catch (err) {
      showToast('Step error: ' + err.message, 'error');
    }
  }

  async function investigateTest(testId) {
    if (!state.currentSessionId) return;
    try {
      const ep = await apiPost('/api/episode/investigate', {
        session_id: state.currentSessionId,
        test_id: testId,
      });
      state.currentEpisode = ep;
      updateEpisodeUI();
      showToast(`Investigated ${testId}.`);
    } catch (err) {
      showToast('Investigation error: ' + err.message, 'error');
    }
  }

  async function runToCompletion() {
    if (!state.currentSessionId) return;
    let ep = state.currentEpisode;
    let guard = 0;
    while (ep && !ep.is_terminal && guard < 15) {
      ep = await apiPost('/api/episode/step', { session_id: state.currentSessionId });
      state.currentEpisode = ep;
      guard++;
    }
    updateEpisodeUI();
    showToast(`Simulation ran to completion (${ep.steps_taken} steps).`);
  }

  async function resetEpisode() {
    if (!state.currentSessionId) return;
    try {
      const ep = await apiPost('/api/episode/reset', { session_id: state.currentSessionId });
      state.currentEpisode = ep;
      updateEpisodeUI();
      showToast('Episode reset to initial priors.');
    } catch (err) {
      showToast('Reset error: ' + err.message, 'error');
    }
  }

  function resetConfigForm() {
    document.getElementById('cfg-seed').value = 42;
    document.getElementById('cfg-budget').value = 10.0;
    document.getElementById('cfg-policy').value = 'greedy_information_gain';
    document.getElementById('cfg-threshold').value = 0.90;
  }

  // Update Episode UI across dashboard & workbench
  function updateEpisodeUI() {
    const ep = state.currentEpisode;
    if (!ep) return;

    // KPI Cards
    document.getElementById('kpi-total-budget').textContent = `$${ep.initial_budget.toFixed(2)}`;
    document.getElementById('kpi-available-budget').textContent = `$${ep.remaining_budget.toFixed(2)}`;
    const pct = ((ep.remaining_budget / ep.initial_budget) * 100).toFixed(0);
    document.getElementById('kpi-remaining-budget-sub').textContent = `${pct}% remaining`;
    document.getElementById('kpi-steps-taken').textContent = `${ep.steps_taken} / ${ep.max_steps}`;
    document.getElementById('kpi-confidence').textContent = ep.current_max_posterior.toFixed(3);

    // Status Badge
    const wbBadge = document.getElementById('wb-status-badge');
    if (wbBadge) {
      if (ep.is_terminal) {
        wbBadge.className = 'status-badge badge-warning';
        wbBadge.textContent = '● Concluded';
      } else {
        wbBadge.className = 'status-badge badge-active';
        wbBadge.textContent = '● In Progress';
      }
    }

    // Entropy Badge
    document.getElementById('entropy-badge').textContent = `Entropy: ${ep.current_entropy_bits.toFixed(3)} bits`;

    // Overview Charts & Timelines
    renderOverviewBeliefChart();
    renderOverviewTimeline();
    renderCandidatesTable();

    // Rationale Summary
    const latestRec = ep.history[ep.history.length - 1];
    const summaryText = latestRec
      ? `Step ${latestRec.step_index + 1}: ${latestRec.decision_summary}`
      : 'Initial state: priors active. Select a test or step policy.';
    document.getElementById('workbench-decision-summary').textContent = summaryText;

    // Full Workbench update if visible
    renderFullWorkbench();
  }

  // Render Overview Belief Bar Chart (SVG)
  function renderOverviewBeliefChart() {
    const container = document.getElementById('overview-belief-chart');
    if (!container || !state.currentEpisode) return;

    const beliefs = state.currentEpisode.current_beliefs;
    const hypotheses = Object.keys(beliefs);
    const width = container.clientWidth || 500;
    const height = 220;
    const padding = { top: 20, right: 30, bottom: 40, left: 50 };

    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;
    const barWidth = Math.min(50, chartW / (hypotheses.length * 1.6));
    const stepX = chartW / hypotheses.length;

    let barsSvg = '';
    hypotheses.forEach((hyp, idx) => {
      const prob = beliefs[hyp];
      const barH = prob * chartH;
      const x = padding.left + idx * stepX + (stepX - barWidth) / 2;
      const y = padding.top + (chartH - barH);
      const color = COLORS[`H${idx + 1}`] || COLORS.default;

      // Prior ghost bar comparison
      const priorProb = state.config?.hypotheses.find((h) => h.id === hyp)?.prior || prob;
      const priorH = priorProb * chartH;
      const priorY = padding.top + (chartH - priorH);

      barsSvg += `
        <!-- Ghost Prior Outline -->
        <rect x="${x - 4}" y="${priorY}" width="${barWidth + 8}" height="${priorH}" 
              fill="none" stroke="#CBD5E1" stroke-width="1.5" stroke-dasharray="3 3" rx="4" />
        
        <!-- Posterior Bar -->
        <rect x="${x}" y="${y}" width="${barWidth}" height="${barH}" 
              fill="${color}" rx="4" opacity="0.9">
          <title>${hyp}: ${prob.toFixed(4)} (Prior: ${priorProb.toFixed(4)})</title>
        </rect>
        
        <!-- Value Label -->
        <text x="${x + barWidth / 2}" y="${y - 6}" text-anchor="middle" 
              font-family="JetBrains Mono" font-size="11" font-weight="600" fill="#0F172A">
          ${prob.toFixed(2)}
        </text>
        
        <!-- X Axis Label -->
        <text x="${x + barWidth / 2}" y="${height - 15}" text-anchor="middle" 
              font-family="Inter" font-size="12" font-weight="600" fill="#475569">
          ${hyp}
        </text>
      `;
    });

    // Y Axis Gridlines (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    let gridSvg = '';
    [0.0, 0.25, 0.5, 0.75, 1.0].forEach((tick) => {
      const tickY = padding.top + (chartH - tick * chartH);
      gridSvg += `
        <line x1="${padding.left}" y1="${tickY}" x2="${width - padding.right}" y2="${tickY}" 
              stroke="#F1F5F9" stroke-width="1" />
        <text x="${padding.left - 10}" y="${tickY + 4}" text-anchor="end" 
              font-family="JetBrains Mono" font-size="10" fill="#94A3B8">
          ${tick.toFixed(2)}
        </text>
      `;
    });

    container.innerHTML = `
      <svg width="${width}" height="${height}" style="overflow: visible;">
        ${gridSvg}
        ${barsSvg}
      </svg>
    `;
  }

  // Render Recent Activity Timeline
  function renderOverviewTimeline() {
    const list = document.getElementById('overview-timeline-list');
    if (!list || !state.currentEpisode) return;

    const hist = state.currentEpisode.history;
    if (hist.length === 0) {
      list.innerHTML = `
        <div class="empty-state">
          <span class="empty-icon">⏳</span>
          <p>Initial state active. Conduct investigations to observe evidence.</p>
        </div>
      `;
      return;
    }

    // Show recent steps in reverse order
    const itemsHtml = [...hist].reverse().slice(0, 4).map((rec) => {
      const maxP = Math.max(...Object.values(rec.posterior_beliefs));
      return `
        <div class="timeline-item">
          <div class="timeline-dot"></div>
          <div style="flex-grow: 1;">
            <div style="display: flex; justify-content: space-between;">
              <span class="tl-step">Step ${rec.step_index + 1}</span>
              <span class="text-muted" style="font-size: 11px;">Cost: $${rec.cost.toFixed(2)}</span>
            </div>
            <div class="tl-content">
              <strong>${rec.test_label}</strong> → outcome: <span class="tl-highlight">${rec.outcome}</span>
            </div>
            <div style="font-size: 11px; color: #64748B; margin-top: 2px;">
              Max Posterior: <strong style="color: #0F172A;">${maxP.toFixed(3)}</strong>
            </div>
          </div>
        </div>
      `;
    }).join('');

    list.innerHTML = itemsHtml;
  }

  // Render Candidates Table in Dashboard Preview
  function renderCandidatesTable() {
    const tbody = document.getElementById('workbench-candidates-tbody');
    if (!tbody || !state.currentEpisode) return;

    const candidates = state.currentEpisode.candidates;
    if (!candidates || candidates.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">All tests conducted or budget exhausted.</td></tr>';
      return;
    }

    const rows = candidates.map((c, idx) => {
      const color = ['#0EA5E9', '#F43F5E', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899'][idx % 6];
      const disabled = !c.is_affordable || state.currentEpisode.is_terminal;

      return `
        <tr>
          <td>
            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: ${color}; margin-right: 6px;"></span>
            <strong>${c.test_id}</strong>
          </td>
          <td>$${c.cost.toFixed(2)}</td>
          <td>${c.expected_information_gain_bits.toFixed(3)}</td>
          <td style="color: #0EA5E9; font-weight: 600;">${c.information_gain_per_cost.toFixed(3)}</td>
          <td>
            <button class="btn btn-sm ${disabled ? 'btn-outline' : 'btn-primary'}" 
                    data-test="${c.test_id}" ${disabled ? 'disabled' : ''}>
              Select
            </button>
          </td>
        </tr>
      `;
    }).join('');

    tbody.innerHTML = rows;

    tbody.querySelectorAll('button[data-test]').forEach((b) => {
      b.addEventListener('click', () => investigateTest(b.dataset.test));
    });

    // Populate Beliefs tab preview
    const beliefsDiv = document.getElementById('workbench-beliefs-preview');
    if (beliefsDiv) {
      const bHtml = Object.entries(state.currentEpisode.current_beliefs).map(([h, p]) => `
        <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #F1F5F9; font-size: 12px;">
          <span><strong>${h}</strong></span>
          <span style="font-family: JetBrains Mono; font-weight: 600;">${p.toFixed(4)}</span>
        </div>
      `).join('');
      beliefsDiv.innerHTML = bHtml;
    }
  }

  // Render Full Workbench View
  function renderFullWorkbench() {
    const ep = state.currentEpisode;
    if (!ep) return;

    // Full Candidates Table
    const fullTbody = document.getElementById('full-workbench-candidates-tbody');
    if (fullTbody) {
      const rows = ep.candidates.map((c, idx) => {
        const disabled = !c.is_affordable || ep.is_terminal;
        const outcomesStr = c.outcome_probabilities
          .map((o) => `${o.outcome}: ${(o.probability * 100).toFixed(0)}%`)
          .join(', ');

        return `
          <tr>
            <td>
              <strong>${c.label}</strong>
              <div class="text-muted" style="font-size: 10px;">ID: ${c.test_id}</div>
            </td>
            <td>$${c.cost.toFixed(2)}</td>
            <td>${c.expected_information_gain_bits.toFixed(4)} bits</td>
            <td style="color: #0EA5E9; font-weight: 700;">${c.information_gain_per_cost.toFixed(4)}</td>
            <td style="font-size: 11px; color: #475569;">${outcomesStr}</td>
            <td>
              <button class="btn btn-sm ${disabled ? 'btn-outline' : 'btn-primary'}" 
                      data-test="${c.test_id}" ${disabled ? 'disabled' : ''}>
                Investigate
              </button>
            </td>
          </tr>
        `;
      }).join('');
      fullTbody.innerHTML = rows || '<tr><td colspan="6" class="text-center text-muted">No available investigations.</td></tr>';

      fullTbody.querySelectorAll('button[data-test]').forEach((b) => {
        b.addEventListener('click', () => investigateTest(b.dataset.test));
      });
    }

    // Status Badge
    const termBadge = document.getElementById('full-wb-terminal-badge');
    if (termBadge) {
      if (ep.is_terminal) {
        termBadge.className = 'badge badge-warning';
        termBadge.textContent = `Terminated: ${ep.termination_reason || 'Complete'}`;
      } else {
        termBadge.className = 'badge badge-info';
        termBadge.textContent = 'Active Episode';
      }
    }

    // Rationale Box
    document.getElementById('rc-policy-tag').textContent = ep.policy;
    document.getElementById('rc-status-tag').textContent = ep.is_terminal ? 'Concluded' : 'Exploring';
    document.getElementById('rc-rem-budget').textContent = `$${ep.remaining_budget.toFixed(2)}`;
    document.getElementById('rc-entropy').textContent = `${ep.current_entropy_bits.toFixed(3)} bits`;
    document.getElementById('rc-max-p').textContent = `${ep.current_max_posterior.toFixed(3)}`;
    document.getElementById('rc-stop-reason').textContent = ep.stop_reason || 'None';

    const latestRec = ep.history[ep.history.length - 1];
    document.getElementById('rc-summary').textContent = latestRec
      ? latestRec.decision_summary
      : 'Initial belief state established from prior base rates. No actions taken yet.';

    // Hidden Truth Protection
    const truthBox = document.getElementById('hidden-truth-container');
    if (truthBox) {
      if (ep.is_terminal && ep.hidden_truth) {
        const correct = ep.hidden_truth.is_correct;
        truthBox.innerHTML = `
          <div class="truth-revealed" style="background: ${correct ? '#ECFDF5' : '#FEF2F2'}; border: 1px solid ${correct ? '#A7F3D0' : '#FECDD3'}; padding: 12px; border-radius: 8px;">
            <span class="tr-title" style="color: ${correct ? '#065F46' : '#991B1B'};">
              ${correct ? '✓ Correct Classification' : '✗ Misclassification'}
            </span>
            <div class="tr-label">True Hypothesis: <strong>${ep.hidden_truth.label}</strong> (${ep.hidden_truth.hypothesis_id})</div>
            <div style="font-size: 12px; color: #475569; margin-top: 4px;">
              Predicted: <strong>${ep.predicted_label}</strong> with posterior confidence <strong>${ep.current_max_posterior.toFixed(3)}</strong>
            </div>
          </div>
        `;
      } else {
        truthBox.innerHTML = `
          <div class="truth-locked">
            <span>🔒 Ground Truth Isolated</span>
            <small>Hidden synthetic hypothesis is strictly concealed until episode termination.</small>
          </div>
        `;
      }
    }

    // Full History Log
    const histLog = document.getElementById('full-wb-history-log');
    if (histLog) {
      if (ep.history.length === 0) {
        histLog.innerHTML = '<p class="text-muted text-center">No investigations performed yet.</p>';
      } else {
        histLog.innerHTML = ep.history.map((rec) => `
          <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 10px; border-radius: 6px; font-size: 12px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
              <strong>Step ${rec.step_index + 1}: ${rec.test_label}</strong>
              <span style="font-family: JetBrains Mono; color: #F43F5E;">Cost: $${rec.cost.toFixed(2)}</span>
            </div>
            <div style="color: #475569;">
              Observed outcome: <strong style="color: #0F172A;">${rec.outcome}</strong> | Remaining Budget: <strong>$${rec.remaining_budget.toFixed(2)}</strong>
            </div>
          </div>
        `).join('');
      }
    }

    // Prior vs Posterior Comparison Chart
    renderPriorPosteriorComparison();
  }

  // Render Dual Bar Prior vs Posterior Chart (SVG)
  function renderPriorPosteriorComparison() {
    const container = document.getElementById('full-wb-belief-compare-chart');
    if (!container || !state.currentEpisode) return;

    const ep = state.currentEpisode;
    const beliefs = ep.current_beliefs;
    const hypotheses = Object.keys(beliefs);
    const width = container.clientWidth || 400;
    const height = 220;
    const padding = { top: 20, right: 30, bottom: 40, left: 50 };

    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;
    const slotW = chartW / hypotheses.length;
    const barW = Math.min(22, slotW / 2.5);

    // Initial priors
    const priors = state.config ? Object.fromEntries(state.config.hypotheses.map((h) => [h.id, h.prior])) : beliefs;

    let bars = '';
    hypotheses.forEach((hyp, idx) => {
      const priorP = priors[hyp] || 0.0;
      const postP = beliefs[hyp];

      const priorH = priorP * chartH;
      const postH = postP * chartH;

      const groupX = padding.left + idx * slotW;
      const priorX = groupX + slotW / 2 - barW - 2;
      const postX = groupX + slotW / 2 + 2;

      const priorY = padding.top + (chartH - priorH);
      const postY = padding.top + (chartH - postH);

      bars += `
        <!-- Prior Bar -->
        <rect x="${priorX}" y="${priorY}" width="${barW}" height="${priorH}" 
              fill="#94A3B8" rx="3" opacity="0.7">
          <title>Prior: ${priorP.toFixed(3)}</title>
        </rect>
        
        <!-- Posterior Bar -->
        <rect x="${postX}" y="${postY}" width="${barW}" height="${postH}" 
              fill="#F43F5E" rx="3">
          <title>Posterior: ${postP.toFixed(3)}</title>
        </rect>

        <!-- X Label -->
        <text x="${groupX + slotW / 2}" y="${height - 15}" text-anchor="middle" 
              font-family="Inter" font-size="11" font-weight="600" fill="#475569">
          ${hyp}
        </text>
      `;
    });

    container.innerHTML = `
      <svg width="${width}" height="${height}">
        <!-- Y Grid -->
        <line x1="${padding.left}" y1="${padding.top}" x2="${width - padding.right}" y2="${padding.top}" stroke="#F1F5F9" />
        <line x1="${padding.left}" y1="${padding.top + chartH}" x2="${width - padding.right}" y2="${padding.top + chartH}" stroke="#E2E8F0" />
        ${bars}
      </svg>
      <div style="display: flex; justify-content: center; gap: 16px; font-size: 11px; margin-top: 4px;">
        <span style="color: #94A3B8;">■ Initial Prior</span>
        <span style="color: #F43F5E;">■ Current Posterior</span>
      </div>
    `;
  }

  // Contingent Plan Solver & Visual Tree
  async function solveContingentPlan() {
    const budget = parseFloat(document.getElementById('cfg-budget')?.value) || 10.0;
    const threshold = parseFloat(document.getElementById('cfg-threshold')?.value) || 0.90;

    try {
      const plan = await apiPost('/api/plan/solve', {
        budget,
        confidence_threshold: threshold,
      });

      state.planData = plan;

      // Update Preview Card Footer
      const expEl = document.getElementById('plan-preview-expansions');
      if (expEl) expEl.textContent = `Expansions: ${plan.expanded_states}`;
      const stEl = document.getElementById('plan-preview-status');
      if (stEl) stEl.textContent = `Status: ${plan.status}`;

      // Update Inspector Header
      const badge = document.getElementById('tree-status-badge');
      if (badge) badge.textContent = plan.status;
      const expVal = document.getElementById('tree-expansions-val');
      if (expVal) expVal.textContent = `${plan.expanded_states} states`;
      const utVal = document.getElementById('tree-utility-val');
      if (utVal) utVal.textContent = plan.expected_utility ? plan.expected_utility.toFixed(4) : '--';

      renderPlanTree(plan);
      renderPlanTreePreview(plan);
      showToast(`Contingent plan solved (${plan.expanded_states} states expanded).`);
    } catch (err) {
      showToast('Plan solve error: ' + err.message, 'error');
    }
  }

  // Render Plan Tree Preview (Dashboard Card)
  function renderPlanTreePreview(plan) {
    const container = document.getElementById('plan-tree-preview');
    if (!container || !plan || !plan.policy_tree) return;

    container.innerHTML = `
      <div style="padding: 16px; width: 100%; text-align: center;">
        <div style="background: #EDE9FE; color: #6D28D9; border: 1px solid #DDD6FE; padding: 6px 12px; border-radius: 20px; display: inline-block; font-size: 12px; font-weight: 700; margin-bottom: 10px;">
          Root Action: ${plan.policy_tree.action_label} (Cost: $${plan.policy_tree.cost.toFixed(2)})
        </div>
        <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;">
          ${plan.policy_tree.branches.map((b) => `
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 6px 10px; border-radius: 6px; font-size: 11px;">
              <span style="color: #F59E0B; font-weight: 700;">${b.outcome}</span> 
              <span style="color: #94A3B8;">(${(b.probability * 100).toFixed(0)}%)</span>
              <div style="font-weight: 600; color: #0F172A; margin-top: 2px;">
                ${b.child?.action ? `→ ${b.child.action_label}` : `Stop: ${b.child?.terminal_label}`}
              </div>
            </div>
          `).join('')}
        </div>
        <div style="margin-top: 12px;">
          <button class="btn btn-sm btn-outline" id="btn-goto-plan-inspector">Inspect Full Tree →</button>
        </div>
      </div>
    `;

    document.getElementById('btn-goto-plan-inspector')?.addEventListener('click', () => switchView('plan-inspector'));
  }

  // Render Interactive Full Contingent Tree (SVG Layout)
  function renderPlanTree(plan) {
    const container = document.getElementById('plan-tree-canvas-container');
    if (!container || !plan || !plan.policy_tree) return;

    const root = plan.policy_tree;
    const svgW = 960;
    const svgH = 540;

    // Build hierarchical layout coordinates
    let nodesSvg = '';
    let linksSvg = '';

    const rootX = svgW / 2;
    const rootY = 50;

    // Draw Root Node
    nodesSvg += `
      <g class="tree-node" onclick="window.selectPlanNode('root')" transform="translate(${rootX}, ${rootY})">
        <rect x="-85" y="-22" width="170" height="44" rx="22" fill="#EDE9FE" stroke="#8B5CF6" stroke-width="2" />
        <text y="-2" text-anchor="middle" font-family="Inter" font-size="12" font-weight="700" fill="#5B21B6">
          ${root.action_label}
        </text>
        <text y="14" text-anchor="middle" font-family="JetBrains Mono" font-size="10" fill="#7C3AED">
          Cost: $${root.cost.toFixed(2)} | Exp: ${plan.expanded_states}
        </text>
      </g>
    `;

    const branchCount = root.branches.length;
    const branchSpacing = svgW / (branchCount + 1);

    root.branches.forEach((branch, bIdx) => {
      const childX = (bIdx + 1) * branchSpacing;
      const childY = 190;

      // Link from Root to Branch
      linksSvg += `
        <path d="M ${rootX} ${rootY + 22} C ${rootX} ${rootY + 80}, ${childX} ${childY - 60}, ${childX} ${childY - 20}" 
              class="tree-link" />
        <rect x="${(rootX + childX) / 2 - 28}" y="${(rootY + childY) / 2 - 12}" width="56" height="20" rx="10" fill="#FEF3C7" stroke="#F59E0B" />
        <text x="${(rootX + childX) / 2}" y="${(rootY + childY) / 2 + 2}" text-anchor="middle" font-family="JetBrains Mono" font-size="10" font-weight="600" fill="#92400E">
          ${branch.outcome} (${(branch.probability * 100).toFixed(0)}%)
        </text>
      `;

      // Branch child node
      const child = branch.child;
      if (child) {
        const isTerminal = !child.action;
        const fillColor = isTerminal ? '#ECFDF5' : '#E0F2FE';
        const strokeColor = isTerminal ? '#10B981' : '#0EA5E9';
        const label = isTerminal ? `Stop: ${child.terminal_label}` : child.action_label;
        const sub = isTerminal ? `Utility: ${child.terminal_value.toFixed(3)}` : `Cost: $${child.cost.toFixed(2)}`;

        nodesSvg += `
          <g class="tree-node" onclick="window.selectPlanNode('b_${bIdx}')" transform="translate(${childX}, ${childY})">
            <rect x="-80" y="-22" width="160" height="44" rx="12" fill="${fillColor}" stroke="${strokeColor}" stroke-width="1.5" />
            <text y="-2" text-anchor="middle" font-family="Inter" font-size="12" font-weight="700" fill="#0F172A">
              ${label}
            </text>
            <text y="14" text-anchor="middle" font-family="JetBrains Mono" font-size="10" fill="#475569">
              ${sub}
            </text>
          </g>
        `;

        // If non-terminal, render secondary terminal leaves
        if (!isTerminal && child.branches) {
          const grandCount = child.branches.length;
          const grandSpacing = 160 / Math.max(1, grandCount);

          child.branches.forEach((gb, gIdx) => {
            const grandX = childX + (gIdx - (grandCount - 1) / 2) * 120;
            const grandY = 360;

            linksSvg += `
              <path d="M ${childX} ${childY + 22} C ${childX} ${childY + 60}, ${grandX} ${grandY - 60}, ${grandX} ${grandY - 18}" 
                    class="tree-link" />
              <text x="${(childX + grandX) / 2}" y="${(childY + grandY) / 2}" text-anchor="middle" font-family="JetBrains Mono" font-size="9" fill="#F59E0B">
                ${gb.outcome}
              </text>
            `;

            const gChild = gb.child;
            nodesSvg += `
              <g class="tree-node" onclick="window.selectPlanNode('gb_${bIdx}_${gIdx}')" transform="translate(${grandX}, ${grandY})">
                <rect x="-65" y="-18" width="130" height="36" rx="8" fill="#ECFDF5" stroke="#10B981" stroke-width="1.5" />
                <text y="-1" text-anchor="middle" font-family="Inter" font-size="11" font-weight="700" fill="#065F46">
                  Stop: ${gChild?.terminal_label || 'Decide'}
                </text>
                <text y="12" text-anchor="middle" font-family="JetBrains Mono" font-size="9" fill="#047857">
                  Val: ${gChild ? gChild.terminal_value.toFixed(2) : '--'}
                </text>
              </g>
            `;
          });
        }
      }
    });

    container.innerHTML = `
      <svg width="${svgW}" height="${svgH}" style="display: block; margin: auto;">
        ${linksSvg}
        ${nodesSvg}
      </svg>
    `;

    // Global hook for selecting tree nodes
    window.selectPlanNode = (key) => {
      renderNodeInspector(key, plan);
    };
  }

  // Render Node Inspector Panel
  function renderNodeInspector(key, plan) {
    const body = document.getElementById('node-inspector-body');
    if (!body || !plan || !plan.policy_tree) return;

    let node = plan.policy_tree;
    let title = 'Root Decision Node';

    if (key.startsWith('gb_')) {
      const parts = key.split('_');
      const bIdx = parseInt(parts[1], 10);
      const gIdx = parseInt(parts[2], 10);
      const branch = plan.policy_tree.branches[bIdx];
      const gBranch = branch?.child?.branches[gIdx];
      node = gBranch?.child;
      title = `Terminal Decision Node (${branch?.outcome} → ${gBranch?.outcome})`;
    } else if (key.startsWith('b_')) {
      const idx = parseInt(key.split('_')[1], 10);
      const branch = plan.policy_tree.branches[idx];
      node = branch?.child;
      title = `Branch Node (${branch?.outcome} outcome)`;
    }

    if (!node) return;

    body.innerHTML = `
      <div class="node-detail-card">
        <h4 style="font-size: 14px; font-weight: 700; color: #0F172A; margin-bottom: 8px;">${title}</h4>
        <div class="nd-row">
          <span class="nd-lbl">Planned Action</span>
          <span class="nd-val" style="color: #F43F5E;">${node.action_label || 'Terminal Decision'}</span>
        </div>
        <div class="nd-row">
          <span class="nd-lbl">Remaining Budget</span>
          <span class="nd-val">$${node.remaining_budget.toFixed(2)}</span>
        </div>
        <div class="nd-row">
          <span class="nd-lbl">Terminal Decision</span>
          <span class="nd-val">${node.terminal_label}</span>
        </div>
        <div class="nd-row">
          <span class="nd-lbl">Terminal Value</span>
          <span class="nd-val">${node.terminal_value.toFixed(4)}</span>
        </div>
        <div class="nd-row">
          <span class="nd-lbl">Branch Value</span>
          <span class="nd-val">${node.value.toFixed(4)}</span>
        </div>
        <div style="margin-top: 8px;">
          <span class="nd-lbl" style="font-weight: 600; display: block; margin-bottom: 4px;">Candidate Action Values:</span>
          ${node.action_values.map((av) => `
            <div style="display: flex; justify-content: space-between; font-size: 11px; padding: 2px 0;">
              <span>${av.action}</span>
              <span style="font-family: JetBrains Mono; font-weight: 600;">${av.value.toFixed(4)}</span>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  // Algorithm Lab Budget Curves
  async function fetchBudgetCurves() {
    try {
      const data = await apiGet('/api/algorithm-lab/budget-curve?episodes=10');
      state.curveData = data;
      renderBudgetCurves(data);
      populateAlgorithmLabTable(data);
    } catch (err) {
      console.warn('Failed to fetch budget curves:', err);
    }
  }

  function renderBudgetCurves(data) {
    const container = document.getElementById('algo-budget-curve-chart');
    if (!container || !data) return;

    const width = container.clientWidth || 700;
    const height = 260;
    const padding = { top: 30, right: 120, bottom: 40, left: 60 };

    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const budgets = data.budgets;
    const minB = budgets[0];
    const maxB = budgets[budgets.length - 1];

    const getX = (b) => padding.left + ((b - minB) / (maxB - minB)) * chartW;
    const getY = (acc) => padding.top + (chartH - acc * chartH);

    const colors = {
      contingent_optimal: '#8B5CF6',
      greedy_information_gain: '#F43F5E',
      cheapest_first: '#0EA5E9',
      fixed_order: '#10B981',
    };

    let pathsSvg = '';
    let dotsSvg = '';
    let legendSvg = '';

    let legIdx = 0;
    for (const [policy, points] of Object.entries(data.curve_data)) {
      const color = colors[policy] || '#64748B';
      const pathD = points
        .map((p, i) => `${i === 0 ? 'M' : 'L'} ${getX(p.budget)} ${getY(p.overall_accuracy)}`)
        .join(' ');

      pathsSvg += `
        <path d="${pathD}" fill="none" stroke="${color}" stroke-width="2.5" />
      `;

      points.forEach((p) => {
        dotsSvg += `
          <circle cx="${getX(p.budget)}" cy="${getY(p.overall_accuracy)}" r="4" fill="${color}">
            <title>${policy} at $${p.budget}: ${(p.overall_accuracy * 100).toFixed(1)}%</title>
          </circle>
        `;
      });

      legendSvg += `
        <g transform="translate(${width - padding.right + 10}, ${padding.top + legIdx * 20})">
          <line x1="0" y1="5" x2="16" y2="5" stroke="${color}" stroke-width="2.5" />
          <circle cx="8" cy="5" r="3" fill="${color}" />
          <text x="22" y="9" font-family="Inter" font-size="11" fill="#475569">${POLICY_NAMES_MAP[policy] || policy}</text>
        </g>
      `;
      legIdx++;
    }

    // Gridlines (Accuracy from 0.0 to 1.0)
    let gridSvg = '';
    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach((tick) => {
      const y = getY(tick);
      gridSvg += `
        <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="#F1F5F9" />
        <text x="${padding.left - 10}" y="${y + 4}" text-anchor="end" font-family="JetBrains Mono" font-size="10" fill="#94A3B8">
          ${tick.toFixed(1)}
        </text>
      `;
    });

    // X Axis ticks
    budgets.forEach((b) => {
      const x = getX(b);
      gridSvg += `
        <line x1="${x}" y1="${height - padding.bottom}" x2="${x}" y2="${height - padding.bottom + 5}" stroke="#CBD5E1" />
        <text x="${x}" y="${height - padding.bottom + 18}" text-anchor="middle" font-family="JetBrains Mono" font-size="10" fill="#64748B">
          $${b.toFixed(0)}
        </text>
      `;
    });

    container.innerHTML = `
      <svg width="${width}" height="${height}">
        ${gridSvg}
        ${pathsSvg}
        ${dotsSvg}
        ${legendSvg}
        <!-- Axis labels -->
        <text x="${padding.left + chartW / 2}" y="${height - 6}" text-anchor="middle" font-family="Inter" font-size="11" font-weight="600" fill="#475569">
          Investigation Budget ($)
        </text>
        <text transform="rotate(-90)" x="${-(padding.top + chartH / 2)}" y="18" text-anchor="middle" font-family="Inter" font-size="11" font-weight="600" fill="#475569">
          Classification Accuracy
        </text>
      </svg>
    `;
  }

  function populateAlgorithmLabTable(data) {
    const tbody = document.getElementById('algo-lab-tbody');
    if (!tbody || !data) return;

    // Use latest budget point for summary
    const rows = [
      {
        id: 'fixed_order',
        name: 'Fixed Order',
        horizon: 'None (static)',
        costAware: 'No',
        desc: 'Evaluates hardcoded order; ignores intermediate posteriors.',
      },
      {
        id: 'cheapest_first',
        name: 'Cheapest First',
        horizon: '1-step (greedy)',
        costAware: 'Yes (cost only)',
        desc: 'Sorts tests by minimum cost; diagnostic value ignored.',
      },
      {
        id: 'greedy_information_gain',
        name: 'Greedy Info Gain',
        horizon: '1-step (myopic)',
        costAware: 'Yes (EIG/Cost)',
        desc: 'Maximizes immediate Shannon information gain per dollar.',
      },
      {
        id: 'contingent_optimal',
        name: 'Contingent Optimal',
        horizon: 'Full Bellman lookahead',
        costAware: 'Optimal budget allocation',
        desc: 'True contingent policy tree branching on observed outcomes.',
      },
    ];

    tbody.innerHTML = rows.map((r) => {
      const curvePts = data.curve_data[r.id] || [];
      const pt = curvePts[curvePts.length - 1] || {};
      const cost = pt.mean_cost ? `$${pt.mean_cost.toFixed(2)}` : '--';
      const tests = pt.mean_tests ? pt.mean_tests.toFixed(1) : '--';
      const acc = pt.overall_accuracy !== undefined ? `${(pt.overall_accuracy * 100).toFixed(0)}%` : '--';

      return `
        <tr>
          <td><strong>${r.name}</strong></td>
          <td>${r.horizon}</td>
          <td>${r.costAware}</td>
          <td>${cost}</td>
          <td>${tests}</td>
          <td style="color: #F43F5E; font-weight: 700;">${acc}</td>
          <td style="font-size: 11px; color: #64748B;">${r.desc}</td>
        </tr>
      `;
    }).join('');
  }

  // Evaluation Runner
  async function runEvaluation() {
    const numEpisodes = parseInt(document.getElementById('eval-episodes').value, 10) || 20;
    const seed = parseInt(document.getElementById('eval-seed').value, 10) || 42;
    const budget = parseFloat(document.getElementById('eval-budget').value) || 10.0;
    const threshold = parseFloat(document.getElementById('eval-threshold').value) || 0.90;

    const statusEl = document.getElementById('eval-status-text');
    const runBtn = document.getElementById('btn-run-evaluation');
    const exportBtn = document.getElementById('btn-export-csv');

    try {
      if (statusEl) statusEl.textContent = 'Running simulation across policies...';
      if (runBtn) runBtn.disabled = true;

      const report = await apiPost('/api/evaluation/run', {
        num_episodes: numEpisodes,
        seed,
        budget,
        confidence_threshold: threshold,
      });

      state.evalData = report;
      if (exportBtn) exportBtn.disabled = false;
      if (statusEl) statusEl.textContent = `Completed ${report.total_runs} runs.`;

      populateEvaluationReport(report);
      showToast(`Evaluation completed (${report.total_runs} total episodes simulated).`);
    } catch (err) {
      if (statusEl) statusEl.textContent = 'Evaluation failed';
      showToast('Evaluation error: ' + err.message, 'error');
    } finally {
      if (runBtn) runBtn.disabled = false;
    }
  }

  function populateEvaluationReport(report) {
    if (!report) return;

    // Macro Stats (across all policies)
    const summaries = Object.values(report.summaries);
    const meanAcc = summaries.reduce((s, x) => s + (x.overall_accuracy || 0), 0) / summaries.length;
    const meanResAcc = summaries.reduce((s, x) => s + (x.resolved_accuracy || 0), 0) / summaries.length;
    const meanResRate = summaries.reduce((s, x) => s + (x.resolution_rate || 0), 0) / summaries.length;
    const meanCost = summaries.reduce((s, x) => s + (x.mean_cost || 0), 0) / summaries.length;
    const meanTests = summaries.reduce((s, x) => s + (x.mean_investigations || 0), 0) / summaries.length;
    const meanEnt = summaries.reduce((s, x) => s + (x.mean_final_entropy_bits || 0), 0) / summaries.length;

    document.getElementById('esc-overall-acc').textContent = `${(meanAcc * 100).toFixed(1)}%`;
    document.getElementById('esc-resolved-acc').textContent = `${(meanResAcc * 100).toFixed(1)}%`;
    document.getElementById('esc-unresolved-rate').textContent = `${((1 - meanResRate) * 100).toFixed(1)}%`;
    document.getElementById('esc-mean-cost').textContent = `$${meanCost.toFixed(2)}`;
    document.getElementById('esc-mean-tests').textContent = meanTests.toFixed(1);
    document.getElementById('esc-mean-entropy').textContent = `${meanEnt.toFixed(3)} bits`;

    // Policy Summaries Table
    const sumTbody = document.getElementById('eval-summaries-tbody');
    if (sumTbody) {
      sumTbody.innerHTML = Object.entries(report.summaries).map(([p, s]) => `
        <tr>
          <td><strong>${POLICY_NAMES_MAP[p] || p}</strong></td>
          <td>${s.attempted_episodes}</td>
          <td>${s.completed_episodes}</td>
          <td style="color: #F43F5E; font-weight: 700;">${((s.overall_accuracy || 0) * 100).toFixed(1)}%</td>
          <td>${((s.resolved_accuracy || 0) * 100).toFixed(1)}%</td>
          <td>${((s.resolution_rate || 0) * 100).toFixed(1)}%</td>
          <td>$${(s.mean_cost || 0).toFixed(2)}</td>
          <td>${(s.mean_investigations || 0).toFixed(1)}</td>
          <td>${(s.mean_final_entropy_bits || 0).toFixed(3)}</td>
        </tr>
      `).join('');
    }

    // Populate Episodes Table
    filterEvaluationEpisodes();
  }

  function filterEvaluationEpisodes() {
    const tbody = document.getElementById('eval-episodes-tbody');
    if (!tbody || !state.evalData) return;

    const filterPol = document.getElementById('eval-episode-filter-policy').value;
    const episodes = state.evalData.episodes.filter((e) => {
      return filterPol === 'all' || e.policy === filterPol;
    });

    if (episodes.length === 0) {
      tbody.innerHTML = '<tr><td colspan="10" class="text-center text-muted">No episodes match filter.</td></tr>';
      return;
    }

    tbody.innerHTML = episodes.slice(0, 100).map((e) => `
      <tr>
        <td>${e.seed}</td>
        <td><strong>${POLICY_NAMES_MAP[e.policy] || e.policy}</strong></td>
        <td>${e.hidden_label}</td>
        <td>${e.predicted_label}</td>
        <td style="color: ${e.is_correct ? '#065F46' : '#DC2626'}; font-weight: 700;">
          ${e.is_correct ? '✓ Yes' : '✗ No'}
        </td>
        <td>${e.is_resolved ? 'Yes' : 'No'}</td>
        <td>$${e.total_cost.toFixed(2)}</td>
        <td>${e.investigations_count}</td>
        <td style="font-size: 11px; color: #475569;">${e.performed_investigations.join(', ') || 'None'}</td>
        <td style="font-size: 11px;">${e.termination_reason}</td>
      </tr>
    `).join('');
  }

  function exportEvaluationCSV() {
    if (!state.evalData || !state.evalData.report_id) {
      showToast('No evaluation results to export', 'error');
      return;
    }
    const url = `/api/evaluation/export?report_id=${encodeURIComponent(state.evalData.report_id)}`;
    window.location.href = url;
  }

  // Model Specification & Quick Stats
  function populateModelSpec() {
    if (!state.config) return;

    // Hypotheses
    const hypTbody = document.getElementById('spec-hypotheses-tbody');
    if (hypTbody) {
      hypTbody.innerHTML = state.config.hypotheses.map((h) => `
        <tr>
          <td><code>${h.id}</code></td>
          <td><strong>${h.label}</strong></td>
          <td style="font-family: JetBrains Mono; font-weight: 600;">${h.prior.toFixed(4)}</td>
        </tr>
      `).join('');
    }

    // Investigations
    const invTbody = document.getElementById('spec-investigations-tbody');
    if (invTbody) {
      invTbody.innerHTML = state.config.investigations.map((i) => `
        <tr>
          <td><code>${i.id}</code></td>
          <td><strong>${i.label}</strong></td>
          <td style="font-family: JetBrains Mono; color: #F43F5E; font-weight: 600;">$${i.cost.toFixed(2)}</td>
          <td>${i.outcomes.join(', ')}</td>
          <td style="font-size: 11px; color: #64748B;">
            ${Object.keys(i.likelihoods).length} hypotheses mapped over ${i.outcomes.length} outcomes
          </td>
        </tr>
      `).join('');
    }
  }

  function updateQuickStats() {
    if (!state.config) return;
    document.getElementById('qs-hypotheses').textContent = state.config.hypotheses.length;
    document.getElementById('qs-investigations').textContent = state.config.investigations.length;
    document.getElementById('qs-min-cost').textContent = `$${state.config.planner_defaults.min_cost.toFixed(2)}`;
    document.getElementById('qs-max-cost').textContent = `$${state.config.planner_defaults.max_cost.toFixed(2)}`;
  }

  // =========================================================================
  // AI Concepts Lab: Bayesian Networks & Propositional Logic
  // =========================================================================

  let aiLabInitialized = false;
  let bnSpecData = null;

  async function initAiLab() {
    if (aiLabInitialized) return;
    aiLabInitialized = true;

    // Sub-tab switching
    const tabBtns = document.querySelectorAll('.sub-tab-btn');
    tabBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        tabBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const target = btn.dataset.target;
        document.querySelectorAll('.ai-lab-tab-content').forEach((c) => {
          c.classList.toggle('active', c.id === target);
        });
      });
    });

    try {
      await setupBayesNetLab();
      setupLogicLab();
    } catch (err) {
      console.error('Failed to initialize AI Concepts Lab:', err);
      showToast('Error loading AI Concepts Lab: ' + err.message, 'error');
    }
  }

  // --- 1. Bayesian Network Explorer ---
  async function setupBayesNetLab() {
    bnSpecData = await apiGet('/api/bayes-net/spec');
    renderBayesNetCpts(bnSpecData);
    renderBayesNetSvg('Burglary', { JohnCalls: true });

    // Controls
    document.getElementById('btn-bn-run-query')?.addEventListener('click', () => {
      runBayesNetQuery();
    });

    // Evidence change updates SVG live
    document.querySelectorAll('.bn-evidence-select').forEach((sel) => {
      sel.addEventListener('change', () => {
        const queryVar = document.getElementById('bn-query-select')?.value || 'Burglary';
        const ev = getSelectedBnEvidence();
        renderBayesNetSvg(queryVar, ev);
      });
    });

    document.getElementById('bn-query-select')?.addEventListener('change', (e) => {
      const queryVar = e.target.value;
      const ev = getSelectedBnEvidence();
      renderBayesNetSvg(queryVar, ev);
    });

    // Preset buttons
    document.getElementById('btn-bn-demo-alarm')?.addEventListener('click', () => {
      setBnState('Alarm', {});
      runBayesNetQuery();
    });

    document.getElementById('btn-bn-demo-john')?.addEventListener('click', () => {
      setBnState('JohnCalls', { Alarm: true });
      runBayesNetQuery();
    });

    document.getElementById('btn-bn-demo-burglary')?.addEventListener('click', () => {
      setBnState('Burglary', { JohnCalls: true });
      runBayesNetQuery();
    });

    document.getElementById('btn-bn-demo-explaining')?.addEventListener('click', () => {
      setBnState('Earthquake', { Alarm: true, Burglary: true });
      runBayesNetQuery();
    });

    // Run initial query
    setBnState('Burglary', { JohnCalls: true });
    await runBayesNetQuery();
  }

  function getSelectedBnEvidence() {
    const evidence = {};
    document.querySelectorAll('.bn-evidence-select').forEach((sel) => {
      const varName = sel.dataset.var;
      const val = sel.value;
      if (val === 'true') evidence[varName] = true;
      else if (val === 'false') evidence[varName] = false;
    });
    return evidence;
  }

  function setBnState(queryVar, evidence) {
    const querySelect = document.getElementById('bn-query-select');
    if (querySelect) querySelect.value = queryVar;

    document.querySelectorAll('.bn-evidence-select').forEach((sel) => {
      const varName = sel.dataset.var;
      if (evidence[varName] === true) sel.value = 'true';
      else if (evidence[varName] === false) sel.value = 'false';
      else sel.value = 'none';
    });

    renderBayesNetSvg(queryVar, evidence);
  }

  function renderBayesNetSvg(queryVar, evidence) {
    const wrapper = document.getElementById('bn-dag-svg-wrapper');
    if (!wrapper) return;

    // Node coordinates in 460x280 viewBox
    const coords = {
      Burglary: { x: 90, y: 50 },
      Earthquake: { x: 370, y: 50 },
      Alarm: { x: 230, y: 145 },
      JohnCalls: { x: 230, y: 235 },
    };

    let nodesSvg = '';
    Object.entries(coords).forEach(([name, pt]) => {
      const isQuery = name === queryVar;
      const isEv = evidence[name] !== undefined;
      const evVal = evidence[name];

      let stroke = '#CBD5E1';
      let fill = '#FFFFFF';
      let badgeText = 'Unobserved';
      let badgeFill = '#F1F5F9';
      let badgeColor = '#64748B';

      if (isQuery) {
        stroke = '#8B5CF6';
        fill = '#F5F3FF';
        badgeText = 'QUERY';
        badgeFill = '#EDE9FE';
        badgeColor = '#6D28D9';
      } else if (isEv) {
        stroke = evVal ? '#10B981' : '#64748B';
        fill = evVal ? '#ECFDF5' : '#F8FAFC';
        badgeText = `Ev: ${evVal ? 'True' : 'False'}`;
        badgeFill = evVal ? '#D1FAE5' : '#E2E8F0';
        badgeColor = evVal ? '#065F46' : '#334155';
      }

      nodesSvg += `
        <g transform="translate(${pt.x - 65}, ${pt.y - 25})">
          <rect width="130" height="50" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="${isQuery ? 2.5 : 1.5}" filter="drop-shadow(0 2px 4px rgba(0,0,0,0.06))"/>
          <text x="65" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#0F172A" font-family="Inter, sans-serif">${name}</text>
          <rect x="25" y="30" width="80" height="14" rx="4" fill="${badgeFill}"/>
          <text x="65" y="40" text-anchor="middle" font-size="9" font-weight="700" fill="${badgeColor}" font-family="Inter, sans-serif">${badgeText}</text>
        </g>
      `;
    });

    const svg = `
      <svg width="100%" height="100%" viewBox="0 0 460 280" style="display: block;">
        <defs>
          <marker id="bn-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748B"/>
          </marker>
        </defs>
        <!-- Directed Edges -->
        <!-- Burglary -> Alarm -->
        <line x1="120" y1="75" x2="195" y2="125" stroke="#94A3B8" stroke-width="2" marker-end="url(#bn-arrow)" />
        <!-- Earthquake -> Alarm -->
        <line x1="340" y1="75" x2="265" y2="125" stroke="#94A3B8" stroke-width="2" marker-end="url(#bn-arrow)" />
        <!-- Alarm -> JohnCalls -->
        <line x1="230" y1="170" x2="230" y2="210" stroke="#94A3B8" stroke-width="2" marker-end="url(#bn-arrow)" />
        ${nodesSvg}
      </svg>
    `;

    wrapper.innerHTML = svg;
  }

  function renderBayesNetCpts(spec) {
    const container = document.getElementById('bn-cpt-container');
    if (!container || !spec || !spec.nodes) return;

    let html = '';
    spec.nodes.forEach((node) => {
      html += `
        <div class="mb-3">
          <div class="font-bold text-xs mb-1" style="color: var(--text-primary);">P(${node.name}${node.parents.length ? ' | ' + node.parents.join(', ') : ''})</div>
          <table class="data-table" style="font-size: 11px;">
            <thead>
              <tr>
                ${node.parents.map((p) => `<th>${p}</th>`).join('')}
                <th>P(True)</th>
                <th>P(False)</th>
              </tr>
            </thead>
            <tbody>
              ${node.cpt
                .map((row) => {
                  const parentCols = node.parents.map((p) => `<td><code>${row.parents[p]}</code></td>`).join('');
                  return `
                  <tr>
                    ${parentCols}
                    <td><strong>${parseFloat(row.probabilities['True']).toFixed(4)}</strong></td>
                    <td class="text-muted">${parseFloat(row.probabilities['False']).toFixed(4)}</td>
                  </tr>
                `;
                })
                .join('')}
            </tbody>
          </table>
        </div>
      `;
    });
    container.innerHTML = html;
  }

  async function runBayesNetQuery() {
    const queryVar = document.getElementById('bn-query-select')?.value || 'Burglary';
    const evidence = getSelectedBnEvidence();

    renderBayesNetSvg(queryVar, evidence);

    try {
      const res = await apiPost('/api/bayes-net/query', {
        query_var: queryVar,
        evidence: evidence,
      });

      // Update badge
      const badge = document.getElementById('bn-result-query-badge');
      const evKeys = Object.keys(evidence);
      const evLabel = evKeys.length ? evKeys.map((k) => `${k}=${evidence[k] ? 'T' : 'F'}`).join(', ') : 'None';
      if (badge) badge.textContent = `P(${queryVar} | ${evLabel})`;

      // Render Distribution bars
      const barsContainer = document.getElementById('bn-result-distribution-bars');
      if (barsContainer) {
        const pTruePct = (res.p_true * 100).toFixed(2);
        const pFalsePct = (res.p_false * 100).toFixed(2);
        barsContainer.innerHTML = `
          <div class="mb-3">
            <div class="flex-between text-xs mb-1">
              <span><strong>True</strong> (${pTruePct}%)</span>
              <span class="font-mono">${res.p_true.toFixed(6)}</span>
            </div>
            <div style="background: #E2E8F0; border-radius: 4px; height: 18px; overflow: hidden;">
              <div style="width: ${pTruePct}%; background: #10B981; height: 100%; transition: width 0.3s ease;"></div>
            </div>
          </div>
          <div>
            <div class="flex-between text-xs mb-1">
              <span><strong>False</strong> (${pFalsePct}%)</span>
              <span class="font-mono">${res.p_false.toFixed(6)}</span>
            </div>
            <div style="background: #E2E8F0; border-radius: 4px; height: 18px; overflow: hidden;">
              <div style="width: ${pFalsePct}%; background: #64748B; height: 100%; transition: width 0.3s ease;"></div>
            </div>
          </div>
        `;
      }

      // Update Explanation Box
      const expBox = document.getElementById('bn-result-explanation-box');
      if (expBox) {
        expBox.className = 'counterexample-callout valid';
        expBox.innerHTML = `<strong>Inference Result:</strong> ${res.explanation}`;
      }
    } catch (err) {
      const expBox = document.getElementById('bn-result-explanation-box');
      if (expBox) {
        expBox.className = 'counterexample-callout invalid';
        expBox.textContent = `Inference Error: ${err.message}`;
      }
      showToast('Bayesian Network inference failed: ' + err.message, 'error');
    }
  }

  // --- 2. Propositional Logic Explorer ---
  function setupLogicLab() {
    // Symbol buttons
    document.querySelectorAll('.logic-sym-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const input = document.getElementById('logic-input-expr');
        if (!input) return;
        const insertText = btn.dataset.insert;
        const start = input.selectionStart || input.value.length;
        const end = input.selectionEnd || input.value.length;
        input.value = input.value.substring(0, start) + insertText + input.value.substring(end);
        input.focus();
        input.setSelectionRange(start + insertText.length, start + insertText.length);
      });
    });

    // Formula Presets
    document.getElementById('btn-logic-demo-taut')?.addEventListener('click', () => {
      setLogicExpr('P | ~P');
      runLogicTruthTable();
    });
    document.getElementById('btn-logic-demo-equiv')?.addEventListener('click', () => {
      setLogicExpr('(P -> Q) <-> (~P | Q)');
      runLogicTruthTable();
    });
    document.getElementById('btn-logic-demo-demorgan')?.addEventListener('click', () => {
      setLogicExpr('~(P & Q) <-> (~P | ~Q)');
      runLogicTruthTable();
    });
    document.getElementById('btn-logic-demo-contra')?.addEventListener('click', () => {
      setLogicExpr('P & ~P');
      runLogicTruthTable();
    });

    // Generate button
    document.getElementById('btn-logic-generate-tt')?.addEventListener('click', () => {
      runLogicTruthTable();
    });

    // Entailment Presets
    document.getElementById('btn-logic-demo-mp')?.addEventListener('click', () => {
      setEntailmentInput('P -> Q, P', 'Q');
      runLogicEntailment();
    });
    document.getElementById('btn-logic-demo-ac')?.addEventListener('click', () => {
      setEntailmentInput('P -> Q, Q', 'P');
      runLogicEntailment();
    });
    document.getElementById('btn-logic-demo-mt')?.addEventListener('click', () => {
      setEntailmentInput('P -> Q, ~Q', '~P');
      runLogicEntailment();
    });
    document.getElementById('btn-logic-demo-da')?.addEventListener('click', () => {
      setEntailmentInput('P -> Q, ~P', '~Q');
      runLogicEntailment();
    });

    // Entailment button
    document.getElementById('btn-logic-check-entailment')?.addEventListener('click', () => {
      runLogicEntailment();
    });

    // Run initial demonstrations
    runLogicTruthTable();
    runLogicEntailment();
  }

  function setLogicExpr(expr) {
    const input = document.getElementById('logic-input-expr');
    if (input) input.value = expr;
  }

  function setEntailmentInput(premises, conclusion) {
    const pInput = document.getElementById('logic-input-premises');
    const cInput = document.getElementById('logic-input-conclusion');
    if (pInput) pInput.value = premises;
    if (cInput) cInput.value = conclusion;
  }

  async function runLogicTruthTable() {
    const input = document.getElementById('logic-input-expr');
    if (!input) return;
    const expr = input.value.trim();
    if (!expr) {
      showToast('Please enter a logical expression.', 'error');
      return;
    }

    try {
      const data = await apiPost('/api/logic/truth-table', { expression: expr });

      // Metadata card
      const metaBox = document.getElementById('logic-formula-meta');
      if (metaBox) {
        metaBox.style.display = 'block';
        document.getElementById('logic-parsed-str').textContent = data.parsed;
        document.getElementById('logic-prefix-str').textContent = data.prefix_ast;

        const badgesRow = document.getElementById('logic-badges-row');
        badgesRow.innerHTML = `
          <span class="status-badge ${data.is_tautology ? 'badge-green' : 'badge-amber'} mr-2">
            ${data.is_tautology ? '✓ Tautology (Valid)' : 'Contingent / Not Tautology'}
          </span>
          <span class="status-badge ${data.is_satisfiable ? 'badge-blue' : 'badge-rose'} mr-2">
            ${data.is_satisfiable ? '✓ Satisfiable' : '✗ Unsatisfiable (Contradiction)'}
          </span>
          ${data.satisfying_model ? `<span class="text-xs text-muted">Model: ${JSON.stringify(data.satisfying_model)}</span>` : ''}
        `;
      }

      // Render Table
      const thead = document.getElementById('logic-tt-thead');
      const tbody = document.getElementById('logic-tt-tbody');

      thead.innerHTML = `
        <tr>
          ${data.variables.map((v) => `<th>${v}</th>`).join('')}
          <th style="color: var(--coral-primary);">${data.parsed}</th>
        </tr>
      `;

      tbody.innerHTML = data.rows
        .map((row) => {
          const varCells = data.variables
            .map((v) => `<td><span class="${row.assignment[v] ? 'truth-val-true' : 'truth-val-false'}">${row.assignment[v] ? 'T' : 'F'}</span></td>`)
            .join('');
          const val = row.values[0];
          return `
          <tr class="${val ? 'row-satisfying' : ''}">
            ${varCells}
            <td><span class="${val ? 'truth-val-true' : 'truth-val-false'}">${val ? 'TRUE' : 'FALSE'}</span></td>
          </tr>
        `;
        })
        .join('');
    } catch (err) {
      showToast('Truth Table Error: ' + err.message, 'error');
      const tbody = document.getElementById('logic-tt-tbody');
      if (tbody) tbody.innerHTML = `<tr><td colspan="10" class="text-rose text-center">Syntax / Parser Error: ${err.message}</td></tr>`;
    }
  }

  async function runLogicEntailment() {
    const pInput = document.getElementById('logic-input-premises');
    const cInput = document.getElementById('logic-input-conclusion');
    if (!pInput || !cInput) return;

    const rawPremises = pInput.value
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    const conclusion = cInput.value.trim();

    if (!conclusion) {
      showToast('Please enter a conclusion to test entailment.', 'error');
      return;
    }

    try {
      const data = await apiPost('/api/logic/entailment', {
        premises: rawPremises,
        conclusion: conclusion,
      });

      // Callout box
      const callout = document.getElementById('logic-entailment-callout');
      if (callout) {
        callout.className = data.is_entailed ? 'counterexample-callout valid mb-3' : 'counterexample-callout invalid mb-3';
        if (data.is_entailed) {
          callout.innerHTML = `
            <strong>✓ ENTAILMENT HOLDS (KB ⊨ α)</strong><br>
            ${data.explanation}
          `;
        } else {
          const ceStrs = data.counterexamples.map((ce) => JSON.stringify(ce)).join(', ');
          callout.innerHTML = `
            <strong>✗ INVALID INFERENCE (KB ⊭ α)</strong><br>
            ${data.explanation}<br>
            <strong>Counterexample Model(s):</strong> <code>${ceStrs}</code>
          `;
        }
      }

      // Render Table
      const thead = document.getElementById('logic-entailment-thead');
      const tbody = document.getElementById('logic-entailment-tbody');

      thead.innerHTML = `
        <tr>
          ${data.truth_table.variables.map((v) => `<th>${v}</th>`).join('')}
          ${data.premises.map((p) => `<th>${p}</th>`).join('')}
          <th style="color: var(--coral-primary);">α: ${data.conclusion}</th>
          <th>Status</th>
        </tr>
      `;

      tbody.innerHTML = data.truth_table.rows
        .map((row) => {
          const varCells = data.truth_table.variables
            .map((v) => `<td><span class="${row.assignment[v] ? 'truth-val-true' : 'truth-val-false'}">${row.assignment[v] ? 'T' : 'F'}</span></td>`)
            .join('');

          const premiseVals = row.values.slice(0, data.premises.length);
          const conclusionVal = row.values[row.values.length - 1];
          const allPremisesTrue = premiseVals.every((v) => v === true);

          const isCounterexample = allPremisesTrue && !conclusionVal;

          const pCells = premiseVals
            .map((v) => `<td><span class="${v ? 'truth-val-true' : 'truth-val-false'}">${v ? 'T' : 'F'}</span></td>`)
            .join('');
          const cCell = `<td><span class="${conclusionVal ? 'truth-val-true' : 'truth-val-false'}">${conclusionVal ? 'T' : 'F'}</span></td>`;

          let statusTag = '';
          if (isCounterexample) {
            statusTag = `<span class="status-badge badge-rose">Counterexample</span>`;
          } else if (allPremisesTrue) {
            statusTag = `<span class="status-badge badge-green">KB Model</span>`;
          } else {
            statusTag = `<span class="text-muted text-xs">—</span>`;
          }

          let rowClass = '';
          if (isCounterexample) rowClass = 'row-counterexample';
          else if (allPremisesTrue) rowClass = 'row-satisfying';

          return `
          <tr class="${rowClass}">
            ${varCells}
            ${pCells}
            ${cCell}
            <td>${statusTag}</td>
          </tr>
        `;
        })
        .join('');
    } catch (err) {
      showToast('Entailment Error: ' + err.message, 'error');
      const tbody = document.getElementById('logic-entailment-tbody');
      if (tbody) tbody.innerHTML = `<tr><td colspan="10" class="text-rose text-center">Entailment Error: ${err.message}</td></tr>`;
    }
  }

  // Simple Notification Toast
  function showToast(msg, type = 'info') {
    const existing = document.querySelector('.toast-banner');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = `toast-banner ${type}`;
    toast.style.position = 'fixed';
    toast.style.bottom = '24px';
    toast.style.right = '24px';
    toast.style.padding = '10px 18px';
    toast.style.background = type === 'error' ? '#EF4444' : '#0F172A';
    toast.style.color = '#FFFFFF';
    toast.style.borderRadius = '8px';
    toast.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
    toast.style.fontSize = '13px';
    toast.style.zIndex = '9999';
    toast.style.fontWeight = '500';
    toast.style.transition = 'opacity 0.2s ease';
    toast.textContent = msg;

    document.body.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 200);
    }, 3000);
  }

  // Start app on DOMContentLoaded
  document.addEventListener('DOMContentLoaded', init);
})();
