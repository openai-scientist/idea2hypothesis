// idea2hypothesis developer dashboard. Renders window.I2H_DATA (see build_data.py).
// Nothing is invented here: a missing artifact is shown as missing, never replaced by a default.
document.addEventListener('DOMContentLoaded', () => {
  const data = window.I2H_DATA || null;
  const STAGES = [
    { num: 1, key: 'stage-01', name: 'Topic Scoping' },
    { num: 2, key: 'stage-02', name: 'Problem Tree' },
    { num: 3, key: 'stage-03', name: 'Search Strategy' },
    { num: 4, key: 'stage-04', name: 'Literature Collect' },
    { num: 5, key: 'stage-05', name: 'Literature Screen' },
    { num: 6, key: 'stage-06', name: 'Knowledge Cards' },
    { num: 7, key: 'stage-07', name: 'Synthesis & Gaps' },
    { num: 8, key: 'stage-08', name: 'Hypotheses & Novelty' },
  ];
  const $ = (id) => document.getElementById(id);

  let activeStageKey = 'stage-01';
  let literatureMode = 'shortlist';
  let literatureQuery = '';
  let currentPage = 1;
  const pageSize = 12;
  const charts = {};

  // --- helpers ---------------------------------------------------------
  function esc(value) {
    const div = document.createElement('div');
    div.textContent = value === null || value === undefined ? '' : String(value);
    return div.innerHTML;
  }

  function md(text) {
    if (!text) return '<p style="color:var(--text-subtle);">Not available.</p>';
    if (typeof marked === 'undefined' || !marked.parse) return `<pre>${esc(text)}</pre>`;
    const html = marked.parse(text);
    return typeof DOMPurify !== 'undefined' ? DOMPurify.sanitize(html) : `<pre>${esc(text)}</pre>`;
  }

  function stageOf(key) { return (data && data.stages && data.stages[key]) || {}; }
  function shortlist() { return stageOf('stage-05').shortlist || []; }
  function candidates() { return (data && data.literature && data.literature.candidates) || []; }
  function fmt(value, digits) {
    return typeof value === 'number' ? value.toFixed(digits === undefined ? 0 : digits) : '-';
  }
  function safeUrl(url) { return /^https?:\/\//i.test(url || '') ? url : ''; }
  function badge(text, cls) { return `<span class="pill-badge ${cls || 'mono'}">${esc(text)}</span>`; }
  function note(text, tone) {
    const bg = tone === 'error' ? '#fef2f2' : tone === 'warn' ? '#fffbeb' : '#f0f9ff';
    const bd = tone === 'error' ? '#fecaca' : tone === 'warn' ? '#fde68a' : '#bae6fd';
    return `<div style="padding:0.85rem 1rem; background:${bg}; border:1px solid ${bd}; border-radius:8px; font-size:0.85rem; margin-bottom:1rem;">${text}</div>`;
  }
  function list(items) {
    if (!items || !items.length) return '<span style="color:var(--text-subtle);">none</span>';
    return `<ul>${items.map((i) => `<li>${esc(i)}</li>`).join('')}</ul>`;
  }

  // --- header / banner ---------------------------------------------------
  function initHeader() {
    $('btn-export').onclick = exportJson;
    $('btn-open-runner').onclick = openRunner;
    document.querySelectorAll('[data-goto]').forEach((b) => { b.onclick = () => switchTab(b.dataset.goto); });
    if (!data) {
      $('banner-topic').textContent = 'No run found. Start one with "Run phase 1" or check I2H_RUNS_DIR.';
      return;
    }
    $('header-run-id').textContent = data.run_id;
    $('header-status').textContent = data.status || 'unknown';
    $('header-status').className = `pill-badge ${data.status === 'completed' ? 'green' : 'mono'}`;
    $('banner-topic').textContent = data.topic;
    $('banner-mode').textContent = data.review_mode || '-';
    $('banner-attempt').textContent = data.attempt === undefined || data.attempt === null ? '-' : data.attempt;
    const hw = stageOf('stage-01').hardware;
    if (hw) {
      const vram = hw.vram_mb ? ` (${hw.vram_mb} MB)` : '';
      $('banner-hardware').textContent = `${hw.gpu_name || hw.gpu_type || 'unknown'}${vram}`;
    }
    const select = $('run-select');
    if (data.runs && data.runs.length > 1) {
      select.style.display = '';
      select.innerHTML = data.runs.map((r) =>
        `<option value="${esc(r.run_id)}" ${r.run_id === data.run_id ? 'selected' : ''}>${esc(r.run_id)} (${esc(r.status)})</option>`).join('');
      select.onchange = () => { window.location.search = `?run=${encodeURIComponent(select.value)}`; };
    }
    const alertBox = $('run-alert');
    if (data.error) {
      alertBox.style.display = '';
      alertBox.innerHTML = note(`<strong>Run failed:</strong> ${esc(data.error.code)}: ${esc(data.error.message)}`, 'error');
    } else if (data.gate && data.gate.status === 'open') {
      alertBox.style.display = '';
      alertBox.innerHTML = note(`<strong>Waiting for review:</strong> gate <code>${esc(data.gate.gate_id)}</code> (${esc(data.gate.kind)}). Answer it through the API.`, 'warn');
    } else if (data.pause_reason) {
      alertBox.style.display = '';
      alertBox.innerHTML = note(`<strong>Paused:</strong> ${esc(data.pause_reason)}`, 'warn');
    }
  }

  function initStepper() {
    const track = $('stepper-track');
    track.innerHTML = '';
    STAGES.forEach((s) => {
      const info = stageOf(s.key);
      const status = info.status || 'pending';
      const dur = typeof info.duration_sec === 'number' ? `${Math.round(info.duration_sec)}s` : status;
      const btn = document.createElement('button');
      btn.className = `step-node ${s.key === activeStageKey ? 'active' : ''}`;
      btn.id = `step-node-${s.key}`;
      btn.onclick = () => { switchTab('stages'); selectStage(s.key); };
      btn.innerHTML = `<div class="step-circle">${s.num}</div><div class="step-label">${esc(s.name)}</div><div class="step-dur">${esc(dur)}</div>`;
      track.appendChild(btn);
    });
  }

  // --- tabs --------------------------------------------------------------
  function switchTab(name) {
    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
    document.querySelectorAll('.view-section').forEach((s) => s.classList.toggle('active', s.id === `view-${name}`));
    if (name === 'overview') setTimeout(() => Object.values(charts).forEach((c) => c && c.resize()), 50);
  }
  function initTabs() {
    document.querySelectorAll('.tab-btn').forEach((b) => { b.onclick = () => switchTab(b.dataset.tab); });
  }

  // --- overview ------------------------------------------------------------
  function initKpis() {
    if (!data) return;
    const lit = data.literature || {};
    const meta = stageOf('stage-04').search_meta;
    $('kpi-total-candidates').textContent = lit.total_candidates ? lit.total_candidates.toLocaleString() : '-';
    if (meta) $('kpi-candidates-sub').textContent = `${meta.raw} raw, ${meta.duplicates} duplicates removed`;
    const s5 = stageOf('stage-05');
    if (s5.shortlist) {
      $('kpi-shortlisted').textContent = s5.shortlist.length;
      const sum = s5.review_summary;
      if (sum) $('kpi-shortlisted-sub').textContent = `${sum.rejected} rejected, ${sum.unscored} unscored`;
    }
    const cards = stageOf('stage-06').cards;
    if (cards) $('kpi-cards-count').textContent = cards.length;
    const hyp = stageOf('stage-08').hypotheses;
    if (hyp && hyp.hypotheses) $('kpi-hypotheses').textContent = hyp.hypotheses.length;
    const nov = stageOf('stage-08').novelty_report;
    if (nov) $('kpi-novelty-sub').textContent = `novelty (heuristic): ${nov.assessment}`;
    const total = Object.values(data.stages || {}).reduce((a, s) => a + (s.duration_sec || 0), 0);
    if (total > 0) $('kpi-total-duration').textContent = `${(total / 60).toFixed(1)}m`;
    const u = data.usage;
    if (u && typeof u.prompt_tokens === 'number') {
      $('kpi-usage-sub').textContent = `${(u.prompt_tokens + (u.completion_tokens || 0)).toLocaleString()} tokens` +
        (typeof u.cost_usd === 'number' ? `, $${u.cost_usd.toFixed(4)}` : ', cost not priced');
    }
  }

  function barChart(canvasId, key, labels, values, colors) {
    const el = $(canvasId);
    if (!el || typeof Chart === 'undefined') return;
    charts[key] = new Chart(el, {
      type: 'bar',
      data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 4 }] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { grid: { color: '#f1f5f9' }, beginAtZero: true }, x: { grid: { display: false } } },
      },
    });
  }

  function initCharts() {
    if (!data || typeof Chart === 'undefined') return;
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size = 11;
    Chart.defaults.color = '#64748b';
    const years = (data.literature && data.literature.year_distribution) || {};
    const yLabels = Object.keys(years);
    barChart('chart-year-dist', 'year', yLabels, Object.values(years),
      yLabels.map((y) => (parseInt(y, 10) >= 2024 ? '#0284c7' : '#94a3b8')));
    barChart('chart-stage-durations', 'duration', STAGES.map((s) => `S${s.num}`),
      STAGES.map((s) => stageOf(s.key).duration_sec || 0), '#059669');
  }

  function initTopPapers() {
    const tbody = $('overview-top-papers-tbody');
    const pool = shortlist().length ? shortlist() : candidates();
    if (!pool.length) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--text-muted); padding:1.5rem;">No papers yet.</td></tr>';
      return;
    }
    const top = [...pool].sort((a, b) => (b.citation_count || 0) - (a.citation_count || 0)).slice(0, 10);
    tbody.innerHTML = top.map((p, i) => `
      <tr>
        <td style="font-weight:700; color:var(--text-muted);">${i + 1}</td>
        <td style="font-weight:600; cursor:pointer;" data-paper="${esc(p.paper_id)}">${esc(p.title || 'Untitled')}</td>
        <td>${esc(p.year || 'n/a')}</td>
        <td style="color:var(--text-muted);">${esc(p.venue || 'unspecified')}</td>
        <td style="text-align:right; font-weight:700;">${(p.citation_count || 0).toLocaleString()}</td>
      </tr>`).join('');
    tbody.querySelectorAll('[data-paper]').forEach((td) => { td.onclick = () => openPaper(td.dataset.paper); });
  }

  // --- stage viewer ----------------------------------------------------------
  const DESCRIPTIONS = {
    'stage-01': 'Goal, scope and success criteria (goal.json).',
    'stage-02': 'Prioritised sub-questions and the topic evaluation.',
    'stage-03': 'Search strategies, queries and sources.',
    'stage-04': 'Deduplicated candidates with per-source results.',
    'stage-05': 'Screening decisions with reasons and scores.',
    'stage-06': 'One evidence card per shortlisted paper.',
    'stage-07': 'Clusters and research gaps linked to cards.',
    'stage-08': 'Falsifiable hypotheses, perspectives and a heuristic novelty assessment.',
  };

  function initStageViewer() {
    const nav = $('stage-nav-list');
    nav.innerHTML = '';
    STAGES.forEach((s) => {
      const info = stageOf(s.key);
      const btn = document.createElement('button');
      btn.className = `stage-menu-btn ${s.key === activeStageKey ? 'active' : ''}`;
      btn.id = `menu-btn-${s.key}`;
      btn.onclick = () => selectStage(s.key);
      btn.innerHTML = `<span>Stage ${s.num}: ${esc(s.name)}</span><span style="font-family:var(--font-mono); font-size:0.7rem; color:var(--text-subtle);">${esc(info.status || 'pending')}</span>`;
      nav.appendChild(btn);
    });
    renderStage(activeStageKey);
  }

  function selectStage(key) {
    activeStageKey = key;
    document.querySelectorAll('.stage-menu-btn').forEach((b) => b.classList.toggle('active', b.id === `menu-btn-${key}`));
    document.querySelectorAll('.step-node').forEach((n) => n.classList.toggle('active', n.id === `step-node-${key}`));
    renderStage(key);
  }

  function renderStage(key) {
    const s = stageOf(key);
    const meta = STAGES.find((x) => x.key === key);
    $('stage-view-title').textContent = `Stage ${meta.num}: ${meta.name}`;
    $('stage-view-desc').textContent = DESCRIPTIONS[key];
    $('stage-view-status').textContent = s.status || 'pending';
    $('stage-view-status').className = `pill-badge ${s.status === 'completed' ? 'green' : 'mono'}`;
    $('stage-view-duration').textContent = typeof s.duration_sec === 'number' ? `${s.duration_sec.toFixed(1)}s` : '';
    let html = '';
    if (s.error) html += note(`<strong>${esc(s.error.code)}:</strong> ${esc(s.error.message)}`, 'error');
    if (!s.status || s.status === 'pending') {
      html += '<p style="color:var(--text-subtle);">This stage has not run.</p>';
    } else {
      html += RENDERERS[key](s);
    }
    $('stage-view-body').innerHTML = html;
    $('stage-view-body').querySelectorAll('[data-paper]').forEach((el) => { el.onclick = () => openPaper(el.dataset.paper); });
    const toLit = $('stage-view-body').querySelector('[data-goto]');
    if (toLit) toLit.onclick = () => switchTab('literature');
  }

  function paperRow(p, idx) {
    return `
      <div class="paper-item" data-paper="${esc(p.paper_id)}">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:0.75rem;">
          <div class="paper-item-title">${idx + 1}. ${esc(p.title)}</div>
          <div style="display:flex; gap:0.35rem; flex-shrink:0;">
            ${typeof p.relevance_score === 'number' ? badge(`rel ${p.relevance_score.toFixed(2)}`, 'green') : ''}
            ${typeof p.quality_score === 'number' ? badge(`q ${p.quality_score.toFixed(2)}`) : ''}
          </div>
        </div>
        <div class="paper-item-meta">${esc(p.year || 'n/a')} &bull; ${esc(p.venue || 'unspecified venue')}</div>
        ${p.keep_reason ? `<div class="paper-item-snippet">${esc(p.keep_reason)}</div>` : ''}
      </div>`;
  }

  const RENDERERS = {
    'stage-01': (s) => `<div class="markdown-body">${md(s.goal_md)}</div>`,
    'stage-02': (s) => {
      const ev = s.topic_evaluation;
      const scores = ev
        ? `<div style="display:flex; gap:0.4rem; flex-wrap:wrap; margin-bottom:1rem;">${['novelty', 'specificity', 'feasibility', 'overall'].map((k) => badge(`${k} ${fmt(ev[k], 1)}`, k === 'overall' ? 'green' : 'mono')).join('')}</div>`
        : '';
      return scores + `<div class="markdown-body">${md(s.problem_tree_md)}</div>`;
    },
    'stage-03': (s) => {
      const queries = (s.queries && s.queries.queries) || [];
      const sources = (s.sources && s.sources.sources) || [];
      return `
        <p style="font-size:0.8rem; color:var(--text-muted);">${queries.length} queries${s.queries && s.queries.year_min ? `, year >= ${esc(s.queries.year_min)}` : ''}; ${sources.length} sources.</p>
        <div style="display:flex; flex-direction:column; gap:0.4rem;">
          ${queries.map((q, i) => `
            <div style="padding:0.6rem 0.85rem; background:#f8fafc; border:1px solid var(--border); border-radius:6px; font-size:0.85rem;">
              <span style="font-weight:700; color:var(--primary); margin-right:0.5rem;">#${i + 1}</span><code>${esc(q.text)}</code>
              <span style="color:var(--text-subtle); margin-left:0.5rem;">${esc(q.strategy)} / ${esc((q.sub_question_ids || []).join(', '))}</span>
            </div>`).join('')}
        </div>`;
    },
    'stage-04': (s) => {
      const m = s.search_meta;
      if (!m) return '<p style="color:var(--text-subtle);">search_meta.json is missing.</p>';
      const rows = Object.entries(m.per_source || {}).map(([name, st]) =>
        `<tr><td>${esc(name)}</td><td>${esc(st.requests)}</td><td>${esc(st.papers)}</td><td>${esc((st.errors || []).join('; ') || 'none')}</td></tr>`).join('');
      return `
        ${note(`<strong>${esc(m.unique)}</strong> unique papers from ${esc(m.raw)} raw records (${esc(m.duplicates)} duplicates, ${esc(m.dropped_without_title)} without title).`)}
        <table class="minimal-table"><thead><tr><th>Source</th><th>Requests</th><th>Papers</th><th>Errors</th></tr></thead><tbody>${rows}</tbody></table>
        <p><button class="btn btn-primary" data-goto="literature">Open the literature explorer</button></p>`;
    },
    'stage-05': (s) => {
      const sum = s.review_summary;
      const head = sum
        ? note(`${esc(sum.candidates)} candidates: <strong>${esc(sum.kept)} kept</strong>, ${esc(sum.rejected)} rejected, ${esc(sum.unscored)} unscored, ${esc(sum.prefiltered)} prefiltered.`)
        : '';
      const rejected = (s.decisions || []).filter((d) => d.decision !== 'kept').slice(0, 40);
      return head +
        `<div style="display:flex; flex-direction:column; gap:0.6rem;">${(s.shortlist || []).map(paperRow).join('')}</div>` +
        (rejected.length
          ? `<h3 style="margin:1.2rem 0 0.5rem; font-size:0.95rem;">Not kept (first ${rejected.length})</h3>
             <table class="minimal-table"><thead><tr><th>Paper</th><th>Decision</th><th>Reason</th></tr></thead><tbody>
             ${rejected.map((d) => `<tr><td>${esc(d.title || d.paper_id)}</td><td>${esc(d.decision)}</td><td>${esc(d.reason)}</td></tr>`).join('')}
             </tbody></table>`
          : '');
    },
    'stage-06': (s) => `
      <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:0.85rem;">
        ${(s.cards || []).map((c) => `
          <div style="border:1px solid var(--border); border-radius:8px; padding:1rem; background:#fff; font-size:0.8rem;">
            <div style="font-weight:700; color:var(--primary); margin-bottom:0.4rem;">${esc(c.title || c.card_id)} ${badge(c.evidence_scope || 'unknown scope')}</div>
            ${['problem', 'method', 'data', 'metrics', 'findings', 'limitations'].map((f) =>
              `<div style="margin-bottom:0.3rem;"><strong>${f}:</strong> ${c[f] === null || c[f] === undefined ? '<span style="color:var(--text-subtle);">not stated</span>' : esc(c[f])}</div>`).join('')}
          </div>`).join('')}
      </div>`,
    'stage-07': (s) => `<div class="markdown-body">${md(s.synthesis_md)}</div>`,
    'stage-08': (s) => {
      const nov = s.novelty_report;
      const novBox = nov
        ? note(`<strong>Novelty assessment (heuristic):</strong> ${esc(nov.assessment)}, score ${fmt(nov.novelty_score, 2)}. ${esc(nov.disclaimer || '')}`)
        : '';
      const persp = Object.keys(s.perspectives || {});
      return novBox +
        (persp.length ? `<p style="font-size:0.8rem; color:var(--text-muted);">Perspective outputs: ${persp.map((p) => badge(p)).join(' ')}</p>` : '') +
        `<div class="markdown-body">${md(s.hypotheses_md)}</div>`;
    },
  };

  // --- literature explorer ------------------------------------------------------
  function filteredLiterature() {
    let pool = literatureMode === 'shortlist' ? shortlist() : candidates();
    if (literatureQuery) {
      pool = pool.filter((p) => `${p.title || ''} ${p.abstract || ''} ${p.venue || ''}`.toLowerCase().includes(literatureQuery));
    }
    return pool;
  }

  function renderLiterature() {
    const items = filteredLiterature();
    const pages = Math.max(1, Math.ceil(items.length / pageSize));
    currentPage = Math.min(currentPage, pages);
    const start = (currentPage - 1) * pageSize;
    const slice = items.slice(start, start + pageSize);
    $('lit-count-label').textContent = `${items.length} papers`;
    $('lit-page-info').textContent = `Page ${currentPage} / ${pages}`;
    $('lit-prev-page').disabled = currentPage <= 1;
    $('lit-next-page').disabled = currentPage >= pages;
    $('lit-tab-count').textContent = candidates().length;
    const box = $('lit-papers-list');
    if (!slice.length) {
      box.innerHTML = '<div style="text-align:center; padding:2rem; color:var(--text-subtle);">No matching papers.</div>';
      return;
    }
    box.innerHTML = slice.map((p, i) => `
      <div class="paper-item" data-paper="${esc(p.paper_id)}">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:0.75rem;">
          <div class="paper-item-title">${start + i + 1}. ${esc(p.title)}</div>
          ${badge(`${(p.citation_count || 0).toLocaleString()} citations`)}
        </div>
        <div class="paper-item-meta">${esc(p.year || 'n/a')} &bull; ${esc(p.venue || 'unspecified venue')}${p.providers ? ` &bull; ${esc((p.providers || []).join(', '))}` : ''}</div>
        <div class="paper-item-snippet">${esc((p.abstract || 'No abstract.').slice(0, 400))}</div>
      </div>`).join('');
    box.querySelectorAll('[data-paper]').forEach((el) => { el.onclick = () => openPaper(el.dataset.paper); });
  }

  function initLiterature() {
    $('lit-search-input').oninput = (e) => { literatureQuery = e.target.value.toLowerCase().trim(); currentPage = 1; renderLiterature(); };
    document.querySelectorAll('.lit-pill-btn').forEach((btn) => {
      btn.onclick = () => {
        document.querySelectorAll('.lit-pill-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        literatureMode = btn.dataset.mode;
        currentPage = 1;
        renderLiterature();
      };
    });
    $('lit-prev-page').onclick = () => { if (currentPage > 1) { currentPage--; renderLiterature(); } };
    $('lit-next-page').onclick = () => { currentPage++; renderLiterature(); };
    renderLiterature();
  }

  // --- paper modal ------------------------------------------------------------
  function openPaper(paperId) {
    const full = candidates().find((p) => p.paper_id === paperId);
    const screened = shortlist().find((p) => p.paper_id === paperId) || {};
    const p = full || screened;
    if (!p || !p.paper_id) return;
    const url = safeUrl(p.url);
    $('paper-modal-body').innerHTML = `
      <div style="margin-bottom:1rem;">
        ${(p.providers || []).map((x) => badge(x, 'green')).join(' ')}
        <h2 style="font-size:1.15rem; font-weight:700; margin-top:0.4rem; line-height:1.4;">${esc(p.title)}</h2>
      </div>
      <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:1rem;">
        ${badge(`${(p.citation_count || 0).toLocaleString()} citations`)}
        ${badge(`year ${p.year || 'n/a'}`)}
        ${typeof screened.relevance_score === 'number' ? badge(`relevance ${screened.relevance_score.toFixed(2)}`, 'green') : ''}
        ${typeof screened.quality_score === 'number' ? badge(`quality ${screened.quality_score.toFixed(2)}`) : ''}
      </div>
      <div style="background:#f8fafc; border:1px solid var(--border); border-radius:6px; padding:0.75rem 1rem; margin-bottom:1rem; font-size:0.8rem; line-height:1.5;">
        <div><strong>Venue:</strong> ${esc(p.venue || 'unspecified')}</div>
        ${p.doi ? `<div><strong>DOI:</strong> <code>${esc(p.doi)}</code></div>` : ''}
        ${p.arxiv_id ? `<div><strong>arXiv:</strong> <code>${esc(p.arxiv_id)}</code></div>` : ''}
        ${screened.keep_reason ? `<div><strong>Kept because:</strong> ${esc(screened.keep_reason)}</div>` : ''}
        ${url ? `<div style="margin-top:6px;"><a href="${esc(url)}" target="_blank" rel="noopener noreferrer" style="color:var(--primary); font-weight:600;">Open source page</a></div>` : ''}
      </div>
      <div style="font-weight:700; font-size:0.85rem; margin-bottom:0.35rem;">Abstract</div>
      <p style="font-size:0.82rem; line-height:1.6; color:var(--text-muted);">${esc(p.abstract || 'No abstract.')}</p>`;
    $('paper-modal-overlay').classList.add('active');
  }

  function initModals() {
    const paper = $('paper-modal-overlay');
    $('modal-close-btn').onclick = () => paper.classList.remove('active');
    paper.onclick = (e) => { if (e.target === paper) paper.classList.remove('active'); };
  }

  // --- runner modal: starts a run through the API (via server.py) ---------------------
  const runner = $('runner-modal-overlay');
  let pollTimer = null;

  function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }
  $('runner-close-btn').onclick = () => { runner.classList.remove('active'); stopPolling(); };

  async function api(url, options) {
    const res = await fetch(url, options);
    let body = {};
    try { body = await res.json(); } catch (e) { /* non-JSON error body */ }
    return { ok: res.ok, status: res.status, body };
  }

  function renderLogs(lines) {
    const box = $('runner-terminal-box');
    box.innerHTML = lines.map((l) => {
      const color = /failed|cancelled/i.test(l) ? '#f87171' : /completed/i.test(l) ? '#34d399' : '#f8fafc';
      return `<div style="color:${color};">&gt; ${esc(l)}</div>`;
    }).join('');
    box.scrollTop = box.scrollHeight;
  }

  async function openRunner() {
    const input = $('runner-topic-input');
    if (!input.value.trim() && data) input.value = data.topic || '';
    runner.classList.add('active');
    const badgeEl = $('backend-status-badge');
    const start = $('btn-start-pipeline');
    try {
      const res = await api('/api/status');
      if (!res.ok) throw new Error('no dashboard server');
      badgeEl.textContent = res.body.is_running ? 'run in progress' : 'ready';
      badgeEl.className = `pill-badge ${res.body.is_running ? 'mono' : 'green'}`;
      start.disabled = !!res.body.is_running;
      if (res.body.is_running) startPolling();
    } catch (e) {
      badgeEl.textContent = 'static mode (no server)';
      badgeEl.className = 'pill-badge mono';
      start.disabled = true;
      $('runner-terminal-box').innerHTML = '<div style="color:#f59e0b;">Static page: run tools/dashboard/server.py to start runs from here.</div>';
    }
  }

  async function startRun() {
    const topic = $('runner-topic-input').value.trim();
    if (!topic) { alert('Enter a research topic.'); return; }
    const domains = $('runner-domains-input').value.split(',').map((d) => d.trim()).filter(Boolean);
    const start = $('btn-start-pipeline');
    start.disabled = true;
    renderLogs(['Sending POST /api/phase1/start ...']);
    try {
      const res = await api('/api/run-phase-1', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, domains }),
      });
      if (!res.ok) {
        renderLogs([`Error: ${res.body.error || res.body.detail || `HTTP ${res.status}`}`]);
        start.disabled = false;
        return;
      }
      renderLogs([`Started run ${res.body.run_id || ''}`]);
      startPolling();
    } catch (e) {
      renderLogs([`Cannot reach the dashboard server: ${e.message}`]);
      start.disabled = false;
    }
  }
  $('btn-start-pipeline').onclick = startRun;

  function startPolling() {
    stopPolling();
    pollTimer = setInterval(async () => {
      try {
        const [status, logs] = await Promise.all([api('/api/status'), api('/api/logs')]);
        renderLogs(logs.body.logs || []);
        if (!status.body.is_running) {
          stopPolling();
          $('btn-start-pipeline').disabled = false;
          const state = status.body.status;
          if (state === 'completed') setTimeout(() => window.location.reload(), 2000);
        }
      } catch (e) {
        stopPolling();
      }
    }, 2000);
  }

  function exportJson() {
    if (!data) return;
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `i2h_${data.run_id}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
  }

  // --- boot --------------------------------------------------------------------------
  initHeader();
  initStepper();
  initTabs();
  initKpis();
  initCharts();
  initTopPapers();
  initStageViewer();
  initLiterature();
  initModals();
});
