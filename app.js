/* =====================================================================
   CONFIG
   API_BASE is the address of your backend, e.g. "http://localhost:8000".
   Leave it as null to use the built-in offline demo.

   The widget calls three addresses on the backend:
     POST /partner  -> { reply }      hints while the student is debugging
     POST /quiz     -> { questions }  a quiz written from the text on the page
     POST /checkin  -> { prompt }     a short check-in question
   If the backend can't be reached, the widget quietly falls back to the
   offline demo versions and labels them as demo.
   ===================================================================== */
const CONFIG = {
  API_BASE: null,
  CHECKIN_EVERY_SEC: 90
};

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const isMobile = () => window.matchMedia('(max-width: 640px)').matches;

/* ---------- State ---------- */
const state = {
  goal: null,            // { text, confidence }
  threads: [],           // debug sessions
  pending: null,         // current text selection { text, paragraph, range }
  selectionNote: '',
  checkin: { on: false, everySec: CONFIG.CHECKIN_EVERY_SEC, nextAt: 0, pending: null, i: 0, responses: [], fetching: false, shown: [] },
  answers: {},           // quiz answers by question index
  quiz: { status: 'idle', questions: [], source: null, error: '' },
  usingDemoText: true,
  startedAt: Date.now(),
  log: []
};
function logEvent(type, data = {}) {
  state.log.push({ t: new Date().toISOString(), type, ...data });
}

/* ---------- Offline demo quiz (used only if the backend is unavailable, and only for the demo story) ---------- */
const OFFLINE_QUIZ = [
  { kind: 'Literal', q: 'How many steps does Mara climb to reach the lamp?',
    options: ['Nineteen', 'Ninety-one', 'Ninety-nine', 'One hundred nine'], answer: 1,
    why: 'The first sentence states the number directly.' },
  { kind: 'Inference', q: 'Why does Mara use the lamp shutter instead of waving or shouting?',
    options: ['She is afraid of the fisherman', 'The lamp is broken', 'The flashes are a known signal that guides boats through the channel', 'She wants to save oil'], answer: 2,
    why: 'The text never says "it was a code", but the boat straightens and enters the channel right after the pattern, which tells us the signal carried meaning.' },
  { kind: 'Inference', q: 'What does "he knew exactly who was up there" suggest?',
    options: ['Mara told him in a letter', 'He recognized the keepers\u2019 traditional signal', 'He could see her face from the boat', 'He was her grandfather\u2019s friend'], answer: 1,
    why: 'He had never met her, so he must have known her from something else: the old signal that the keepers use.' }
];

/* ---------- The four kinds of trouble a student can pick ---------- */
const PROBLEMS = {
  vocab:   { label: 'A word I don\u2019t know', sub: 'Unfamiliar or tricky word' },
  lost:    { label: 'I lost the point',       sub: 'I read it but it didn\u2019t land' },
  connect: { label: 'Ideas don\u2019t connect', sub: 'How does this link to before?' },
  why:     { label: 'Why did this happen?',   sub: 'Something is implied, not stated' }
};

/* ---------- Offline demo replies (used ONLY if the backend can't be reached) ---------- */
const DEMO_REPLIES = {
  vocab: [
    'Pick the one word that blocks you. Reread the sentence with that word skipped. Do you still get the gist? Then look at the sentences around it for clues: an example, a contrast, or a restatement.',
    'Which word is it? Try to guess its meaning from the sentence around it. What would make sense there? Then check your guess against the next sentence.'
  ],
  lost: [
    'Reread just this sentence once, slowly. Then say it in your own words in ten words or fewer. Who is doing what?',
    'Try this: what is this sentence adding that the previous one did not say? Say the main idea out loud before you move on.'
  ],
  connect: [
    'Look at the sentence right before this one. How are they linked: one causes the other, they contrast, or this adds more? Try putting \u201Cbecause\u201D, \u201Cbut\u201D, or \u201Cand then\u201D between them. Which fits?',
    'Find the last thing you clearly understood. What changed between that point and this one? Name the link in one short phrase.'
  ],
  why: [
    'Split it in two. What does the text say directly, and what are you filling in yourself? Which earlier detail supports your guess?',
    'Ask: what would have to be true for this to make sense? Look back for a clue that points that way.'
  ],
  followup: [
    'Compare your version with the original: did you keep who did what? If something is missing, add it. Then tell me below if it makes sense now.',
    'Check one thing: does your explanation still hold when you read the next sentence? If yes, mark it as clear.'
  ],
  stuck: [
    'That\u2019s okay. This one is flagged so your teacher can see it. Break it down: what is the subject, what is it doing, and which earlier line does it depend on?'
  ]
};
const pick = arr => arr[Math.floor(Math.random() * arr.length)];

/* ---------- Talking to the backend ---------- */
// Sends JSON to the backend. Returns the answer, or null if the backend is off or fails.
async function callBackend(path, body) {
  if (!CONFIG.API_BASE) return null;
  try {
    const r = await fetch(CONFIG.API_BASE.replace(/\/$/, '') + path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
    });
    if (r.ok) return await r.json();
    console.warn('The backend answered with an error:', r.status, path);
  } catch (e) {
    console.warn('Could not reach the backend, using the offline demo instead.', e);
  }
  return null;
}

// Returns { reply, demo }. demo = true means the offline fallback was used.
async function askPartner(payload) {
  const d = await callBackend('/partner', payload);
  if (d && d.reply) return { reply: d.reply, demo: false };
  await new Promise(r => setTimeout(r, 450));
  let reply;
  if (payload.mode === 'debug') reply = pick(DEMO_REPLIES[payload.problem] || DEMO_REPLIES.lost);
  else if (payload.mode === 'stuck') reply = pick(DEMO_REPLIES.stuck);
  else reply = pick(DEMO_REPLIES.followup);
  return { reply, demo: true };
}

// Builds what we send to /partner, including the conversation so far about this spot.
function partnerPayload(mode, t, message) {
  return {
    mode, problem: t.problem, passage: t.text, paragraph: t.paragraph,
    goal: state.goal && state.goal.text, message: message || null,
    history: t.msgs.filter(m => !m.typing).map(m => ({ role: m.from === 'you' ? 'student' : 'coach', text: m.text }))
  };
}

/* ---------- Offline check-in prompts (used only if the backend is unavailable) ---------- */
const OFFLINE_CHECKINS = [
  'Pause for a moment. In one sentence, what have you read so far?',
  'Is what you just read matching your goal? What is still missing?',
  'What is the most confusing part so far? Try selecting it in the text.',
  'What do you think happens or is explained next?',
  'Which sentence would you reread if you had to explain this to a friend?'
];

/* ---------- Elements ---------- */
const widget = $('#widget'), head = $('#dragHandle'), launcher = $('#launcher');
const reader = $('#reader');

/* ---------- Widget: position, drag, minimize, close ---------- */
const POS_KEY = 'rp-widget-pos';
function clampPos(x, y) {
  const w = widget.offsetWidth, h = widget.offsetHeight;
  return [Math.min(Math.max(8, x), Math.max(8, innerWidth - w - 8)), Math.min(Math.max(8, y), Math.max(8, innerHeight - h - 8))];
}
function setPos(x, y) {
  if (isMobile()) return;
  const [cx, cy] = clampPos(x, y);
  widget.style.left = cx + 'px'; widget.style.top = cy + 'px';
  widget.style.right = 'auto'; widget.style.bottom = 'auto';
}
function initPos() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(POS_KEY) || 'null'); } catch (e) {}
  if (saved) setPos(saved.x, saved.y);
  else setPos(innerWidth - widget.offsetWidth - 24, innerHeight - widget.offsetHeight - 24);
}
let drag = null;
head.addEventListener('pointerdown', e => {
  if (isMobile() || e.target.closest('button')) return;
  drag = { dx: e.clientX - widget.offsetLeft, dy: e.clientY - widget.offsetTop };
  head.setPointerCapture(e.pointerId);
});
head.addEventListener('pointermove', e => { if (drag) setPos(e.clientX - drag.dx, e.clientY - drag.dy); });
head.addEventListener('pointerup', () => {
  if (!drag) return; drag = null;
  try { localStorage.setItem(POS_KEY, JSON.stringify({ x: widget.offsetLeft, y: widget.offsetTop })); } catch (e) {}
});
addEventListener('resize', () => { if (!widget.hidden) setPos(widget.offsetLeft, widget.offsetTop); });

function openWidget() { widget.hidden = false; launcher.hidden = true; widget.classList.remove('min'); updateBadges(); }
function minimize() { widget.classList.toggle('min'); updateBadges(); }
function closeWidget() { widget.hidden = true; launcher.hidden = false; updateBadges(); }
$('#minBtn').addEventListener('click', minimize);
$('#closeBtn').addEventListener('click', closeWidget);
launcher.addEventListener('click', openWidget);
head.addEventListener('dblclick', e => { if (!e.target.closest('button')) minimize(); });

/* ---------- Tabs ---------- */
function setTab(name) {
  $$('.tab').forEach(t => t.setAttribute('aria-selected', String(t.dataset.tab === name)));
  $$('.panel').forEach(p => p.hidden = p.id !== 'p-' + name);
  if (name === 'stats') renderStats();
  if (name === 'check') renderCheck();
  if (name === 'monitor') renderCheckin();
}
$$('.tab').forEach(t => t.addEventListener('click', () => setTab(t.dataset.tab)));
function currentTab() { return ($$('.tab').find(t => t.getAttribute('aria-selected') === 'true') || {}).dataset?.tab; }

function updateBadges() {
  const has = !!state.checkin.pending;
  $('#monDot').hidden = !(has && currentTab() !== 'monitor');
  const hiddenAway = widget.hidden || widget.classList.contains('min');
  $('#headDot').hidden = !(has && widget.classList.contains('min'));
  $('#launchDot').hidden = !(has && widget.hidden);
}

/* ---------- Goal tab ---------- */
function renderGoal() {
  const g = state.goal;
  const el = $('#p-goal');
  el.innerHTML = `
    <h3>Before you read</h3>
    <p class="lead">Set a purpose. Readers who know what they want from a text notice sooner when they lose it.</p>
    <label class="f" for="goalText">What do you want to get out of this text?</label>
    <textarea id="goalText" placeholder="Example: Understand why Mara uses the lamp signal.">${g ? esc(g.text) : ''}</textarea>
    <label class="f">How well do you expect to understand it?</label>
    <div class="scale" id="conf">
      ${[1,2,3,4,5].map(n => `<button type="button" data-n="${n}" aria-pressed="${g && g.confidence === n}">${n}</button>`).join('')}
    </div>
    <div class="scale-cap"><span>Not at all</span><span>Very well</span></div>
    <div class="row" style="margin-top:14px">
      <button class="btn" id="setGoal" type="button">${g ? 'Update goal' : 'Set goal and start reading'}</button>
    </div>
    ${g ? '<p class="small" style="margin-top:10px">Goal saved. Check-ins are on. Change them in the Check-in tab.</p>' : ''}`;
  let conf = g ? g.confidence : null;
  $$('#conf button', el).forEach(b => b.addEventListener('click', () => {
    conf = +b.dataset.n; $$('#conf button', el).forEach(x => x.setAttribute('aria-pressed', String(x === b)));
  }));
  $('#setGoal', el).addEventListener('click', () => {
    const text = $('#goalText', el).value.trim();
    if (!text) { $('#goalText', el).focus(); return; }
    state.goal = { text, confidence: conf };
    logEvent('goal_set', { goal: text, confidence: conf });
    $('#goalLine').textContent = 'Goal: ' + text;
    if (!state.checkin.on) startCheckins();
    renderGoal();
    setTab('debug');
  });
}

/* ---------- Text selection -> Debug ---------- */
let selTimer = null;
document.addEventListener('selectionchange', () => {
  clearTimeout(selTimer);
  selTimer = setTimeout(captureSelection, 350);
});
function captureSelection() {
  const sel = getSelection();
  if (!sel || sel.isCollapsed || !sel.rangeCount) return;
  const range = sel.getRangeAt(0);
  if (!reader.contains(range.commonAncestorContainer)) return;
  const text = sel.toString().replace(/\s+/g, ' ').trim();
  if (text.length < 2) return;
  const pA = closestP(range.startContainer), pB = closestP(range.endContainer);
  state.selectionNote = pA !== pB ? 'Select inside a single paragraph so the mark stays clean.' : '';
  state.pending = pA === pB ? { text, paragraph: pA ? pA.textContent : '', range: range.cloneRange() } : null;
  openWidget();
  setTab('debug');
  renderDebug();
}
function closestP(node) {
  const el = node.nodeType === 1 ? node : node.parentElement;
  return el ? el.closest('#reader p') : null;
}

/* ---------- Debug tab ---------- */
function shorten(s, n = 90) { return s.length > n ? s.slice(0, n - 1) + '\u2026' : s; }

function renderDebug() {
  const el = $('#p-debug');
  const p = state.pending;
  const chooser = p ? `
    <div class="card">
      <p class="small" style="margin:0 0 4px">You selected</p>
      <blockquote class="quote" style="margin:0 0 10px; padding-left:10px; border-left:3px solid var(--accent); font-family:var(--serif); font-size:15px">${esc(shorten(p.text, 160))}</blockquote>
      <p class="small" style="margin:0 0 8px">What is going wrong?</p>
      <div class="chips">
        ${Object.entries(PROBLEMS).map(([k, v]) => `<button type="button" data-problem="${k}">${v.label}<small>${v.sub}</small></button>`).join('')}
      </div>
      <div class="row" style="margin-top:10px"><button class="btn alt" id="cancelSel" type="button">Cancel</button></div>
    </div>` : `
    <div class="card">
      <b>Debug a confusing spot</b>
      <p class="small" style="margin:4px 0 0">${state.selectionNote ? esc(state.selectionNote) : 'Select a word, phrase, or sentence in the text. It gets a red squiggle and we work it out together.'}</p>
    </div>`;
  const threads = state.threads.slice().reverse().map(t => `
    <div class="card ${t.status}" data-id="${t.id}">
      <span class="tag ${t.status}">${t.status === 'resolved' ? 'Clear now' : t.status === 'stuck' ? 'Still stuck' : PROBLEMS[t.problem].label}</span>
      <blockquote class="quote">${esc(shorten(t.text, 140))}</blockquote>
      ${t.msgs.map(m => `<div class="msg ${m.from}${m.typing ? ' typing' : ''}">${esc(m.text)}${m.demo ? '<div class="small" style="margin-top:4px">Offline demo reply (backend not reached)</div>' : ''}</div>`).join('')}
      ${t.status === 'open' && !t.busy ? `
        <form class="reply" data-id="${t.id}">
          <input type="text" placeholder="Explain it in your own words" aria-label="Your reply" autocomplete="off">
          <button class="btn" type="submit">Send</button>
        </form>
        <div class="row" style="margin-top:8px">
          <button class="btn alt" type="button" data-act="resolve" data-id="${t.id}">Makes sense now</button>
          <button class="btn alt" type="button" data-act="stuck" data-id="${t.id}">Still stuck</button>
        </div>` : ''}
    </div>`).join('');
  el.innerHTML = chooser + threads;

  $$('.chips button', el).forEach(b => b.addEventListener('click', () => startDebug(b.dataset.problem)));
  const c = $('#cancelSel', el); if (c) c.addEventListener('click', () => { state.pending = null; getSelection().removeAllRanges(); renderDebug(); });
}
$('#p-debug').addEventListener('submit', e => {
  const f = e.target.closest('form.reply'); if (!f) return;
  e.preventDefault();
  const v = f.querySelector('input').value.trim(); if (!v) return;
  sendFollowup(+f.dataset.id, v);
});
$('#p-debug').addEventListener('click', e => {
  const b = e.target.closest('button[data-act]'); if (!b) return;
  const id = +b.dataset.id;
  if (b.dataset.act === 'resolve') setStatus(id, 'resolved');
  if (b.dataset.act === 'stuck') markStuck(id);
});

let threadSeq = 0;
function startDebug(problem) {
  const p = state.pending; if (!p) return;
  const id = ++threadSeq;
  const mark = wrapRange(p.range, id);
  getSelection().removeAllRanges();
  const t = { id, problem, text: p.text, paragraph: p.paragraph, status: 'open', busy: true, msgs: [{ from: 'ai', text: 'Thinking\u2026', typing: true }], mark };
  state.threads.push(t);
  state.pending = null;
  logEvent('debug_start', { id, problem, passage: p.text });
  renderDebug();
  askPartner(partnerPayload('debug', t))
    .then(r => { t.busy = false; t.msgs = [{ from: 'ai', text: r.reply, demo: r.demo }]; renderDebug(); });
}
function wrapRange(range, id) {
  const mark = document.createElement('mark');
  mark.className = 'dbg'; mark.dataset.id = id; mark.tabIndex = 0;
  try { mark.appendChild(range.extractContents()); range.insertNode(mark); return mark; }
  catch (e) { return null; }
}
async function sendFollowup(id, message) {
  const t = state.threads.find(x => x.id === id); if (!t) return;
  t.msgs.push({ from: 'you', text: message });
  const payload = partnerPayload('followup', t, message);   // built before the "Thinking" line is added
  t.busy = true; t.msgs.push({ from: 'ai', text: 'Thinking\u2026', typing: true });
  logEvent('debug_reply', { id, message });
  renderDebug();
  const r = await askPartner(payload);
  t.msgs = t.msgs.filter(m => !m.typing); t.msgs.push({ from: 'ai', text: r.reply, demo: r.demo }); t.busy = false;
  renderDebug();
}
function setStatus(id, status) {
  const t = state.threads.find(x => x.id === id); if (!t) return;
  t.status = status;
  if (t.mark) { t.mark.classList.remove('stuck'); t.mark.classList.toggle('resolved', status === 'resolved'); }
  logEvent(status === 'resolved' ? 'debug_resolved' : 'debug_stuck', { id });
  renderDebug();
}
async function markStuck(id) {
  const t = state.threads.find(x => x.id === id); if (!t) return;
  t.status = 'stuck'; t.busy = true;
  if (t.mark) t.mark.classList.add('stuck');
  logEvent('debug_stuck', { id });
  const r = await askPartner(partnerPayload('stuck', t));
  t.msgs.push({ from: 'ai', text: r.reply, demo: r.demo }); t.busy = false;
  renderDebug();
}
reader.addEventListener('click', e => {
  const m = e.target.closest('mark.dbg'); if (!m) return;
  openWidget(); setTab('debug');
  const card = $(`#p-debug .card[data-id="${m.dataset.id}"]`);
  if (card) { card.scrollIntoView({ block: 'nearest' }); m.classList.add('active'); setTimeout(() => m.classList.remove('active'), 1200); }
});

/* ---------- Check-in tab (periodic monitoring prompts) ---------- */
function startCheckins() {
  const c = state.checkin; c.on = true; c.nextAt = Date.now() + c.everySec * 1000;
}
async function fireCheckin() {
  const c = state.checkin;
  if (c.fetching || c.pending) return;
  c.fetching = true;
  if (currentTab() === 'monitor' && !widget.hidden) renderCheckin();
  const d = await callBackend('/checkin', {
    goal: state.goal && state.goal.text,
    text: reader.innerText.trim().slice(0, 6000),
    struggles: state.threads.map(t => t.text).slice(-5),
    previous: c.shown.slice(-5)
  });
  let prompt, source;
  if (d && d.prompt) { prompt = d.prompt; source = 'ai'; }
  else { prompt = OFFLINE_CHECKINS[c.i++ % OFFLINE_CHECKINS.length]; source = 'demo'; }
  c.fetching = false; c.pending = prompt; c.shown.push(prompt);
  logEvent('checkin_prompt', { prompt, source });
  updateBadges();
  if (currentTab() === 'monitor' && !widget.hidden) renderCheckin();
}
function renderCheckin() {
  const c = state.checkin, el = $('#p-monitor');
  el.innerHTML = `
    <h3>Check-ins while you read</h3>
    <p class="lead">A short pause now and then helps you notice if you have drifted.</p>
    <div class="switch">
      <div><b>Remind me</b><div class="small" id="cd"></div></div>
      <button type="button" id="cinToggle" role="switch" aria-checked="${c.on}" aria-label="Turn check-ins on or off"></button>
    </div>
    <label class="f" for="cinEvery">How often</label>
    <select id="cinEvery">
      ${[30, 60, 90, 180, 300].map(s => `<option value="${s}" ${s === c.everySec ? 'selected' : ''}>Every ${s < 60 ? s + ' seconds' : (s / 60) + (s === 60 ? ' minute' : ' minutes')}</option>`).join('')}
    </select>
    <div class="row" style="margin:12px 0"><button class="btn alt" id="askNow" type="button">Ask me now</button></div>
    <div id="cinCard"></div>
    ${c.responses.length ? `<p class="small">${c.responses.length} check-in${c.responses.length > 1 ? 's' : ''} answered</p>` : ''}`;

  $('#cinToggle', el).addEventListener('click', () => { c.on ? (c.on = false) : startCheckins(); renderCheckin(); });
  $('#cinEvery', el).addEventListener('change', e => { c.everySec = +e.target.value; if (c.on) c.nextAt = Date.now() + c.everySec * 1000; });
  $('#askNow', el).addEventListener('click', () => { if (c.pending || c.fetching) renderCheckin(); else fireCheckin(); });
  drawCheckinCard();
  updateCountdown();
  updateBadges();
}
function drawCheckinCard() {
  const c = state.checkin, box = $('#cinCard'); if (!box) return;
  if (c.fetching) { box.innerHTML = '<p class="small">Thinking of a question\u2026</p>'; return; }
  if (!c.pending) { box.innerHTML = ''; return; }
  box.innerHTML = `
    <div class="card">
      <b>${esc(c.pending)}</b>
      <textarea id="cinAnswer" style="margin-top:8px" placeholder="Type a quick answer (optional)"></textarea>
      <p class="small" style="margin:10px 0 6px">How is it going?</p>
      <div class="row">
        <button class="btn alt" data-level="clear" type="button">I follow it</button>
        <button class="btn alt" data-level="fuzzy" type="button">A bit fuzzy</button>
        <button class="btn alt" data-level="lost" type="button">I'm lost</button>
      </div>
    </div>`;
  $$('button[data-level]', box).forEach(b => b.addEventListener('click', () => {
    const level = b.dataset.level;
    c.responses.push({ level, prompt: c.pending, answer: $('#cinAnswer', box).value.trim() });
    logEvent('checkin_response', { level, prompt: c.pending, answer: $('#cinAnswer', box).value.trim() });
    c.pending = null;
    if (c.on) c.nextAt = Date.now() + c.everySec * 1000;
    renderCheckin();
    if (level !== 'clear') {
      $('#cinCard').innerHTML = `<div class="card"><b>Try this</b><p style="margin:6px 0 0">Find the last sentence you understood, then the first one you didn't. Select that spot in the text and debug it.</p></div>`;
    }
  }));
}
function updateCountdown() {
  const c = state.checkin, el = $('#cd'); if (!el) return;
  if (!c.on) { el.textContent = 'Off'; return; }
  if (c.pending) { el.textContent = 'A check-in is waiting'; return; }
  const s = Math.max(0, Math.ceil((c.nextAt - Date.now()) / 1000));
  el.textContent = `Next in ${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}
setInterval(() => {
  const c = state.checkin;
  if (c.on && !c.pending && Date.now() >= c.nextAt) fireCheckin();
  if (currentTab() === 'monitor') updateCountdown();
}, 1000);

/* ---------- Quiz tab ---------- */
// The quiz is written from whatever text is on the page, when the student asks for it.
async function loadQuiz() {
  const qz = state.quiz;
  qz.status = 'loading'; qz.error = ''; state.answers = {};
  renderCheck();
  const d = await callBackend('/quiz', {
    text: reader.innerText.trim(),
    goal: state.goal && state.goal.text,
    n: 4
  });
  if (d && Array.isArray(d.questions) && d.questions.length) {
    qz.questions = d.questions; qz.source = 'ai'; qz.status = 'ready';
    logEvent('quiz_generated', { source: 'ai', count: d.questions.length });
  } else if (state.usingDemoText) {
    qz.questions = OFFLINE_QUIZ; qz.source = 'demo'; qz.status = 'ready';
    logEvent('quiz_generated', { source: 'demo', count: OFFLINE_QUIZ.length });
  } else {
    qz.status = 'error';
    qz.error = CONFIG.API_BASE
      ? 'The backend could not make a quiz. Look at the server window for the error, then try again.'
      : 'Making a quiz from your own text needs the backend. Set API_BASE at the top of the script.';
  }
  renderCheck();
}

function renderCheck() {
  const el = $('#p-check'), qz = state.quiz;
  if (qz.status === 'loading') {
    el.innerHTML = `<h3>Quiz</h3><p class="lead">Writing questions from your text\u2026 this can take several seconds.</p>`;
    return;
  }
  if (qz.status !== 'ready') {
    el.innerHTML = `
      <h3>Check your understanding</h3>
      <p class="lead">Finish reading first. Then make a short quiz based on this exact text.</p>
      ${qz.status === 'error' ? `<div class="card"><b>Couldn't make the quiz</b><p style="margin:6px 0 0">${esc(qz.error)}</p></div>` : ''}
      <div class="row"><button class="btn" id="makeQuiz" type="button">${qz.status === 'error' ? 'Try again' : 'Make my quiz'}</button></div>`;
    $('#makeQuiz', el).addEventListener('click', loadQuiz);
    return;
  }
  const qs = qz.questions;
  const done = Object.keys(state.answers).length;
  const score = Object.entries(state.answers).filter(([i, a]) => a === qs[i].answer).length;
  el.innerHTML = `
    <h3>Check your understanding</h3>
    <p class="lead">${done ? `${score} of ${done} correct so far.` : 'Pick the best answer for each question.'}${qz.source === 'demo' ? ' (Offline demo quiz)' : ''}</p>
    ${qs.map((q, i) => {
      const a = state.answers[i]; const answered = a !== undefined;
      return `<div class="card">
        <span class="tag">${esc(q.kind || 'Question')}</span>
        <div><b>${esc(q.q)}</b></div>
        ${q.options.map((o, j) => `<button class="opt ${answered ? (j === q.answer ? 'right' : j === a ? 'wrong' : '') : ''}" data-q="${i}" data-o="${j}" ${answered ? 'disabled' : ''}>${esc(o)}</button>`).join('')}
        ${answered ? `<div class="explain">${esc(q.why || '')}${q.evidence ? ` Look at: \u201C${esc(q.evidence)}\u201D` : ''}</div>` : ''}
      </div>`;
    }).join('')}
    <div class="row"><button class="btn alt" id="newQuiz" type="button">Make new questions</button></div>`;
  $$('.opt', el).forEach(b => b.addEventListener('click', () => {
    const i = +b.dataset.q, j = +b.dataset.o;
    state.answers[i] = j;
    logEvent('quiz_answer', { question: qs[i].q, chosen: qs[i].options[j], correct: j === qs[i].answer });
    renderCheck();
  }));
  $('#newQuiz', el).addEventListener('click', loadQuiz);
}

/* ---------- Progress tab ---------- */
function renderStats() {
  const t = state.threads;
  const resolved = t.filter(x => x.status === 'resolved').length;
  const stuck = t.filter(x => x.status === 'stuck').length;
  const mins = Math.max(1, Math.round((Date.now() - state.startedAt) / 60000));
  const qs = state.quiz.questions;
  const answered = Object.keys(state.answers).length;
  const right = Object.entries(state.answers).filter(([i, a]) => qs[i] && a === qs[i].answer).length;
  const ci = state.checkin.responses;
  $('#p-stats').innerHTML = `
    <h3>Your session</h3>
    <p class="lead">${state.goal ? 'Goal: ' + esc(state.goal.text) : 'No goal set yet.'}</p>
    <div class="stats">
      <div class="stat"><b>${t.length}</b><span>Spots debugged</span></div>
      <div class="stat"><b>${resolved}</b><span>Cleared up</span></div>
      <div class="stat"><b>${stuck}</b><span>Still stuck</span></div>
      <div class="stat"><b>${ci.length}</b><span>Check-ins answered</span></div>
      <div class="stat"><b>${answered ? right + '/' + answered : '\u2013'}</b><span>Quiz score</span></div>
      <div class="stat"><b>${mins} min</b><span>Time in session</span></div>
    </div>
    <div class="row">
      <button class="btn" id="exportBtn" type="button">Download session data</button>
      <button class="btn alt" id="resetBtn" type="button">Start over</button>
    </div>
    <p class="small" style="margin-top:10px">The download is a JSON file of everything above. The teacher dashboard will read the same shape.</p>`;
  $('#exportBtn').addEventListener('click', exportLog);
  $('#resetBtn').addEventListener('click', () => { if (confirm('Clear marks, goal, and progress for this session?')) resetSession(); });
}
function exportLog() {
  const data = { exportedAt: new Date().toISOString(), goal: state.goal, events: state.log };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'reading-session.json';
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 500);
}
function resetSession() {
  $$('mark.dbg', reader).forEach(m => { const p = m.parentNode; while (m.firstChild) p.insertBefore(m.firstChild, m); p.removeChild(m); p.normalize(); });
  Object.assign(state, { goal: null, threads: [], pending: null, selectionNote: '', answers: {}, quiz: { status: 'idle', questions: [], source: null, error: '' }, startedAt: Date.now(), log: [] });
  Object.assign(state.checkin, { on: false, pending: null, responses: [], i: 0, fetching: false, shown: [] });
  $('#goalLine').textContent = 'No goal set yet';
  renderGoal(); renderDebug(); setTab('goal'); updateBadges();
}

/* ---------- Use your own text ---------- */
const dlg = $('#pasteDlg');
$('#pasteBtn').addEventListener('click', () => dlg.showModal());
$('#pasteOk').addEventListener('click', () => {
  const raw = $('#pasteText').value.trim();
  if (!raw) return;
  resetSession();
  reader.innerHTML = raw.split(/\n\s*\n/).map(p => `<p>${esc(p.replace(/\s*\n\s*/g, ' ').trim())}</p>`).join('');
  $('#docTitle').textContent = 'Your text';
  $('#docBy').textContent = 'Pasted passage';
  state.usingDemoText = false;
  renderCheck();
});

/* ---------- Theme ---------- */
$('#themeBtn').addEventListener('click', () => {
  const root = document.documentElement;
  const dark = matchMedia('(prefers-color-scheme: dark)').matches;
  const cur = root.dataset.theme || (dark ? 'dark' : 'light');
  root.dataset.theme = cur === 'dark' ? 'light' : 'dark';
});

/* ---------- Boot ---------- */
renderGoal(); renderDebug(); renderCheck(); renderStats();
initPos();
