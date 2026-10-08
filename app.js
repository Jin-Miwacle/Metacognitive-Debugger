/* CONFIG -- set API_BASE to your backend's URL, or leave it null to run
   on the built-in offline demo. Three endpoints get called: POST /partner
   (debug hints), POST /quiz (quiz questions), POST /checkin (check-in
   prompts). If the backend's unreachable, the app quietly falls back to
   offline demo answers. */
const CONFIG = {
  API_BASE: null,
  CHECKIN_EVERY_SEC: 90
};

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

/* Built-in practice readings: plain text, blank line between paragraphs.
   Work in both Readings and Quizzes without a teacher or database. */
const DEMO_READINGS = [
  {
    id: 'demo-lamp',
    title: 'The Lamp on Harrow Point',
    text:
`Every evening at dusk, Mara climbed the ninety-one steps of the lighthouse to light the lamp, just as her grandfather had done before her. The lamp itself was old, but it never failed. What worried her was the harbor below, where fewer boats returned each season.

Tonight, a single fishing boat wobbled toward the rocks. Mara did not wave or shout. She turned the lamp's shutter slowly, three long flashes and one short, the way her grandfather had taught her when she was small. The boat straightened and slid into the channel between the stones.

At the dock, the fisherman looked up at the tower and lifted his cap. He had never met Mara, yet he knew exactly who was up there. The old signal had told him that the person keeping the light still remembered the old ways, and that the channel was safe to trust.`
  },
  {
    id: 'demo-seed',
    title: 'The Last Jar of Seeds',
    text:
`When the rains failed for the second year, the community garden shrank to a single raised bed behind the old schoolhouse. Teodora kept one jar of bean seeds on the highest shelf in her kitchen, untouched, while the rest of the garden's seeds went into the ground early, before anyone was sure the soil could hold them.

Her neighbors asked why she didn't plant her share like everyone else. She only said that a garden needed one thing kept in reserve, in case the first planting failed entirely. When a late frost killed half the seedlings in May, it was her jar, and no one else's, that refilled the empty rows by the second week of June.

Nobody suggested she explain herself again after that. The following spring, three other families on her street had jars of their own on their highest shelves, untouched, waiting.`
  },
  {
    id: 'demo-machine',
    title: 'The Quiet Machine',
    text:
`The record player had sat in the back room of the library for eleven years, donated along with a box of records nobody had catalogued. When Priya finally opened the cabinet to clear space, she found a needle worn down to almost nothing and a handwritten label taped inside the lid: "Return to Mr. Oyelaran when fixed."

She looked him up before she decided what to do with any of it. Mr. Oyelaran had taught music at the school across the street until it closed in the nineties; the library had no record of him after that. Still, she ordered a new needle, cleaned the turntable herself over a weekend, and played the first record in the box just to test it.

It was a recording of a school choir, slightly out of tune, dated the same year the label was written. Priya didn't know whose voice was whose. She reshelved the player in the front room anyway, with a handwritten card of her own: "Ask at the desk to listen."`
  }
];
const DEMO_READING_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 3.6 4.8 7.4l7.2 3.8 7.2-3.8Z"/><path d="M4.8 7.4v9l7.2 3.8 7.2-3.8v-9"/><path d="M12 11.2v9"/></svg>`;

/* ---- Reading workspace state (Goal / Debug / Check-in / Progress) ---- */
const state = {
  goal: null,            // { text, confidence }
  threads: [],           // debug sessions
  pending: null,         // current text selection { text, paragraph, range }
  selectionNote: '',
  checkin: { on: false, everySec: CONFIG.CHECKIN_EVERY_SEC, nextAt: 0, pending: null, i: 0, responses: [], fetching: false, shown: [] },
  support: { score: 0, level: 'steady' },  // see nudgeSupport() below
  startedAt: Date.now(),
  log: [],
  user: null,                  // { id, role, full_name } once logged in
  reading: { id: null },       // the teacher reading this session is on (null = demo/pasted text)
  session: { id: null }        // the row in the "sessions" table for this reading attempt
};
function logEvent(type, data = {}) {
  state.log.push({ t: new Date().toISOString(), type, ...data });
}

/* ---- Support level: makes goals/check-ins actually change the session ----
   Score rises on a low-confidence goal, a "lost" check-in, or a stuck
   thread; falls on "clear" or resolved. The level it maps to (steady /
   closer / extra) speeds up check-ins, shows a banner, and rides along
   to the backend as a hint. */
function nudgeSupport(delta) {
  const s = state.support;
  s.score = Math.max(0, Math.min(6, s.score + delta));
  const prevLevel = s.level;
  s.level = s.score >= 4 ? 'extra' : s.score >= 2 ? 'closer' : 'steady';
  if (s.level !== prevLevel) logEvent('support_level_change', { from: prevLevel, to: s.level, score: s.score });
  updateSupportBanner();
  if (currentTab() === 'monitor') renderCheckin();
  if (currentTab() === 'stats') renderStats();
}
function effectiveCheckinSec() {
  const base = state.checkin.everySec;
  if (state.support.level === 'extra') return Math.max(30, Math.round(base * 0.5));
  if (state.support.level === 'closer') return Math.max(30, Math.round(base * 0.75));
  return base;
}
function updateSupportBanner() {
  const el = $('#supportBanner'); if (!el) return;
  const s = state.support;
  if (s.level === 'steady') { el.hidden = true; return; }
  el.hidden = false;
  el.textContent = s.level === 'extra'
    ? 'Extra support: check-ins are more frequent while we work through this together.'
    : 'Checking in a little more often — a few spots have been tricky.';
}
/* Quick word-overlap check against the goal (stopwords stripped) -- no
   AI call, so it still works offline. */
const STOPWORDS = new Set(['the','a','an','to','of','in','on','for','and','or','is','are','was','were','it','its','that','this','what','why','how','do','does','did','i','me','my','understand','about','with','as','be','at','so','not']);
function keywordsOf(text) {
  return new Set((text.toLowerCase().match(/[a-z']+/g) || []).filter(w => w.length > 2 && !STOPWORDS.has(w)));
}
function goalOverlapWord(passage) {
  if (!state.goal) return null;
  const gw = keywordsOf(state.goal.text), pw = keywordsOf(passage);
  for (const w of pw) if (gw.has(w)) return w;
  return null;
}

/* ---- Saving progress to the database ----
   A "session" is one student's attempt at one reading, created on first
   real use (a goal, a debug spot, a check-in) rather than the moment the
   reading opens -- so just browsing doesn't clutter the teacher's data.
   A failed save just warns to console; the session keeps working locally. */
async function ensureSession() {
  if (state.session.id) return state.session.id;
  if (!state.user) return null;
  const { data, error } = await supa.from('sessions').insert({
    student_id: state.user.id,
    reading_id: state.reading.id,
    goal: state.goal ? state.goal.text : null,
    confidence: state.goal ? state.goal.confidence : null
  }).select('id').single();
  if (error) { console.warn('Could not start a session in the database:', error.message); return null; }
  state.session.id = data.id;
  return data.id;
}
async function dbUpdateSession(fields) {
  const id = state.session.id; if (!id) return;
  const { error } = await supa.from('sessions').update(fields).eq('id', id);
  if (error) console.warn('Could not update the session:', error.message);
}
async function dbInsertDebugEvent(t) {
  const sid = await ensureSession(); if (!sid) return;
  const { data, error } = await supa.from('debug_events')
    .insert({ session_id: sid, passage: t.text, problem: t.problem, status: 'open' })
    .select('id').single();
  if (error) { console.warn('Could not save the debug spot:', error.message); return; }
  t.dbId = data.id;
}
async function dbUpdateDebugStatus(t) {
  if (!t.dbId) return;
  const { error } = await supa.from('debug_events').update({ status: t.status }).eq('id', t.dbId);
  if (error) console.warn('Could not update the debug spot:', error.message);
}
/* Saves the actual back-and-forth for this debug thread -- needs
   hint_transcript_migration.sql run once. Without it the column doesn't
   exist and this just fails quietly, same as any other save error. */
async function dbSaveTranscript(t) {
  if (!t.dbId) return;
  const transcript = t.msgs.filter(m => !m.typing).map(m => ({ from: m.from, text: m.text, demo: !!m.demo }));
  const { error } = await supa.from('debug_events').update({ transcript }).eq('id', t.dbId);
  if (error) console.warn('Could not save the hint transcript:', error.message);
}
async function dbInsertCheckin(prompt, level, answer) {
  const sid = await ensureSession(); if (!sid) return;
  const { error } = await supa.from('checkin_events').insert({ session_id: sid, prompt, level, answer });
  if (error) console.warn('Could not save the check-in:', error.message);
}

/* ---- Offline demo quiz: backend-unavailable fallback, lamp story only ---- */
const OFFLINE_QUIZ = [
  { kind: 'Literal', q: 'How many steps does Mara climb to reach the lamp?',
    options: ['Nineteen', 'Ninety-one', 'Ninety-nine', 'One hundred nine'], answer: 1,
    why: 'The first sentence states the number directly.' },
  { kind: 'Inference', q: 'Why does Mara use the lamp shutter instead of waving or shouting?',
    options: ['She is afraid of the fisherman', 'The lamp is broken', 'The flashes are a known signal that guides boats through the channel', 'She wants to save oil'], answer: 2,
    why: 'The text never says "it was a code", but the boat straightens and enters the channel right after the pattern, which tells us the signal carried meaning.' },
  { kind: 'Inference', q: 'What does "he knew exactly who was up there" suggest?',
    options: ['Mara told him in a letter', 'He recognized the keepers’ traditional signal', 'He could see her face from the boat', 'He was her grandfather’s friend'], answer: 1,
    why: 'He had never met her, so he must have known her from something else: the old signal that the keepers use.' }
];

/* ---- The four debug problem types ---- */
const PROBLEMS = {
  vocab:   { label: 'A word I don’t know', sub: 'Unfamiliar or tricky word' },
  lost:    { label: 'I lost the point',       sub: 'I read it but it didn’t land' },
  connect: { label: 'Ideas don’t connect', sub: 'How does this link to before?' },
  why:     { label: 'Why did this happen?',   sub: 'Something is implied, not stated' }
};

/* ---- Offline demo replies: backend-unavailable fallback ---- */
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
    'Look at the sentence right before this one. How are they linked: one causes the other, they contrast, or this adds more? Try putting “because”, “but”, or “and then” between them. Which fits?',
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
    'That’s okay. This one is flagged so your teacher can see it. Break it down: what is the subject, what is it doing, and which earlier line does it depend on?'
  ]
};
const pick = arr => arr[Math.floor(Math.random() * arr.length)];

/* ---- Backend calls ---- */
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

function partnerPayload(mode, t, message) {
  return {
    mode, problem: t.problem, passage: t.text, paragraph: t.paragraph,
    goal: state.goal && state.goal.text, support: state.support.level, message: message || null,
    history: t.msgs.filter(m => !m.typing).map(m => ({ role: m.from === 'you' ? 'student' : 'coach', text: m.text }))
  };
}

/* ---- Offline check-in prompts: backend-unavailable fallback ---- */
const OFFLINE_CHECKINS = [
  'Pause for a moment. In one sentence, what have you read so far?',
  'Is what you just read matching your goal? What is still missing?',
  'What is the most confusing part so far? Try selecting it in the text.',
  'What do you think happens or is explained next?',
  'Which sentence would you reread if you had to explain this to a friend?'
];

/* ---- Elements ---- */
const reader = $('#reader');

/* ---- Side-panel tabs (Goal / Debug / Check-in / Progress) ---- */
function setTab(name) {
  $$('.tab').forEach(t => t.setAttribute('aria-selected', String(t.dataset.tab === name)));
  $$('.panel').forEach(p => p.hidden = p.id !== 'p-' + name);
  if (name === 'stats') renderStats();
  if (name === 'monitor') renderCheckin();
}
$$('.tab').forEach(t => t.addEventListener('click', () => setTab(t.dataset.tab)));
function currentTab() { return ($$('.tab').find(t => t.getAttribute('aria-selected') === 'true') || {}).dataset?.tab; }

function updateBadges() {
  const has = !!state.checkin.pending;
  $('#monDot').hidden = !(has && currentTab() !== 'monitor');
}

/* ---- Goal tab ---- */
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
    if (state.session.id) dbUpdateSession({ goal: text, confidence: conf });
    else ensureSession();
    // Low confidence starts check-ins closer together; high confidence starts relaxed.
    nudgeSupport({1:2, 2:1, 3:0, 4:-1, 5:-1}[conf ?? 3]);
    if (!state.checkin.on) startCheckins();
    renderGoal();
    renderDebug();
    setTab('debug');
  });
}

/* ---- Text selection -> Debug ---- */
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
  setTab('debug');
  renderDebug();
}
function closestP(node) {
  const el = node.nodeType === 1 ? node : node.parentElement;
  return el ? el.closest('#reader p') : null;
}

/* ---- Debug tab ---- */
function shorten(s, n = 90) { return s.length > n ? s.slice(0, n - 1) + '…' : s; }

function renderDebug() {
  const el = $('#p-debug');
  const p = state.pending;
  const goalWord = p ? goalOverlapWord(p.text) : null;
  const chooser = p ? `
    <div class="card">
      <p class="small" style="margin:0 0 4px">You selected</p>
      <blockquote class="quote" style="margin:0 0 10px; padding-left:10px; border-left:3px solid var(--accent); font-family:var(--serif); font-size:15px">${esc(shorten(p.text, 160))}</blockquote>
      ${goalWord ? `<p class="small" style="margin:0 0 8px; color:var(--accent)">This looks related to your goal (“${esc(goalWord)}”) — worth sorting out before moving on.</p>` : ''}
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
  const t = { id, problem, text: p.text, paragraph: p.paragraph, status: 'open', busy: true, msgs: [{ from: 'ai', text: 'Thinking…', typing: true }], mark };
  state.threads.push(t);
  state.pending = null;
  logEvent('debug_start', { id, problem, passage: p.text });
  dbInsertDebugEvent(t);
  renderDebug();
  askPartner(partnerPayload('debug', t))
    .then(r => { t.busy = false; t.msgs = [{ from: 'ai', text: r.reply, demo: r.demo }]; renderDebug(); dbSaveTranscript(t); });
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
  const payload = partnerPayload('followup', t, message);
  t.busy = true; t.msgs.push({ from: 'ai', text: 'Thinking…', typing: true });
  logEvent('debug_reply', { id, message });
  renderDebug();
  const r = await askPartner(payload);
  t.msgs = t.msgs.filter(m => !m.typing); t.msgs.push({ from: 'ai', text: r.reply, demo: r.demo }); t.busy = false;
  renderDebug();
  dbSaveTranscript(t);
}
function setStatus(id, status) {
  const t = state.threads.find(x => x.id === id); if (!t) return;
  t.status = status;
  if (t.mark) { t.mark.classList.remove('stuck'); t.mark.classList.toggle('resolved', status === 'resolved'); }
  logEvent(status === 'resolved' ? 'debug_resolved' : 'debug_stuck', { id });
  nudgeSupport(status === 'resolved' ? -1 : 2);
  dbUpdateDebugStatus(t);
  renderDebug();
}
async function markStuck(id) {
  const t = state.threads.find(x => x.id === id); if (!t) return;
  t.status = 'stuck'; t.busy = true;
  if (t.mark) t.mark.classList.add('stuck');
  logEvent('debug_stuck', { id });
  nudgeSupport(2);
  dbUpdateDebugStatus(t);
  const r = await askPartner(partnerPayload('stuck', t));
  t.msgs.push({ from: 'ai', text: r.reply, demo: r.demo }); t.busy = false;
  renderDebug();
  dbSaveTranscript(t);
}
reader.addEventListener('click', e => {
  const m = e.target.closest('mark.dbg'); if (!m) return;
  setTab('debug');
  const card = $(`#p-debug .card[data-id="${m.dataset.id}"]`);
  if (card) { card.scrollIntoView({ block: 'nearest' }); m.classList.add('active'); setTimeout(() => m.classList.remove('active'), 1200); }
});

/* ---- Check-in tab ---- */
function startCheckins() {
  const c = state.checkin; c.on = true; c.nextAt = Date.now() + effectiveCheckinSec() * 1000;
}
async function fireCheckin() {
  const c = state.checkin;
  if (c.fetching || c.pending) return;
  c.fetching = true;
  if (currentTab() === 'monitor') renderCheckin();
  const d = await callBackend('/checkin', {
    goal: state.goal && state.goal.text,
    support: state.support.level,
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
  if (currentTab() === 'monitor') renderCheckin();
}
function renderCheckin() {
  const c = state.checkin, el = $('#p-monitor');
  const supportNote = state.support.level !== 'steady'
    ? `<p class="small" style="color:var(--accent)">${state.support.level === 'extra' ? 'Extra support is on, so these are firing faster than your setting below.' : 'Firing a bit faster than your setting below, since a few spots have been tricky.'}</p>`
    : '';
  el.innerHTML = `
    <h3>Check-ins while you read</h3>
    <p class="lead">A short pause now and then helps you notice if you have drifted. How you answer changes how often the next one comes, and whether the Debug tab steps in.</p>
    ${supportNote}
    <div class="switch">
      <div><b>Remind me</b><div class="small" id="cd"></div></div>
      <button type="button" id="cinToggle" role="switch" aria-checked="${c.on}" aria-label="Turn check-ins on or off"></button>
    </div>
    <label class="f" for="cinEvery">How often (baseline)</label>
    <select id="cinEvery">
      ${[30, 60, 90, 180, 300].map(s => `<option value="${s}" ${s === c.everySec ? 'selected' : ''}>Every ${s < 60 ? s + ' seconds' : (s / 60) + (s === 60 ? ' minute' : ' minutes')}</option>`).join('')}
    </select>
    <div class="row" style="margin:12px 0"><button class="btn alt" id="askNow" type="button">Ask me now</button></div>
    <div id="cinCard"></div>
    ${c.responses.length ? `<p class="small">${c.responses.length} check-in${c.responses.length > 1 ? 's' : ''} answered</p>` : ''}`;

  $('#cinToggle', el).addEventListener('click', () => { c.on ? (c.on = false) : startCheckins(); renderCheckin(); });
  $('#cinEvery', el).addEventListener('change', e => { c.everySec = +e.target.value; if (c.on) c.nextAt = Date.now() + effectiveCheckinSec() * 1000; });
  $('#askNow', el).addEventListener('click', () => { if (c.pending || c.fetching) renderCheckin(); else fireCheckin(); });
  drawCheckinCard();
  updateCountdown();
  updateBadges();
}
function drawCheckinCard() {
  const c = state.checkin, box = $('#cinCard'); if (!box) return;
  if (c.fetching) { box.innerHTML = '<p class="small">Thinking of a question…</p>'; return; }
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
    const answerText = $('#cinAnswer', box).value.trim();
    c.responses.push({ level, prompt: c.pending, answer: answerText });
    logEvent('checkin_response', { level, prompt: c.pending, answer: answerText });
    dbInsertCheckin(c.pending, level, answerText);
    nudgeSupport(level === 'clear' ? -1 : level === 'fuzzy' ? 1 : 2);
    c.pending = null;
    if (c.on) c.nextAt = Date.now() + effectiveCheckinSec() * 1000;
    renderCheckin();
    if (level !== 'clear') {
      $('#cinCard').innerHTML = `<div class="card"><b>Try this</b><p style="margin:6px 0 0">Find the last sentence you understood, then the first one you didn't. Select that spot in the text and debug it.</p>
        <div class="row" style="margin-top:8px"><button class="btn alt" type="button" id="goDebugFromCheckin">Open Debug tab</button></div></div>`;
      const go = $('#goDebugFromCheckin'); if (go) go.addEventListener('click', () => setTab('debug'));
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

/* ---- Progress tab ---- */
const SUPPORT_LABEL = { steady: 'Steady — check-ins at your normal pace', closer: 'Checking in a bit closer together', extra: 'Extra support — check-ins are more frequent' };
function renderStats() {
  const t = state.threads;
  const resolved = t.filter(x => x.status === 'resolved').length;
  const stuck = t.filter(x => x.status === 'stuck').length;
  const mins = Math.max(1, Math.round((Date.now() - state.startedAt) / 60000));
  const ci = state.checkin.responses;
  const trend = ci.length ? `
    <p class="small" style="margin:14px 0 4px">How check-ins have gone, in order:</p>
    <div class="ci-trend">${ci.map(r => `<i class="ci-dot ${r.level}" title="${r.level}"></i>`).join('')}</div>` : '';
  const g = state.goal;
  const goalRating = !g ? '' : g.finalConfidence == null ? `
    <div class="row" style="margin-top:10px"><button class="btn alt" id="rateGoalBtn" type="button">Rate how well you met your goal</button></div>` : `
    <p class="small" style="margin-top:10px">Expected to understand it: <b>${g.confidence ?? '—'}/5</b> &middot; Actually did: <b>${g.finalConfidence}/5</b></p>`;
  $('#p-stats').innerHTML = `
    <h3>Your session</h3>
    <p class="lead">${g ? 'Goal: ' + esc(g.text) : 'No goal set yet.'}</p>
    <p class="small">Support level: <b>${SUPPORT_LABEL[state.support.level]}</b></p>
    <div class="stats">
      <div class="stat"><b>${t.length}</b><span>Spots debugged</span></div>
      <div class="stat"><b>${resolved}</b><span>Cleared up</span></div>
      <div class="stat"><b>${stuck}</b><span>Still stuck</span></div>
      <div class="stat"><b>${ci.length}</b><span>Check-ins answered</span></div>
      <div class="stat" style="grid-column:1/-1"><b>${mins} min</b><span>Time in session</span></div>
    </div>
    ${trend}
    ${goalRating}
    <div class="row" style="margin-top:14px">
      <button class="btn" id="exportBtn" type="button">Download session data</button>
      <button class="btn alt" id="resetBtn" type="button">Start over</button>
    </div>
    <p class="small" style="margin-top:10px">The download is a JSON file of everything above. The teacher dashboard reads from the database directly.</p>`;
  $('#exportBtn').addEventListener('click', exportLog);
  $('#resetBtn').addEventListener('click', () => { if (confirm('Clear marks, goal, and progress for this session?')) resetSession(); });
  const rateBtn = $('#rateGoalBtn');
  if (rateBtn) rateBtn.addEventListener('click', () => {
    $('#p-stats').insertAdjacentHTML('beforeend', `
      <div class="card" id="goalRateCard" style="margin-top:10px">
        <p class="small" style="margin:0 0 8px">How well did you end up understanding it?</p>
        <div class="scale" id="finalConf">${[1,2,3,4,5].map(n => `<button type="button" data-n="${n}">${n}</button>`).join('')}</div>
      </div>`);
    $$('#finalConf button').forEach(b => b.addEventListener('click', () => {
      g.finalConfidence = +b.dataset.n;
      logEvent('goal_final_rating', { confidence: g.confidence, finalConfidence: g.finalConfidence });
      if (state.session.id) dbUpdateSession({ final_confidence: g.finalConfidence });
      renderStats();
    }));
  });
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
  Object.assign(state, {
    goal: null, threads: [], pending: null, selectionNote: '',
    session: { id: null },   // fresh reading = fresh DB session
    startedAt: Date.now(), log: []
  });
  Object.assign(state.checkin, { on: false, pending: null, responses: [], i: 0, fetching: false, shown: [] });
  Object.assign(state.support, { score: 0, level: 'steady' });
  $('#goalLine').textContent = 'No goal set yet';
  updateSupportBanner();
  renderGoal(); renderDebug(); setTab('goal'); updateBadges();
}

function setReaderText(title, byline, bodyParagraphsHtml) {
  $('#docTitle').textContent = title;
  $('#docBy').textContent = byline;
  reader.innerHTML = bodyParagraphsHtml;
}
function paragraphsToHtml(raw) {
  return raw.split(/\n\s*\n/).map(p => `<p>${esc(p.replace(/\s*\n\s*/g, ' ').trim())}</p>`).join('');
}

/* ---- Paste/upload dialog, shared by Readings and Quizzes -- pasteTarget tracks which one opened it ---- */
let pasteTarget = 'reader';
function openPasteDialog(target) { pasteTarget = target; $('#pasteFileMsg').textContent = ''; $('#pasteDlg').showModal(); }
$('#pasteBtn').addEventListener('click', () => openPasteDialog('reader'));

$('#pasteFile').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = ''; // so picking the same file twice still fires "change"
  if (!file) return;
  const msg = $('#pasteFileMsg');
  msg.style.color = 'var(--muted)';
  msg.textContent = `Reading ${file.name}…`;
  try {
    const text = await extractTextFromFile(file);
    $('#pasteText').value = text;
    msg.style.color = 'var(--ok)';
    msg.textContent = `Loaded text from ${file.name}. Review it below, then click "Load text".`;
  } catch (err) {
    msg.style.color = 'var(--bug)';
    msg.textContent = err.message || 'Could not read that file.';
  }
});
$('#pasteOk').addEventListener('click', () => {
  const raw = $('#pasteText').value.trim();
  if (!raw) return;
  $('#pasteText').value = '';
  if (pasteTarget === 'reader') {
    enterReader({ title: 'Your text', byline: 'Pasted passage', bodyHtml: paragraphsToHtml(raw), readingId: null });
  } else {
    quizCtx = { readingId: null, isDemo: false, demoId: null, isPasted: true, title: 'Your text', text: raw, sessionId: null, questions: [], answers: {}, status: 'idle', error: '', source: null };
    renderQuizWorkspace();
  }
});

/* Sidebar nav: each button's data-view matches a <section id="<name>View">,
   so a new section just needs a new button. */
function showView(name) {
  $$('.side-link').forEach(b => {
    const active = b.dataset.view === name;
    b.classList.toggle('active', active);
    b.setAttribute('aria-current', active ? 'page' : 'false');
  });
  $$('.view').forEach(v => { v.hidden = v.id !== name + 'View'; });
  if (name === 'quizzes') renderQuizzesView();
  if (name === 'classes') renderClassesView();
  if (name === 'settings') renderSettings('settingsPane', state.user);
}
$$('.side-link').forEach(b => b.addEventListener('click', () => showView(b.dataset.view)));

/* CLASSES: join a teacher's class with a code. The database (not just the
   UI) limits visible readings to teachers whose class you've joined. */
async function renderClassesView() {
  const el = $('#classesPane');
  el.innerHTML = `<h1>Your classes</h1><p class="hint">Loading…</p>`;

  const { data: memberships, error } = await supa
    .from('class_members')
    .select('class_id, joined_at, classes(id, name, teacher_id)')
    .eq('student_id', state.user.id);

  if (error) {
    el.innerHTML = `<h1>Your classes</h1><p class="small" style="color:var(--bug)">${esc(error.message)}</p>`;
    return;
  }

  // classes and profiles aren't linked in the schema, so teacher names need a separate lookup
  const teacherIds = [...new Set((memberships || []).map(m => m.classes && m.classes.teacher_id).filter(Boolean))];
  let teacherNames = {};
  if (teacherIds.length) {
    const { data: teachers } = await supa.from('profiles').select('id, full_name').in('id', teacherIds);
    (teachers || []).forEach(t => { teacherNames[t.id] = t.full_name || 'Teacher'; });
  }

  const rows = (memberships || []).filter(m => m.classes).map(m => `
    <div class="reading-row-card" data-class="${m.classes.id}">
      <div>
        <b>${esc(m.classes.name)}</b>
        <div class="small">Taught by ${esc(teacherNames[m.classes.teacher_id] || 'Teacher')}</div>
      </div>
      <button class="tiny danger" type="button" data-act="leave">Leave</button>
    </div>`).join('');

  el.innerHTML = `
    <h1>Your classes</h1>
    <p class="hint">Joining a class lets you see the readings that teacher adds. Ask your teacher for their class's join code.</p>

    <div class="card" style="max-width:420px">
      <h3 style="margin:0 0 10px">Join a class</h3>
      <label class="f" for="joinCodeInput">Join code</label>
      <input type="text" id="joinCodeInput" placeholder="e.g. 7K4PXM" maxlength="8" style="text-transform:uppercase; letter-spacing:2px">
      <div class="row" style="margin-top:12px">
        <button class="btn" id="joinClassBtn" type="button">Join class</button>
      </div>
    </div>

    ${rows ? `<div style="margin-top:18px; display:flex; flex-direction:column; gap:10px">${rows}</div>`
           : '<p class="small" style="margin-top:16px">You haven\'t joined any classes yet. The demo story and pasting your own text still work either way.</p>'}`;

  $('#joinClassBtn').addEventListener('click', async () => {
    const codeInput = $('#joinCodeInput');
    const code = codeInput.value.trim().toUpperCase();
    if (!code) return;
    const { data: cls, error: findErr } = await supa.from('classes').select('id').eq('join_code', code).maybeSingle();
    if (findErr || !cls) { showToast("That join code didn't match a class. Double-check it with your teacher.", 'bug'); return; }
    const { error: joinErr } = await supa.from('class_members').insert({ class_id: cls.id, student_id: state.user.id });
    if (joinErr) {
      showToast(/duplicate|unique/i.test(joinErr.message) ? "You're already in that class." : 'Could not join: ' + joinErr.message, 'bug');
      return;
    }
    codeInput.value = '';
    showToast('Joined the class.', 'ok');
    renderClassesView();
  });

  el.querySelectorAll('[data-act="leave"]').forEach(btn => btn.addEventListener('click', async () => {
    const row = btn.closest('[data-class]');
    if (!confirm("Leave this class? You'll lose access to that teacher's readings until you rejoin.")) return;
    const { error: leaveErr } = await supa.from('class_members').delete().eq('class_id', row.dataset.class).eq('student_id', state.user.id);
    if (leaveErr) { showToast('Could not leave: ' + leaveErr.message, 'bug'); return; }
    showToast('Left the class.', 'ok');
    renderClassesView();
  }));
}

function handleLogin(profile) {
  state.user = profile; // { id, role, full_name }
  $('#userName').textContent = profile.full_name || 'Student';
  showView('readings');
  showReadingPicker();
}

/* READINGS: library grid -> reader + side panel.
   QUIZZES reuses this same data but renders as a flat list instead of a
   card grid (pass opts.kind = 'quiz') -- so the two pickers read as
   different kinds of screens, not the same page with a different title. */
function readingCardsHtml(readings, iconSvg, opts = {}) {
  if (opts.kind === 'quiz') {
    return `<div class="quiz-list">${readings.map(r => {
      const words = r.body.trim().split(/\s+/).length;
      const qCount = Math.max(3, Math.min(6, Math.round(words / 180)));
      const cls = opts.demo ? 'quiz-row demo' : 'quiz-row';
      const attr = opts.demo ? `data-demo="${r.id}"` : `data-id="${r.id}"`;
      return `<button class="${cls}" ${attr}>
        <span class="quiz-row-icon">${iconSvg}</span>
        <span class="quiz-row-body">
          <span class="quiz-row-title">${esc(r.title)}${opts.demo ? '<span class="quiz-row-badge">Built-in</span>' : ''}</span>
          <span class="quiz-row-meta">About ${qCount} questions</span>
        </span>
        <span class="quiz-row-cta">Start quiz &rarr;</span>
      </button>`;
    }).join('')}</div>`;
  }
  return `<div class="lib-grid">${readings.map(r => {
    const words = r.body.trim().split(/\s+/).length;
    const mins = Math.max(1, Math.round(words / 150));
    const snippet = r.body.replace(/\s+/g, ' ').trim().slice(0, 110);
    const cls = opts.demo ? 'lib-card demo' : 'lib-card';
    const attr = opts.demo ? `data-demo="${r.id}"` : `data-id="${r.id}"`;
    return `<button class="${cls}" ${attr}>
      ${opts.demo ? '<span class="lib-card-badge">Built-in</span>' : ''}
      <span class="lib-card-icon">${iconSvg}</span>
      <span class="lib-card-title">${esc(r.title)}</span>
      <span class="lib-card-snippet">${esc(snippet)}…</span>
      <span class="lib-card-meta">${mins} min read</span>
    </button>`;
  }).join('')}</div>`;
}
/* Demo readings are shaped like {id, title, body} too, so the same card markup renders both. */
function demoReadingObjs() { return DEMO_READINGS.map(d => ({ id: d.id, title: d.title, body: d.text })); }
const BOOK_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 5.2c2.3-1 5-1 7 .3v12.8c-2-1.3-4.7-1.3-7-.3V5.2Z"/><path d="M18.5 5.2c-2.3-1-5-1-7 .3v12.8c2-1.3 4.7-1.3 7-.3V5.2Z"/></svg>`;

$('#changeReadingBtn').addEventListener('click', () => showReadingPicker());

async function showReadingPicker() {
  $('#readerWorkspace').hidden = true;
  const picker = $('#pickerPane');
  picker.hidden = false;
  picker.innerHTML = `<h1>Choose a reading</h1><p class="hint">Loading…</p>`;

  const { data: readings, error } = await supa
    .from('readings').select('id, title, body, created_at').order('created_at', { ascending: false });

  const teacherGrid = (!error && readings && readings.length)
    ? readingCardsHtml(readings, BOOK_ICON)
    : `<p class="small">${error ? 'Could not load readings right now.'
        : 'No readings from your classes yet. <a href="#" id="goToClasses">Join a class</a> to see what your teacher adds.'}</p>`;

  picker.innerHTML = `
    <h1>Choose a reading</h1>
    <p class="hint">Pick a built-in practice text, something your teacher added, or paste your own.</p>
    <h3 style="margin:22px 0 0">Built-in practice readings</h3>
    ${readingCardsHtml(demoReadingObjs(), DEMO_READING_ICON, { demo: true })}
    <h3 style="margin:28px 0 0">From your classes</h3>
    ${teacherGrid}
    <div class="row" style="margin-top:16px">
      <button class="btn alt" id="pickPaste" type="button">Paste or upload my own text</button>
    </div>`;

  $$('button[data-id]', picker).forEach(b => b.addEventListener('click', async () => {
    b.disabled = true;
    const { data: r, error: rErr } = await supa.from('readings').select('*').eq('id', b.dataset.id).single();
    if (rErr || !r) { picker.insertAdjacentHTML('beforeend', `<p class="small" style="color:var(--bug)">Could not open that reading.</p>`); return; }
    enterReader({ title: r.title, byline: 'Added by your teacher', bodyHtml: paragraphsToHtml(r.body), readingId: r.id });
  }));
  $$('button[data-demo]', picker).forEach(b => b.addEventListener('click', () => {
    const d = DEMO_READINGS.find(x => x.id === b.dataset.demo);
    if (!d) return;
    enterReader({ title: d.title, byline: 'Built-in practice text', bodyHtml: paragraphsToHtml(d.text), readingId: null });
  }));
  $('#pickPaste', picker).addEventListener('click', () => openPasteDialog('reader'));
  const classesLink = $('#goToClasses', picker);
  if (classesLink) classesLink.addEventListener('click', (e) => { e.preventDefault(); showView('classes'); });
}

function enterReader({ title, byline, bodyHtml, readingId }) {
  resetSession();
  setReaderText(title, byline, bodyHtml);
  state.reading = { id: readingId };

  $('#pickerPane').hidden = true;
  $('#readerWorkspace').hidden = false;
  $('#pasteBtn').hidden = false;

  renderGoal(); renderDebug();
}

/* QUIZZES: a separate top-level view. Fetches the reading's text directly,
   so a quiz works without opening that reading in Readings first. */
let quizCtx = { readingId: null, isDemo: false, demoId: null, isPasted: false, title: '', text: '', sessionId: null, questions: [], answers: {}, status: 'idle', error: '', source: null };

async function ensureQuizSession() {
  if (quizCtx.sessionId) return quizCtx.sessionId;
  if (!state.user) return null;
  const { data, error } = await supa.from('sessions').insert({
    student_id: state.user.id, reading_id: quizCtx.readingId
  }).select('id').single();
  if (error) { console.warn('Could not start a quiz session in the database:', error.message); return null; }
  quizCtx.sessionId = data.id;
  return data.id;
}
async function saveQuizAnswer(question, chosen, correct) {
  const sid = await ensureQuizSession(); if (!sid) return;
  const { error } = await supa.from('quiz_results').insert({ session_id: sid, question, chosen, correct });
  if (error) console.warn('Could not save the quiz answer:', error.message);
}

function renderQuizzesView() {
  if (quizCtx.readingId || quizCtx.isDemo || quizCtx.isPasted) renderQuizWorkspace();
  else renderQuizPicker();
}

const QUIZ_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="5.5" y="4.5" width="13" height="16" rx="2"/><path d="M9 4.5V3.3a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1V4.5M8.5 12l2 2 4.5-4.5"/></svg>`;

async function renderQuizPicker() {
  const el = $('#quizzesPane');
  el.innerHTML = `<h1>Quizzes</h1><p class="hint">Loading…</p>`;

  const { data: readings, error } = await supa
    .from('readings').select('id, title, body, created_at').order('created_at', { ascending: false });

  const teacherGrid = (!error && readings && readings.length)
    ? readingCardsHtml(readings, QUIZ_ICON, { kind: 'quiz' })
    : `<p class="small">${error ? 'Could not load readings right now.'
        : 'No readings from your classes yet. <a href="#" id="goToClassesQuiz">Join a class</a> to see what your teacher adds.'}</p>`;

  el.innerHTML = `
    <h1>Quizzes</h1>
    <p class="hint">Pick a reading to test yourself on &mdash; a built-in text, something your teacher added, or your own paste.</p>
    <h3 style="margin:22px 0 0">Built-in practice texts</h3>
    ${readingCardsHtml(demoReadingObjs(), DEMO_READING_ICON, { demo: true, kind: 'quiz' })}
    <h3 style="margin:28px 0 0">From your classes</h3>
    ${teacherGrid}
    <div class="row" style="margin-top:16px">
      <button class="btn alt" id="quizPaste" type="button">Paste or upload a text to quiz</button>
    </div>`;

  $$('button[data-id]', el).forEach(b => b.addEventListener('click', async () => {
    b.disabled = true;
    const { data: r, error: rErr } = await supa.from('readings').select('*').eq('id', b.dataset.id).single();
    if (rErr || !r) { el.insertAdjacentHTML('beforeend', `<p class="small" style="color:var(--bug)">Could not open that reading.</p>`); return; }
    quizCtx = { readingId: r.id, isDemo: false, demoId: null, isPasted: false, title: r.title, text: r.body, sessionId: null, questions: [], answers: {}, status: 'idle', error: '', source: null };
    renderQuizWorkspace();
  }));
  $$('button[data-demo]', el).forEach(b => b.addEventListener('click', () => {
    const d = DEMO_READINGS.find(x => x.id === b.dataset.demo);
    if (!d) return;
    quizCtx = { readingId: null, isDemo: true, demoId: d.id, isPasted: false, title: d.title, text: d.text, sessionId: null, questions: [], answers: {}, status: 'idle', error: '', source: null };
    renderQuizWorkspace();
  }));
  $('#quizPaste', el).addEventListener('click', () => openPasteDialog('quiz'));
  const classesLink = $('#goToClassesQuiz', el);
  if (classesLink) classesLink.addEventListener('click', (e) => { e.preventDefault(); showView('classes'); });
}

async function loadQuizQuestions() {
  quizCtx.status = 'loading'; quizCtx.error = ''; quizCtx.answers = {};
  renderQuizWorkspace();
  const d = await callBackend('/quiz', { text: quizCtx.text, goal: null, n: 4 });
  if (d && Array.isArray(d.questions) && d.questions.length) {
    quizCtx.questions = d.questions; quizCtx.source = 'ai'; quizCtx.status = 'ready';
  } else if (quizCtx.demoId === 'demo-lamp') {
    // The hand-written offline fallback quiz only covers this one story.
    quizCtx.questions = OFFLINE_QUIZ; quizCtx.source = 'demo'; quizCtx.status = 'ready';
  } else {
    quizCtx.status = 'error';
    quizCtx.error = CONFIG.API_BASE
      ? 'The backend could not make a quiz. Look at the server window for the error, then try again.'
      : 'Making a quiz needs the backend (this built-in text doesn\'t have a ready-made offline quiz). Set API_BASE at the top of app.js.';
  }
  renderQuizWorkspace();
}

function renderQuizWorkspace() {
  const el = $('#quizzesPane');
  const back = `<button class="ghost" id="quizBack" type="button">&larr; All readings</button>`;

  if (quizCtx.status === 'loading') {
    el.innerHTML = `${back}<h1>${esc(quizCtx.title)}</h1><p class="hint">Writing questions from this text… this can take several seconds.</p>`;
    wireQuizBack(el);
    return;
  }
  if (quizCtx.status !== 'ready') {
    el.innerHTML = `
      ${back}<h1>${esc(quizCtx.title)}</h1>
      <p class="hint">Make a short quiz based on this exact text.</p>
      ${quizCtx.status === 'error' ? `<div class="card"><b>Couldn't make the quiz</b><p style="margin:6px 0 0">${esc(quizCtx.error)}</p></div>` : ''}
      <div class="row"><button class="btn" id="makeQuiz" type="button">${quizCtx.status === 'error' ? 'Try again' : 'Make my quiz'}</button></div>`;
    wireQuizBack(el);
    $('#makeQuiz', el).addEventListener('click', loadQuizQuestions);
    return;
  }

  const qs = quizCtx.questions;
  const done = Object.keys(quizCtx.answers).length;
  const score = Object.entries(quizCtx.answers).filter(([i, a]) => a === qs[i].answer).length;
  el.innerHTML = `
    ${back}<h1>${esc(quizCtx.title)}</h1>
    <p class="hint">${done ? `${score} of ${done} correct so far.` : 'Pick the best answer for each question.'}${quizCtx.source === 'demo' ? ' (Offline demo quiz)' : ''}</p>
    ${qs.map((q, i) => {
      const a = quizCtx.answers[i]; const answered = a !== undefined;
      return `<div class="card">
        <span class="tag">${esc(q.kind || 'Question')}</span>
        <div><b>${esc(q.q)}</b></div>
        ${q.options.map((o, j) => `<button class="opt ${answered ? (j === q.answer ? 'right' : j === a ? 'wrong' : '') : ''}" data-q="${i}" data-o="${j}" ${answered ? 'disabled' : ''}>${esc(o)}</button>`).join('')}
        ${answered ? `<div class="explain">${esc(q.why || '')}${q.evidence ? ` Look at: “${esc(q.evidence)}”` : ''}</div>` : ''}
      </div>`;
    }).join('')}
    <div class="row"><button class="btn alt" id="newQuiz" type="button">Make new questions</button></div>`;
  wireQuizBack(el);
  $$('.opt', el).forEach(b => b.addEventListener('click', () => {
    const i = +b.dataset.q, j = +b.dataset.o;
    quizCtx.answers[i] = j;
    const correct = j === qs[i].answer;
    saveQuizAnswer(qs[i].q, qs[i].options[j], correct);
    renderQuizWorkspace();
  }));
  $('#newQuiz', el).addEventListener('click', loadQuizQuestions);
}
function wireQuizBack(el) {
  $('#quizBack', el).addEventListener('click', () => {
    quizCtx = { readingId: null, isDemo: false, demoId: null, isPasted: false, title: '', text: '', sessionId: null, questions: [], answers: {}, status: 'idle', error: '', source: null };
    renderQuizzesView();
  });
}
