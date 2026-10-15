/* CONFIG -- these two settings now live in config.js, which is the one
   small file that differs between running on your own machine and a
   deployed site. Editing a 2-line file at deploy time is a lot harder to
   get wrong than editing this one. The defaults below apply only if an
   older config.js is still in place and does not define them.

   API_BASE is the backend's address. Three endpoints get called: POST
   /partner (debug hints), POST /quiz (quiz questions), POST /checkin
   (check-in prompts).

   Two ways this goes wrong quietly, both worth knowing, because neither
   shows an error. callBackend() returns null on any failure and the app
   falls back to built-in demo answers, so the app keeps working and
   simply stops being the real AI:

   1. Pointing at localhost from a deployed site. "http://localhost:8000"
      means the visitor's own machine, so it only ever works for whoever
      is running the backend on the same computer.
   2. Mixed content. If the site is served over https and API_BASE is
      plain http, the browser blocks every call before it is sent. The
      backend needs its own https address, not an IP with a port. */
const CONFIG = Object.assign(
  { API_BASE: "http://localhost:8000", CHECKIN_EVERY_SEC: 90 },
  window.APP_CONFIG || {}
);

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

/* Built-in practice readings -- plain text, blank line between paragraphs --
   so both the Readings view and the Quizzes view (which has no reader open)
   can use them without needing a teacher or the database at all. */
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
  },
  {
    id: 'demo-noli',
    title: 'Kabanata 1',
    text:
`Isang marangyang salu-salo ang ipinag-anyaya ni Don Santiago de los Santos na higit na popular sa taguring Kapitan Tiago.  Ang handaan ay gagawin sa kanyang bahay na nasa daang Anluwage na karatig ng Ilog-Binundok.

Ang paayaya ay madaling kumalat sa lahat ng sulok ng Maynila.  Bawat isa ay gustong dumalo sapagkat ang mayamang Kapitan ay kilala bilang isang mabuting tao, mapagbigay at laging bukas ang palad sa mga nangangailangan.  Dahil dito, ang iba ay nababalino kung ano ang isusuot at sasabihin sa mismong araw ng handaan.

Nang gabing iyon dagsa ang mga panauhin na gaya ng dapat asahan.  Puno ang bulwagan.  Ang nag-iistima sa mgta bisita ay si Tiya Isabel, isang matandang babae na pinsan ng may-bahay.  Kabilang sa mga bisita sina Tinyente ng guardia civil, Pari Sibyla, ang Kura paroko ng Binundok, si Padre Damaso na madaldal at mahahayap ang mga salita at dalawang Paisano.  Ang isa ay kararating lamang sa Pilipinas.`
  }
];
const DEMO_READING_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 3.6 4.8 7.4l7.2 3.8 7.2-3.8Z"/><path d="M4.8 7.4v9l7.2 3.8 7.2-3.8v-9"/><path d="M12 11.2v9"/></svg>`;

/* ---------- State for the reading workspace (Goal / Debug / Check-in / Progress) ---------- */
const state = {
  goal: null,            // { text, confidence }
  threads: [],           // debug sessions
  pending: null,         // current text selection { text, paragraph, range }
  selectionNote: '',
  checkin: {
    on: false, everySec: CONFIG.CHECKIN_EVERY_SEC, nextAt: 0, pending: null, i: 0, responses: [], fetching: false, shown: [],
    // Whether a check-in shows as a floating popup the moment it's ready, instead of
    // only a quiet dot on the Monitor tab -- the popup used to be very easy to miss.
    popupOn: (() => { try { return localStorage.getItem('checkinPopupOn') !== 'off'; } catch (e) { return true; } })()
  },
  support: { score: 0, level: 'steady' },  // see nudgeSupport() below
  responseLanguage: (() => {              // 'auto' | 'en' | 'fil' -- which language the coach replies in
    try { return localStorage.getItem('responseLanguage') || 'auto'; } catch (e) { return 'auto'; }
  })(),
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
    : 'Checking in a little more often: a few spots have been tricky.';
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

   A failed save still lets the session keep working locally, because losing
   the reading mid-way would be worse for the student. But it is no longer
   silent. Every write below reports through noteSaveFailure(), which warns
   to the console as before, tells the student once, and leaves a marker in
   the sidebar for the rest of the session.

   The reason this matters more than a normal app: ensureSession() is the
   first step of every other write, so one failed insert here turns all of
   them into no-ops. Without a visible marker a student can read a whole
   text, flag passages, work through hints and answer check-ins, and nothing
   is recorded, with nobody finding out until the dashboard is opened later
   and the session is not there. */

/* What has failed to save this session. Not reset when a later save works:
   the rows already lost stay lost, and that is what the marker is about. */
const saveHealth = { failures: 0, offline: false, last: '' };

function renderSaveWarning() {
  let box = $('#saveWarning');
  if (!box) {
    box = document.createElement('div');
    box.id = 'saveWarning';
    box.className = 'save-warning';
    // The sidebar is on every signed-in view; fall back to the body so the
    // marker still appears if the layout changes.
    ($('.sidebar-foot') || document.body).prepend(box);
  }
  const n = saveHealth.failures;
  box.innerHTML = `<b>Not saving</b><span>${n} write${n === 1 ? '' : 's'} failed. `
    + `Your work is on screen but not in the database.</span>`;
  box.title = saveHealth.last;
}

/* Called instead of a bare console.warn wherever a write can fail. */
function noteSaveFailure(what, error) {
  const message = (error && error.message) || String(error || 'unknown error');
  console.warn('Could not save ' + what + ':', message);
  saveHealth.failures += 1;
  saveHealth.last = what + ': ' + message;
  if (saveHealth.failures === 1 && typeof showToast === 'function') {
    showToast('Your work is not being saved. Tell your teacher before you carry on.', 'bug');
  }
  renderSaveWarning();
}

/* The offline fallback is its own kind of quiet: the app keeps answering,
   but from the built-in demo replies rather than the model, so a session
   collected this way is not measuring the model at all. */
function noteOfflineFallback() {
  if (saveHealth.offline) return;
  saveHealth.offline = true;
  if (typeof showToast === 'function') {
    showToast('The backend could not be reached, so replies are coming from the built-in offline examples.', 'bug');
  }
}
async function ensureSession() {
  if (state.session.id) return state.session.id;
  if (!state.user) return null;
  const { data, error } = await supa.from('sessions').insert({
    student_id: state.user.id,
    reading_id: state.reading.id,
    goal: state.goal ? state.goal.text : null,
    confidence: state.goal ? state.goal.confidence : null
  }).select('id').single();
  if (error) { noteSaveFailure('the reading session', error); return null; }
  state.session.id = data.id;
  return data.id;
}
async function dbUpdateSession(fields) {
  const id = state.session.id; if (!id) return;
  const { error } = await supa.from('sessions').update(fields).eq('id', id);
  if (error) noteSaveFailure('the session update', error);
}
async function dbInsertDebugEvent(t) {
  const sid = await ensureSession(); if (!sid) return;
  const { data, error } = await supa.from('debug_events')
    .insert({ session_id: sid, passage: t.text, problem: t.problem,
              problem_text: t.problemText || null, status: 'open' })
    .select('id').single();
  if (error) { noteSaveFailure('this flagged passage', error); return; }
  t.dbId = data.id;
}
async function dbUpdateDebugStatus(t) {
  if (!t.dbId) return;
  const { error } = await supa.from('debug_events').update({ status: t.status }).eq('id', t.dbId);
  if (error) noteSaveFailure('the change to this flagged passage', error);
}
/* Saves the actual back-and-forth for this debug thread -- needs
   hint_transcript_migration.sql run once. Without it the column doesn't
   exist and this just fails quietly, same as any other save error. */
async function dbSaveTranscript(t) {
  if (!t.dbId) return;
  const transcript = t.msgs.filter(m => !m.typing).map(m => ({ from: m.from, text: m.text, demo: !!m.demo }));
  const { error } = await supa.from('debug_events').update({ transcript }).eq('id', t.dbId);
  if (error) noteSaveFailure('the conversation about this passage', error);
}
async function dbInsertCheckin(prompt, level, answer) {
  const sid = await ensureSession(); if (!sid) return;
  const { error } = await supa.from('checkin_events').insert({ session_id: sid, prompt, level, answer });
  if (error) noteSaveFailure('this check-in answer', error);
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
    options: ['Mara told him in a letter', 'He recognized the keepers’ traditional signal', 'He could see her face from the boat', 'He was her grandfather’s friend'], answer: 1,
    why: 'He had never met her, so he must have known her from something else: the old signal that the keepers use.' }
];

/* ---------- The four kinds of trouble a student can pick ---------- */
const PROBLEMS = {
  vocab:   { label: 'A word I don’t know', sub: 'Unfamiliar or tricky word' },
  lost:    { label: 'I lost the point',       sub: 'I read it but it didn’t land' },
  connect: { label: 'Ideas don’t connect', sub: 'How does this link to before?' },
  why:     { label: 'Why did this happen?',   sub: 'Something is implied, not stated' }
};
/* A student whose trouble is none of the four types above types it into
   the box under them instead. That is sent as problem = "other" with the
   wording in problem_text, so "other" is a real problem key even though it
   is deliberately not a button here. */

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
    'Look at the sentence right before this one. How are they linked: one causes the other, they contrast, or this adds more? Try putting “because”, “but”, or “and then” between them. Which fits?',
    'Find the last thing you clearly understood. What changed between that point and this one? Name the link in one short phrase.'
  ],
  why: [
    'Split it in two. What does the text say directly, and what are you filling in yourself? Which earlier detail supports your guess?',
    'Ask: what would have to be true for this to make sense? Look back for a clue that points that way.'
  ],
  other: [
    'Say the trouble in one sentence, as plainly as you can. Then reread just this part and watch for the exact word or idea it sits on.',
    'Start from what you do follow here. Then name the first thing that stops making sense. That spot is usually the one worth working on.'
  ],
  followup: [
    'Compare your version with the original: did you keep who did what? If something is missing, add it. Then tell me below if it makes sense now.',
    'Check one thing: does your explanation still hold when you read the next sentence? If yes, mark it as clear.'
  ],
  stuck: [
    'That’s okay. This one is flagged so your teacher can see it. Break it down: what is the subject, what is it doing, and which earlier line does it depend on?'
  ],
  // Shown after a student has struggled through a couple of follow-ups in a
  // row -- a more direct nudge instead of yet another open question.
  giveup: [
    'No worries -- here is the key idea plainly: focus on who is doing what in this sentence, and what changed right before it. Reread it once more with that in mind, then tell me if it clicks.',
    'That is okay, let us just walk through it directly: look at the subject of the sentence and the one earlier detail it depends on. Put those two together and see if it makes sense now.'
  ]
};
const pick = arr => arr[Math.floor(Math.random() * arr.length)];

/* ---------- Talking to the backend ---------- */
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
    noteOfflineFallback();
  }
  return null;
}

/* The backend reports which kind of turn it decided this was (debug,
   followup, stuck, success, repeat) alongside the reply. That has to be
   kept and sent back with the history: the backend works out how stuck a
   student is by counting how many earlier coach turns were "stuck", so if
   the history arrives without it the count is always zero, the escalation
   stays on its first rung, and a student who keeps saying "I don't know"
   is asked an easier question forever instead of eventually being given
   the explanation. */
async function askPartner(payload) {
  const d = await callBackend('/partner', payload);
  if (d && d.reply) return { reply: d.reply, mode: d.mode || null, demo: false };
  await new Promise(r => setTimeout(r, 450));
  let reply;
  if (payload.mode === 'debug') reply = pick(DEMO_REPLIES[payload.problem] || DEMO_REPLIES.lost);
  else if (payload.mode === 'giveup') reply = pick(DEMO_REPLIES.giveup);
  else if (payload.mode === 'stuck') reply = pick(DEMO_REPLIES.stuck);
  else reply = pick(DEMO_REPLIES.followup);
  // Offline replies carry the mode the frontend asked for, so the ladder
  // still behaves sensibly when the backend is unreachable.
  return { reply, mode: payload.mode || null, demo: true };
}

function partnerPayload(mode, t, message) {
  return {
    mode, problem: t.problem, problem_text: t.problemText || null,
    passage: t.text, paragraph: t.paragraph,
    goal: state.goal && state.goal.text, goal_hint: t.goalHint || null,
    support: state.support.level, message: message || null,
    // 'auto' means let the backend detect the passage's language itself;
    // anything else is an explicit override from the Progress tab toggle.
    response_language: state.responseLanguage === 'auto' ? null : state.responseLanguage,
    history: t.msgs.filter(m => !m.typing).map(m => ({
      role: m.from === 'you' ? 'student' : 'coach',
      text: m.text,
      // the backend counts coach turns with mode "stuck" to decide how
      // much help to give next, so this has to survive the round trip
      mode: m.mode || null
    }))
  };
}

function setResponseLanguage(lang) {
  state.responseLanguage = lang;
  try { localStorage.setItem('responseLanguage', lang); } catch (e) { /* private browsing etc -- still works for this session */ }
  renderStats();
}

/* ---- Offline check-in prompts: backend-unavailable fallback ---- */
const OFFLINE_CHECKINS = [
  'Pause for a moment. In one sentence, what have you read so far?',
  'Is what you just read matching your goal? What is still missing?',
  'What is the most confusing part so far? Try selecting it in the text.',
  'What do you think happens or is explained next?',
  'Which sentence would you reread if you had to explain this to a friend?'
];

/* ---------- Elements ---------- */
const reader = $('#reader');

/* ---------- Side-panel tabs (Goal / Debug / Check-in / Progress) ---------- */
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
function goalFormHtml(g) {
  return `
    <label class="f" for="goalText">What do you want to get out of this text?</label>
    <textarea id="goalText" placeholder="Example: Understand why Mara uses the lamp signal.">${g ? esc(g.text) : ''}</textarea>
    <label class="f">How well do you expect to understand it?</label>
    <div class="scale" id="conf">
      ${[1,2,3,4,5].map(n => `<button type="button" data-n="${n}" aria-pressed="${g && g.confidence === n}">${n}</button>`).join('')}
    </div>
    <div class="scale-cap"><span>Not at all</span><span>Very well</span></div>`;
}
/* Shared by the Goal tab and the start-of-session popup, so the two never
   drift out of sync. `scope` is the element the form lives in; `onDone`
   runs after the goal is saved (closing the popup, switching tabs, etc). */
function wireGoalForm(scope, g, onDone) {
  let conf = g ? g.confidence : null;
  $$('#conf button', scope).forEach(b => b.addEventListener('click', () => {
    conf = +b.dataset.n; $$('#conf button', scope).forEach(x => x.setAttribute('aria-pressed', String(x === b)));
  }));
  return () => {
    const text = $('#goalText', scope).value.trim();
    if (!text) { $('#goalText', scope).focus(); return false; }
    state.goal = { text, confidence: conf };
    logEvent('goal_set', { goal: text, confidence: conf });
    $('#goalLine').textContent = 'Goal: ' + text;
    if (state.session.id) dbUpdateSession({ goal: text, confidence: conf });
    else ensureSession();
    // Low confidence starts check-ins closer together; high confidence starts
    // relaxed. This has to run before startCheckins(), or the first countdown
    // is set from the old level and never reflects what they just told us.
    nudgeSupport({1:2, 2:1, 3:0, 4:-1, 5:-1}[conf ?? 3]);
    if (!state.checkin.on) startCheckins();
    if (onDone) onDone();
    return true;
  };
}
function renderGoal() {
  const g = state.goal;
  const el = $('#p-goal');
  el.innerHTML = `
    <h3>Before you read</h3>
    <p class="lead">Set a purpose. Readers who know what they want from a text notice sooner when they lose it.</p>
    ${goalFormHtml(g)}
    <div class="row" style="margin-top:14px">
      <button class="btn" id="setGoal" type="button">${g ? 'Update goal' : 'Set goal and start reading'}</button>
    </div>
    ${g ? '<p class="small" style="margin-top:10px">Goal saved. Check-ins are on. Change them in the Check-in tab.</p>' : ''}`;
  const save = wireGoalForm(el, g, () => { renderGoal(); renderDebug(); setTab('debug'); });
  $('#setGoal', el).addEventListener('click', save);
}

/* ---- Goal popup: shown once per session the moment a reading opens, so
   setting a purpose isn't something a student can just scroll past. The
   Goal tab still exists for changing it later. ---- */
function ensureGoalDialog() {
  let dlg = $('#goalDlg');
  if (dlg) return dlg;
  dlg = document.createElement('dialog');
  dlg.id = 'goalDlg';
  document.body.appendChild(dlg);
  return dlg;
}
function showGoalPopup() {
  const dlg = ensureGoalDialog();
  dlg.innerHTML = `
    <form method="dialog">
      <h3 style="margin:0 0 4px">Before you start reading</h3>
      <p class="small" style="margin:0 0 14px">Set a purpose for this text -- it helps you notice sooner if you lose the thread.</p>
      ${goalFormHtml(null)}
      <div class="row" style="margin-top:14px; justify-content:flex-end">
        <button class="btn alt" type="button" id="goalSkip">Skip for now</button>
        <button class="btn" type="button" id="goalSave">Start reading</button>
      </div>
    </form>`;
  const save = wireGoalForm(dlg, null, () => { dlg.close(); renderGoal(); renderDebug(); });
  $('#goalSave', dlg).addEventListener('click', () => { if (save()) dlg.close(); });
  $('#goalSkip', dlg).addEventListener('click', () => dlg.close());
  dlg.showModal();
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
  setTab('debug');
  renderDebug();
}
function closestP(node) {
  const el = node.nodeType === 1 ? node : node.parentElement;
  return el ? el.closest('#reader p') : null;
}

/* ---------- Debug tab ---------- */
function shorten(s, n = 90) { return s.length > n ? s.slice(0, n - 1) + '…' : s; }

/* What the tag on an open thread says. For the four fixed types that's their
   label. For a typed one it's the student's own wording, which tells them (and
   the teacher reading it back later) far more than the word "Other" would.
   The fallback matters: a thread restored from an older session may carry a
   problem key this build no longer has, and a bare PROBLEMS[key].label throws
   on that and takes the whole Debug tab down with it. */
function threadProblemLabel(t) {
  if (t.problem === 'other' && t.problemText) return shorten(t.problemText, 48);
  return (PROBLEMS[t.problem] || PROBLEMS.lost).label;
}

function renderDebug() {
  const el = $('#p-debug');
  const p = state.pending;
  const goalWord = p ? goalOverlapWord(p.text) : null;
  const chooser = p ? `
    <div class="card">
      <p class="small" style="margin:0 0 4px">You selected</p>
      <blockquote class="quote" style="margin:0 0 10px; padding-left:10px; border-left:3px solid var(--accent); font-family:var(--serif); font-size:15px">${esc(shorten(p.text, 160))}</blockquote>
      ${goalWord ? `<p class="small" style="margin:0 0 8px; color:var(--accent)">This looks related to your goal (“${esc(goalWord)}”), worth sorting out before moving on.</p>` : ''}
      <p class="small" style="margin:0 0 8px">What is going wrong?</p>
      <div class="chips">
        ${Object.entries(PROBLEMS).map(([k, v]) => `<button type="button" data-problem="${k}">${v.label}<small>${v.sub}</small></button>`).join('')}
      </div>
      <form id="ownProblemForm" style="margin-top:12px">
        <label class="f" for="ownProblem">Or say it in your own words</label>
        <div class="row">
          <input type="text" id="ownProblem" style="flex:1" placeholder="What is confusing about it?" autocomplete="off">
          <button class="btn" type="submit">Start</button>
        </div>
      </form>
      <div class="row" style="margin-top:10px"><button class="btn alt" id="cancelSel" type="button">Cancel</button></div>
    </div>` : `
    <div class="card">
      <b>Debug a confusing spot</b>
      <p class="small" style="margin:4px 0 0">${state.selectionNote ? esc(state.selectionNote) : 'Select a word, phrase, or sentence in the text. It gets a red squiggle and we work it out together.'}</p>
    </div>`;
  const threads = state.threads.slice().reverse().map(t => `
    <div class="card ${t.status}" data-id="${t.id}">
      <span class="tag ${t.status}">${t.status === 'resolved' ? 'Clear now' : t.status === 'stuck' ? 'Still stuck' : esc(threadProblemLabel(t))}</span>
      <blockquote class="quote">${esc(shorten(t.text, 140))}</blockquote>
      ${t.msgs.map(m => `<div class="msg ${m.from}${m.typing ? ' typing' : ''}">${esc(m.text)}${m.demo ? '<div class="small" style="margin-top:4px">Offline demo reply (backend not reached)</div>' : ''}</div>`).join('')}
      ${t.status !== 'resolved' && !t.busy ? `
        <form class="reply" data-id="${t.id}">
          <input type="text" placeholder="Explain it in your own words" aria-label="Your reply" autocomplete="off">
          <button class="btn" type="submit">Send</button>
        </form>
        <div class="row" style="margin-top:8px">
          <button class="btn alt" type="button" data-act="resolve" data-id="${t.id}">Makes sense now</button>
          ${t.gaveUp ? '' : `<button class="btn alt" type="button" data-act="stuck" data-id="${t.id}">Still stuck</button>`}
        </div>` : ''}
      ${t.status === 'resolved' && t.autoResolved ? `
        <div class="row" style="margin-top:8px; align-items:center">
          <span class="small">Marked as cleared up, since your answer showed you had it.</span>
          <button class="btn alt" type="button" data-act="reopen" data-id="${t.id}">Still confused</button>
        </div>` : ''}
    </div>`).join('');
  el.innerHTML = chooser + threads;

  $$('.chips button', el).forEach(b => b.addEventListener('click', () => startDebug(b.dataset.problem)));
  const c = $('#cancelSel', el); if (c) c.addEventListener('click', () => { state.pending = null; getSelection().removeAllRanges(); renderDebug(); });
}
$('#p-debug').addEventListener('submit', e => {
  const own = e.target.closest('#ownProblemForm');
  if (own) {
    e.preventDefault();
    const v = own.querySelector('input').value.trim();
    if (!v) return;
    startDebug('other', v);
    return;
  }
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
  // Undoing an automatic close. The thread goes back to open and the reply
  // box comes back, and the reopen is logged, so a coach that closes threads
  // too eagerly shows up in the data instead of hiding as a clean resolve.
  if (b.dataset.act === 'reopen') {
    const t = state.threads.find(x => x.id === id);
    if (t) { t.autoResolved = false; setStatus(id, 'open'); }
  }
});

let threadSeq = 0;
function startDebug(problem, problemText) {
  const p = state.pending; if (!p) return;
  const id = ++threadSeq;
  const mark = wrapRange(p.range, id);
  getSelection().removeAllRanges();
  const t = {
    id, problem, problemText: problemText || null,
    text: p.text, paragraph: p.paragraph, status: 'open', busy: true,
    attempts: 0, gaveUp: false, goalHint: goalOverlapWord(p.text),
    msgs: [{ from: 'ai', text: 'Thinking…', typing: true }], mark
  };
  state.threads.push(t);
  state.pending = null;
  logEvent('debug_start', { id, problem, problem_text: problemText || null, passage: p.text });
  dbInsertDebugEvent(t);
  renderDebug();
  askPartner(partnerPayload('debug', t))
    .then(r => { t.busy = false; t.msgs = [{ from: 'ai', text: r.reply, mode: r.mode, demo: r.demo }]; renderDebug(); dbSaveTranscript(t); });
}
function wrapRange(range, id) {
  const mark = document.createElement('mark');
  mark.className = 'dbg'; mark.dataset.id = id; mark.tabIndex = 0;
  try { mark.appendChild(range.extractContents()); range.insertNode(mark); return mark; }
  catch (e) { return null; }
}
async function sendFollowup(id, message) {
  const t = state.threads.find(x => x.id === id); if (!t) return;
  t.attempts = (t.attempts || 0) + 1;
  // After a couple of follow-ups with no resolution, stop asking another
  // question and have the coach just explain the key idea directly.
  const mode = t.attempts >= 2 ? 'giveup' : 'followup';
  const payload = partnerPayload(mode, t, message);
  t.msgs.push({ from: 'you', text: message });
  t.busy = true; t.msgs.push({ from: 'ai', text: 'Thinking…', typing: true });
  logEvent('debug_reply', { id, message, mode });
  renderDebug();
  const r = await askPartner(payload);
  t.msgs = t.msgs.filter(m => !m.typing); t.msgs.push({ from: 'ai', text: r.reply, mode: r.mode, demo: r.demo }); t.busy = false;
  if (mode === 'giveup') t.gaveUp = true;
  // The backend reports mode "success" when the student's reply shows they
  // have understood this spot. Closing the thread on that is the whole point
  // of the signal: otherwise a spot the student clearly got stays "open"
  // until they remember to press a button, and an open spot looks the same
  // as an abandoned one in the teacher dashboard and in the 3.6.1 counts.
  // A backend that does not send a mode leaves r.mode undefined, so this is
  // simply inert there and the buttons still work as before.
  if (r.mode === 'success' && t.status === 'open' && !r.demo) {
    t.autoResolved = true;
    setStatus(id, 'resolved', 'auto');   // re-renders
  } else {
    renderDebug();
  }
  dbSaveTranscript(t);
}
/* `how` records who decided: "student" when they pressed a button, "auto"
   when the backend classified their reply as understanding it. Both end up
   as status "resolved", and the resolved counts are evaluation data, so the
   log has to keep them apart. A thread can also come back to "open", which
   the old version could not express: anything that was not "resolved" was
   logged as stuck and pushed the support level up. */
function setStatus(id, status, how = 'student') {
  const t = state.threads.find(x => x.id === id); if (!t) return;
  t.status = status;
  if (t.mark) {
    t.mark.classList.toggle('stuck', status === 'stuck');
    t.mark.classList.toggle('resolved', status === 'resolved');
  }
  const EVENTS = { resolved: 'debug_resolved', stuck: 'debug_stuck', open: 'debug_reopened' };
  logEvent(EVENTS[status] || 'debug_status', { id, how });
  if (status === 'resolved') nudgeSupport(-1);
  else if (status === 'stuck') nudgeSupport(2);
  dbUpdateDebugStatus(t);
  renderDebug();
}
async function markStuck(id) {
  const t = state.threads.find(x => x.id === id); if (!t) return;
  t.status = 'stuck'; t.busy = true; t.gaveUp = true;
  if (t.mark) t.mark.classList.add('stuck');
  logEvent('debug_stuck', { id });
  dbUpdateDebugStatus(t);
  // Pressing "Still stuck" has to reach the backend as something the
  // student said. The backend decides what kind of turn this is from
  // ask.message, and its first branch is "no message means a fresh first
  // hint" -- so with an empty message this button produced an opening
  // hint rather than the escalation it exists for. Recording it as a
  // student line also keeps the saved transcript faithful, which matters
  // because that transcript is evaluation data.
  const stuckSays = "I'm still stuck.";
  t.msgs.push({ from: 'you', text: stuckSays });
  const r = await askPartner(partnerPayload('stuck', t, stuckSays));
  t.msgs.push({ from: 'ai', text: r.reply, mode: r.mode, demo: r.demo }); t.busy = false;
  renderDebug();
  dbSaveTranscript(t);
}
reader.addEventListener('click', e => {
  const m = e.target.closest('mark.dbg'); if (!m) return;
  setTab('debug');
  const card = $(`#p-debug .card[data-id="${m.dataset.id}"]`);
  if (card) { card.scrollIntoView({ block: 'nearest' }); m.classList.add('active'); setTimeout(() => m.classList.remove('active'), 1200); }
});

/* ---- Check-in tab + floating popup ---- */
function startCheckins() {
  const c = state.checkin; c.on = true; c.nextAt = Date.now() + c.everySec * 1000;
}
async function fireCheckin() {
  const c = state.checkin;
  if (c.fetching || c.pending) return;
  c.fetching = true;
  if (currentTab() === 'monitor') renderCheckin();
  const d = await callBackend('/checkin', {
    goal: state.goal && state.goal.text,
    text: reader.innerText.trim().slice(0, 6000),
    struggles: state.threads.map(t => t.text).slice(-5),
    previous: c.shown.slice(-5),
    response_language: state.responseLanguage === 'auto' ? null : state.responseLanguage
  });
  let prompt, source;
  if (d && d.prompt) { prompt = d.prompt; source = 'ai'; }
  else { prompt = OFFLINE_CHECKINS[c.i++ % OFFLINE_CHECKINS.length]; source = 'demo'; }
  c.fetching = false; c.pending = prompt; c.shown.push(prompt);
  logEvent('checkin_prompt', { prompt, source });
  updateBadges();
  if (currentTab() === 'monitor') renderCheckin();
  // A popup is more likely to actually get seen than a dot on a tab the
  // student isn't looking at -- that was the main complaint about check-ins.
  // Always show it (even on the Monitor tab itself) so testing via "Ask me
  // now" is never a silent no-op.
  if (c.popupOn) showCheckinPopup();
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
    <div class="switch">
      <div><b>Pop up when ready</b><div class="small">Show it right away instead of waiting for you to open this tab</div></div>
      <button type="button" id="cinPopupToggle" role="switch" aria-checked="${c.popupOn}" aria-label="Turn the check-in popup on or off"></button>
    </div>
    <label class="f" for="cinEvery">How often (baseline)</label>
    <select id="cinEvery">
      ${[30, 60, 90, 180, 300].map(s => `<option value="${s}" ${s === c.everySec ? 'selected' : ''}>Every ${s < 60 ? s + ' seconds' : (s / 60) + (s === 60 ? ' minute' : ' minutes')}</option>`).join('')}
    </select>
    <div class="row" style="margin:12px 0"><button class="btn alt" id="askNow" type="button">Ask me now</button></div>
    <div id="cinCard"></div>
    ${c.responses.length ? `<p class="small">${c.responses.length} check-in${c.responses.length > 1 ? 's' : ''} answered</p>` : ''}
    <p class="small" style="margin-top:16px; padding-top:12px; border-top:1px solid var(--w-line)">
      Check-ins are part of this project: short, occasional pauses that help you notice sooner if you've
      drifted from the text, and help your teacher see where students are getting stuck. Answering honestly
      (including "I'm lost") is what makes them useful -- it's not graded.
    </p>`;

  $('#cinToggle', el).addEventListener('click', () => { c.on ? (c.on = false) : startCheckins(); renderCheckin(); });
  $('#cinPopupToggle', el).addEventListener('click', () => {
    c.popupOn = !c.popupOn;
    try { localStorage.setItem('checkinPopupOn', c.popupOn ? 'on' : 'off'); } catch (e) { /* private browsing etc */ }
    renderCheckin();
  });
  $('#cinEvery', el).addEventListener('change', e => { c.everySec = +e.target.value; if (c.on) c.nextAt = Date.now() + effectiveCheckinSec() * 1000; });
  $('#askNow', el).addEventListener('click', () => { if (c.pending || c.fetching) renderCheckin(); else fireCheckin(); });
  drawCheckinCard();
  updateCountdown();
  updateBadges();
}
/* Shared by the Monitor-tab card and the floating popup. `onAfter` lets
   the popup close itself once answered; the tab card just re-renders. */
function checkinCardHtml(prompt) {
  return `
    <b>${esc(prompt)}</b>
    <textarea class="cinAnswer" style="margin-top:8px" placeholder="Type a quick answer (optional)"></textarea>
    <p class="small" style="margin:10px 0 6px">How is it going?</p>
    <div class="row">
      <button class="btn alt" data-level="clear" type="button">I follow it</button>
      <button class="btn alt" data-level="fuzzy" type="button">A bit fuzzy</button>
      <button class="btn alt" data-level="lost" type="button">I'm lost</button>
    </div>`;
}
function wireCheckinCard(box, onAfter) {
  $$('button[data-level]', box).forEach(b => b.addEventListener('click', () => {
    const c = state.checkin;
    const level = b.dataset.level;
    const answerText = ($('.cinAnswer', box) || {}).value?.trim() || '';
    c.responses.push({ level, prompt: c.pending, answer: answerText });
    logEvent('checkin_response', { level, prompt: c.pending, answer: answerText });
    dbInsertCheckin(c.pending, level, answerText);
    c.pending = null;
    if (c.on) c.nextAt = Date.now() + effectiveCheckinSec() * 1000;
    if (currentTab() === 'monitor') renderCheckin(); else updateBadges();
    if (onAfter) onAfter(level);
  }));
}
function drawCheckinCard() {
  const c = state.checkin, box = $('#cinCard'); if (!box) return;
  if (c.fetching) { box.innerHTML = '<p class="small">Thinking of a question…</p>'; return; }
  if (!c.pending) { box.innerHTML = ''; return; }
  box.innerHTML = `<div class="card">${checkinCardHtml(c.pending)}</div>`;
  wireCheckinCard(box, (level) => {
    if (level !== 'clear') {
      $('#cinCard').innerHTML = `<div class="card"><b>Try this</b><p style="margin:6px 0 0">Find the last sentence you understood, then the first one you didn't. Select that spot in the text and debug it.</p>
        <div class="row" style="margin-top:8px"><button class="btn alt" type="button" id="goDebugFromCheckin">Open Debug tab</button></div></div>`;
      const go = $('#goDebugFromCheckin'); if (go) go.addEventListener('click', () => setTab('debug'));
    } else {
      drawCheckinCard();
    }
  });
}
/* Floating, non-blocking popup -- sits in the corner so a student can keep
   reading, answer right there, or dismiss it and answer later from the
   Monitor tab (the dot badge stays on until it's answered). */
function ensureCheckinPopupHost() {
  let host = $('#checkinPopupHost');
  if (!host) {
    host = document.createElement('div');
    host.id = 'checkinPopupHost';
    document.body.appendChild(host);
  }
  return host;
}
function showCheckinPopup() {
  const c = state.checkin;
  if (!c.pending) return;
  const host = ensureCheckinPopupHost();
  host.innerHTML = `
    <div class="checkin-popup" role="dialog" aria-label="Check-in">
      <button class="checkin-popup-close" type="button" aria-label="Dismiss, answer later">✕</button>
      ${checkinCardHtml(c.pending)}
    </div>`;
  const popup = $('.checkin-popup', host);
  $('.checkin-popup-close', popup).addEventListener('click', () => { host.innerHTML = ''; });
  wireCheckinCard(popup, () => { host.innerHTML = ''; });
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
const SUPPORT_LABEL = { steady: 'Steady: check-ins at your normal pace', closer: 'Checking in a bit closer together', extra: 'Extra support: check-ins are more frequent' };
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
    <p class="small" style="margin-top:10px">Expected to understand it: <b>${g.confidence ?? 'N/A'}/5</b> &middot; Actually did: <b>${g.finalConfidence}/5</b></p>`;
  $('#p-stats').innerHTML = `
    <h3>Your session</h3>
    <p class="lead">${state.goal ? 'Goal: ' + esc(state.goal.text) : 'No goal set yet.'}</p>
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
      <button class="btn" id="endSessionBtn" type="button">End session</button>
      <button class="btn alt" id="exportBtn" type="button">Download session data</button>
    </div>
    <p class="small" style="margin-top:10px">The download is a JSON file of everything above. The teacher dashboard reads from the database directly.</p>
    <p class="small" style="margin-top:4px"><a href="#" id="startOverLink">Start over on this same reading instead</a></p>`;
  $('#endSessionBtn').addEventListener('click', endSession);
  $('#exportBtn').addEventListener('click', exportLog);
  $('#startOverLink').addEventListener('click', (e) => { e.preventDefault(); if (confirm('Clear marks, goal, and progress for this reading and start it over?')) resetSession(); });
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
    session: { id: null },   // a new reading means a new database session
    startedAt: Date.now(), log: []
  });
  Object.assign(state.checkin, { on: false, pending: null, responses: [], i: 0, fetching: false, shown: [] });
  $('#goalLine').textContent = 'No goal set yet';
  renderGoal(); renderDebug(); setTab('goal'); updateBadges();
  const host = $('#checkinPopupHost'); if (host) host.innerHTML = '';
}
/* "End session" leaves this reading entirely and returns to the picker
   (distinct from "Start over", which clears progress but stays on the
   same text) -- what gets saved is already saved either way. */
function endSession() {
  if (!confirm('End this reading session and go back to all readings?')) return;
  resetSession();
  showReadingPicker();
}

function setReaderText(title, byline, bodyParagraphsHtml) {
  $('#docTitle').textContent = title;
  $('#docBy').textContent = byline;
  reader.innerHTML = bodyParagraphsHtml;
}
function paragraphsToHtml(raw) {
  return raw.split(/\n\s*\n/).map(p => `<p>${esc(p.replace(/\s*\n\s*/g, ' ').trim())}</p>`).join('');
}

/* ---------- Use your own text: one dialog, shared by the Readings and
   Quizzes pickers. pasteTarget says which one is currently open. ---------- */
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
    quizCtx = { readingId: null, isDemo: false, demoId: null, isPasted: true, title: 'Your text', text: raw, sessionId: null, questions: [], answers: {}, status: 'idle', error: '', source: null, quizId: null, instructions: null };
    renderQuizWorkspace();
  }
});

/* =====================================================================
   Sidebar navigation: Readings <-> Quizzes <-> Settings
   Each nav button's data-view matches a <section id="<name>View">, so
   adding another section later just means adding another button.
   ===================================================================== */
function showView(name) {
  $$('.side-link').forEach(b => {
    const active = b.dataset.view === name;
    b.classList.toggle('active', active);
    b.setAttribute('aria-current', active ? 'page' : 'false');
  });
  $$('.view').forEach(v => { v.hidden = v.id !== name + 'View'; });
  if (name === 'quizzes') renderQuizzesView();
  if (name === 'classes') renderClassesView();
  if (name === 'history') renderHistoryView();
  if (name === 'settings') renderSettings('settingsPane', state.user);
}
$$('.side-link').forEach(b => b.addEventListener('click', () => showView(b.dataset.view)));

/* CLASSES: join a teacher's class with a code. Visible readings are
   filtered to general readings plus classes you've joined in
   getVisibleReadings() above -- that's enforced in this query, not by a
   Row Level Security policy, so it's only as strong as this code path
   (every reading fetch should go through getVisibleReadings(), never a
   raw .from('readings') select). */
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

  // Look up each class's teacher name separately -- classes and profiles
  // aren't directly linked as far as the database schema goes, so this
  // can't be fetched in the same query.
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

/* HISTORY: every past session for this student, own data only -- row-level
   security already scopes this to state.user.id, same as every other
   student-side query, so no RPC is needed here (unlike the teacher-side
   class drill-down, which has to read across students). */
async function renderHistoryView() {
  const el = $('#historyPane');
  el.innerHTML = `<h1>Your history</h1><p class="hint">Loading…</p>`;

  const { data: sessions, error } = await supa
    .from('sessions')
    .select('id, goal, confidence, final_confidence, created_at, readings(title)')
    .eq('student_id', state.user.id)
    .order('created_at', { ascending: false });

  if (error) {
    // A missing column is a setup problem, not something the student did
    // wrong, so say which migration fixes it instead of showing them the
    // raw Postgres message and leaving them stuck.
    const needsMigration = /created_at|column .* does not exist/i.test(error.message);
    el.innerHTML = `<h1>Your history</h1><p class="small" style="color:var(--bug)">Could not load your history: ${esc(error.message)}${
      needsMigration ? '. Run sessions_created_at_migration.sql in Supabase to add the missing column.' : ''
    }</p>`;
    return;
  }
  if (!sessions || !sessions.length) {
    el.innerHTML = `<h1>Your history</h1><p class="hint">Once you start a reading, your past sessions will show up here.</p>`;
    return;
  }

  // Two batched queries instead of one per session row.
  const ids = sessions.map(s => s.id);
  const [{ data: debugRows }, { data: checkinRows }] = await Promise.all([
    supa.from('debug_events').select('session_id, status').in('session_id', ids),
    supa.from('checkin_events').select('session_id, level').in('session_id', ids)
  ]);
  const statsFor = (id) => {
    const d = (debugRows || []).filter(r => r.session_id === id);
    const c = (checkinRows || []).filter(r => r.session_id === id);
    return { resolved: d.filter(r => r.status === 'resolved').length, stuck: d.filter(r => r.status === 'stuck').length, total: d.length, checkins: c.length };
  };

  const rows = sessions.map(s => {
    const st = statsFor(s.id);
    const when = s.created_at ? new Date(s.created_at).toLocaleDateString() : '';
    const title = (s.readings && s.readings.title) || 'A pasted or deleted text';
    return `<div class="reading-row-card">
      <div>
        <b>${esc(title)}</b>
        <div class="small">
          ${when ? when + ' &middot; ' : ''}${s.goal ? 'Goal: ' + esc(s.goal) + ' &middot; ' : ''}
          ${st.total} spot${st.total === 1 ? '' : 's'} debugged (${st.resolved} cleared up, ${st.stuck} still stuck)
          &middot; ${st.checkins} check-in${st.checkins === 1 ? '' : 's'}
          ${s.confidence != null && s.final_confidence != null ? ` &middot; Confidence ${s.confidence} &rarr; ${s.final_confidence}` : ''}
        </div>
      </div>
    </div>`;
  }).join('');

  el.innerHTML = `
    <h1>Your history</h1>
    <p class="hint">Every reading session you've started, most recent first.</p>
    <div style="display:flex; flex-direction:column; gap:10px; margin-top:12px">${rows}</div>`;
}

function handleLogin(profile) {
  state.user = profile; // { id, role, full_name }
  $('#userName').textContent = profile.full_name || 'Student';
  showView('readings');
  showReadingPicker();
}

/* =====================================================================
   READINGS: library grid -> reader + side panel
   ===================================================================== */
function readingCardsHtml(readings, iconSvg, opts = {}) {
  return `<div class="lib-grid">${readings.map(r => {
    const words = r.body.trim().split(/\s+/).length;
    const mins = Math.max(1, Math.round(words / 150));
    const snippet = r.body.replace(/\s+/g, ' ').trim().slice(0, 110);
    const cls = opts.demo ? 'lib-card demo' : 'lib-card';
    const attr = opts.demo ? `data-demo="${r.id}"` : `data-id="${r.id}"`;
    return `<button class="${cls}" ${attr}>
      ${opts.demo || opts.builtIn ? '<span class="lib-card-badge">Built-in</span>' : ''}
      <span class="lib-card-icon">${iconSvg}</span>
      <span class="lib-card-title">${esc(r.title)}</span>
      <span class="lib-card-snippet">${esc(snippet)}…</span>
      <span class="lib-card-meta">${mins} min read</span>
    </button>`;
  }).join('')}</div>`;
}
/* Built-in demo readings reuse the same card markup -- they're shaped
   the same as a database reading ({id, title, body}) so one function
   can render both. */
function demoReadingObjs() { return DEMO_READINGS.map(d => ({ id: d.id, title: d.title, body: d.text })); }
const BOOK_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 5.2c2.3-1 5-1 7 .3v12.8c-2-1.3-4.7-1.3-7-.3V5.2Z"/><path d="M18.5 5.2c-2.3-1-5-1-7 .3v12.8c2-1.3 4.7-1.3 7-.3V5.2Z"/></svg>`;

/* Readings a student can actually see: general readings (class_id is
   null, every teacher's "Add a reading" default) plus readings from
   classes they've joined. Shared by the reading picker and the quiz
   picker so both match the promise made on the Classes tab -- "only
   students who join can see the readings you add". */
async function getVisibleReadings() {
  if (!state.user) return { data: [], error: null };
  const { data: memberships, error: memErr } = await supa
    .from('class_members').select('class_id').eq('student_id', state.user.id);
  if (memErr) return { data: null, error: memErr };
  const classIds = (memberships || []).map(m => m.class_id);
  let query = supa.from('readings').select('id, title, body, created_at')
    .not('teacher_id', 'is', null)   // built-in readings are listed separately
    .order('created_at', { ascending: false });
  query = classIds.length ? query.or(`class_id.is.null,class_id.in.(${classIds.join(',')})`) : query.is('class_id', null);
  return await query;
}

/* The built-in practice readings, from the database rather than the copies
   below. Reading one from the database means the session is saved with a real
   reading_id, so the work can be traced to a text and a teacher can see it.
   If the database cannot be reached, the hardcoded copies are used instead
   and everything still works, which is the whole point of keeping them.
   Needs builtin_readings_migration.sql. */
async function getBuiltInReadings() {
  const { data, error } = await supa
    .from('readings').select('id, title, body')
    .is('teacher_id', null).order('created_at');
  if (error || !data || !data.length) {
    if (error) console.warn('Falling back to the built-in copies:', error.message);
    return { rows: demoReadingObjs(), fromDb: false };
  }
  return { rows: data, fromDb: true };
}

$('#changeReadingBtn').addEventListener('click', () => showReadingPicker());

async function showReadingPicker() {
  $('#readerWorkspace').hidden = true;
  const picker = $('#pickerPane');
  picker.hidden = false;
  picker.innerHTML = `<h1>Choose a reading</h1><p class="hint">Loading…</p>`;

  const { data: readings, error } = await getVisibleReadings();
  const builtIn = await getBuiltInReadings();

  const teacherGrid = (!error && readings && readings.length)
    ? readingCardsHtml(readings, BOOK_ICON)
    : `<p class="small">${error ? 'Could not load readings right now.'
        : 'No readings from your classes yet. <a href="#" id="goToClasses">Join a class</a> to see what your teacher adds.'}</p>`;

  picker.innerHTML = `
    <h1>Choose a reading</h1>
    <p class="hint">Pick a built-in practice text, something your teacher added, or paste your own.</p>
    <h3 style="margin:22px 0 0">Built-in practice readings</h3>
    ${readingCardsHtml(builtIn.rows, DEMO_READING_ICON, { demo: !builtIn.fromDb, builtIn: true })}
    <h3 style="margin:28px 0 0">From your classes</h3>
    ${teacherGrid}
    <div class="row" style="margin-top:16px">
      <button class="btn alt" id="pickPaste" type="button">Paste or upload my own text</button>
    </div>`;

  $$('.lib-card[data-id]', picker).forEach(b => b.addEventListener('click', async () => {
    b.disabled = true;
    const { data: r, error: rErr } = await supa.from('readings').select('*').eq('id', b.dataset.id).single();
    if (rErr || !r) { picker.insertAdjacentHTML('beforeend', `<p class="small" style="color:var(--bug)">Could not open that reading.</p>`); return; }
    // Both grids on this page render data-id cards, so the byline has to come
    // from the row itself. A built-in reading has no teacher_id, and telling a
    // student their teacher added it would be wrong.
    enterReader({
      title: r.title,
      byline: r.teacher_id ? 'Added by your teacher' : 'Built-in practice text',
      bodyHtml: paragraphsToHtml(r.body),
      readingId: r.id
    });
  }));
  $$('.lib-card[data-demo]', picker).forEach(b => b.addEventListener('click', () => {
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
  showGoalPopup();
}

/* =====================================================================
   QUIZZES: its own top-level view, decoupled from the reading workspace.
   Picking a reading here fetches its text directly from the database,
   so a quiz can be taken without ever opening that reading in Readings.
   ===================================================================== */
let quizCtx = { readingId: null, isDemo: false, demoId: null, isPasted: false, title: '', text: '', sessionId: null, questions: [], answers: {}, status: 'idle', error: '', source: null, quizId: null, instructions: null };

async function ensureQuizSession() {
  if (quizCtx.sessionId) return quizCtx.sessionId;
  if (!state.user) return null;
  // kind: 'quiz' keeps this out of the reading-session counts. A quiz
  // attempt has no goal, no flagged passages and no check-ins, so counting
  // it as a reading session inflates every per-session figure the teacher
  // dashboard and the evaluation rely on. Needs session_kind_migration.sql.
  const { data, error } = await supa.from('sessions').insert({
    student_id: state.user.id, reading_id: quizCtx.readingId, kind: 'quiz'
  }).select('id').single();
  if (error) { noteSaveFailure('the quiz attempt', error); return null; }
  quizCtx.sessionId = data.id;
  return data.id;
}
async function saveQuizAnswer(question, chosen, correct, questionId) {
  const sid = await ensureQuizSession(); if (!sid) return;
  const { error } = await supa.from('quiz_results').insert({
    session_id: sid, question, chosen, correct,
    quiz_id: quizCtx.quizId, question_id: questionId || null });
  if (error) noteSaveFailure('this quiz answer', error);
}

function renderQuizzesView() {
  if (quizCtx.quizId || quizCtx.readingId || quizCtx.isDemo || quizCtx.isPasted) renderQuizWorkspace();
  else renderQuizPicker();
}

const QUIZ_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="5.5" y="4.5" width="13" height="16" rx="2"/><path d="M9 4.5V3.3a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1V4.5M8.5 12l2 2 4.5-4.5"/></svg>`;

async function renderQuizPicker() {
  const el = $('#quizzesPane');
  el.innerHTML = `<h1>Quizzes</h1><p class="hint">Loading…</p>`;

  const { data: readings, error } = await getVisibleReadings();

  const assigned = await getAssignedQuizzes();

  const teacherGrid = (!error && readings && readings.length)
    ? readingCardsHtml(readings, QUIZ_ICON)
    : `<p class="small">${error ? 'Could not load readings right now.'
        : 'No readings from your classes yet. <a href="#" id="goToClassesQuiz">Join a class</a> to see what your teacher adds.'}</p>`;

  el.innerHTML = `
    <h1>Quizzes</h1>
    <p class="hint">Pick a reading to test yourself on: a built-in text, something your teacher added, or your own paste.</p>
    ${assigned.length ? `<h3 style="margin:22px 0 0">Assigned by your teacher</h3>
    <div class="quiz-list">${assigned.map(z => `
      <button class="quiz-row" data-assigned="${z.id}">
        <span class="quiz-row-icon">${QUIZ_ICON}</span>
        <span class="quiz-row-body">
          <span class="quiz-row-title">${esc(z.title)}<span class="quiz-row-badge">Assigned</span></span>
          <span class="quiz-row-meta">${z.n} question${z.n === 1 ? '' : 's'}${z.instructions ? ' &middot; ' + esc(z.instructions) : ''}</span>
        </span>
        <span class="quiz-row-cta">Start quiz &rarr;</span>
      </button>`).join('')}</div>` : ''}
    <h3 style="margin:22px 0 0">Built-in practice texts</h3>
    ${readingCardsHtml(builtIn.rows, DEMO_READING_ICON, { demo: !builtIn.fromDb, builtIn: true, kind: 'quiz' })}
    <h3 style="margin:28px 0 0">From your classes</h3>
    ${teacherGrid}
    <div class="row" style="margin-top:16px">
      <button class="btn alt" id="quizPaste" type="button">Paste or upload my own text</button>
    </div>`;

  $$('.lib-card[data-id]', el).forEach(b => b.addEventListener('click', async () => {
    b.disabled = true;
    const { data: r, error: rErr } = await supa.from('readings').select('*').eq('id', b.dataset.id).single();
    if (rErr || !r) { el.insertAdjacentHTML('beforeend', `<p class="small" style="color:var(--bug)">Could not open that reading.</p>`); return; }
    quizCtx = { readingId: r.id, isDemo: false, demoId: null, isPasted: false, title: r.title, text: r.body, sessionId: null, questions: [], answers: {}, status: 'idle', error: '', source: null, quizId: null, instructions: null };
    renderQuizWorkspace();
  }));
  $$('.lib-card[data-demo]', el).forEach(b => b.addEventListener('click', () => {
    const d = DEMO_READINGS.find(x => x.id === b.dataset.demo);
    if (!d) return;
    quizCtx = { readingId: null, isDemo: true, demoId: d.id, isPasted: false, title: d.title, text: d.text, sessionId: null, questions: [], answers: {}, status: 'idle', error: '', source: null, quizId: null, instructions: null };
    renderQuizWorkspace();
  }));
  $$('button[data-assigned]', el).forEach(b => b.addEventListener('click', async () => {
    b.disabled = true;
    await openAssignedQuiz(b.dataset.assigned);
  }));
  $('#quizPaste', el).addEventListener('click', () => openPasteDialog('quiz'));
  const classesLink = $('#goToClassesQuiz', el);
  if (classesLink) classesLink.addEventListener('click', (e) => { e.preventDefault(); showView('classes'); });
}

/* Quizzes a teacher has assigned to a class this student has joined. Row
   level security already limits the table to assigned quizzes in joined
   classes, so there is nothing to filter here. Needs
   teacher_quizzes_migration.sql; until that runs the table does not exist,
   this returns nothing, and the section simply does not appear. */
async function getAssignedQuizzes() {
  if (!state.user) return [];
  const { data, error } = await supa
    .from('quizzes')
    .select('id, title, instructions, created_at, quiz_questions(id)')
    .order('created_at', { ascending: false });
  if (error) { console.warn('Could not load assigned quizzes:', error.message); return []; }
  return (data || [])
    .map(z => ({ id: z.id, title: z.title, instructions: z.instructions,
                 n: (z.quiz_questions || []).length }))
    .filter(z => z.n > 0);
}

/* Answers this student already gave on the assigned quiz now open. Only the
   earliest answer to each question is kept, the same rule the teacher's
   figures use, so what the student sees and what the teacher sees agree.
   Needs quiz_stats_migration.sql for created_at and question_id; without it
   this returns nothing and the quiz just opens blank. */
async function loadPreviousAnswers() {
  if (!quizCtx.quizId || !state.user) return;
  const { data, error } = await supa
    .from('quiz_results')
    .select('question_id, chosen, created_at, sessions!inner(student_id)')
    .eq('quiz_id', quizCtx.quizId)
    .eq('sessions.student_id', state.user.id)
    .order('created_at', { ascending: true });
  if (error) { console.warn('Could not load earlier answers:', error.message); return; }

  const firstFor = {};
  (data || []).forEach(row => {
    if (row.question_id && !(row.question_id in firstFor)) firstFor[row.question_id] = row.chosen;
  });
  quizCtx.questions.forEach((q, i) => {
    if (!(q.id in firstFor)) return;
    const j = q.options.findIndex(o => o === firstFor[q.id]);
    if (j > -1) quizCtx.answers[i] = j;
  });
}

/* An assigned quiz is answered exactly as the teacher assigned it, so its
   questions are loaded rather than generated. Every student in the class
   answers the same set, which is what makes their scores comparable. */
async function openAssignedQuiz(quizId) {
  const el = $('#quizzesPane');
  el.innerHTML = `<h1>Quizzes</h1><p class="hint">Opening the quiz…</p>`;
  const [{ data: quiz, error: zErr }, { data: rows, error: qErr }] = await Promise.all([
    supa.from('quizzes').select('id, title, instructions, reading_id').eq('id', quizId).single(),
    supa.from('quiz_questions').select('id, position, kind, q, options, answer, why, evidence')
        .eq('quiz_id', quizId).order('position')
  ]);
  if (zErr || qErr || !quiz || !rows || !rows.length) {
    const e = zErr || qErr;
    el.innerHTML = `<h1>Quizzes</h1><p class="small" style="color:var(--bug)">Could not open that quiz${
      e ? ': ' + esc(e.message) : '.'}</p>`;
    return;
  }
  quizCtx = {
    readingId: quiz.reading_id, isDemo: false, demoId: null, isPasted: false,
    title: quiz.title, text: '', sessionId: null,
    questions: rows.map(r => ({ id: r.id, kind: r.kind, q: r.q, options: r.options,
                                answer: r.answer, why: r.why, evidence: r.evidence })),
    answers: {}, status: 'ready', error: '', source: 'teacher',
    quizId: quiz.id, instructions: quiz.instructions
  };
  // Show what they answered before rather than a blank quiz. Only the first
  // answer to each question counts, here and in the teacher's figures, so
  // re-answering would only mislead them about their own score.
  await loadPreviousAnswers();
  renderQuizWorkspace();
}

/* How many questions a passage can carry. The same rule lives in teacher.html
   and in main.py as a backstop, so all three agree: one question per 25 words,
   in steps of five, never fewer than 5 or more than 20. A 150-word built-in
   reading therefore offers 5 and nothing else.

   The ceiling is not arbitrary. The model writes the whole quiz as one piece
   of JSON, so a long one is likelier to be cut off mid-string, and the parse
   is all or nothing: a truncated answer loses the entire quiz rather than the
   last question. Short passages also run out of material for believable wrong
   options, which is the failure that looks like a working quiz. */
const WORDS_PER_QUESTION = 25;
const QUESTION_COUNTS = [5, 10, 15, 20];

function countWords(text) {
  return (text || '').trim().split(/\s+/).filter(Boolean).length;
}
function questionsSupported(words) {
  return Math.max(5, Math.min(QUESTION_COUNTS[QUESTION_COUNTS.length - 1],
                              Math.floor(Math.floor(words / WORDS_PER_QUESTION) / 5) * 5));
}

/* What the student picked on the "Make my quiz" screen. Kept outside quizCtx
   so switching readings does not silently reset their choice. */
const quizPrefs = { n: 5, kinds: 'mixed' };

async function loadQuizQuestions() {
  quizCtx.status = 'loading'; quizCtx.error = ''; quizCtx.answers = {};
  renderQuizWorkspace();
  const supported = questionsSupported(countWords(quizCtx.text));
  const d = await callBackend('/quiz', {
    text: quizCtx.text,
    goal: null,
    n: Math.min(quizPrefs.n, supported),
    kinds: quizPrefs.kinds,
    // Follows the coach-language setting they already chose in Settings rather
    // than asking the same question again here. 'auto' means follow the text.
    language: state.responseLanguage === 'auto' ? null : state.responseLanguage
  });
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
    const words = countWords(quizCtx.text);
    const supported = questionsSupported(words);
    if (quizPrefs.n > supported) quizPrefs.n = supported;
    el.innerHTML = `
      ${back}<h1>${esc(quizCtx.title)}</h1>
      <p class="hint">Make a short quiz based on this exact text.</p>
      ${quizCtx.status === 'error' ? `<div class="card"><b>Couldn't make the quiz</b><p style="margin:6px 0 0">${esc(quizCtx.error)}</p></div>` : ''}

      <label class="f" for="myQuizN">How many questions</label>
      <select id="myQuizN">${QUESTION_COUNTS.map(n => `<option value="${n}"${
        n === quizPrefs.n ? ' selected' : ''}${n > supported ? ' disabled' : ''}>${n} questions${
        n > supported ? ' (text is too short)' : ''}</option>`).join('')}</select>

      <label class="f" for="myQuizKinds">What kind of questions</label>
      <select id="myQuizKinds">
        <option value="mixed"${quizPrefs.kinds === 'mixed' ? ' selected' : ''}>A mix of both</option>
        <option value="literal"${quizPrefs.kinds === 'literal' ? ' selected' : ''}>Answers stated in the text</option>
        <option value="inferential"${quizPrefs.kinds === 'inferential' ? ' selected' : ''}>Answers you have to work out</option>
      </select>
      <p class="small" style="margin:6px 0 0">This text is ${words} words, so it can carry up to ${supported} question${supported === 1 ? '' : 's'}. Questions come out in the language of the reading; the answer choices follow your language setting.</p>

      <div class="row" style="margin-top:14px"><button class="btn" id="makeQuiz" type="button">${quizCtx.status === 'error' ? 'Try again' : 'Make my quiz'}</button></div>`;
    wireQuizBack(el);
    const nSel = $('#myQuizN', el), kSel = $('#myQuizKinds', el);
    nSel.addEventListener('change', () => { quizPrefs.n = +nSel.value || 5; });
    kSel.addEventListener('change', () => { quizPrefs.kinds = kSel.value; });
    $('#makeQuiz', el).addEventListener('click', loadQuizQuestions);
    return;
  }

  const qs = quizCtx.questions;
  const done = Object.keys(quizCtx.answers).length;
  const score = Object.entries(quizCtx.answers).filter(([i, a]) => a === qs[i].answer).length;
  el.innerHTML = `
    ${back}<h1>${esc(quizCtx.title)}</h1>
    <p class="hint">${done ? `${score} of ${done} correct so far.` : 'Pick the best answer for each question.'}${quizCtx.source === 'demo' ? ' (Offline demo quiz)' : ''}${quizCtx.quizId ? ' Assigned by your teacher.' : ''}</p>
    ${quizCtx.instructions ? `<p class="small" style="margin:0 0 10px">${esc(quizCtx.instructions)}</p>` : ''}
    ${qs.map((q, i) => {
      const a = quizCtx.answers[i]; const answered = a !== undefined;
      return `<div class="card">
        <span class="tag">${esc(q.kind || 'Question')}</span>
        <div><b>${esc(q.q)}</b></div>
        ${q.options.map((o, j) => `<button class="opt ${answered ? (j === q.answer ? 'right' : j === a ? 'wrong' : '') : ''}" data-q="${i}" data-o="${j}" ${answered ? 'disabled' : ''}>${esc(o)}</button>`).join('')}
        ${answered ? `<div class="explain">${esc(q.why || '')}${q.evidence ? ` Look at: “${esc(q.evidence)}”` : ''}</div>` : ''}
      </div>`;
    }).join('')}
    ${quizCtx.quizId ? '' : '<div class="row"><button class="btn alt" id="newQuiz" type="button">Make new questions</button></div>'}`;
  wireQuizBack(el);
  $$('.opt', el).forEach(b => b.addEventListener('click', () => {
    const i = +b.dataset.q, j = +b.dataset.o;
    quizCtx.answers[i] = j;
    const correct = j === qs[i].answer;
    saveQuizAnswer(qs[i].q, qs[i].options[j], correct, qs[i].id || null);
    renderQuizWorkspace();
  }));
  const again = $('#newQuiz', el);
  if (again) again.addEventListener('click', loadQuizQuestions);
}
function wireQuizBack(el) {
  $('#quizBack', el).addEventListener('click', () => {
    quizCtx = { readingId: null, isDemo: false, demoId: null, isPasted: false, title: '', text: '', sessionId: null, questions: [], answers: {}, status: 'idle', error: '', source: null, quizId: null, instructions: null };
    renderQuizzesView();
  });
}
