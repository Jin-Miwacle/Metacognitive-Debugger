# Metacognitive Debugger: AI Reading Partner

A web-based reading helper for students. While a student reads, a small floating window (the "widget") helps them notice when they stop understanding a text and guides them to fix it.

This is our thesis project (Reading Comprehension track). This README explains **what we have so far** and **what we still need to do**.

---

## 1. The idea in simple words

Many students read a whole page without noticing they didn't understand it. Our system acts like a **debugger for reading**, the way programmers use a debugger to find bugs in code:

- The student **sets a goal** before reading.
- The student **marks confusing parts** ("debug" them) and gets hints.
- The system **checks in** every so often ("What have you read so far?").
- The student answers a **short quiz** at the end.
- The teacher can see **where students got stuck**.

**Why it's different from ChatGPT or Quizizz:** ChatGPT only answers when asked and doesn't know which part of the text you're stuck on. Quizizz gives the same quiz to everyone. Ours works alongside the reader, and it is built on our own fine-tuned model.

---

## 2. What we have right now

One file: **`index.html`**. It is a working **design prototype** of the widget. Open it in a browser and try it.

### What the widget does

| Tab | What it does |
|---|---|
| **Goal** | Student writes what they want from the text and rates how well they expect to understand it. |
| **Debug** | Student selects a word, phrase, or sentence. It gets a red wavy underline. They choose what's wrong (unknown word, lost the point, ideas don't connect, why did this happen), get a hint, reply, then mark it "Makes sense now" (green) or "Still stuck" (flagged for the teacher). |
| **Check-in** | Timed prompts while reading, with a "how is it going?" rating. Can be turned off or set to a different timing. |
| **Quiz** | A practice quiz with literal and inference questions. |
| **Progress** | Session numbers and a "Download session data" button (JSON file). |

Other things it can/will do:

- Drag it around, resize it, minimize it, or close it to a small button.
- Light and dark theme.
- "Use my own text" button to paste any passage.

### What is still fake (hard-coded)

This is a **prototype**, so some parts are placeholders:

- The partner's hints are **pre-written examples**, not a real AI.
- The quiz has **3 fixed questions** about one demo story ("The Lamp on Harrow Point").
- The check-in prompts come from a fixed list.
- There are **no accounts, no saved progress, and no teacher dashboard** yet.

---

## 3. How to run it

1. Download `index.html`.
2. Double-click it. It opens in your browser.

## 4. How to put it online with GitHub Pages

1. Create a new repository on github.com.
2. Upload `index.html` (and this README).
3. Go to **Settings, then Pages**.
4. Choose the **main** branch and click **Save**.
5. After about a minute, GitHub shows a link. Anyone can open it.

---

## 5. How the full system will fit together

```
Student's browser            Our server                    AI
+----------------+  ask   +----------------+  ask   +-------------------+
|  Widget        | -----> |  Backend       | -----> |  Fine-tuned model |
|  (index.html)  | <----- |  (FastAPI)     | <----- |                   |
+----------------+ answer +-------+--------+ answer +-------------------+
                                   |
                                   v
                             +------------+
                             |  Database  |  accounts, texts, progress
                             +------------+
```

- **Frontend (screen):** the widget. Shows things and takes clicks.
- **Backend (brain):** a Python program (FastAPI) that receives requests, calls the AI, and talks to the database.
- **Database:** stores accounts, readings, and student progress.
- **AI model:** our fine-tuned open-source model.

### The plug-in point

At the top of the script in `index.html` there is a setting:

```js
const CONFIG = {
  API_URL: null,            // put the backend address here later
  CHECKIN_EVERY_SEC: 90
};
```

When `API_URL` is set, the widget sends this to the backend:

```json
{
  "mode": "debug",
  "problem": "vocab",
  "passage": "the words the student selected",
  "paragraph": "the paragraph around it",
  "goal": "the student's goal",
  "message": "the student's reply (for follow-ups)"
}
```

`mode` can be `"debug"`, `"followup"`, or `"stuck"`. `problem` can be `"vocab"`, `"lost"`, `"connect"`, or `"why"`.

and expects this back:

```json
{ "reply": "what the partner should say" }
```

If the backend can't be reached, the widget falls back to the demo replies, so it never breaks.

### What replaces each hard-coded part

| Hard-coded now | Will be replaced by |
|---|---|
| Demo story | Readings saved in the database (teacher adds them) |
| Pre-written hints | Replies from our fine-tuned model |
| 3 fixed quiz questions | Question generator (AQG module) |
| Fixed check-in prompts | Model-generated or teacher-approved prompts |
| Download-only progress | Progress saved automatically in the database |
| No login | Student and teacher accounts |

---

## 6. What we need to do next

Work in this order. Each step leaves us with something that still works.

### Build steps

- [x] **Step 1: Widget design prototype** (`index.html`)
- [ ] **Step 2: Tiny backend with a fake reply.** One address (`/partner`) that returns a simple message. Connect the widget to it. This proves the connection works.
- [ ] **Step 3: Connect a regular AI model** (before our own). Now the hints are real.
- [ ] **Step 4: Database and logins.** Student and teacher accounts. Save readings and progress.
- [ ] **Step 5: Question generator.** Make quiz questions from any text.
- [ ] **Step 6: Teacher dashboard.** Show where students got stuck.
- [ ] **Step 7: Swap in our fine-tuned model.** The widget already knows how to talk to the backend, so nothing else changes.
- [ ] **Step 8: Testing.** Confusion matrix, accuracy/F1, ISO 25010 checklist, small student pilot.
- [ ] **Step 9: Fixes from the pilot, documentation, defense preparation.**

### AI / model track (runs at the same time as the build steps)

- [ ] Decide exactly what the fine-tuned model does. Suggested: (a) spot which sentences are likely confusing, and (b) write the hint or check-question.
- [ ] Collect data: open datasets (for example FairytaleQA) plus samples labeled by our partner teacher.
- [ ] Write a short labeling guide so labels are consistent. Have two people label some of the same samples.
- [ ] Test a regular (not fine-tuned) model first. This is our "before" number.
- [ ] Fine-tune a small open-source model (LoRA/QLoRA on Colab or a school GPU).
- [ ] Compare "before" and "after" on a test set the model never trained on.
- [ ] Decide where the model will run for the demo (school GPU, cloud server, or a smaller model on CPU).

### Non-coding tasks (start these now)

- [ ] Present the scoped plan to the panel for approval.
- [ ] Prepare consent and ethics forms for students and teachers.
- [ ] Confirm a partner teacher and class for labeling, interviews, and the pilot.
- [ ] Assign roles: ML/fine-tuning, frontend, backend, QA/documentation.
- [ ] Check GPU access and the school's thesis IP rules.
- [ ] Keep student data anonymous (Philippine Data Privacy Act of 2012).

---

## 7. Panel requirements checklist

- [ ] Interviews or consultations with students and teachers
- [ ] Student and teacher accounts (admin optional)
- [ ] Custom fine-tuned LLM (not just an off-the-shelf model)
- [ ] Formal testing: confusion matrix and accuracy
- [ ] ISO 25010 compliance

---

## 8. Planned tech

| Part | Choice |
|---|---|
| Screen | Plain HTML/CSS/JS now, React + TailwindCSS later |
| Backend | Python (FastAPI or Flask) |
| Database | Firebase or PostgreSQL |
| AI | Small open-source LLM, fine-tuned; spaCy / Sentence-Transformers for light tasks |
| Hosting | GitHub Pages for the screen; free tiers (Render, Supabase, etc.) for backend and database |

## 9. To do

| Lock scope with panel, start consent/ethics paperwork |
| Teacher interviews, data labeling, build the dataset |
| Fine-tune the model, build core reading screen and question module |
| Accounts, teacher dashboard, progress tracking |
| Testing (confusion matrix, F1, ISO 25010) and student pilot |
| Fixes, documentation, defense |

## 10. Out of scope for now

- Eye-tracking (GCAS)
- Full story/character map (Angel-StoryMapper)
- Native mobile app
- Optional extra if time allows: SQ3R-style question gating

## 11. Suggested folder layout (when the project grows)

```
/frontend    the widget (index.html now, React later)
/backend     FastAPI server
/ml          datasets, training notebooks, evaluation scripts
/docs        study, interview notes, test results
README.md
```

## 12. Team

| Name | Role |
|---|---|
| (Gamit/Tonguia/Velasquez) | ML / fine-tuning |
| (Gamit) | Frontend |
| (Gamit/Tonguia/Velasquez) | Backend |
| (Repalda/Dela Cruz) | QA / documentation |
