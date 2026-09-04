/* Diary Companion UI — JSON API client with token auth. No build step. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const chatlog = $("#chatlog");
const form = $("#chatform");
const input = $("#message");
const sendBtn = $("#sendbtn");
const logstatus = $("#logstatus");

let sessionCounter = 0; // index of the last exchange (for re-log)

/* ---------------- auth ---------------- */

const TOKEN_KEY = "diary_token";

function getToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

function saveToken(t) {
  localStorage.setItem(TOKEN_KEY, t);
}

function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

function showLock(msg) {
  $("#layout").classList.add("hidden");
  $("#lock").classList.remove("hidden");
  const err = $("#lockerror");
  if (msg) {
    err.textContent = msg;
    err.classList.remove("hidden");
  } else {
    err.classList.add("hidden");
  }
  $("#tokeninput").value = "";
  setTimeout(() => $("#tokeninput").focus(), 50);
}

function showApp() {
  $("#lock").classList.add("hidden");
  $("#layout").classList.remove("hidden");
  refreshSidebar();
}

async function authedFetch(url, opts = {}) {
  const headers = Object.assign({ "Content-Type": "application/json" }, opts.headers || {});
  const t = getToken();
  if (t) headers["Authorization"] = "Bearer " + t;
  const res = await fetch(url, Object.assign({}, opts, { headers }));
  if (res.status === 401) {
    const needAuth = await fetch("/api/health", { headers: { "Authorization": "Bearer x" } })
      .then((r) => r.status === 401)
      .catch(() => false);
    if (needAuth) {
      showLock("Session expired — enter your token again.");
      throw new Error("unauthorized");
    }
  }
  return res;
}

$("#lockform").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const t = $("#tokeninput").value.trim();
  if (!t) return;
  const res = await fetch("/api/health", { headers: { "Authorization": "Bearer " + t } }).catch(() => null);
  if (res && res.status === 200) {
    saveToken(t);
    showApp();
  } else if (res && res.status === 401) {
    showLock("Wrong token — try again.");
  } else {
    showLock("Server unreachable — check the address and try again.");
  }
});

$("#lockbtn").addEventListener("click", () => {
  clearToken();
  showLock();
});

/* ---------------- chat ---------------- */

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

function addMsg(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  chatlog.appendChild(el);
  chatlog.scrollTop = chatlog.scrollHeight;
  return el;
}

function statusHTML(decision, reason, index) {
  const tags = {
    logged: '<span class="tag logged">✓ logged</span>',
    skipped: '<span class="tag skipped">⊘ skipped</span>',
    error: '<span class="tag error">⚠ log error</span>',
  };
  const actions =
    decision === "skipped" ? ' <button class="relog" data-index="' + index + '">log anyway</button>' :
    decision === "error" ? ' <button class="relog" data-index="' + index + '">retry</button>' : "";
  const reasonTxt = reason && decision === "error" ? " " + esc(reason) : "";
  return (tags[decision] || tags.error) + reasonTxt + actions;
}

function addStatus(decision, reason, index) {
  const el = document.createElement("div");
  el.className = "status";
  el.dataset.index = String(index);
  el.innerHTML = statusHTML(decision, reason, index);
  const btn = el.querySelector("button.relog");
  if (btn) {
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      btn.textContent = "…";
      try {
        const res = await authedFetch("/api/relog", {
          method: "POST",
          body: JSON.stringify({ index, session_id: "default" }),
        });
        if (res.status === 401) return;
        const data = await res.json();
        el.innerHTML = statusHTML(data.decision, data.reason || "", index);
        bindRelog(el);
        refreshSidebar();
      } catch (e) {
        if (e.message !== "unauthorized") btn.textContent = "retry failed";
      }
    });
  }
  chatlog.appendChild(el);
  return el;
}

function bindRelog(el) {
  const btn = el.querySelector("button.relog");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "…";
    try {
      const res = await authedFetch("/api/relog", {
        method: "POST",
        body: JSON.stringify({ index: el.dataset.index, session_id: "default" }),
      });
      if (res.status === 401) return;
      const data = await res.json();
      el.innerHTML = statusHTML(data.decision, data.reason || "", el.dataset.index);
      bindRelog(el);
    } catch (e) {
      if (e.message !== "unauthorized") btn.textContent = "retry failed";
    }
  });
}

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  sendBtn.disabled = true;
  logstatus.textContent = "thinking…";
  addMsg("user", text);
  try {
    const res = await authedFetch("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message: text, session_id: "default" }),
    });
    if (res.status === 401) return;
    const data = await res.json();
    if (!res.ok) {
      addMsg("error", data.error || "request failed");
      return;
    }
    addMsg("companion", data.reply);
    sessionCounter += 1;
    addStatus(data.decision, data.reason, sessionCounter - 1);
    logstatus.textContent = "";
    refreshSidebar();
  } catch (e) {
    if (e.message !== "unauthorized") addMsg("error", "network error: " + e.message);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});

async function refreshSidebar() {
  try {
    const [dayRes, healthRes] = await Promise.all([
      authedFetch("/api/day").then((r) => (r.ok ? r.json() : null)),
      authedFetch("/api/health").then((r) => (r.ok ? r.json() : null)),
    ]);
    if (!dayRes || !healthRes) return;
    $("#todaylog").textContent = dayRes.today_log || "(nothing logged yet today)";
    $("#standing").textContent = dayRes.standing || "(no standing sections yet)";
    $("#health").textContent =
      `model: ${healthRes.model}\njournal pending: ${healthRes.journal_pending}\nchunks indexed: ${healthRes.retrieval.chunks}\nvec available: ${healthRes.retrieval.vec_available}`;
  } catch (e) {
    if (e.message !== "unauthorized") {
      $("#health").textContent = "sidebar refresh failed: " + e.message;
    }
  }
}

/* ---------------- boot ---------------- */

(async function boot() {
  const t = getToken();
  if (!t) {
    // Probe: is auth even required?
    const res = await fetch("/api/health").catch(() => null);
    if (res && res.status === 200) {
      showApp(); // open mode (no token configured server-side)
    } else {
      showLock();
    }
    return;
  }
  const res = await fetch("/api/health", { headers: { "Authorization": "Bearer " + t } }).catch(() => null);
  if (res && res.status === 200) {
    showApp();
  } else if (res && res.status === 401) {
    showLock();
  } else {
    showLock("Server unreachable — check the address and try again.");
  }
})();
