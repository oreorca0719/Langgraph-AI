// /static/js/chat.js

const chatBox   = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn   = document.getElementById('send-btn');
const attachBtn = document.getElementById('attach-btn');
const fileInput = document.getElementById('file-input');
const fileBadge = document.getElementById('file-badge');

let pendingFile = null;

// ── 파일 첨부 ──────────────────────────────────────────
attachBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => {
  const f = fileInput.files[0];
  if (!f) return;
  pendingFile = f;
  fileBadge.textContent = f.name + ' ✕';
  fileBadge.style.display = 'inline-flex';
  fileInput.value = '';
});

fileBadge.addEventListener('click', () => {
  pendingFile = null;
  fileBadge.style.display = 'none';
  fileBadge.textContent = '';
});

// ── 유틸 ───────────────────────────────────────────────
function escapeHtml(str) {
  return (str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function escapeTilde(text) {
  return (text ?? '').replace(/~/g, '\\~');
}

function renderSources(sources) {
  if (!Array.isArray(sources) || sources.length === 0) return '';
  const items = sources.map((s) => {
    const id      = s?.id ?? '';
    const title   = escapeHtml(s?.title || `출처 ${id}`);
    const snippet = escapeHtml(s?.snippet || '');
    let metaRaw   = (s?.path || '').toString();
    if (
      metaRaw.startsWith('/app/') || metaRaw.startsWith('/') ||
      metaRaw.includes(':\\') || metaRaw.includes(':/')
    ) {
      metaRaw = '';
    }
    const meta = escapeHtml(metaRaw);
    return `
      <div class="source-item">
        <div class="source-title">[${id}] ${title}</div>
        ${meta    ? `<div class="source-meta">${meta}</div>`       : ''}
        ${snippet ? `<div class="source-snippet">${snippet}</div>` : ''}
      </div>`;
  }).join('');
  return `<details class="sources"><summary>출처 보기</summary>${items}</details>`;
}

function rfpDraftToMarkdown(draft) {
  let body = draft == null ? '' : typeof draft === 'string' ? draft
    : Array.isArray(draft)
      ? draft.map(x => (x && typeof x === 'object' ? (x.text ?? '') : String(x ?? ''))).join('\n').trim()
      : JSON.stringify(draft, null, 2);
  return ['### RFP 초안', '', '---', '', body.trim() || '(본문 없음)'].join('\n');
}

function emailDraftToMarkdown(draft) {
  const d = draft || {};
  let to = (d.to ?? '').toString();
  let cc = (d.cc ?? '').toString();
  let subject = (d.subject ?? '').toString();
  let body = d.body;

  if (Array.isArray(body)) {
    const t = body
      .map(x => (x && typeof x === 'object' ? (x.text ?? '') : String(x ?? '')))
      .join('\n').trim();
    if (t.startsWith('{') && t.endsWith('}')) {
      try {
        const n = JSON.parse(t);
        to = (n.to ?? to).toString();
        cc = (n.cc ?? cc).toString();
        subject = (n.subject ?? subject).toString();
        body = (n.body ?? '').toString();
      } catch { body = t; }
    } else { body = t; }
  } else if (body && typeof body === 'object') {
    body = JSON.stringify(body, null, 2);
  } else {
    body = (body ?? '').toString();
  }

  return [
    '### 이메일 초안', '',
    `- **To**: ${to || '(미지정)'}`,
    `- **CC**: ${cc || '(없음)'}`,
    `- **Subject**: ${subject || '(제목 없음)'}`,
    '', '---', '',
    body || '(본문 없음)',
  ].join('\n');
}

// ── 메시지 렌더 헬퍼 ──────────────────────────────────
function addLoading() {
  const id = 'loading-' + Date.now();
  chatBox.innerHTML += `
    <div id="${id}" class="message bot">
      <div class="bot-avatar"></div>
      <div class="content">응답 생성 중입니다...</div>
    </div>`;
  chatBox.scrollTop = chatBox.scrollHeight;
  return id;
}

function removeLoading(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function appendBot(htmlContent) {
  chatBox.innerHTML += `
    <div class="message bot">
      <div class="bot-avatar"></div>
      <div class="content">${htmlContent}</div>
    </div>`;
  chatBox.scrollTop = chatBox.scrollHeight;
}

function showError(loadingId, msg) {
  const el = document.getElementById(loadingId);
  if (el) el.querySelector('.content').textContent = '오류: ' + msg;
  chatBox.scrollTop = chatBox.scrollHeight;
}

// ── 전송 메인 ─────────────────────────────────────────
async function send() {
  const message = userInput.value.trim();
  const hasFile = !!pendingFile;
  if (!message && !hasFile) return;

  // 사용자 메시지 표시
  if (hasFile && message) {
    chatBox.innerHTML += `<div class="message user">${escapeHtml(message)}<br><span class="file-label">📎 ${escapeHtml(pendingFile.name)}</span></div>`;
  } else if (hasFile) {
    chatBox.innerHTML += `<div class="message user"><span class="file-label">📎 ${escapeHtml(pendingFile.name)}</span></div>`;
  } else {
    chatBox.innerHTML += `<div class="message user">${escapeHtml(message)}</div>`;
  }
  userInput.value = '';

  const loadingId = addLoading();

  // ── 파일 포함 요청 ───────────────────────────────────
  if (hasFile) {
    const file = pendingFile;
    pendingFile = null;
    fileBadge.style.display = 'none';
    fileBadge.textContent = '';

    try {
      const form = new FormData();
      form.append('file', file);

      let endpoint = '/upload';
      if (message) {
        form.append('message', message);
        endpoint = '/chat-with-file';
      }

      const res = await fetch(endpoint, { method: 'POST', body: form });

      if (res.redirected) { window.location.href = res.url; return; }

      const data = await res.json();
      removeLoading(loadingId);

      if (!res.ok) throw new Error(data?.detail || '처리 실패');

      let mdText;
      if (data.type === 'file_qa') {
        // 파일 + 질문 → LLM 답변
        mdText = data.answer || '';
      } else if (data.summary) {
        // 파일만 → LLM 요약 성공
        mdText = `### 📎 ${escapeHtml(file.name)}\n\n${data.summary}`;
      } else {
        // 파일만 → LLM 요약 실패 시 원문 fallback
        mdText = `### 📎 ${escapeHtml(file.name)} 추출 결과\n\n` +
          (data.text ? data.text : '*(추출된 텍스트 없음)*');
      }

      appendBot(marked.parse(escapeTilde(mdText)));
    } catch (e) {
      showError(loadingId, e?.message || String(e));
    }
    return;
  }

  // ── 텍스트 전용 → /chat ───────────────────────────────
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });

    if (res.redirected) { window.location.href = res.url; return; }

    const contentType = (res.headers.get('content-type') || '').toLowerCase();
    if (!contentType.includes('application/json')) {
      const text = await res.text();
      throw new Error('서버가 JSON이 아닌 응답을 반환했습니다.\n' + text.slice(0, 300));
    }

    const data = await res.json();
    removeLoading(loadingId);

    if (!res.ok) throw new Error(data?.detail || data?.message || JSON.stringify(data));

    let markdownText = '';
    if (data.type === 'email_draft') {
      markdownText = emailDraftToMarkdown(data.draft);
    } else if (data.type === 'rfp_draft') {
      const answer = (data?.answer ?? '').toString().trim();
      markdownText = (answer ? answer + '\n\n' : '') + rfpDraftToMarkdown(data.draft);
    } else if (data.type === 'file_extract') {
      markdownText =
        '### 파일 추출 결과\n\n' +
        (data.text || '(추출 텍스트 없음)') +
        '\n\n```json\n' + JSON.stringify(data.meta ?? {}, null, 2) + '\n```';
    } else {
      markdownText = data?.answer ?? '';
    }

    const sourcesHtml = renderSources(data?.sources);
    appendBot(marked.parse(escapeTilde(markdownText)) + sourcesHtml);
  } catch (e) {
    showError(loadingId, e?.message || String(e));
  }
}

sendBtn.addEventListener('click', send);
userInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') send();
});